"""Input-gradient providers for white-box attacks.

sklearn's MLPClassifier exposes no input gradients, so the forward/backward pass
is re-implemented here from its fitted coefs_/intercepts_ (ReLU hidden layers +
softmax/logistic output). Verified against numerical differentiation in
tests/test_attacks.py. The MLP also serves as the surrogate model for
transfer attacks on non-differentiable families (trees, kernel SVMs).
"""

from __future__ import annotations

import numpy as np
from sklearn.neural_network import MLPClassifier


def mlp_forward(mlp: MLPClassifier, X: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
    """Forward pass; returns (activations per layer, output probabilities)."""
    if mlp.activation != "relu" or mlp.out_activation_ not in ("softmax", "logistic"):
        raise ValueError("gradient path implemented for relu hidden + softmax/logistic output")
    acts = [X]
    a = X
    n_layers = len(mlp.coefs_)
    for k in range(n_layers - 1):
        a = np.maximum(0.0, a @ mlp.coefs_[k] + mlp.intercepts_[k])
        acts.append(a)
    z = a @ mlp.coefs_[-1] + mlp.intercepts_[-1]
    if mlp.out_activation_ == "softmax":
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        proba = e / e.sum(axis=1, keepdims=True)
    else:  # logistic, single output unit
        p1 = 1.0 / (1.0 + np.exp(-z[:, 0]))
        proba = np.column_stack([1 - p1, p1])
    return acts, proba


def mlp_loss_input_grad(mlp: MLPClassifier, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """d(cross-entropy)/dX for the fitted sklearn MLP."""
    acts, proba = mlp_forward(mlp, X)
    if mlp.out_activation_ == "softmax":
        delta = proba.copy()
        delta[np.arange(len(y)), y] -= 1.0  # dCE/dz for softmax
    else:
        delta = (proba[:, 1] - y).reshape(-1, 1)  # dCE/dz for logistic
    for k in range(len(mlp.coefs_) - 1, 0, -1):
        delta = (delta @ mlp.coefs_[k].T) * (acts[k] > 0)
    return delta @ mlp.coefs_[0].T


def rbf_svm_loss_input_grad(svc, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """d(BCE(sigmoid(f), y))/dX for a fitted binary RBF-kernel SVC.

    The CLASSICAL-KERNEL CONTROL for the quantum-kernel white-box attack
    (attacks/qsvm_whitebox.py): without it, "the fidelity kernel collapses under
    a targeted attack" has no matched-strength classical baseline, and the
    comparison reduces to attack availability. Both are kernel machines
    f(x) = sum_i alpha_i k(x, x_i) + b, so the only difference under test is k.

    For the RBF kernel k(x, x_i) = exp(-gamma ||x - x_i||^2),
        df/dx = sum_i alpha_i k(x, x_i) * (-2 gamma) (x - x_i),
    and the zoo derives probabilities as sigmoid(f), so dL/df = sigmoid(f) - y.
    Verified against central finite differences in tests/test_attacks.py.
    """
    if len(getattr(svc, "classes_", [])) != 2:
        raise ValueError("rbf_svm_loss_input_grad implements the binary case only")
    sv = svc.support_vectors_
    alpha = svc.dual_coef_[0]
    gamma = svc._gamma if hasattr(svc, "_gamma") else svc.gamma
    d2 = ((X[:, None, :] - sv[None, :, :]) ** 2).sum(axis=2)  # (B, n_sv)
    Kx = np.exp(-gamma * d2)
    f = Kx @ alpha + svc.intercept_[0]
    dL_df = 1.0 / (1.0 + np.exp(-f)) - y  # sigmoid(f) - y
    w = (alpha[None, :] * Kx) * (-2.0 * gamma)  # (B, n_sv) per-SV coefficient
    # df/dx = sum_i w_i (x - x_i)  ->  (sum_i w_i) x - w @ sv
    df_dx = w.sum(axis=1)[:, None] * X - w @ sv
    return dL_df[:, None] * df_dx


def rbf_svm_decision_input_grad(svc, X: np.ndarray) -> np.ndarray:
    """df/dX for a fitted binary RBF-kernel SVC — the loss-free half of the gradient.

    Factored out of `rbf_svm_loss_input_grad` so the margin objective below can reuse the
    identical, hand-derived df/dx and differ from BCE in exactly one factor (dL/df). Any
    difference the objective comparison reports is then the objective, not two derivations.
    """
    if len(getattr(svc, "classes_", [])) != 2:
        raise ValueError("rbf_svm_decision_input_grad implements the binary case only")
    sv = svc.support_vectors_
    alpha = svc.dual_coef_[0]
    gamma = svc._gamma if hasattr(svc, "_gamma") else svc.gamma
    Kx = np.exp(-gamma * ((X[:, None, :] - sv[None, :, :]) ** 2).sum(axis=2))
    w = (alpha[None, :] * Kx) * (-2.0 * gamma)
    return w.sum(axis=1)[:, None] * X - w @ sv


def rbf_svm_margin_input_grad(svc, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """d(margin loss)/dX for a fitted binary RBF-kernel SVC.

    THE OBJECTIVE CONTROL (P14a). Every attack in this benchmark ascends BCE, whose
    dL/df = sigmoid(f) - y VANISHES on confidently-correct points. Kernel machines fit at
    large C carry large |f|, so that factor can underflow and leave those points effectively
    unattacked — the failure mode the CW attack was introduced to fix. The margin loss

        L(x) = -(2y - 1) f(x)

    has dL/df = -+1 identically, so it cannot saturate. Only the dL/df factor differs from
    `rbf_svm_loss_input_grad`; df/dx is the same hand-derived expression.
    """
    return (-(2.0 * np.asarray(y, dtype=np.float64) - 1.0))[:, None] * (
        rbf_svm_decision_input_grad(svc, X)
    )


class GradProvider:
    """Uniform loss_input_grad interface over attackable models.

    `objective` selects which loss the returned gradient ascends: "bce" (the benchmark's
    standard, used for every published number) or "margin" (the non-saturating control).
    """

    def __init__(self, model, kind: str, objective: str = "bce"):
        if kind not in ("mlp", "vqc", "rbf_svm"):
            raise ValueError(f"no white-box gradient for kind '{kind}'")
        if objective not in ("bce", "margin"):
            raise ValueError(f"unknown objective '{objective}'")
        self.model, self.kind, self.objective = model, kind, objective

    def __call__(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        if self.objective == "margin":
            if self.kind == "rbf_svm":
                return rbf_svm_margin_input_grad(self.model, X, y)
            # QSVMWhiteBox carries its own; the MLP margin path is not needed by P14a and
            # is not implemented rather than silently falling back to BCE.
            return self.model.margin_input_grad(X, y)
        if self.kind == "mlp":
            return mlp_loss_input_grad(self.model, X, y)
        if self.kind == "rbf_svm":
            return rbf_svm_loss_input_grad(self.model, X, y)
        return self.model.loss_input_grad(X, y)
