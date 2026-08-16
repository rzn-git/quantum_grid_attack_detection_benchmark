"""White-box input gradients for the fidelity-kernel QSVM (P5a).

The fitted precomputed-kernel SVC decides with
    f(x) = sum_j alpha_j * |<psi(bw * a(x)) | psi_j>|^2 + b
over its support vectors j, where a(.) is the train-fitted AngleScaler (linear
map + clip to [-pi, pi]) and bw the tuned encoding bandwidth. This module
rebuilds that exact function as one differentiable torch graph and returns
d(loss)/dx in the shared PRE-ENCODING (PCA) feature space, keeping the
comparability contract (CLAUDE.md section 7).

Engine: the SAME supported path the VQC uses — PennyLane's torch interface on
`default.qubit` with `diff_method="backprop"`, batched via parameter
broadcasting (decision log 2026-08-13). No custom simulation.

NUMERIC GATE (non-negotiable): `assert_matches_reference` compares this graph's
decision values against the lightning.qubit reference path
(`compute_states` + `fidelity_kernel` + `SVC.decision_function`) and its
gradients against central finite differences. tests/test_attacks.py runs the
gate on a toy fixture; every runner must call it on real data BEFORE any
reported number depends on this path.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml
import torch

from qgridbench.models.quantum.qkernel import ENCODINGS, compute_states, fidelity_kernel
from qgridbench.utils.run_tracking import get_logger

log = get_logger(__name__)


def _state_qnode(encoding: str, n_qubits: int):
    """Statevector QNode mirroring qkernel._circuit_state_fn, torch/backprop."""
    if encoding not in ENCODINGS:
        raise ValueError(f"unknown encoding '{encoding}'")
    dev = qml.device("default.qubit", wires=n_qubits)

    if encoding == "angle_ry":

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def state_fn(x):
            for i in range(n_qubits):
                qml.RY(x[..., i], wires=i)
            return qml.state()

    else:  # zz

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def state_fn(x):
            for _ in range(2):  # 2 repetitions, ring entanglement — mirror qkernel
                for i in range(n_qubits):
                    qml.Hadamard(wires=i)
                    qml.RZ(2.0 * x[..., i], wires=i)
                for i in range(n_qubits):
                    j = (i + 1) % n_qubits
                    qml.CNOT(wires=[i, j])
                    qml.RZ(2.0 * x[..., i] * x[..., j], wires=j)
                    qml.CNOT(wires=[i, j])
            return qml.state()

    return state_fn


class QSVMWhiteBox:
    """Differentiable decision function of a fitted binary fidelity-kernel SVC.

    Parameters mirror the FittedZoo QSVM exactly: `svc` is the binary
    SVC(kernel="precomputed") fit on K_tr, `states_train` the statevectors the
    Gram was built from (bandwidth already applied inside compute_states), and
    `angle_scaler` the train-fitted AngleScaler the zoo routes inputs through.
    """

    def __init__(self, svc, states_train, angle_scaler, encoding, bandwidth, chunk=256):
        if len(getattr(svc, "classes_", [])) != 2:
            raise ValueError("QSVMWhiteBox implements the binary decision function only")
        self.encoding, self.bandwidth, self.chunk = encoding, float(bandwidth), int(chunk)
        # support-vector states, dual coefficients, intercept — the whole decision fn
        self._psi_sv = torch.as_tensor(
            np.asarray(states_train)[svc.support_], dtype=torch.complex128
        )
        self._alpha = torch.as_tensor(svc.dual_coef_[0], dtype=torch.float64)
        self._b = float(svc.intercept_[0])
        # AngleScaler.transform(Z) == clip(Z * scale_ + min_, -pi, pi)
        mm = angle_scaler._mm
        self._a_scale = torch.as_tensor(mm.scale_, dtype=torch.float64)
        self._a_min = torch.as_tensor(mm.min_, dtype=torch.float64)
        self.n_qubits = len(mm.scale_)
        self._qnode = _state_qnode(encoding, self.n_qubits)

    # ------------------------------------------------------------------ forward

    def _decision_t(self, x: torch.Tensor) -> torch.Tensor:
        """f(x) for a batch of PRE-ENCODING (PCA-space) rows; torch, differentiable."""
        angles = torch.clamp(x * self._a_scale + self._a_min, -np.pi, np.pi)
        psi = self._qnode(self.bandwidth * angles)
        psi = psi.reshape(-1, 2**self.n_qubits).to(torch.complex128)
        K = (psi.conj() @ self._psi_sv.T).abs() ** 2  # (B, n_sv) fidelities
        return K.real.to(torch.float64) @ self._alpha + self._b

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        out = []
        with torch.no_grad():
            for s in range(0, len(X), self.chunk):
                xb = torch.as_tensor(X[s : s + self.chunk], dtype=torch.float64)
                out.append(self._decision_t(xb).numpy())
        return np.concatenate(out)

    # ------------------------------------------------------------ attack grads

    def loss_input_grad(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """d(BCE(sigmoid(f), y))/dX — the same loss shape every family is attacked
        with (the zoo derives QSVM probabilities as sigmoid(decision))."""
        X = np.asarray(X, dtype=np.float64)
        yt = torch.as_tensor(np.asarray(y), dtype=torch.float64)
        grads = np.empty_like(X)
        for s in range(0, len(X), self.chunk):
            xb = torch.tensor(X[s : s + self.chunk], dtype=torch.float64, requires_grad=True)
            p = torch.sigmoid(self._decision_t(xb))
            eps = 1e-9
            yb = yt[s : s + self.chunk]
            loss = -(yb * torch.log(p + eps) + (1 - yb) * torch.log(1 - p + eps)).sum()
            (g,) = torch.autograd.grad(loss, xb)
            grads[s : s + self.chunk] = g.detach().numpy()
        return grads

    def margin_input_grad(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """d(-(2y-1) f(x))/dX — the non-saturating objective control (P14a).

        BCE's dL/df = sigmoid(f) - y vanishes on confidently-correct points, and this
        model's duals are pinned at ~1443, so |f| is large and that factor underflows.
        The margin loss has dL/df = -+1 identically. Same decision graph, one factor
        different, so an objective comparison isolates the objective.
        """
        X = np.asarray(X, dtype=np.float64)
        sgn = -(2.0 * np.asarray(y, dtype=np.float64) - 1.0)
        grads = np.empty_like(X)
        for s in range(0, len(X), self.chunk):
            xb = torch.tensor(X[s : s + self.chunk], dtype=torch.float64, requires_grad=True)
            sb = torch.as_tensor(sgn[s : s + self.chunk], dtype=torch.float64)
            (g,) = torch.autograd.grad((sb * self._decision_t(xb)).sum(), xb)
            grads[s : s + self.chunk] = g.detach().numpy()
        return grads

    def saturation_fraction(self, X: np.ndarray, y: np.ndarray, tol: float = 1e-6) -> float:
        """Fraction of points where BCE's dL/df has underflowed to |sigmoid(f) - y| < tol.

        The proposed mechanism behind P14a, measured rather than assumed: if this is ~0 and
        the margin attack still wins, the explanation is something else.
        """
        f = self.decision_function(np.asarray(X, dtype=np.float64))
        return float(np.mean(np.abs(1.0 / (1.0 + np.exp(-f)) - np.asarray(y)) < tol))

    def __call__(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.loss_input_grad(X, y)


# ------------------------------------------------------------------ numeric gate


def assert_matches_reference(
    wb: QSVMWhiteBox,
    svc,
    states_train,
    angle_scaler,
    X_check: np.ndarray,
    fwd_tol: float = 1e-4,
    grad_tol: float = 1e-3,
    deployed_tol: float = 5e-3,
    fd_step: float = 1e-5,
    n_grad_rows: int = 3,
) -> dict:
    """Fail-loud equivalence gate; returns the measured discrepancies for logging.

    Forward, two comparisons at different precisions — measured, not assumed
    (tmp probe 2026-08-14, decision log): the deployed kernel path stores
    statevectors as complex64, which by itself moves the decision value by
    ~2e-3 at 8 qubits, so a single strict tolerance would fail for a reason that
    has nothing to do with this graph.
      1. vs the EXACT (complex128) Gram over the same states — the mathematical
         identity this graph must satisfy; strict `fwd_tol`.
      2. vs the DEPLOYED complex64 path the rest of the study runs on — bounded
         by the looser `deployed_tol` and returned, so growth is caught.
    Predictions from both paths must additionally agree on every checked row;
    disagreement means the attacked model is not the evaluated model. That check is
    unconditional and is the one that actually protects the comparison.

    `deployed_tol=None` records the deployed discrepancy WITHOUT blocking on it. That is
    legitimate for exactly one kind of caller: one that also EVALUATES through the exact
    complex128 Gram (`fidelity_kernel(..., exact=True)`), so attack and evaluation share a
    precision and the deployed comparison is a diagnostic rather than a correctness
    condition. It is not a way to wave through a failing gate — comparison 1 stays strict,
    prediction agreement stays mandatory, and the measured value is still returned for the
    caller to record. P9a needs it because the near-degenerate Gram at small bandwidth
    amplifies complex64 accumulation error into the decision value (see `fidelity_kernel`).

    Gradient: central finite differences of the torch decision function on
    `n_grad_rows` rows (all input dims) — an independent check of the autograd
    path through scaler, bandwidth, circuit, and Gram.
    """
    A = angle_scaler.transform(X_check)
    states_check = compute_states(A, wb.encoding, bandwidth=wb.bandwidth)
    states_tr = np.asarray(states_train)

    f_deployed = svc.decision_function(fidelity_kernel(states_check, states_tr))
    K_exact = np.clip(
        np.abs(states_check.astype(np.complex128).conj() @ states_tr.astype(np.complex128).T) ** 2,
        0.0,
        1.0,
    )
    f_exact = svc.decision_function(K_exact)
    f_wb = wb.decision_function(X_check)

    fwd_err = float(np.max(np.abs(f_wb - f_exact) / (1.0 + np.abs(f_exact))))
    if fwd_err > fwd_tol:
        raise AssertionError(f"QSVM white-box forward mismatch: rel err {fwd_err:.2e} > {fwd_tol}")
    deployed_err = float(np.max(np.abs(f_wb - f_deployed) / (1.0 + np.abs(f_deployed))))
    if deployed_tol is not None and deployed_err > deployed_tol:
        raise AssertionError(
            f"QSVM white-box vs deployed float32 path: rel err {deployed_err:.2e} > {deployed_tol}"
        )
    # "The attacked model must BE the evaluated model" — so this blocks on agreement with
    # whichever path the caller evaluates on, and records the other. Callers on the deployed
    # path (P5a/P7b/P8) are checked against it; a caller that evaluates through the exact
    # complex128 Gram (deployed_tol=None) is checked against that, because comparing it to a
    # path it never uses would test the wrong reference. Neither branch is optional.
    n_agree_deployed = int((np.sign(f_wb) == np.sign(f_deployed)).sum())
    n_agree_exact = int((np.sign(f_wb) == np.sign(f_exact)).sum())
    reference, n_agree = (
        ("exact complex128", n_agree_exact)
        if deployed_tol is None
        else ("deployed float32", n_agree_deployed)
    )
    if n_agree != len(f_wb):
        raise AssertionError(
            f"white-box and {reference} paths disagree on "
            f"{len(f_wb) - n_agree}/{len(f_wb)} predictions"
        )

    rows = X_check[:n_grad_rows].astype(np.float64)
    xb = torch.tensor(rows, dtype=torch.float64, requires_grad=True)
    (g_auto,) = torch.autograd.grad(wb._decision_t(xb).sum(), xb)
    g_auto = g_auto.numpy()
    g_fd = np.empty_like(rows)
    for j in range(rows.shape[1]):
        hi, lo = rows.copy(), rows.copy()
        hi[:, j] += fd_step
        lo[:, j] -= fd_step
        g_fd[:, j] = (wb.decision_function(hi) - wb.decision_function(lo)) / (2 * fd_step)
    denom = np.maximum(np.abs(g_fd), np.abs(g_auto)).max()
    grad_err = float(np.max(np.abs(g_auto - g_fd)) / max(denom, 1e-12))
    if grad_err > grad_tol:
        raise AssertionError(
            f"QSVM white-box gradient mismatch: rel err {grad_err:.2e} > {grad_tol}"
        )
    log.info(
        "qsvm white-box gate passed: fwd(exact) %.2e, fwd(deployed float32) %.2e, grad %.2e",
        fwd_err,
        deployed_err,
        grad_err,
    )
    return {
        "forward_rel_err": fwd_err,
        "deployed_float32_rel_err": deployed_err,
        "grad_rel_err": grad_err,
        "predictions_agree": n_agree,
        "blocking_reference": reference,
        # both are recorded regardless of which one blocked, so a caller that evaluates
        # exactly still surfaces how far the deployed pipeline sits from it
        "predictions_agree_deployed": n_agree_deployed,
        "predictions_agree_exact": n_agree_exact,
        "n_check": int(len(X_check)),
    }
