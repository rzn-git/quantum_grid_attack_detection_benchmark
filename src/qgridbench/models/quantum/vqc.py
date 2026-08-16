"""Variational quantum classifier: AngleEmbedding(RY) + StronglyEntanglingLayers.

ONE simulation engine, end to end: PennyLane's supported **torch interface** on
`default.qubit` with `diff_method="backprop"`. That is the library's own
batched-autodiff path — no hand-rolled simulation, no custom CUDA kernels — and
it serves training, inference, and attack input-gradients identically, so every
number in the study comes from the same code path.

Why this engine (measured, decision log 2026-08-13):
  - it BATCHES (parameter broadcasting), which is where the speed actually comes
    from: ~4 s per 2,000-sample epoch at 8 qubits, vs 75 s for per-sample
    lightning+adjoint;
  - it is stable at every qubit count we run (a custom GPU statevector kernel was
    trialled and abandoned: it crashed with CUBLAS_STATUS_EXECUTION_FAILED at 16
    qubits);
  - `parameter-shift` remains available on the same QNode for the hardware-faithful
    gradient, and the two are asserted equal in tests.

Multiclass strategy (uniform across all experiments, logged): multi-qubit
measurement heads — PauliZ expectations on the first n_classes qubits (1 for
binary), passed through a trainable per-head scale+bias and sigmoid/softmax.
Loss is class-weighted cross-entropy (mirrors the classical class-weight policy).

Trainability logging: per-epoch mean and variance of batch gradient norms are
recorded; a collapse of gradient variance with depth is the barren-plateau
signature and is reported as a finding.

Attack interface: `loss_input_grad` differentiates the loss w.r.t. the INPUT
angles, i.e. through the encoding, keeping attacks in the shared pre-encoding
feature space (comparability contract, CLAUDE.md section 7).
"""

from __future__ import annotations

import os
import time

import numpy as np
import pennylane as qml
import torch
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight

from qgridbench.utils.run_tracking import get_logger

log = get_logger(__name__)

# Statevector memory per sample doubles with every qubit; chunk the batch so a
# high-qubit backward pass cannot exhaust RAM. Detected at runtime, overridable.
_BYTES_PER_AMP = 8  # complex64


def resolve_chunk(n_qubits: int, depth: int, budget_gib: float | None = None) -> int:
    """Largest safe forward/backward chunk for this circuit on this host.

    Backprop stores one statevector per elementary op, so peak memory is roughly
    chunk * 2^n * (depth * n_qubits) * 8 bytes. Sizes the chunk to fit a budget
    that defaults to a share of AVAILABLE (not total) RAM, leaving headroom for
    the OS and any concurrent jobs. QGB_VQC_CHUNK overrides.
    """
    override = os.environ.get("QGB_VQC_CHUNK")
    if override:
        return max(1, int(override))
    if budget_gib is None:
        try:
            import psutil

            budget_gib = max(1.0, psutil.virtual_memory().available / 2**30 * 0.35)
        except ImportError:
            budget_gib = 2.0
    per_sample = (2**n_qubits) * max(1, depth * n_qubits) * _BYTES_PER_AMP
    return int(np.clip((budget_gib * 2**30) // per_sample, 1, 4096))


class VQClassifier:
    """sklearn-flavoured wrapper (fit / predict_proba / predict)."""

    def __init__(
        self,
        n_qubits: int,
        depth: int = 4,
        n_classes: int = 2,
        lr: float = 0.05,
        batch_size: int = 64,
        max_epochs: int = 60,
        patience: int = 10,
        seed: int = 0,
        device: str = "default.qubit",
        diff_method: str = "backprop",
        grad_method: str | None = None,
    ):
        self.n_qubits, self.depth, self.n_classes = n_qubits, depth, n_classes
        self.lr, self.batch_size = lr, batch_size
        self.max_epochs, self.patience = max_epochs, patience
        self.seed = seed
        self.device_name, self.diff_method = device, diff_method
        # attack input-gradients reuse the training engine unless told otherwise
        # ("parameter-shift" = hardware-faithful reference, same analytic value)
        self.grad_method = grad_method or diff_method
        self.n_out = 1 if n_classes == 2 else n_classes
        self.history: dict[str, list] = {
            "epoch_loss": [],
            "val_macro_f1": [],
            "grad_norm_mean": [],
            "grad_norm_var": [],
        }
        self.fit_seconds: float | None = None
        self.chunk = resolve_chunk(n_qubits, depth)
        self._build_qnodes()

        g = torch.Generator().manual_seed(seed)
        shape = qml.StronglyEntanglingLayers.shape(n_layers=depth, n_wires=n_qubits)
        self.weights = (0.1 * torch.randn(*shape, generator=g, dtype=torch.float64)).requires_grad_(
            True
        )
        self.head_scale = torch.ones(self.n_out, dtype=torch.float64, requires_grad=True)
        self.head_bias = torch.zeros(self.n_out, dtype=torch.float64, requires_grad=True)

    # ------------------------------------------------------------------ circuit

    def _build_qnodes(self) -> None:
        def circuit(x, weights):
            qml.AngleEmbedding(x, wires=range(self.n_qubits), rotation="Y")
            qml.StronglyEntanglingLayers(weights, wires=range(self.n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_out)]

        self._circuit = circuit
        dev = qml.device(self.device_name, wires=self.n_qubits)
        self._qnode = qml.QNode(circuit, dev, interface="torch", diff_method=self.diff_method)
        if self.grad_method == self.diff_method:
            self._qnode_grad = self._qnode
        else:
            dev_g = qml.device(
                "lightning.qubit" if self.grad_method == "parameter-shift" else self.device_name,
                wires=self.n_qubits,
            )
            self._qnode_grad = qml.QNode(
                circuit, dev_g, interface="torch", diff_method=self.grad_method
            )

    def _expvals(self, X: torch.Tensor, weights: torch.Tensor, qnode=None) -> torch.Tensor:
        """Batched PauliZ expectations -> (B, n_out)."""
        res = (qnode or self._qnode)(X, weights)
        if isinstance(res, (list, tuple)):
            return torch.stack(list(res), dim=-1).reshape(-1, self.n_out)
        return res.reshape(-1, self.n_out)

    def _logits(self, X: torch.Tensor, weights=None, qnode=None) -> torch.Tensor:
        w = self.weights if weights is None else weights
        return self._expvals(X, w, qnode) * self.head_scale + self.head_bias

    def _ce_loss(self, logits: torch.Tensor, y: torch.Tensor, class_w=None) -> torch.Tensor:
        eps = 1e-9
        if self.n_out == 1:
            p = torch.sigmoid(logits[:, 0])
            yf = y.double()
            per = -(yf * torch.log(p + eps) + (1 - yf) * torch.log(1 - p + eps))
        else:
            per = -torch.log_softmax(logits, dim=1)[torch.arange(len(y)), y] + eps * 0
        if class_w is not None:
            per = per * class_w[y]
        return per.mean()

    # -------------------------------------------------------------------- train

    def fit(self, X, y, X_val=None, y_val=None):
        t0 = time.perf_counter()
        rng = np.random.default_rng(self.seed)
        torch.manual_seed(self.seed)

        Xt = torch.as_tensor(np.asarray(X), dtype=torch.float64)
        y = np.asarray(y)
        yt = torch.as_tensor(y, dtype=torch.long)
        classes = np.unique(y)
        cw = compute_class_weight("balanced", classes=classes, y=y)
        w_full = np.ones(int(classes.max()) + 1)
        w_full[classes.astype(int)] = cw
        class_w = torch.as_tensor(w_full, dtype=torch.float64)

        params = [self.weights, self.head_scale, self.head_bias]
        opt = torch.optim.Adam(params, lr=self.lr)
        best_f1, best_state, best_epoch, stale = -1.0, None, 0, 0
        # chunk bounds peak memory; batches larger than the chunk accumulate grads
        chunk = min(self.chunk, self.batch_size)

        for epoch in range(self.max_epochs):
            order = rng.permutation(len(y))
            batch_losses, grad_norms = [], []
            for s in range(0, len(y), self.batch_size):
                b = order[s : s + self.batch_size]
                opt.zero_grad(set_to_none=True)
                total = 0.0
                for c0 in range(0, len(b), chunk):
                    cb = b[c0 : c0 + chunk]
                    loss = self._ce_loss(self._logits(Xt[cb]), yt[cb], class_w)
                    (loss * (len(cb) / len(b))).backward()
                    total += float(loss.detach()) * len(cb) / len(b)
                gnorm = float(
                    torch.sqrt(sum((p.grad**2).sum() for p in params if p.grad is not None))
                )
                grad_norms.append(gnorm)
                opt.step()
                batch_losses.append(total)

            self.history["epoch_loss"].append(float(np.mean(batch_losses)))
            self.history["grad_norm_mean"].append(float(np.mean(grad_norms)))
            self.history["grad_norm_var"].append(float(np.var(grad_norms)))

            if X_val is not None:
                f1 = f1_score(y_val, self.predict(X_val), average="macro")
                self.history["val_macro_f1"].append(float(f1))
                if f1 > best_f1:
                    best_f1, best_epoch, stale = f1, epoch, 0
                    best_state = [p.detach().clone() for p in params]
                else:
                    stale += 1
                    if stale >= self.patience:
                        break

        if best_state is not None:
            with torch.no_grad():
                for p, b in zip(params, best_state):
                    p.copy_(b)
            self.best_epoch_, self.best_val_f1_ = best_epoch, best_f1
        self.fit_seconds = time.perf_counter() - t0
        return self

    # ---------------------------------------------------------------- inference

    def predict_proba(self, X, shots: int | None = None) -> np.ndarray:
        X = np.asarray(X)
        if shots is None:
            outs = []
            with torch.no_grad():
                for s in range(0, len(X), self.chunk):
                    xb = torch.as_tensor(X[s : s + self.chunk], dtype=torch.float64)
                    outs.append(self._logits(xb).numpy())
            logits = np.vstack(outs)
        else:
            dev = qml.device("lightning.qubit", wires=self.n_qubits)
            qn = qml.set_shots(qml.QNode(self._circuit, dev), shots=shots)
            w = self.weights.detach()
            zs = np.array([np.atleast_1d(qn(torch.as_tensor(x), w)) for x in X], dtype=float)
            logits = zs * self.head_scale.detach().numpy() + self.head_bias.detach().numpy()

        if self.n_out == 1:
            p1 = 1.0 / (1.0 + np.exp(-logits[:, 0]))
            return np.column_stack([1 - p1, p1])
        z = logits - logits.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def predict(self, X, shots: int | None = None) -> np.ndarray:
        return self.predict_proba(X, shots=shots).argmax(axis=1)

    # ------------------------------------------------------------ attack grads

    def loss_input_grad(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """d(unweighted CE)/dX — batched, differentiating THROUGH the encoding.

        Uses the same QNode as training (`grad_method`, default backprop);
        setting grad_method="parameter-shift" gives the hardware-faithful
        gradient, asserted numerically identical in tests.
        """
        X = np.asarray(X, dtype=np.float64)
        yt = torch.as_tensor(np.asarray(y), dtype=torch.long)
        qnode = self._qnode_grad
        # parameter-shift cannot differentiate a broadcasted tape (PennyLane
        # #4462), so that engine is driven one sample at a time
        step = 1 if self.grad_method == "parameter-shift" else self.chunk
        grads = np.empty_like(X)
        for s in range(0, len(X), step):
            xb = torch.tensor(X[s : s + step], dtype=torch.float64, requires_grad=True)
            xin = xb[0] if step == 1 else xb
            loss = self._ce_loss(self._logits(xin, qnode=qnode), yt[s : s + step])
            (g,) = torch.autograd.grad(loss * len(xb), xb)
            grads[s : s + step] = g.detach().numpy()
        return grads
