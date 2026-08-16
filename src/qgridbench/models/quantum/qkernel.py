"""Fidelity quantum kernels via statevector simulation, with caching + diagnostics.

Method: for pure-state feature maps, the fidelity kernel K[i,j] = |<psi_i|psi_j>|^2
is computed by simulating one statevector per sample (N circuit evaluations on
lightning.qubit) and taking the Gram matrix of the statevector matrix — an exact
identity that replaces the naive O(N^2) circuit-pair evaluation. Shot noise is
modeled post-hoc: a SWAP/overlap test with S shots yields a Binomial(S, K)/S
estimate per entry, which is sampled exactly.

Feature maps (bandwidth c multiplies the angle-scaled inputs, [-pi, pi] range
enforced upstream by AngleScaler — see features/reduce.py):
  - angle_ry: RY(c * x_i) on each wire (product state)
  - zz:       2 repetitions of [H^n; RZ(2 c x_i); ring CZZ(2 c^2 x_i x_j)]
              (IQP-style ZZ feature map, ring entanglement)

Diagnostics (Huang et al., "Power of data in quantum machine learning", 2021):
  - centered kernel-target alignment (KTA)
  - geometric difference g(K_classical -> K_quantum) with ridge regularization
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pennylane as qml

from qgridbench.utils.run_tracking import REPO_ROOT, get_logger

log = get_logger(__name__)

KERNEL_CACHE = REPO_ROOT / "models" / "kernels"

ENCODINGS = ("angle_ry", "zz")


def _circuit_state_fn(
    encoding: str, n_qubits: int, entangle_strength: float = 1.0, n_reps: int = 2
):
    """Feature-map statevector circuits.

    `entangle_strength` scales ONLY the ZZ map's pair rotation. It defaults to 1.0, which is
    the deployed circuit, so no existing caller or published number is affected. At 0.0 the
    pair rotation becomes the identity and the enclosing CNOTs cancel exactly, leaving the
    same H+RZ product map with the same repetitions and the same single-qubit scaling — so
    sweeping it varies entanglement and nothing else (P17b).

    `n_reps` sets the ZZ map's repetition count, default 2 = the deployed circuit. It exists
    because repetition is what makes the map NON-STATIONARY: at one repetition the overlap is
    prod_i cos^2(c(x_i - y_i)), a function of x - y alone, while at two the per-wire state
    e^{-it}cos t|0> - i e^{it} sin t|1> gives an overlap depending on t_x and t_y separately.
    RY cannot show this — RY(t)RY(t) = RY(2t) merely doubles the bandwidth — so the H between
    the RZs is what the extra repetition buys (P17c).
    """
    dev = qml.device("lightning.qubit", wires=n_qubits)

    if encoding == "angle_ry":

        @qml.qnode(dev)
        def state_fn(x):
            for i in range(n_qubits):
                qml.RY(x[i], wires=i)
            return qml.state()

    elif encoding == "zz":

        @qml.qnode(dev)
        def state_fn(x):
            for _ in range(n_reps):
                for i in range(n_qubits):
                    qml.Hadamard(wires=i)
                    qml.RZ(2.0 * x[i], wires=i)
                for i in range(n_qubits):  # ring entanglement
                    j = (i + 1) % n_qubits
                    qml.CNOT(wires=[i, j])
                    qml.RZ(2.0 * entangle_strength * x[i] * x[j], wires=j)
                    qml.CNOT(wires=[i, j])
            return qml.state()

    else:
        raise ValueError(f"unknown encoding '{encoding}'")
    return state_fn


def compute_states(
    X_angles: np.ndarray,
    encoding: str,
    bandwidth: float = 1.0,
    dtype=np.complex64,
    entangle_strength: float = 1.0,
    n_reps: int = 2,
) -> np.ndarray:
    """Statevectors for every row of X_angles (already scaled to [-pi, pi]).

    The simulator returns complex128; storage is complex64 by DEFAULT (unchanged, so no
    existing number moves) because the Gram is 2**d wide per sample. Pass
    dtype=np.complex128 where the comparison itself must be exact — a numerical gate
    against an analytic kernel otherwise measures the storage epsilon (~1e-7) and not
    the quantity under test.
    """
    n, d = X_angles.shape
    state_fn = _circuit_state_fn(encoding, d, entangle_strength, n_reps)
    out = np.empty((n, 2**d), dtype=dtype)
    Xb = bandwidth * X_angles
    for i in range(n):
        out[i] = state_fn(Xb[i])
    return out


def fidelity_kernel(
    states_a: np.ndarray, states_b: np.ndarray, chunk: int = 512, exact: bool = False
) -> np.ndarray:
    """K[i,j] = |<a_i|b_j>|^2, chunked to bound memory.

    `exact=True` accumulates the overlap in complex128 instead of the stored complex64.
    It costs memory and time, and the DEFAULT IS UNCHANGED so no existing number moves.

    Why the option exists (measured, tmp probes 2026-08-14): at small encoding bandwidth
    every state crowds toward |0...0>, the Gram approaches all-ones, the SVC compensates
    with dual coefficients pinned at C, and f(x) becomes a difference of nearly-equal
    large terms. Cancellation then amplifies complex64 accumulation error over thousands
    of support vectors — up to 1.9e-2 relative on the decision value at bandwidth 0.05,
    against 2.0e-3 at the tuned bandwidth. PREDICTIONS are essentially unaffected
    (200/200 agreement at bw 0.05/0.1, 199/200 at 0.2), which is why the shipped
    accuracy and flip-rate exhibits stand on the default path. Callers that report
    decision-MAGNITUDE quantities (prediction spread, kernel mass) at small bandwidth
    should pass exact=True so evaluation matches the complex128 white-box attack path.
    """
    na, nb = len(states_a), len(states_b)
    K = np.empty((na, nb), dtype=np.float64)
    b = states_b.astype(np.complex128) if exact else states_b
    for s in range(0, na, chunk):
        a = states_a[s : s + chunk]
        block = (a.astype(np.complex128) if exact else a).conj() @ b.T
        K[s : s + chunk] = np.abs(block) ** 2
    return np.clip(K, 0.0, 1.0)


def cosine_fidelity_kernel(
    angles_a: np.ndarray, angles_b: np.ndarray, bandwidth: float = 1.0
) -> np.ndarray:
    """Closed form of the `angle_ry` fidelity kernel: prod_i cos^2(c*(a_i - b_i)/2).

    The product RY feature map builds a product state, so its overlap factorizes per wire
    and the Gram needs no simulator at all — O(N M d) arithmetic instead of N statevector
    simulations. `fidelity_kernel(compute_states(A, "angle_ry", c), ...)` must reproduce
    this to floating-point tolerance; callers that rely on the identity should assert it
    (the ablation that uses this one does).

    It exists to separate two properties the RBF/fidelity contrast confounds: this kernel
    is BOUNDED and PERIODIC like a quantum fidelity kernel but carries no entanglement and
    runs classically, so a sweep over `bandwidth` isolates which of those axes an effect
    actually rides on.
    """
    a = np.asarray(angles_a, dtype=np.float64)
    b = np.asarray(angles_b, dtype=np.float64)
    if a.shape[1] != b.shape[1]:
        raise ValueError(f"feature-dim mismatch: {a.shape[1]} vs {b.shape[1]}")
    # accumulate the product one wire at a time: (N, M) working set, never (N, M, d)
    K = np.ones((len(a), len(b)), dtype=np.float64)
    for i in range(a.shape[1]):
        K *= np.cos(bandwidth * (a[:, i, None] - b[None, :, i]) / 2.0) ** 2
    return np.clip(K, 0.0, 1.0)


def shot_noise_kernel(
    K: np.ndarray, shots: int, rng: np.random.Generator, symmetric: bool | None = None
) -> np.ndarray:
    """Exact sampling model of an overlap-test estimate with `shots` shots/entry."""
    Kn = rng.binomial(shots, K) / shots
    if symmetric is None:
        symmetric = K.shape[0] == K.shape[1] and np.allclose(K, K.T, atol=1e-6)
    if symmetric:
        iu = np.triu_indices_from(Kn, k=1)
        Kn[(iu[1], iu[0])] = Kn[iu]
        np.fill_diagonal(Kn, 1.0)
    return Kn


def _mixed_density_fn(encoding: str, n_qubits: int, p_depol: float):
    """Same feature map on `default.mixed`, with per-qubit depolarizing noise.

    Noise is applied after every encoding layer, which is where it lands on real
    hardware (after the physical gates), not once at the end.
    """
    dev = qml.device("default.mixed", wires=n_qubits)

    @qml.qnode(dev)
    def density_fn(x):
        if encoding == "angle_ry":
            for i in range(n_qubits):
                qml.RY(x[i], wires=i)
            for i in range(n_qubits):
                qml.DepolarizingChannel(p_depol, wires=i)
        elif encoding == "zz":
            for _ in range(2):
                for i in range(n_qubits):
                    qml.Hadamard(wires=i)
                    qml.RZ(2.0 * x[i], wires=i)
                for i in range(n_qubits):
                    j = (i + 1) % n_qubits
                    qml.CNOT(wires=[i, j])
                    qml.RZ(2.0 * x[i] * x[j], wires=j)
                    qml.CNOT(wires=[i, j])
                for i in range(n_qubits):
                    qml.DepolarizingChannel(p_depol, wires=i)
        else:
            raise ValueError(f"unknown encoding '{encoding}'")
        return qml.density_matrix(wires=range(n_qubits))

    return density_fn


def compute_density_matrices(
    X_angles: np.ndarray, encoding: str, p_depol: float, bandwidth: float = 1.0
) -> np.ndarray:
    """Flattened density matrices, one row per sample (n, 4**d) complex64.

    Flattened so the noisy kernel below is one BLAS call instead of an O(N^2)
    python loop — the same Gram-matrix trick the pure-state path uses. Memory is
    4**d complex entries per sample, i.e. 256x the statevector path at the same
    qubit count, which is why the noise ablation runs on a reduced subset.
    """
    n, d = X_angles.shape
    density_fn = _mixed_density_fn(encoding, d, p_depol)
    out = np.empty((n, 4**d), dtype=np.complex64)
    Xb = bandwidth * X_angles
    for i in range(n):
        out[i] = np.asarray(density_fn(Xb[i])).reshape(-1)
    return out


def noisy_fidelity_kernel(rho_a: np.ndarray, rho_b: np.ndarray, chunk: int = 128) -> np.ndarray:
    """K[i,j] = Tr(rho_i rho_j), the mixed-state generalisation of |<a|b>|^2.

    For pure states this reduces exactly to the fidelity kernel, so the noiseless
    and noisy kernels are directly comparable. Tr(AB) for Hermitian A, B equals
    <vec(A), vec(B)> (Frobenius inner product), hence the Gram-matrix form.
    Entries are real up to floating error; the imaginary part is discarded after
    an assertion rather than silently.
    """
    na, nb = len(rho_a), len(rho_b)
    K = np.empty((na, nb), dtype=np.float64)
    for s in range(0, na, chunk):
        block = rho_a[s : s + chunk].conj() @ rho_b.T
        if np.max(np.abs(block.imag)) > 1e-3:
            raise ValueError("Tr(rho_i rho_j) has a non-negligible imaginary part")
        K[s : s + chunk] = block.real
    return np.clip(K, 0.0, 1.0)


# ----------------------------- caching ---------------------------------------


def kernel_cache_key(**kwargs) -> str:
    payload = json.dumps(kwargs, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


def cached_kernel(key: str) -> np.ndarray | None:
    p = KERNEL_CACHE / f"{key}.npz"
    if p.exists():
        log.info("kernel cache hit %s", key)
        return np.load(p)["K"]
    return None


def find_cached_kernel(**criteria) -> tuple[np.ndarray, dict]:
    """Look a cached Gram up by metadata (encoding/qubits/tag/...), not hash key.

    Fails loud if no cached kernel matches — callers that tolerate absence must
    catch FileNotFoundError explicitly.
    """
    for meta_path in sorted(KERNEL_CACHE.glob("*.json")):
        meta = json.loads(meta_path.read_text())
        if all(meta.get(k) == v for k, v in criteria.items()):
            return np.load(meta_path.with_suffix(".npz"))["K"], meta
    raise FileNotFoundError(f"no cached kernel matching {criteria}")


def store_kernel(key: str, K: np.ndarray, meta: dict) -> Path:
    KERNEL_CACHE.mkdir(parents=True, exist_ok=True)
    p = KERNEL_CACHE / f"{key}.npz"
    np.savez_compressed(p, K=K)
    (KERNEL_CACHE / f"{key}.json").write_text(json.dumps(meta, indent=2, default=str))
    return p


# --------------------------- diagnostics --------------------------------------


def _center(K: np.ndarray) -> np.ndarray:
    n = len(K)
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H


def kernel_target_alignment(K: np.ndarray, y: np.ndarray) -> float:
    """Centered KTA. Binary y -> {-1,+1} outer product; multiclass -> one-hot Gram."""
    classes = np.unique(y)
    if len(classes) == 2:
        t = np.where(y == classes.max(), 1.0, -1.0)
        Ky = np.outer(t, t)
    else:
        Y = np.eye(len(classes))[np.searchsorted(classes, y)]
        Ky = Y @ Y.T
    Kc, Kyc = _center(K), _center(Ky)
    denom = np.linalg.norm(Kc, "fro") * np.linalg.norm(Kyc, "fro")
    return float(np.sum(Kc * Kyc) / denom)


def _geometric_eigproblem(
    K_classical: np.ndarray, K_quantum: np.ndarray, lam: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Eigen-decomposition of M = sqrt(K_Q) (K_C + lam*n*I)^-1 sqrt(K_Q),
    both kernels trace-normalized to n. Returns (evals ascending, evecs, sqrt(K_Q))."""
    n = len(K_classical)
    Kc = n * K_classical / np.trace(K_classical)
    Kq = n * K_quantum / np.trace(K_quantum)
    # sqrt of PSD matrix via eigh (clip tiny negatives from numerical noise)
    w, V = np.linalg.eigh(Kq)
    sqKq = (V * np.sqrt(np.clip(w, 0, None))) @ V.T
    M = sqKq @ np.linalg.solve(Kc + lam * n * np.eye(n), sqKq)
    evals, evecs = np.linalg.eigh(M)
    return evals, evecs, sqKq


def geometric_difference(
    K_classical: np.ndarray, K_quantum: np.ndarray, lam: float = 1e-6
) -> float:
    """g(K_C -> K_Q) = sqrt(||sqrt(K_Q) (K_C + lam*n*I)^-1 sqrt(K_Q)||_inf),
    with both kernels trace-normalized to n (Huang et al. 2021, regularized)."""
    evals, _, _ = _geometric_eigproblem(K_classical, K_quantum, lam)
    return float(np.sqrt(evals[-1]))


def quantum_easy_labels(
    K_classical: np.ndarray, K_quantum: np.ndarray, lam: float = 1e-6
) -> tuple[np.ndarray, float]:
    """Labels saturating the geometric difference (Huang et al. 2021 construction).

    f = sqrt(K_Q) v with v the top eigenvector of the g eigenproblem; labels are
    f thresholded at its median so the relabeled task is exactly balanced. These
    labels are learnable by the quantum kernel while costing a classical kernel a
    ~g^2 sample penalty — the constructive converse of the g diagnostic, used as
    a positive control for the benchmark pipeline.

    Returns (labels in {0,1}, g) — g from the same decomposition, so callers can
    assert it matches geometric_difference() as an integrity check.
    """
    evals, evecs, sqKq = _geometric_eigproblem(K_classical, K_quantum, lam)
    f = sqKq @ evecs[:, -1]
    return (f > np.median(f)).astype(int), float(np.sqrt(evals[-1]))
