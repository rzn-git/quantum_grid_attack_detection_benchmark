"""Quantum kernel + VQC sanity: PSD, self-fidelity, diagnostics, reproducibility."""

import numpy as np

from qgridbench.models.quantum.qkernel import (
    compute_states,
    cosine_fidelity_kernel,
    fidelity_kernel,
    geometric_difference,
    kernel_target_alignment,
    shot_noise_kernel,
)


def _angles(n=40, d=4, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(-np.pi, np.pi, size=(n, d))


def test_fidelity_kernel_properties():
    X = _angles()
    states = compute_states(X, "angle_ry")
    K = fidelity_kernel(states, states)
    assert np.allclose(np.diag(K), 1.0, atol=1e-5)  # self-fidelity = 1
    assert np.allclose(K, K.T, atol=1e-6)  # symmetric
    assert K.min() >= -1e-9 and K.max() <= 1 + 1e-9  # bounded [0,1]
    evals = np.linalg.eigvalsh(K)
    assert evals.min() > -1e-6  # PSD


def test_fidelity_kernel_exact_matches_complex128_and_keeps_default_behaviour():
    """`exact=True` must be the complex128 accumulation, and must not change the default.

    Added with the option (decision log 2026-08-14T18:05Z): the deployed complex64
    accumulation drifts on near-degenerate Grams, so P9a evaluates with exact=True to match
    its complex128 attack path. Two things have to hold — the option really is the higher
    precision computation, and turning it on is the ONLY thing that changes, so every
    previously published number computed on the default path stays valid.
    """
    X = _angles(n=32, d=4)
    sa, sb = compute_states(X, "zz"), compute_states(_angles(n=24, d=4, seed=1), "zz")

    reference = np.abs(sa.astype(np.complex128).conj() @ sb.astype(np.complex128).T) ** 2
    assert np.allclose(fidelity_kernel(sa, sb, exact=True), np.clip(reference, 0, 1), atol=1e-12)

    default = fidelity_kernel(sa, sb)
    assert default.dtype == np.float64
    assert np.allclose(default, fidelity_kernel(sa, sb, exact=True), atol=1e-5)
    # chunking must not interact with the dtype promotion
    assert np.allclose(
        fidelity_kernel(sa, sb, chunk=7, exact=True),
        fidelity_kernel(sa, sb, exact=True),
        atol=1e-12,
    )


def test_cosine_closed_form_is_the_angle_ry_fidelity_kernel():
    """The P17a control arm is only a valid control if it IS the unentangled quantum kernel.

    Pinned at float64/complex128 on both sides deliberately: the pipeline stores angles as
    float32 and statevectors as complex64, which alone separates the two paths by ~1e-7 and
    would make a loose tolerance here hide a genuine algebraic error.
    """
    X = _angles(n=24, d=4).astype(np.float64)
    for bw in (0.1, 0.8, 3.0):
        s = compute_states(X, "angle_ry", bandwidth=bw, dtype=np.complex128)
        assert (
            np.abs(cosine_fidelity_kernel(X, X, bw) - fidelity_kernel(s, s, exact=True)).max()
            < 1e-12
        )


def test_compute_states_default_dtype_is_unchanged():
    """The dtype parameter must not move any shipped number: complex64 stays the default."""
    assert compute_states(_angles(n=4, d=3), "angle_ry").dtype == np.complex64


def test_entangle_strength_default_is_the_shipped_circuit_and_zero_is_a_product_state():
    """Both P17b gates, pinned: the sweep is only an isolation if its endpoints are exact.

    s=1 must be bit-identical to the deployed ZZ map (else the sweep measures a different
    model than every published ZZ number), and s=0 must be genuinely unentangled rather
    than weakly entangled (else "the ring vanished" is an assumption, not a fact).
    """
    X = _angles(n=8, d=4).astype(np.float64)
    shipped = compute_states(X, "zz", dtype=np.complex128)
    assert (
        np.abs(compute_states(X, "zz", dtype=np.complex128, entangle_strength=1.0) - shipped).max()
        == 0.0
    )

    s0 = compute_states(X, "zz", dtype=np.complex128, entangle_strength=0.0)
    for v in s0:  # Schmidt rank 1 across a 1-vs-rest cut <=> product state
        assert np.linalg.svd(v.reshape(2, -1), compute_uv=False)[1] < 1e-10


def test_single_repetition_zz_is_the_stationary_closed_form_and_two_is_not():
    """The P17c gate, pinned: the decisive arm must BE a stationary kernel.

    At one repetition the H+RZ overlap is prod_i cos^2(c(x_i-y_i)) -- the RY closed form at
    twice the bandwidth, hence a function of x-y alone. At two it is not shift-invariant.
    That split is the entire basis for reading the inversion as a stationarity effect, so
    both halves are asserted, and n_reps must default to 2 (the deployed circuit).
    """
    X = _angles(n=16, d=4).astype(np.float64)
    assert compute_states(X, "zz", dtype=np.complex128, n_reps=2).tolist() == (
        compute_states(X, "zz", dtype=np.complex128).tolist()
    )

    def gram(Z, reps, bw):
        s = compute_states(
            Z, "zz", bandwidth=bw, dtype=np.complex128, entangle_strength=0.0, n_reps=reps
        )
        return fidelity_kernel(s, s, exact=True)

    for bw in (0.1, 0.8, 3.0):
        assert np.abs(gram(X, 1, bw) - cosine_fidelity_kernel(X, X, 2.0 * bw)).max() < 1e-12
        # stationary <=> translating every point leaves the Gram unchanged
        assert np.abs(gram(X + 0.37, 1, bw) - gram(X, 1, bw)).max() < 1e-12
    assert np.abs(gram(X + 0.37, 2, 0.8) - gram(X, 2, 0.8)).max() > 0.1


def test_zz_encoding_kernel_psd():
    X = _angles(d=4)
    K = fidelity_kernel(compute_states(X, "zz"), compute_states(X, "zz"))
    assert np.linalg.eigvalsh(K).min() > -1e-6
    assert np.allclose(np.diag(K), 1.0, atol=1e-5)


def test_state_computation_reproducible():
    X = _angles()
    s1 = compute_states(X, "angle_ry")
    s2 = compute_states(X, "angle_ry")
    assert np.allclose(s1, s2)


def test_kernel_target_alignment_range():
    X = _angles()
    y = (X[:, 0] > 0).astype(int)
    kta = kernel_target_alignment(
        fidelity_kernel(compute_states(X, "angle_ry"), compute_states(X, "angle_ry")), y
    )
    assert -1.0 <= kta <= 1.0


def test_geometric_difference_positive():
    X = _angles()
    Kq = fidelity_kernel(compute_states(X, "zz"), compute_states(X, "zz"))
    Kc = np.exp(-0.5 * ((X[:, None] - X[None]) ** 2).sum(-1))  # toy RBF
    g = geometric_difference(Kc, Kq)
    assert g > 0 and np.isfinite(g)


def test_quantum_easy_labels_balanced_and_consistent():
    """The Huang et al. relabel construction (relabel_control ablation) must return
    exactly balanced labels, a g identical to geometric_difference, and labels that
    align better with the quantum kernel than with the classical one — otherwise it
    is not a positive control for anything."""
    from qgridbench.models.quantum.qkernel import quantum_easy_labels

    X = _angles(n=60, d=4)
    Kq = fidelity_kernel(compute_states(X, "zz"), compute_states(X, "zz"))
    Kc = np.exp(-0.5 * ((X[:, None] - X[None]) ** 2).sum(-1))
    y, g = quantum_easy_labels(Kc, Kq)
    assert set(np.unique(y)) == {0, 1}
    assert abs(y.mean() - 0.5) <= 1 / len(y)  # median threshold -> balanced
    assert np.isclose(g, geometric_difference(Kc, Kq), rtol=1e-8)
    assert kernel_target_alignment(Kq, y) > kernel_target_alignment(Kc, y)


def test_shot_noise_converges_to_analytic():
    X = _angles(n=20)
    K = fidelity_kernel(compute_states(X, "angle_ry"), compute_states(X, "angle_ry"))
    rng = np.random.default_rng(0)
    Kn = shot_noise_kernel(K, shots=8192, rng=rng)
    assert np.abs(Kn - K).mean() < 0.02
    assert np.allclose(np.diag(Kn), 1.0)


def test_vqc_batched_forward_matches_per_sample():
    """The batched (broadcast) forward must equal per-sample execution — the
    batching is the performance lever, so it may never change the numbers."""
    import torch

    from qgridbench.models.quantum.vqc import VQClassifier

    for q in [4, 12]:  # 12 exercises the wide-statevector path
        m = VQClassifier(q, depth=2, n_classes=2, seed=0)
        X = np.random.default_rng(1).uniform(-np.pi, np.pi, (5, q))
        batched = m._logits(torch.as_tensor(X, dtype=torch.float64)).detach().numpy()
        per_sample = np.vstack(
            [
                m._logits(torch.as_tensor(X[i : i + 1], dtype=torch.float64)).detach().numpy()
                for i in range(len(X))
            ]
        )
        assert np.allclose(batched, per_sample, atol=1e-10)


def test_vqc_chunking_bounds_memory_and_preserves_results():
    """Chunked inference must be numerically identical to unchunked."""

    from qgridbench.models.quantum.vqc import VQClassifier, resolve_chunk

    m = VQClassifier(6, depth=2, n_classes=2, seed=0)
    X = np.random.default_rng(4).uniform(-np.pi, np.pi, (37, 6))
    full = m.predict_proba(X)
    m.chunk = 5  # force multiple chunks
    assert np.allclose(full, m.predict_proba(X), atol=1e-12)
    # chunk sizing shrinks as the statevector grows
    assert resolve_chunk(16, 6) < resolve_chunk(8, 4)
    assert resolve_chunk(16, 6) >= 1


def test_noisy_kernel_reduces_to_fidelity_at_zero_noise():
    """Tr(rho_i rho_j) must equal |<psi_i|psi_j>|^2 when p_depol = 0.

    This is the property that makes the noisy and noiseless kernels comparable at
    all: without it the depolarizing ablation would be measuring a different
    quantity, not the same quantity under noise.
    """
    import numpy as np

    from qgridbench.models.quantum.qkernel import (
        compute_density_matrices,
        compute_states,
        fidelity_kernel,
        noisy_fidelity_kernel,
    )

    rng = np.random.default_rng(0)
    X = rng.uniform(-np.pi, np.pi, size=(6, 4))
    for enc in ("angle_ry", "zz"):
        K_pure = fidelity_kernel(compute_states(X, enc), compute_states(X, enc))
        rho = compute_density_matrices(X, enc, p_depol=0.0)
        K_mixed = noisy_fidelity_kernel(rho, rho)
        assert np.allclose(K_pure, K_mixed, atol=1e-4), enc


def test_depolarizing_noise_contracts_the_kernel():
    """Depolarizing noise pulls states toward the maximally mixed state, so the
    self-fidelity Tr(rho^2) (purity) must drop below 1."""
    import numpy as np

    from qgridbench.models.quantum.qkernel import compute_density_matrices, noisy_fidelity_kernel

    rng = np.random.default_rng(0)
    X = rng.uniform(-np.pi, np.pi, size=(4, 4))
    purity = [
        np.diag(noisy_fidelity_kernel(r, r)).mean()
        for r in (compute_density_matrices(X, "angle_ry", p) for p in (0.0, 0.05, 0.2))
    ]
    assert purity[0] > 0.999
    assert purity[0] > purity[1] > purity[2], purity
    assert purity[2] < 0.9
