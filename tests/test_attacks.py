"""Attack epsilon-boundedness, gradient correctness, poisoning, noise."""

import numpy as np
import pytest
from sklearn.neural_network import MLPClassifier

from qgridbench.attacks.evasion import fgsm, gaussian_sensor_noise, pgd
from qgridbench.attacks.gradients import mlp_loss_input_grad
from qgridbench.attacks.label_noise import flip_labels


def _fit_mlp(toy_binary):
    X, y = toy_binary
    mlp = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=300, random_state=0)
    mlp.fit(X, y)
    return mlp, X, y


def test_fgsm_pgd_linf_bounded(toy_binary):
    """Budget holds in STANDARDIZED units: |delta_j| <= eps * scale_j."""
    mlp, X, y = _fit_mlp(toy_binary)

    def grad_fn(x, yy):
        return mlp_loss_input_grad(mlp, x, yy)

    eps = 0.1
    scale = X.std(axis=0)
    for atk in (
        lambda: fgsm(grad_fn, X, y, eps, scale),
        lambda: pgd(grad_fn, X, y, eps, scale, n_steps=10, rng=np.random.default_rng(0)),
    ):
        X_adv = atk()
        assert np.max(np.abs(X_adv - X) / scale) <= eps + 1e-9


def test_attack_scale_is_required_and_validated(toy_binary):
    """The scale contract fails loud: a silent unit mismatch is the bug this guards."""
    import pytest

    mlp, X, y = _fit_mlp(toy_binary)

    def grad_fn(x, yy):
        return mlp_loss_input_grad(mlp, x, yy)

    with pytest.raises(TypeError):  # omitted entirely
        fgsm(grad_fn, X, y, 0.1)
    with pytest.raises(ValueError):  # wrong dimensionality
        fgsm(grad_fn, X, y, 0.1, np.ones(X.shape[1] + 1))
    with pytest.raises(ValueError):  # non-positive
        fgsm(grad_fn, X, y, 0.1, np.zeros(X.shape[1]))


def test_epsilon_is_scale_relative_not_raw(toy_binary):
    """A feature with larger std gets a proportionally larger absolute budget."""
    mlp, X, y = _fit_mlp(toy_binary)

    def grad_fn(x, yy):
        return mlp_loss_input_grad(mlp, x, yy)

    scale = np.linspace(1.0, 4.0, X.shape[1])
    d = np.abs(fgsm(grad_fn, X, y, 0.1, scale) - X).max(axis=0)
    assert np.allclose(d / scale, d[0] / scale[0])  # constant in sigma units
    assert d[-1] > d[0] * 3.5  # but NOT constant in raw units


def test_mlp_gradient_matches_numerical(toy_binary):
    mlp, X, y = _fit_mlp(toy_binary)
    xs, ys = X[:5], y[:5]
    ana = mlp_loss_input_grad(mlp, xs, ys)

    def ce(xrow, yi):
        p = mlp.predict_proba(xrow.reshape(1, -1))[0]
        return -np.log(p[yi] + 1e-9)

    h = 1e-5
    num = np.zeros_like(xs)
    for i in range(len(xs)):
        for j in range(xs.shape[1]):
            xp, xm = xs[i].copy(), xs[i].copy()
            xp[j] += h
            xm[j] -= h
            num[i, j] = (ce(xp, ys[i]) - ce(xm, ys[i])) / (2 * h)
    assert np.allclose(ana, num, atol=1e-3)


def test_label_flip_rate(toy_binary):
    _, y = toy_binary
    rng = np.random.default_rng(0)
    yf = flip_labels(y, 0.2, rng)
    assert (yf != y).sum() == int(round(0.2 * len(y)))
    assert set(np.unique(yf)) <= set(np.unique(y))


def test_gaussian_noise_snr(toy_binary):
    """Noise power tracks the PER-FEATURE signal power, so the SNR label is honest.

    Assuming unit signal power on PCA outputs (std 2-5.6) mislabels every level by
    ~10 dB; this asserts the achieved SNR per feature, not a global std.
    """
    X, _ = toy_binary
    rng = np.random.default_rng(0)
    scale = X.std(axis=0)
    Xn = gaussian_sensor_noise(X, snr_db=20, rng=rng, scale=scale)
    achieved = 20 * np.log10(scale / (Xn - X).std(axis=0))
    assert np.all(np.abs(achieved - 20) < 1.0)


def test_vqc_input_gradient_engines_agree():
    """Adjoint and parameter-shift must return the same analytic input-gradient
    (adjoint is the simulator-fast engine used for the attack sweep)."""
    from qgridbench.models.quantum.vqc import VQClassifier
    from qgridbench.utils.seeding import set_all_seeds

    set_all_seeds(0)
    rng = np.random.default_rng(0)
    X = rng.uniform(-np.pi, np.pi, (40, 4))
    y = (X[:, 0] > 0).astype(int)
    m = VQClassifier(4, depth=2, n_classes=2, max_epochs=2, seed=0)
    m.fit(X, y)

    m.grad_method = "parameter-shift"
    g_ps = m.loss_input_grad(X[:3], y[:3])
    m.grad_method = "adjoint"
    g_adj = m.loss_input_grad(X[:3], y[:3])
    assert np.allclose(g_ps, g_adj, atol=1e-10)
    assert np.isfinite(g_adj).all()


def test_rbf_svm_gradient_matches_numerical(toy_binary):
    """Classical-kernel control gradient: analytic vs central finite differences."""
    from sklearn.svm import SVC

    from qgridbench.attacks.gradients import rbf_svm_loss_input_grad

    X, y = toy_binary
    svc = SVC(C=2.0, gamma=0.3, class_weight="balanced").fit(X, y)
    xs, ys = X[:5], y[:5]
    ana = rbf_svm_loss_input_grad(svc, xs, ys)

    def bce(xrow, yi):
        f = svc.decision_function(xrow.reshape(1, -1))[0]
        p = 1.0 / (1.0 + np.exp(-f))
        return -(yi * np.log(p + 1e-12) + (1 - yi) * np.log(1 - p + 1e-12))

    h = 1e-5
    num = np.zeros_like(xs)
    for i in range(len(xs)):
        for j in range(xs.shape[1]):
            xp, xm = xs[i].copy(), xs[i].copy()
            xp[j] += h
            xm[j] -= h
            num[i, j] = (bce(xp, ys[i]) - bce(xm, ys[i])) / (2 * h)
    assert np.allclose(ana, num, atol=1e-4)


def test_qsvm_whitebox_gate_both_encodings():
    """P5a numeric gate on a toy fixture: the torch decision graph must match the
    lightning.qubit reference forward and central finite differences backward,
    for both feature maps, before any white-box QSVM number is reported."""
    from sklearn.svm import SVC

    from qgridbench.attacks.qsvm_whitebox import QSVMWhiteBox, assert_matches_reference
    from qgridbench.features.reduce import AngleScaler
    from qgridbench.models.quantum.qkernel import compute_states, fidelity_kernel

    rng = np.random.default_rng(0)
    Xtr = rng.standard_normal((24, 4)) * 2.0
    ytr = (Xtr[:, 0] + 0.5 * Xtr[:, 1] > 0).astype(int)
    Xck = rng.standard_normal((6, 4)) * 2.0
    scaler = AngleScaler().fit(Xtr)
    for enc in ("angle_ry", "zz"):
        states = compute_states(scaler.transform(Xtr), enc, bandwidth=0.8)
        K = fidelity_kernel(states, states)
        svc = SVC(kernel="precomputed", C=2.0, class_weight="balanced").fit(K, ytr)
        wb = QSVMWhiteBox(svc, states, scaler, enc, 0.8)
        rep = assert_matches_reference(wb, svc, states, scaler, Xck)
        assert rep["forward_rel_err"] < 1e-4  # vs the exact complex128 Gram
        assert rep["deployed_float32_rel_err"] < 5e-3  # vs the deployed complex64 path
        assert rep["grad_rel_err"] < 1e-3
        assert rep["predictions_agree"] == len(Xck)
        assert rep["blocking_reference"] == "deployed float32"


def test_qsvm_whitebox_gate_still_blocks_when_evaluating_at_exact_precision():
    """`deployed_tol=None` must relax ONLY the deployed comparison, never the gate.

    P9a evaluates through the exact complex128 Gram, so blocking it against the deployed
    float32 path would test a reference it never uses (decision log 2026-08-14T18:05Z).
    The risk in that change is gate erosion, so this pins what must survive it: the strict
    exact-math check still blocks, the gradient check still blocks, prediction agreement
    still blocks (against the exact path now), and BOTH agreement counts are still
    reported so the deployed discrepancy stays visible rather than disappearing.
    """
    from sklearn.svm import SVC

    from qgridbench.attacks.qsvm_whitebox import QSVMWhiteBox, assert_matches_reference
    from qgridbench.features.reduce import AngleScaler
    from qgridbench.models.quantum.qkernel import compute_states, fidelity_kernel

    rng = np.random.default_rng(3)
    Xtr = rng.standard_normal((24, 4)) * 2.0
    ytr = (Xtr[:, 0] - Xtr[:, 1] > 0).astype(int)
    Xck = rng.standard_normal((6, 4)) * 2.0
    scaler = AngleScaler().fit(Xtr)
    states = compute_states(scaler.transform(Xtr), "zz", bandwidth=0.8)
    svc = SVC(kernel="precomputed", C=2.0, class_weight="balanced").fit(
        fidelity_kernel(states, states), ytr
    )
    wb = QSVMWhiteBox(svc, states, scaler, "zz", 0.8)

    rep = assert_matches_reference(wb, svc, states, scaler, Xck, deployed_tol=None)
    assert rep["blocking_reference"] == "exact complex128"
    assert rep["forward_rel_err"] < 1e-4
    assert rep["grad_rel_err"] < 1e-3
    assert rep["predictions_agree"] == len(Xck)
    assert rep["predictions_agree_exact"] == len(Xck)
    assert "deployed_float32_rel_err" in rep and rep["predictions_agree_deployed"] <= len(Xck)

    # the strict math check must STILL raise under deployed_tol=None
    with pytest.raises(AssertionError, match="forward mismatch"):
        assert_matches_reference(wb, svc, states, scaler, Xck, deployed_tol=None, fwd_tol=1e-14)


def test_qsvm_whitebox_attack_is_bounded_and_moves_the_decision():
    """PGD through the kernel graph respects the epsilon ball and increases loss."""
    from sklearn.svm import SVC

    from qgridbench.attacks.qsvm_whitebox import QSVMWhiteBox
    from qgridbench.features.reduce import AngleScaler
    from qgridbench.models.quantum.qkernel import compute_states, fidelity_kernel

    rng = np.random.default_rng(1)
    Xtr = rng.standard_normal((24, 4)) * 2.0
    ytr = (Xtr[:, 0] + 0.5 * Xtr[:, 1] > 0).astype(int)
    Xte = rng.standard_normal((10, 4)) * 2.0
    yte = (Xte[:, 0] + 0.5 * Xte[:, 1] > 0).astype(int)
    scaler = AngleScaler().fit(Xtr)
    states = compute_states(scaler.transform(Xtr), "zz", bandwidth=0.8)
    svc = SVC(kernel="precomputed", C=2.0, class_weight="balanced").fit(
        fidelity_kernel(states, states), ytr
    )
    wb = QSVMWhiteBox(svc, states, scaler, "zz", 0.8)

    eps, scale = 0.3, Xtr.std(axis=0)
    X_adv = pgd(wb, Xte, yte, eps, scale, n_steps=10, rng=np.random.default_rng(0))
    assert np.max(np.abs(X_adv - Xte) / scale) <= eps + 1e-9
    # signed margin toward the wrong class must not improve on average
    sgn = np.where(yte == 1, 1.0, -1.0)
    assert (sgn * wb.decision_function(X_adv)).mean() < (sgn * wb.decision_function(Xte)).mean()


def test_eval_subset_is_stratified_not_a_head_slice():
    """Cost-control subsets must preserve the class balance of the full split.

    Regression guard: the saved split indices are sorted, so X[:n] is the first n
    rows in original file order rather than a sample. On the real dataset that
    slice carried a 93.4% attack rate against a true 71.0%, which deflated every
    Phase-4 macro-F1 and produced "F1 rises under attack" artifacts. This asserts
    the property that was violated, on data with the same pathology: a sorted
    index over a class-clustered population.
    """
    from qgridbench.models.classical.zoo import stratified_cap

    rng = np.random.default_rng(0)
    n = 4000
    # class-clustered population: positives concentrated early, as in file-ordered data
    y = (rng.random(n) < np.linspace(0.95, 0.55, n)).astype(int)
    X = rng.standard_normal((n, 4))

    head = y[:500].mean()
    sampled = y[stratified_cap(X, y, 500, seed=0)].mean()
    assert abs(head - y.mean()) > 0.05, "fixture must reproduce the head-slice bias"
    assert abs(sampled - y.mean()) < 0.01, "stratified subset must track the full balance"
