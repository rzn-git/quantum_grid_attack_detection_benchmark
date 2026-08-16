"""Import-boundary regression: shipped code (src/) must not import dev-only or
experiment orchestration layers in a way that breaks the one-way dependency rule.

For this repo the shipped surface is src/qgridbench; experiments/ and tests/ may
import it but not vice versa.
"""

import ast
import inspect
import re
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src" / "qgridbench"


def _imports(pyfile: Path) -> set[str]:
    tree = ast.parse(pyfile.read_text(encoding="utf-8"))
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods.add(node.module)
    return mods


def test_src_does_not_import_experiments_or_tests():
    for py in SRC.rglob("*.py"):
        mods = _imports(py)
        assert not any(m.startswith("experiments") for m in mods), f"{py} imports experiments"
        assert not any(m.split(".")[0] == "tests" for m in mods), f"{py} imports tests"


# --- specification guards -------------------------------------------------------------
# These exist because two defects reached the paper by being true of the SPECIFICATION
# rather than of any result: an imbalance policy that silently skipped one estimator
# (decision log 2026-08-15T10:15Z) and an attack whose seed argument does nothing
# (2026-08-15T11:05Z). The rest of the suite checks outputs; these check contracts.


def test_class_balance_route_matches_the_estimators():
    """Every model in the zoo has a DECLARED route for the imbalance policy, and the
    declaration matches what the estimator can actually accept.

    The MLP's route is "none" — a known, reported deviation (see zoo.py). This test pins
    it so it cannot change silently in either direction: if someone fixes the default,
    this test fails and forces the Phase 2/4 re-run to be a deliberate act.
    """
    from qgridbench.models.classical.zoo import CLASS_BALANCE_ROUTE, MODEL_NAMES, build

    assert set(CLASS_BALANCE_ROUTE) == set(MODEL_NAMES)
    params = {
        "logreg": {"C": 1.0},
        "rbf_svm": {"C": 1.0, "gamma": 0.1},
        "rf": {"n_estimators": 10, "max_depth": 3, "min_samples_leaf": 1, "max_features": "sqrt"},
        "xgb": {
            "n_estimators": 10,
            "learning_rate": 0.1,
            "max_depth": 3,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "min_child_weight": 1,
            "reg_lambda": 1.0,
        },
        "lgbm": {
            "n_estimators": 10,
            "learning_rate": 0.1,
            "num_leaves": 7,
            "min_child_samples": 5,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "reg_lambda": 1.0,
        },
        "mlp": {"h1": 8, "h2": 4, "alpha": 1e-4, "learning_rate_init": 1e-3, "batch_size": 128},
    }
    for name, route in CLASS_BALANCE_ROUTE.items():
        est = build(name, params[name], seed=0)
        has_cw = "class_weight" in est.get_params()
        if route == "class_weight":
            assert has_cw and est.get_params()["class_weight"] in (
                "balanced",
                "balanced_subsample",
            ), f"{name} declares class_weight but is not balanced"
        else:
            assert not has_cw, f"{name} declares '{route}' but does accept class_weight"


def test_mlp_deviation_is_correctable_not_impossible():
    """The corrected fit must remain available from shipped code, not only from a probe."""
    from sklearn.neural_network import MLPClassifier

    assert "sample_weight" in inspect.signature(MLPClassifier.fit).parameters, (
        "MLPClassifier.fit no longer accepts sample_weight; the corrected-surrogate control "
        "in the paper needs a different implementation"
    )


def test_seeded_attacks_are_actually_reproducible():
    """FGSM/PGD/label-noise take an explicit Generator and must be bit-reproducible.

    HopSkipJump is NOT — ART seeds its initial point from an unseeded RandomState, so
    `attacks/blackbox.py`'s seed argument is inert (decision log 2026-08-15T11:05Z). That
    is documented there rather than asserted here; this test guards the attacks we DO
    control, which is what makes the black-box exception visible instead of assumed.
    """
    from qgridbench.attacks.evasion import fgsm, pgd
    from qgridbench.attacks.label_noise import flip_labels

    rng0 = np.random.default_rng(0)
    X = rng0.standard_normal((32, 5))
    y = (rng0.random(32) > 0.5).astype(int)
    scale = np.abs(X).std(axis=0) + 0.5

    def grad(Xa, ya):
        return np.sign(Xa) * (2.0 * ya - 1.0)[:, None]

    a = pgd(grad, X, y, 0.3, scale, n_steps=5, rng=np.random.default_rng(7))
    b = pgd(grad, X, y, 0.3, scale, n_steps=5, rng=np.random.default_rng(7))
    assert np.array_equal(a, b), "PGD is not reproducible at a fixed generator"
    assert not np.array_equal(
        a, pgd(grad, X, y, 0.3, scale, n_steps=5, rng=np.random.default_rng(8))
    ), "PGD ignores its rng"

    assert np.array_equal(fgsm(grad, X, y, 0.3, scale), fgsm(grad, X, y, 0.3, scale))
    assert np.array_equal(
        flip_labels(y, 0.2, np.random.default_rng(3)),
        flip_labels(y, 0.2, np.random.default_rng(3)),
    )


def test_log_entries_never_predate_their_artifacts():
    """A decision-log entry may not be stamped before a run it reports.

    Header MONOTONICITY is deliberately not asserted: two sessions append here concurrently,
    so a later append can carry an earlier truthful stamp, and nine such inversions exist by
    design (see the log's ordering-semantics header). What IS checkable is causality --
    `results/runs/<UTC>_<name>` is stamped by RunDir at run START, so an entry citing that run
    cannot predate it. That catches the real defect (a stamp written from an estimate, as
    happened 2026-08-15) without forcing truthful out-of-order stamps to be rewritten.
    """
    log = Path(__file__).resolve().parents[1] / "experiment_decision_log.md"
    runs = Path(__file__).resolve().parents[1] / "results" / "runs"
    if not log.exists() or not runs.exists():
        return  # log/artifacts are not part of a fresh checkout
    on_disk = {d.name for d in runs.iterdir() if d.is_dir()}
    if not on_disk:
        return

    def iso(stamp: str) -> str:
        return f"{stamp[0:4]}-{stamp[4:6]}-{stamp[6:8]}T{stamp[9:11]}:{stamp[11:13]}Z"

    text = log.read_text(encoding="utf-8", errors="replace")
    parts = re.split(r"(?m)^## (\S+Z) — .*$", text)
    # parts = [preamble, stamp1, body1, stamp2, body2, ...]
    for header, body in zip(parts[1::2], parts[2::2]):
        cited = {
            c for c in re.findall(r"20\d{6}T\d{6}Z", body) if any(d.startswith(c) for d in on_disk)
        }
        if cited and iso(max(cited)) > header:
            raise AssertionError(
                f"log entry {header} cites run artifact {iso(max(cited))}, which did not exist "
                "yet -- the header was written from an estimate, not the clock"
            )


def test_a_new_attack_source_cannot_disturb_the_ones_before_it():
    """Adding an attack arm must not move the arms crafted ahead of it.

    THE NEAR-MISS THIS PINS (decision log 2026-08-15T20:33Z). The multi-source runners
    (`run_adversarial.py`, `run_surrogate_control.py`) draw every arm from ONE shared
    `rng`, consumed in list order. Appending the P11a RBF-SVM arm ahead of the VQC arm
    therefore shifted the VQC's random PGD start and silently changed a published
    reverse-direction number -- a result moved by where a line was inserted. It was caught
    only because an unrelated re-run happened to be bit-comparable, so it gets a test:
    the prefix of a shared-generator sequence must be invariant to what follows it.
    """
    from qgridbench.attacks.evasion import pgd

    rng0 = np.random.default_rng(0)
    X = rng0.standard_normal((16, 4))
    y = (rng0.random(16) > 0.5).astype(int)
    scale = np.abs(X).std(axis=0) + 0.5

    def grad_a(Xa, ya):
        return np.sign(Xa) * (2.0 * ya - 1.0)[:, None]

    def grad_b(Xa, ya):
        return np.tanh(Xa) * (2.0 * ya - 1.0)[:, None]

    def craft(sources):
        rng = np.random.default_rng(11)  # one stream, shared across arms, as the runners do
        return [pgd(g, X, y, 0.3, scale, n_steps=4, rng=rng) for g in sources]

    two = craft([grad_a, grad_b])
    three = craft([grad_a, grad_b, grad_a])  # a third arm APPENDED
    for i, (before, after) in enumerate(zip(two, three)):
        assert np.array_equal(before, after), (
            f"arm {i} moved when an arm was appended after it -- a new attack source must go "
            "LAST, or every arm needs its own generator"
        )
