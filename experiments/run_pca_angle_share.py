"""E-NS1a: how much of the binary PCA-8 basis do the absolute phase-angle channels carry?

NS1 (BACKLOG, Tier 2) holds that an absolute phase angle is mostly reference-frame noise: its
marginal train std is 95-112 deg against 0.3-26 deg in its relay's positive-sequence frame
(decision log 2026-11-16T06:24Z). If the 8-component PCA spends its components on that noise,
frame-relative angles are a lever on the matched-dimensionality regime. This measures the
first half of that claim and nothing else.

Statistic (pre-registered, decision log 2026-11-16T09:19Z, E-NS1a): on the train-fit PCA(8) of
the shipped binary pipeline (median impute, StandardScaler, PCA), the explained-variance-
weighted share of squared loadings on the 52 angle channels (PA1..PA12 per relay and PA:ZH).
The uniform share is 52/128 = 0.406. Branches: > 0.50 HYPOTHESIS; <= 0.41 KILL; between
INCONCLUSIVE. The unweighted share and the per-component shares are reported beside it.

Control (blocking): the pipeline's PCA reproduces the regime's train features exactly, so the
loadings read are the ones every published PCA-8 model sees.

Usage: uv run python -m experiments.run_pca_angle_share
"""

import json
import re

import numpy as np

from qgridbench.data.preprocess import load_processed
from qgridbench.features.reduce import build_preprocessor, fit_transform_splits
from qgridbench.protocol import prepare_regime
from qgridbench.utils.run_tracking import REPO_ROOT, RunDir, get_logger

log = get_logger(__name__)

ANGLE = re.compile(r"^R[1-4]-PA(\d+:(VH|IH)|:ZH)$")
N_ANGLE = 52
HYPOTHESIS, KILL = 0.50, 0.41


def main() -> None:
    names = json.loads((REPO_ROOT / "data" / "processed" / "meta_binary.json").read_text())[
        "feature_names"
    ]
    is_angle = np.array([bool(ANGLE.match(c)) for c in names])
    if is_angle.sum() != N_ANGLE:
        raise AssertionError(f"{is_angle.sum()} angle channels, expected {N_ANGLE}")

    X, _, splits = load_processed("binary")
    pipe = build_preprocessor("pca", n_components=8, seed=0)
    Z = fit_transform_splits(pipe, X, splits)
    reg = prepare_regime("binary", "pca", n_components=8, seed=0)
    gap = float(np.abs(Z["train"] - reg.X["train"]).max())
    if gap > 0.0:
        raise AssertionError(f"pipeline does not reproduce the regime's train features: {gap}")

    pca = pipe.named_steps["pca"]
    V2 = pca.components_**2  # (8, 128); each row sums to 1
    lam = pca.explained_variance_
    per_comp = V2[:, is_angle].sum(axis=1)
    weighted = float((lam * per_comp).sum() / lam.sum())
    unweighted = float(per_comp.mean())
    branch = (
        "HYPOTHESIS" if weighted > HYPOTHESIS else "KILL" if weighted <= KILL else "INCONCLUSIVE"
    )
    top = {
        f"pc{k + 1}": [[names[j], round(float(V2[k, j]), 4)] for j in np.argsort(V2[k])[::-1][:5]]
        for k in range(len(lam))
    }
    res = {
        "n_channels": len(names),
        "n_angle": int(is_angle.sum()),
        "uniform_share": N_ANGLE / len(names),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "angle_share_per_component": per_comp.tolist(),
        "angle_share_weighted": weighted,
        "angle_share_unweighted": unweighted,
        "top_loadings": top,
        "control_pipeline_reproduces_regime": gap,
        "branch": branch,
        "preregistration": "decision log 2026-11-16T09:19Z (E-NS1a)",
    }
    with RunDir(
        "pca_angle_share_binary_q8", config={"variant": "binary", "n_components": 8}
    ) as run:
        run.write_json("results.json", res)
        run.metrics.update({"angle_share_weighted": weighted, "angle_share_unweighted": unweighted})
    log.info(
        "angle share weighted %.3f, unweighted %.3f (uniform %.3f): %s",
        weighted,
        unweighted,
        N_ANGLE / len(names),
        branch,
    )
    log.info("per component: %s", np.round(per_comp, 3).tolist())


if __name__ == "__main__":
    main()
