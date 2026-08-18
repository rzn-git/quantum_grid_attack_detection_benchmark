"""Time HopSkipJump per model family to size the query budget from measurement.

The configured budget (max_iter=20, max_eval=1000, n=200) failed to finish one
family in 4.2h. This probe measures cost/sample at a small budget on the cheapest
and most expensive families, so the shipped budget is chosen from data.

Run:  python -m experiments.run_hopskipjump_budget

Graduated from tmp/ because its measurement set a SHIPPED value: the blackbox budget in
configs/adversarial.yaml cites this script as the justification for max_iter/max_eval, so
the config's provenance pointer has to resolve to a file the repository actually carries.
"""

from __future__ import annotations

import json
import time

import numpy as np

from qgridbench.attacks.blackbox import hopskipjump_attack
from qgridbench.protocol import kernel_subset, prepare_regime
from qgridbench.utils.run_tracking import REPO_ROOT, get_logger, load_yaml

log = get_logger(__name__)


def main() -> None:
    from experiments.run_adversarial import FittedZoo

    cfg_q = load_yaml(REPO_ROOT / "configs" / "quantum.yaml")
    best = json.loads((REPO_ROOT / "results" / "adv_best_params_binary_q8.json").read_text())
    reg = prepare_regime("binary", "pca", n_components=8, seed=0)

    t0 = time.perf_counter()
    zoo = FittedZoo(reg, "zz", 8, 0, cfg_q["qkernel"]["subset_cap"], best)
    log.info("zoo fit in %.1fs", time.perf_counter() - t0)

    X = reg.X["test"][:20]
    out = {}
    for fam in ("logreg", "rf", "mlp", "qsvm", "vqc"):
        t = time.perf_counter()
        _, budget = hopskipjump_attack(
            lambda z, _f=fam: zoo.predict_proba(_f, z),
            X,
            zoo.n_classes,
            max_iter=5,
            max_eval=100,
            init_eval=10,
        )
        dt = time.perf_counter() - t
        out[fam] = {
            "sec_per_sample": dt / len(X),
            "queries_per_sample": budget["queries_per_sample"],
            "sec_per_1k_queries": dt / max(budget["total_queries"], 1) * 1000,
        }
        log.info(
            "%s: %.2fs/sample, %.0f q/sample",
            fam,
            out[fam]["sec_per_sample"],
            out[fam]["queries_per_sample"],
        )

    out_dir = REPO_ROOT / "tmp"  # gitignored, so absent on a fresh clone
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "hsj_budget_probe.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    # projection at the configured budget (16x the queries of this probe)
    print("\nprojected minutes @ n=200, max_iter=20/max_eval=1000 (~16x queries):")
    for fam, v in out.items():
        print(f"  {fam}: {v['sec_per_sample'] * 16 * 200 / 60:.0f} min")
    print("kernel subset:", len(kernel_subset(reg, cfg_q["qkernel"]["subset_cap"], seed=0)))
    print("test pool:", len(reg.X["test"]), "features:", np.shape(reg.X["test"])[1])


if __name__ == "__main__":
    main()
