"""P17e-H3: the MATCHED CLASSICAL CONTROL that the first P17e run pre-registered and skipped.

Without it, "the fidelity kernel's decision regions interleave at fine scale" is uncontrolled:
any kernel machine at a wide length scale might cross many boundaries along a ray, in which
case the crossing counts describe kernel machines, not this feature map.

Identical protocol to `experiments/run_ray_crossings.py`: same evaluation subset, same rng seed and
radius so the RAYS ARE THE SAME, same dense sampling, same statistic. Only the model changes --
tuned RBF-SVM in the same PCA-8 space, swept over multipliers of its tuned gamma, mirroring the
quantum bandwidth sweep (the same pairing `bandwidth_fragility_classical` uses).

The headline comparison is at each family's OWN accuracy optimum, since that is where the paper
claims the fidelity kernel is fragile.

Run:  python -m experiments.run_ray_crossings_classical

R4-17(b): `--dump-rays` also records each setting's endpoint flip rate and writes per-ray counts,
crossing indices and flips to `rays.npz` under the RunDir
`ray_crossings_classical_dump_binary_q8`; blocking anchor: every archived row
(`20260815T211558Z_probe_ray_crossings_classical_binary_q8`) is reproduced exactly. Serial: the
whole scan took about 1 min in the archive, so a pool would cost more than it saves.

Run:  python -m experiments.run_ray_crossings_classical --dump-rays

Graduated from tmp/ because make_report consumes its run dir for the paper's
ray_crossings_binary_q8 table; the RunDir name keeps its original `probe_` prefix so
already-recorded results still resolve.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from sklearn.metrics import f1_score

from experiments.run_ablations import kernel_subset
from experiments.run_ray_crossings import (
    ARCHIVED_CLASSICAL_RUN,
    ray_set,
    ray_stats,
    save_rays,
)
from qgridbench.models.classical.zoo import build, fit_model
from qgridbench.protocol import prepare_regime
from qgridbench.utils.run_tracking import REPO_ROOT, RunDir, get_logger, load_yaml
from qgridbench.utils.seeding import set_all_seeds

log = get_logger(__name__)

N_SAMPLES, RADIUS = 200, 0.5
GAMMA_MULTS = (0.01, 0.1, 1.0, 10.0, 100.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-rays", action="store_true", help="R4-17(b): write rays.npz + anchor")
    ap.add_argument("--n-rays", type=int, default=None, help="rays per setting (default 100)")
    args = ap.parse_args()
    cfg = load_yaml(REPO_ROOT / "configs" / "quantum.yaml")["ablations"]["bandwidth_fragility"]
    best = json.loads((REPO_ROOT / "results" / "adv_best_params_binary_q8.json").read_text())
    q = cfg["cell"][1]

    reg = prepare_regime("binary", "pca", n_components=q, seed=0)
    sub = kernel_subset(reg, 2000, seed=0)
    Xtr, ytr = reg.X["train"][sub], reg.y["train"][sub]
    # SAME ray set as the quantum arm (block 0 = the archived 100 rays, then any extra rays)
    blocks = ray_set(reg, cfg["eval_subset"], args.n_rays or cfg["eval_subset"], RADIUS)
    X_ev, y_ev, delta = blocks[0]
    ts = np.linspace(0.0, 1.0, N_SAMPLES)

    name = (
        "ray_crossings_classical_dump_binary_q8"
        if args.dump_rays
        else "probe_ray_crossings_classical_binary_q8"
    )
    cfg_run = {
        "gamma_mults": list(GAMMA_MULTS),
        "dump_rays": args.dump_rays,
        "n_rays": args.n_rays,
    }
    with RunDir(name, config=cfg_run) as run:
        rows, dumps = [], {}
        base_gamma = None
        for m in GAMMA_MULTS:
            p = dict(best["rbf_svm"])
            set_all_seeds(0)
            est = build("rbf_svm", p, 0)
            g = est.get_params()["gamma"]
            if isinstance(g, str):  # 'scale' -> resolve numerically so the sweep is meaningful
                g = 1.0 / (Xtr.shape[1] * Xtr.var())
            base_gamma = base_gamma or float(g)
            est.set_params(gamma=base_gamma * m)
            fit_model("rbf_svm", est, Xtr, ytr)

            F = np.stack([est.decision_function(X_ev + t * delta) for t in ts])
            s = np.sign(F)
            s[s == 0] = 1.0
            crossings = (np.diff(s, axis=0) != 0).sum(axis=0)
            if args.dump_rays:
                extra = [
                    np.stack([est.decision_function(X + t * d) for t in ts])
                    for X, _y, d in blocks[1:]
                ]
                F_all = np.concatenate([F, *extra], axis=1)
                dumps[str(float(m))] = ray_stats(F_all)
            rows.append(
                {
                    "gamma_mult": float(m),
                    "gamma": float(base_gamma * m),
                    "clean_macro_f1": float(
                        f1_score(y_ev, (F[0] > 0).astype(int), average="macro")
                    ),
                    "crossings_mean": float(crossings.mean()),
                    "crossings_max": int(crossings.max()),
                    "frac_rays_multi_crossing": float(np.mean(crossings > 1)),
                }
            )
            log.info(
                "gamma x%.2g (=%.4g): F1 %.3f | crossings mean %.2f (max %d, >1 on %.0f%%)",
                m,
                base_gamma * m,
                rows[-1]["clean_macro_f1"],
                rows[-1]["crossings_mean"],
                rows[-1]["crossings_max"],
                100 * rows[-1]["frac_rays_multi_crossing"],
            )
        peak = max(rows, key=lambda r: r["clean_macro_f1"])
        if args.dump_rays:
            save_rays(run.path / "rays.npz", dumps)
            arch = json.loads((REPO_ROOT / ARCHIVED_CLASSICAL_RUN / "results.json").read_text())
            mism = {
                f"{a['gamma_mult']}:{k}": (r[k], a[k])
                for r, a in zip(rows, arch["rows"])
                for k in a
                if r[k] != a[k]
            }
            flips = {k: float(np.mean(v["flip"])) for k, v in dumps.items()}
            run.write_json(
                "results.json",
                {
                    "rows": rows,
                    "flip_rate_all_rays": flips,
                    "anchor": {"mismatches": mism, "pass": not mism},
                },
            )
            run.metrics.update({"anchor_pass": not mism})
            log.info("anchor mismatches %s | flip rates %s", mism, flips)
            if mism:
                raise RuntimeError(f"R4-17(b) classical anchor FAILED: {mism}")
            print(f"-> {run.path}")
            return
        run.write_json("results.json", {"rows": rows, "accuracy_peak": peak})
        log.info("CLASSICAL PEAK\n%s", json.dumps(peak, indent=2))
        print(f"-> {run.path}")


if __name__ == "__main__":
    main()
