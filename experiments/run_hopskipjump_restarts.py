"""P22: is the QSVM's 2.818 sigma HopSkipJump distortion restart-robust?

External review: "HSJ's outcome depends on its random initialization; with interleaved
decision regions (your own finding), restart variance could be large. A few restarts on
the QSVM cells makes the 'attack is fooled' claim restart-robust."

The published black-box table is one HSJ run per family at np.random.seed(0)
(attacks/blackbox.py). HSJ starts from a random adversarial point and binary-searches
back, so the initialisation is a real degree of freedom -- and the paper's own mechanism
claim (decision regions interleave at fine scale, so the line search terminates far away)
predicts that a different start could terminate somewhere else entirely.

H: the QSVM's median L-inf distortion varies materially across HSJ initialisations.
PRE-REGISTERED KILL CRITERION, decided before looking:
  - restart-robust if the QSVM's median distortion has coefficient of variation <= 0.10
    across restarts AND the QSVM stays the highest-distortion family in every restart;
    the 2.818 sigma figure then stands as published.
  - otherwise the number must be reported as mean +- std over restarts, and the
    "hardest target" wording scoped to the restart budget.
CONTROLS: the MLP (0.867 sigma) and RBF-SVM (0.000 sigma) get the identical treatment,
so restart variance specific to the fidelity kernel is separable from HSJ's variance in
general.

Run: python -m experiments.run_hopskipjump_restarts [n_restarts] [family ...] [--variant V]
     [--start pool] [--restarts 0 1 ...] [--max-iter N --init-eval N --max-eval N]

--start pool supplies each input's starting point from training rows the model assigns to
another class (attacks/blackbox.py, other_class_starts), for the models on which the library's
random draws never leave the predicted class. --restarts runs a subset of the restart indices,
so one model's restarts can run as separate processes; experiments/pool_hsj_restarts.py pools
them. --max-iter, --init-eval and --max-eval replace the published attack settings of
configs/adversarial.yaml for a query-budget sweep (HopSkipJump has no query cap as such; the
run records the settings it used and the measured queries per point).

Graduated from tmp/ because the paper reports its numbers: the black-box cells of the
robustness exhibits and the Results section's restart range come from this runner's runs,
pooled by experiments/pool_hsj_restarts.py (experiments/hsj_restart_pool.yaml names them), so
the producer has to be a file the repository carries. The RunDir name is unchanged, so the
existing run directories still match.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from experiments.exports import Exports
from experiments.run_adversarial import FittedZoo
from experiments.tuned_file import add_tuned_file_args, load_tuned_file, run_name
from qgridbench.attacks.blackbox import found_mask, hopskipjump_attack, other_class_starts
from qgridbench.models.classical.zoo import stratified_cap
from qgridbench.protocol import prepare_regime
from qgridbench.utils.run_tracking import REPO_ROOT, RunDir, get_logger, load_yaml
from qgridbench.utils.seeding import spawn_rng

log = get_logger(__name__)

DEFAULT_FAMILIES = ["qsvm", "mlp", "rbf_svm"]
ENCODING = "zz"
SEED_FIT = 0
BUDGET_KEYS = ("max_iter", "init_eval", "max_eval")
PERCENTILES = (10, 30, 45, 50, 70, 90)


def budget_settings(published: dict, overrides: dict) -> dict:
    """The black-box settings of the config with any attack setting given on the command line
    in its place; a setting below 1 is an error."""
    out = dict(published)
    for key in BUDGET_KEYS:
        value = overrides.get(key)
        if value is None:
            continue
        if value < 1:
            raise ValueError(f"{key} must be at least 1, got {value}")
        out[key] = int(value)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "n_restarts",
        nargs="?",
        type=int,
        default=5,
        help="HopSkipJump restarts per family (default 5)",
    )
    ap.add_argument(
        "families",
        nargs="*",
        default=DEFAULT_FAMILIES,
        help=f"families to attack (default: {' '.join(DEFAULT_FAMILIES)})",
    )
    ap.add_argument("--variant", default="binary")
    ap.add_argument(
        "--start",
        choices=["library", "pool"],
        default="library",
        help="library: the random start of the attack library (default, as published); pool: "
        "a training row of another predicted class per input",
    )
    ap.add_argument(
        "--restarts",
        type=int,
        nargs="*",
        default=None,
        help="restart indices to run (default: 0 .. n_restarts-1)",
    )
    for key in BUDGET_KEYS:
        ap.add_argument(
            f"--{key.replace('_', '-')}",
            type=int,
            default=None,
            help=f"HopSkipJump {key} (default: configs/adversarial.yaml, as published)",
        )
    add_tuned_file_args(ap)
    args = ap.parse_args()
    n_restarts = args.n_restarts
    families = args.families or DEFAULT_FAMILIES
    restart_ids = args.restarts if args.restarts else list(range(n_restarts))
    cfg_adv = load_yaml(REPO_ROOT / "configs" / "adversarial.yaml")
    cfg_q = load_yaml(REPO_ROOT / "configs" / "quantum.yaml")
    bb = budget_settings(cfg_adv["blackbox"], {k: getattr(args, k) for k in BUDGET_KEYS})
    pool_size = cfg_adv["blackbox_supplied_start"]["pool_size"]
    bp_rel, best_params = load_tuned_file(args.best_params, args.variant, 8)

    reg = prepare_regime(args.variant, "pca", n_components=8, seed=0)
    scale = reg.feature_scale
    bb_idx = stratified_cap(reg.X["test"], reg.y["test"], bb["eval_subset"], seed=0)
    Xc = reg.X["test"][bb_idx]

    zoo = FittedZoo(reg, ENCODING, 8, SEED_FIT, cfg_q["qkernel"]["subset_cap"], best_params)

    with RunDir(
        run_name(f"probe_hsj_restarts_{args.variant}_q8", args.run_tag),
        config={
            "variant": args.variant,
            "n_restarts": n_restarts,
            "families": families,
            "best_params": bp_rel,
            "start": args.start,
            "restarts": restart_ids,
            **({"pool_size": pool_size} if args.start == "pool" else {}),
            **bb,
        },
        seeds=restart_ids,
    ) as run:
        out = {f: [] for f in families}
        ex = Exports(args.variant, bp_rel)
        yc = reg.y["test"][bb_idx]
        for f in families:
            clean_proba = zoo.clean_proba(f, Xc, yc)
            clean_pred = clean_proba.argmax(axis=1)
            ex.predictions(SEED_FIT, "clean", f, clean_proba, yc, bb_idx)

            def predict(z, _f=f):
                return zoo.predict_proba(_f, z)

            for r in restart_ids:
                t0 = time.perf_counter()
                starts, supplied, start_queries = None, None, 0
                if args.start == "pool":
                    # the same pool rows for every model at one restart; the pick is per model
                    rows = spawn_rng(r, "hsj_start_pool").choice(
                        len(zoo.Xtr), size=pool_size, replace=False
                    )
                    starts, supplied, start_queries = other_class_starts(
                        predict, Xc, zoo.Xtr[rows], spawn_rng(r, "hsj_start_pick")
                    )
                Xa, budget = hopskipjump_attack(
                    predict,
                    Xc,
                    zoo.n_classes,
                    max_iter=bb["max_iter"],
                    max_eval=bb["max_eval"],
                    init_eval=bb["init_eval"],
                    seed=r,  # labels the restart; with --start library the start stays unseeded
                    x_adv_init=starts,
                )
                dist = (np.abs(Xa - Xc) / scale).max(axis=1)
                # a row the attack could not start on comes back unmoved (distortion 0): the
                # distortion is read over the found rows, with their share beside it
                found = found_mask(predict, Xc, Xa)
                median_found = float(np.median(dist[found])) if found.any() else float("nan")
                cond = f"hsj_restart{r}"
                ex.crafted_input(SEED_FIT, cond, f, Xa)
                ex.predictions(SEED_FIT, cond, f, zoo.predict_proba(f, Xa), yc, bb_idx)
                ex.metric(SEED_FIT, f, cond, "median_linf_sigma", float(np.median(dist)))
                ex.metric(SEED_FIT, f, cond, "mean_linf_sigma", float(dist.mean()))
                ex.metric(SEED_FIT, f, cond, "found_share", float(found.mean()))
                ex.metric(SEED_FIT, f, cond, "median_linf_sigma_found", median_found)
                rec = {
                    "restart": r,
                    "median_linf_sigma": float(np.median(dist)),
                    "mean_linf_sigma": float(dist.mean()),
                    "found_share": float(found.mean()),
                    "median_linf_sigma_found": median_found,
                    # the per-point distances' percentiles over all points (a query-budget
                    # sweep reads the lower tail against the random-noise control)
                    "linf_sigma_percentiles": {
                        str(q): float(np.percentile(dist, q)) for q in PERCENTILES
                    },
                    "queries_per_sample": budget["queries_per_sample"],
                    # step-size searches the bounded attack ended (the library would stall)
                    "stalled_steps": budget["stalled_steps"],
                    "seconds": round(time.perf_counter() - t0, 1),
                    # which way the found rows cross: per clean predicted class
                    "by_clean_prediction": {
                        str(c): {
                            "n": int((clean_pred == c).sum()),
                            "found": int((found & (clean_pred == c)).sum()),
                            "median_linf_sigma_found": float(
                                np.median(dist[found & (clean_pred == c)])
                            )
                            if (found & (clean_pred == c)).any()
                            else None,
                        }
                        for c in range(zoo.n_classes)
                    },
                }
                if supplied is not None:
                    start_dist = (np.abs(starts - Xc) / scale).max(axis=1)
                    rec["supplied_share"] = float(supplied.mean())
                    rec["start_queries_per_sample"] = start_queries / len(Xc)
                    rec["start_linf_sigma_median"] = (
                        float(np.median(start_dist[supplied])) if supplied.any() else None
                    )
                    ex.crafted_input(SEED_FIT, f"hsj_start{r}", f, starts)
                out[f].append(rec)
                run.write_json("results_partial.json", out)
                log.info(
                    "%s restart %d -> median %.3f sigma (%.0f s)",
                    f,
                    r,
                    rec["median_linf_sigma"],
                    rec["seconds"],
                )

        summary = {}
        for f in families:
            m = np.array([r["median_linf_sigma"] for r in out[f]])
            summary[f] = {
                "mean": float(m.mean()),
                "std": float(m.std(ddof=1)) if len(m) > 1 else 0.0,
                "min": float(m.min()),
                "max": float(m.max()),
                "cv": float(m.std(ddof=1) / m.mean()) if len(m) > 1 and m.mean() else float("nan"),
            }
            share = np.array([r["found_share"] for r in out[f]])
            mf = np.array([r["median_linf_sigma_found"] for r in out[f]])
            summary[f]["found_share"] = {
                "mean": float(share.mean()),
                "min": float(share.min()),
                "max": float(share.max()),
            }
            summary[f]["found"] = {
                "mean": float(mf.mean()),
                "std": float(mf.std(ddof=1)) if len(mf) > 1 else 0.0,
                "min": float(mf.min()),
                "max": float(mf.max()),
            }
        # does the QSVM stay the hardest target in every restart?
        ordering_holds = (
            all(
                out["qsvm"][r]["median_linf_sigma"]
                > max(out[f][r]["median_linf_sigma"] for f in families if f != "qsvm")
                for r in range(len(restart_ids))
            )
            if "qsvm" in families and len(families) > 1
            else None
        )
        verdict = (
            "RESTART-ROBUST — published figure stands"
            if "qsvm" in summary and summary["qsvm"]["cv"] <= 0.10 and ordering_holds
            else "NOT RESTART-ROBUST — report mean +- std and scope the wording"
        )
        ex.extra["hsj_seed"] = "hopskipjump_attack(seed=restart): ART's random start"
        ex.extra["hsj_start"] = args.start
        ex.write(run.path)
        run.write_json(
            "results.json",
            {
                "per_family": out,
                "summary": summary,
                "qsvm_highest_every_restart": ordering_holds,
                "verdict": verdict,
                "published_qsvm": 2.8185,
            },
        )
        log.info("VERDICT: %s | %s", verdict, json.dumps(summary))
        print(f"-> {run.path}")


if __name__ == "__main__":
    main()
