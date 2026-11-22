"""Per-class reading of Table III's shared transfer attack (E-NS3a, backlog NS3).

Untargeted macro-F1 retention mixes two failures: evasion (attack rows read as natural) and false
alarms (natural rows read as attack). This replays Table III's evasion stage exactly (the same
`FittedZoo`, eval subset and per-seed `spawn_rng(seed, "adv")` stream, calls in the same order as
`run_adversarial.evasion_and_transfer`) and reads every family's predictions on the shared
MLP-surrogate set at the top epsilon (and the MLP's and VQC's own white-box sets) per class:
attack-class recall and natural-class false-alarm rate, clean and attacked. Families are then
ranked by attacked attack-class recall and by shared retention; Spearman rho between the two is
the pre-registered statistic (decision log, "PRE-REGISTRATION ... E-NS3a").

Anchor (blocking): each family's replayed macro-F1 reproduces the published Table III run
(`20260813T183448Z_adversarial_binary_q8`) within 1e-6 at every seed.

Usage: uv run python -m experiments.run_evasion_per_class --workers 3 --threads 3
       uv run python -m experiments.run_evasion_per_class --variant gas_binary
           --anchor-run <stamp>_adversarial_gas_binary_q8 --workers 2 --threads 4

On another variant the anchor is that variant's own run_adversarial artifact (--anchor-run).
"""

import argparse
import os
import sys

_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--threads", type=int, default=3)
THREADS = _pre.parse_known_args(sys.argv[1:])[0].threads
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "LOKY_MAX_CPU_COUNT"):
    os.environ[_var] = str(THREADS)

import json  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from functools import partial  # noqa: E402

import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

from experiments.exports import Exports  # noqa: E402
from experiments.run_adversarial import CLASSICAL, FittedZoo  # noqa: E402
from qgridbench.attacks.evasion import fgsm, pgd  # noqa: E402
from qgridbench.models.classical.zoo import stratified_cap  # noqa: E402
from qgridbench.protocol import best_params_path, prepare_regime  # noqa: E402
from qgridbench.utils.run_tracking import REPO_ROOT, RunDir, get_logger, load_yaml  # noqa: E402
from qgridbench.utils.seeding import spawn_rng  # noqa: E402

log = get_logger(__name__)

TABLE_III_RUN = "20260813T183448Z_adversarial_binary_q8"
ANCHOR_TOL = 1e-6
FAMILIES = CLASSICAL + ["qsvm", "vqc"]
KILL_RHO = 0.9


def _ctx(variant: str):
    cfg_adv = load_yaml(REPO_ROOT / "configs" / "adversarial.yaml")
    cfg_q = load_yaml(REPO_ROOT / "configs" / "quantum.yaml")
    bp = json.loads(best_params_path(variant, 8).read_text())
    reg = prepare_regime(variant, "pca", n_components=8, seed=0)
    return cfg_adv, cfg_q, bp, reg


def per_class(y: np.ndarray, pred: np.ndarray) -> dict:
    att, nat = y == 1, y == 0
    return {
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "attack_recall": float((pred[att] == 1).mean()),
        "false_alarm": float((pred[nat] == 1).mean()),
        "n_attack": int(att.sum()),
        "n_natural": int(nat.sum()),
    }


def run_seed(s: int, variant: str = "binary") -> dict:
    """Replays run_adversarial.evasion_and_transfer's RNG consumption for one seed."""
    if os.environ.get("OMP_NUM_THREADS") != str(THREADS):
        raise AssertionError(f"worker thread cap {os.environ.get('OMP_NUM_THREADS')} != {THREADS}")
    cfg_adv, cfg_q, bp, reg = _ctx(variant)
    ev = stratified_cap(reg.X["test"], reg.y["test"], cfg_adv["eval_subset_adv"], seed=0)
    X, y = reg.X["test"][ev], reg.y["test"][ev]
    scale = reg.feature_scale
    zoo = FittedZoo(reg, "zz", 8, s, cfg_q["qkernel"]["subset_cap"], bp)
    rng = spawn_rng(s, "adv")
    eps_grid = cfg_adv["evasion"]["epsilons"]
    top = max(eps_grid)
    kw = {
        "n_steps": cfg_adv["evasion"]["pgd"]["n_steps"],
        "step_frac": cfg_adv["evasion"]["pgd"]["step_frac"],
    }
    sources = cfg_adv["transfer_matrix"]["sources"]
    wb = {}
    for eps in eps_grid:  # loop 1: each differentiable source attacks itself
        for src in sources:
            gp = zoo.grad_provider(src)
            fgsm(gp, X, y, eps, scale)  # consumes no randomness; kept for call parity
            Xa = pgd(gp, X, y, eps, scale, rng=rng, **kw)
            if eps == top:
                wb[src] = Xa
    gp = zoo.grad_provider(cfg_adv["evasion"]["surrogate_model"])
    shared = None
    for eps in eps_grid:  # loop 2: the MLP surrogate for the non-differentiable families
        Xa = pgd(gp, X, y, eps, scale, rng=rng, **kw)
        if eps == top:
            shared = Xa

    top_s = str(top)
    out = {"seed": s, "families": {}}
    arrays = {
        "ev": ev,
        "y": y,
        "proba": {},
        "crafted": {(f"pgd@{top_s}|shared", "mlp"): shared},
    }
    for src, Xw in wb.items():
        arrays["crafted"][(f"pgd@{top_s}|whitebox", src)] = Xw
    for f in FAMILIES:
        p_clean = zoo.clean_proba(f, X, y)
        p_shared = zoo.predict_proba(f, shared)
        arrays["proba"][("clean", f)] = p_clean
        arrays["proba"][(f"pgd@{top_s}|shared", f)] = p_shared
        rec = {"clean": per_class(y, p_clean.argmax(1)), "shared": per_class(y, p_shared.argmax(1))}
        if f in wb:
            p_wb = zoo.predict_proba(f, wb[f])
            arrays["proba"][(f"pgd@{top_s}|whitebox", f)] = p_wb
            rec["whitebox"] = per_class(y, p_wb.argmax(1))
        out["families"][f] = rec
    out["_arrays"] = arrays
    log.info(
        "seed %d | %s",
        s,
        " ".join(
            f"{f}:{r['clean']['attack_recall']:.2f}->{r['shared']['attack_recall']:.2f}"
            for f, r in out["families"].items()
        ),
    )
    return out


def anchor(rows: list[dict], anchor_run: str = TABLE_III_RUN) -> dict:
    pub = json.loads((REPO_ROOT / "results" / "runs" / anchor_run / "results.json").read_text())[
        "per_seed"
    ]
    diffs = {}
    for r in rows:
        ev = pub[r["seed"]]["evasion"]
        for f, rec in r["families"].items():
            diffs[f"{r['seed']}|{f}|clean"] = rec["clean"]["macro_f1"] - ev["clean"][f]
            published = ev["curves"][f]["pgd"]["0.5"]
            ours = rec["whitebox"]["macro_f1"] if "whitebox" in rec else rec["shared"]["macro_f1"]
            diffs[f"{r['seed']}|{f}|attacked"] = ours - published
    worst = max(diffs, key=lambda k: abs(diffs[k]))
    by_family = {}
    for k, v in diffs.items():
        f = k.split("|")[1]
        by_family[f] = max(by_family.get(f, 0.0), abs(v))
    return {
        "max_abs_diff": abs(diffs[worst]),
        "worst": worst,
        "max_abs_diff_by_family": by_family,
        "tol": ANCHOR_TOL,
        "passed": abs(diffs[worst]) <= ANCHOR_TOL,
    }


def summarize(rows: list[dict]) -> dict:
    agg = {}
    for f in FAMILIES:
        att_kind = "whitebox" if "whitebox" in rows[0]["families"][f] else "shared"

        def m(kind, key, f=f):
            return float(np.mean([r["families"][f][kind][key] for r in rows]))

        agg[f] = {
            "attack_kind_in_table_iii": att_kind,
            "clean_macro_f1": m("clean", "macro_f1"),
            "shared_macro_f1": m("shared", "macro_f1"),
            "shared_retention": m("shared", "macro_f1") / m("clean", "macro_f1"),
            "clean_attack_recall": m("clean", "attack_recall"),
            "shared_attack_recall": m("shared", "attack_recall"),
            "clean_false_alarm": m("clean", "false_alarm"),
            "shared_false_alarm": m("shared", "false_alarm"),
            "evasion_rate_shared": m("clean", "attack_recall") - m("shared", "attack_recall"),
            "false_alarm_increase_shared": m("shared", "false_alarm") - m("clean", "false_alarm"),
        }
        if att_kind == "whitebox":
            agg[f].update(
                {
                    "whitebox_macro_f1": m("whitebox", "macro_f1"),
                    "whitebox_attack_recall": m("whitebox", "attack_recall"),
                    "whitebox_false_alarm": m("whitebox", "false_alarm"),
                }
            )
    ret = [agg[f]["shared_retention"] for f in FAMILIES]
    rec = [agg[f]["shared_attack_recall"] for f in FAMILIES]
    rho = float(spearmanr(ret, rec).statistic)
    order_ret = sorted(FAMILIES, key=lambda f: -agg[f]["shared_retention"])
    order_rec = sorted(FAMILIES, key=lambda f: -agg[f]["shared_attack_recall"])
    return {
        "families": agg,
        "spearman_retention_vs_attack_recall": rho,
        "order_by_retention": order_ret,
        "order_by_attacked_attack_recall": order_rec,
        "branch": "no_reorder (kill)" if rho >= KILL_RHO else "reorder",
    }


def write_exports(run_dir, variant: str, rows: list[dict], arrays: dict) -> None:
    """Round-4 exports (decision log 2026-11-17T02:18Z): per-seed metrics, every prediction,
    every crafted input, provenance."""
    bp_rel = best_params_path(variant, 8).relative_to(REPO_ROOT).as_posix()
    ex = Exports(variant, bp_rel)
    ex.stream("adv")
    for r in rows:
        s, a = r["seed"], arrays[r["seed"]]
        for (cond, src), X in a["crafted"].items():
            ex.crafted_input(s, cond, src, X)
        for (cond, f), proba in a["proba"].items():
            ex.predictions(s, cond, f, proba, a["y"], a["ev"])
        for f, rec in r["families"].items():
            for kind, d in rec.items():
                cond = "clean" if kind == "clean" else f"pgd@0.5|{kind}"
                for k in ("macro_f1", "attack_recall", "false_alarm"):
                    ex.metric(s, f, cond, k, d[k])
    ex.write(run_dir)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=THREADS)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--seeds", type=int, nargs="*", default=list(range(10)))
    ap.add_argument("--variant", default="binary")
    ap.add_argument("--anchor-run", default=None, help="run_adversarial artifact to replay")
    args = ap.parse_args()
    if args.anchor_run is None:
        if args.variant != "binary":
            raise SystemExit("--anchor-run is required for a variant other than binary")
        args.anchor_run = TABLE_III_RUN
    config = {**vars(args), "table_iii_artifact": args.anchor_run, "families": FAMILIES}
    name = f"evasion_per_class_{args.variant}_q8" + ("" if len(args.seeds) == 10 else "_partial")
    with RunDir(name, config=config, seeds=args.seeds) as run:
        rows, arrays = [], {}
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for r in pool.map(partial(run_seed, variant=args.variant), args.seeds):
                arrays[r["seed"]] = r.pop("_arrays")
                rows.append(r)
                run.write_json("results_partial.json", {"rows": rows})
        rows.sort(key=lambda r: r["seed"])
        write_exports(run.path, args.variant, rows, arrays)
        anc = anchor(rows, args.anchor_run)
        summary = summarize(rows)
        run.write_json("results.json", {"anchor": anc, "summary": summary, "rows": rows})
        run.metrics.update(
            {"anchor_passed": anc["passed"], "rho": summary["spearman_retention_vs_attack_recall"]}
        )
        log.info("ANCHOR %s", json.dumps(anc))
        log.info("SUMMARY %s", json.dumps(summary, indent=1))
        print(f"-> {run.path}")
        if not anc["passed"]:
            raise AssertionError(f"anchor failed: {anc['worst']} by {anc['max_abs_diff']:.2e}")


if __name__ == "__main__":
    main()
