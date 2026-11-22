"""E-M2: is the classical robustness ordering a property of the row-level split?

Pre-registered in the decision log (2026-11-16T05:15Z, "E-M2 - whole-file robustness"). The
paper's robustness rankings were measured on models trained under the published row-level
split, which P12a showed rewards memorization of near-duplicates within a source file. This
run refits the full-feature (128 standardized features, binary) classical models under BOTH
protocols and attacks them identically at eps = 0.5 sigma:

  row         the published split, model seeds 0-4
  whole_file  the five held-out partitions of Table I (`run_group_split_control`), seed 0

Families: logreg, rbf_svm (study cap), rf, xgb, lgbm, mlp (as tuned, unweighted) and
mlp_balanced (the same parameters, `fit_model(..., balance_via_sample_weight=True)`), each at
the hyperparameters of its FULL-regime Optuna study `cls_<model>_binary_full`.

Attacks per split on a stratified test subset of `eval_subset_adv` rows (PGD settings from
`configs/adversarial.yaml`, budget unit = the split's train per-feature std):
  shared      PGD crafted on the unweighted MLP surrogate, transferred to every family
  white-box   PGD on mlp_balanced and rbf_svm with their own exact gradients (for mlp the
              shared attack IS white-box, so the shared set is reused)
  random      uniform random-sign perturbation at the same L-inf radius (negative control)

DECISION RULE (fixed before the run): Spearman rho between the row-level and whole-file mean
retention rankings of the six families (logreg, rbf_svm, rf, xgb, lgbm, mlp) under the shared
attack. rho >= 0.6 supports the hypothesis; rho < 0.2 or a top/bottom swap kills it (the
rankings are then protocol-dependent); anything between is intermediate.

BLOCKING CONTROLS (assertions that raise):
  anchor    whole-file clean LightGBM reproduces Table I (0.594 +- 0.008) within 0.01; runs
            first, before any attack (`--anchor-only` runs it alone)
  negative  random-perturbation damage (clean - random macro-F1) is below shared-PGD damage;
            recorded per family and protocol, raises only when most families fail

KNOWN, STATED CHOICES
  * a partition's train split can hold constant (zero-variance) features; their standardized
    train std is 0, which the PGD scale contract rejects. Those features get scale 1.0, the
    convention StandardScaler itself applies to a constant feature. The count is recorded.
  * the RBF-SVM gradient is evaluated in row chunks: the shipped provider materializes a
    (rows x support vectors x 128) tensor, which is ~5 GB at 500 rows. Rows are independent,
    so chunking changes memory only, not the values.

Run:  uv run python -m experiments.run_whole_file_robustness --threads 10
      uv run python -m experiments.run_whole_file_robustness --anchor-only
      uv run python -m experiments.run_whole_file_robustness --smoke
      (resume: --resume results/runs/<stamp>_whole_file_robustness_binary_full/results_partial.json)
"""

from __future__ import annotations

import argparse
import json
import os
import time

# BLAS / OpenMP thread counts must be set before numpy is first imported.
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--threads", type=int, default=10)
THREADS = _pre.parse_known_args()[0].threads
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_var] = str(THREADS)

import numpy as np  # noqa: E402
import optuna  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

from experiments.run_group_split_control import group_splits, reconstruct_groups  # noqa: E402
from qgridbench.attacks.evasion import pgd  # noqa: E402
from qgridbench.attacks.gradients import GradProvider  # noqa: E402
from qgridbench.data.preprocess import load_processed  # noqa: E402
from qgridbench.eval.baselines import trivial_baselines  # noqa: E402
from qgridbench.features.reduce import build_preprocessor, fit_transform_splits  # noqa: E402
from qgridbench.models.classical.zoo import (  # noqa: E402
    build,
    fit_model,
    predict_proba_timed,
    stratified_cap,
)
from qgridbench.protocol import prepare_regime  # noqa: E402
from qgridbench.utils.run_tracking import REPO_ROOT, RunDir, get_logger, load_yaml  # noqa: E402
from qgridbench.utils.seeding import set_all_seeds, spawn_rng  # noqa: E402

log = get_logger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

VARIANT = "binary"
EPS = 0.5
RANKED = ["logreg", "rbf_svm", "rf", "xgb", "lgbm", "mlp"]  # the six ranked families
ALL_FAMILIES = [*RANKED, "mlp_balanced"]
SMOKE_FAMILIES = ["rbf_svm", "lgbm", "mlp", "mlp_balanced"]  # mlp is the surrogate: always fit
ROW_SEEDS = [0, 1, 2, 3, 4]
WHOLE_FILE_SEED = 0
N_PARTITIONS = 5
SMOKE_TRAIN_CAP = 3000
SVM_GRAD_CHUNK = 32  # rows per RBF-SVM gradient call (~1 GB of temporaries at 10k SVs)

TABLE_I_RUN = "20260816T023620Z_group_split_control_binary"  # per-partition anchor artifact
TABLE_I_MEAN = 0.594  # pre-registered anchor value
ANCHOR_TOL = 0.01
HYPOTHESIS_RHO = 0.6
KILL_RHO = 0.2
TREE_SKILL_BAR = 0.05  # hypothesis part 2: whole-file clean F1 - floor for rf, xgb, lgbm

METRICS = [
    "clean_f1",
    "clean_full_f1",
    "clean_minus_floor",
    "shared_f1",
    "shared_retention",
    "shared_minus_floor",
    "shared_damage",
    "wb_f1",
    "wb_retention",
    "wb_minus_floor",
    "random_f1",
    "random_damage",
    "ppr_clean",
    "ppr_shared",
    "ppr_wb",
    "ppr_random",
]


def full_regime_params() -> dict[str, dict]:
    """Best parameters of each `cls_<model>_binary_full` Optuna study (MLP serves both MLPs)."""
    out = {}
    for m in RANKED:
        name = f"cls_{m}_{VARIANT}_full"
        db = REPO_ROOT / "results" / f"optuna_{name}.db"
        if not db.exists():  # load_study would silently create an empty database
            raise FileNotFoundError(db)
        out[m] = optuna.load_study(
            study_name=name, storage=f"sqlite:///{db.as_posix()}"
        ).best_params
    return out


def attack_scale(Z_train: np.ndarray) -> tuple[np.ndarray, int]:
    """Train per-feature std (protocol.Regime.feature_scale); constant features get 1.0."""
    s = Z_train.std(axis=0, ddof=0)
    n_const = int((s <= 0).sum())
    return np.where(s > 0, s, 1.0), n_const


def f1(y: np.ndarray, pred: np.ndarray) -> float:
    return float(f1_score(y, pred, average="macro"))


def floor_f1(ytr: np.ndarray, y: np.ndarray) -> float:
    agg = trivial_baselines(ytr, y)["baseline_stratified_random"]["aggregate"]
    return float(agg["macro_f1"]["mean"])


def fit_family(name, params, seed, Ztr, ytr, threads, train_cap, svm_cap):
    """One estimator at one seed on the full train split (rbf_svm and smoke capped)."""
    base = "mlp" if name == "mlp_balanced" else name
    cap = train_cap
    if base == "rbf_svm":
        cap = min(cap, svm_cap) if cap else svm_cap
    idx = stratified_cap(Ztr, ytr, cap, seed=0) if cap else np.arange(len(ytr))
    set_all_seeds(seed)
    est = build(base, params[base], seed)
    if base in ("rf", "xgb", "lgbm"):
        est.set_params(n_jobs=threads)
    fit_model(base, est, Ztr[idx], ytr[idx], balance_via_sample_weight=name == "mlp_balanced")
    return est


def predict(est, X: np.ndarray) -> np.ndarray:
    return predict_proba_timed(est, X)[0].argmax(1)


def chunked(grad_fn, chunk: int):
    """Row-chunked gradient: identical values, bounded memory (rows are independent)."""

    def g(X, y):
        return np.concatenate(
            [grad_fn(X[i : i + chunk], y[i : i + chunk]) for i in range(0, len(X), chunk)]
        )

    return g


def assert_bounded(X_adv, X, scale, what):
    worst = float((np.abs(X_adv - X) / scale).max())
    assert worst <= EPS + 1e-9, f"{what}: perturbation {worst:.6f} sigma exceeds eps {EPS}"


def run_unit(*, tag, protocol, split, seed, held, Ztr, ytr, Zte, yte, ctx) -> dict:
    """Fit every family on one split/seed, attack it, and return the unit record."""
    t0 = time.perf_counter()
    scale, n_const = attack_scale(Ztr)
    ev = stratified_cap(Zte, yte, ctx["n_eval"], seed=0)
    Xe, ye = Zte[ev], yte[ev]
    floor_eval, floor_full = floor_f1(ytr, ye), floor_f1(ytr, yte)

    models = {
        name: fit_family(
            name,
            ctx["params"],
            seed,
            Ztr,
            ytr,
            ctx["threads"],
            ctx["train_cap"],
            ctx["svm_cap"],
        )
        for name in ctx["families"]
    }
    fit_s = time.perf_counter() - t0

    pgd_kw = {"n_steps": ctx["pgd"]["n_steps"], "step_frac": ctx["pgd"]["step_frac"]}
    shared = pgd(
        GradProvider(models["mlp"], "mlp"),
        Xe,
        ye,
        EPS,
        scale,
        rng=spawn_rng(seed, f"em2_{tag}_shared"),
        **pgd_kw,
    )
    assert_bounded(shared, Xe, scale, "shared PGD")
    wb_sets = {"mlp": shared}  # the shared attack is white-box on its own source
    if "mlp_balanced" in models:
        wb_sets["mlp_balanced"] = pgd(
            GradProvider(models["mlp_balanced"], "mlp"),
            Xe,
            ye,
            EPS,
            scale,
            rng=spawn_rng(seed, f"em2_{tag}_wb_mlp_balanced"),
            **pgd_kw,
        )
    if "rbf_svm" in models:
        wb_sets["rbf_svm"] = pgd(
            chunked(GradProvider(models["rbf_svm"], "rbf_svm"), SVM_GRAD_CHUNK),
            Xe,
            ye,
            EPS,
            scale,
            rng=spawn_rng(seed, f"em2_{tag}_wb_rbf_svm"),
            **pgd_kw,
        )
    for k, Xw in wb_sets.items():
        assert_bounded(Xw, Xe, scale, f"white-box PGD {k}")
    sign = spawn_rng(seed, f"em2_{tag}_random").choice([-1.0, 1.0], size=Xe.shape)
    rand = Xe + EPS * scale * sign
    assert_bounded(rand, Xe, scale, "random control")
    attack_s = time.perf_counter() - t0 - fit_s

    fams = {}
    for name, est in models.items():
        p_clean = predict(est, Xe)
        clean = f1(ye, p_clean)
        assert clean > 0, f"{name}: clean macro-F1 is 0"
        p_shared, p_rand = predict(est, shared), predict(est, rand)
        r = {
            "clean_f1": clean,
            "clean_full_f1": f1(yte, predict(est, Zte)),
            "clean_minus_floor": clean - floor_eval,
            "shared_f1": f1(ye, p_shared),
            "random_f1": f1(ye, p_rand),
            "ppr_clean": float(p_clean.mean()),
            "ppr_shared": float(p_shared.mean()),
            "ppr_random": float(p_rand.mean()),
            "wb_f1": None,
            "wb_retention": None,
            "wb_minus_floor": None,
            "ppr_wb": None,
        }
        r["shared_retention"] = r["shared_f1"] / clean
        r["shared_minus_floor"] = r["shared_f1"] - floor_eval
        r["shared_damage"] = clean - r["shared_f1"]
        r["random_damage"] = clean - r["random_f1"]
        if name in wb_sets:
            p_wb = predict(est, wb_sets[name])
            r["wb_f1"] = f1(ye, p_wb)
            r["wb_retention"] = r["wb_f1"] / clean
            r["wb_minus_floor"] = r["wb_f1"] - floor_eval
            r["ppr_wb"] = float(p_wb.mean())
        fams[name] = r

    rec = {
        "protocol": protocol,
        "split": split,
        "seed": seed,
        "held_out_files": held,
        "n_train": int(len(ytr)),
        "n_test": int(len(yte)),
        "n_eval": int(len(ye)),
        "eval_positive_rate": float(ye.mean()),
        "n_constant_features": n_const,
        "floor_eval": floor_eval,
        "floor_full": floor_full,
        "fit_s": round(fit_s, 1),
        "attack_s": round(attack_s, 1),
        "families": fams,
    }
    log.info(
        "%s seed %d | floor %.3f | %s | fit %.0fs attack %.0fs",
        tag,
        seed,
        floor_eval,
        " ".join(
            f"{n}:{r['clean_f1']:.3f}->{r['shared_f1']:.3f}({r['shared_retention']:.2f})"
            for n, r in fams.items()
        ),
        fit_s,
        attack_s,
    )
    return rec


def anchor(X, y, partitions, lgbm_params, threads) -> dict:
    """Whole-file clean LightGBM against Table I (blocking). Mirrors `score_level`."""
    ref_rows = json.loads(
        (REPO_ROOT / "results" / "runs" / TABLE_I_RUN / "results.json").read_text()
    )["rows"]
    ref = {r["partition"]: r["level_full_lgbm"] for r in ref_rows if r["split"] == "group"}
    ours = {}
    for p, sp, _ in partitions:
        pipe = build_preprocessor("full", n_components=None, seed=0)
        Z = fit_transform_splits(pipe, X, sp)
        set_all_seeds(0)
        est = build("lgbm", lgbm_params, 0)
        est.set_params(n_jobs=threads)
        fit_model("lgbm", est, Z["train"], y[sp["train"]])
        ours[p] = f1(y[sp["test"]], est.predict(Z["test"]))
        log.info("anchor p%d: ours %.4f | Table I %.4f", p, ours[p], ref[p])
    mean_ours = float(np.mean(list(ours.values())))
    mean_ref = float(np.mean([ref[p] for p in ours]))
    res = {
        "per_partition_ours": ours,
        "per_partition_table_i": {p: ref[p] for p in ours},
        "mean_ours": mean_ours,
        "std_ours": float(np.std(list(ours.values()))),
        "mean_table_i_same_partitions": mean_ref,
        "preregistered": TABLE_I_MEAN,
        "tolerance": ANCHOR_TOL,
    }
    assert abs(mean_ours - mean_ref) <= ANCHOR_TOL, f"anchor failed vs Table I artifact: {res}"
    if len(ours) == N_PARTITIONS:
        assert abs(mean_ours - TABLE_I_MEAN) <= ANCHOR_TOL, f"anchor failed vs 0.594: {res}"
    res["passed"] = True
    log.info("ANCHOR passed: mean %.4f (Table I %.4f)", mean_ours, mean_ref)
    return res


def mean_std(vals: list[float]) -> dict:
    a = np.asarray(vals, dtype=float)
    return {
        "mean": float(a.mean()),
        "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
        "n": len(a),
    }


def aggregate(records: list[dict]) -> dict:
    out = {}
    for protocol in ("row", "whole_file"):
        recs = [r for r in records if r["protocol"] == protocol]
        if not recs:
            continue
        out[protocol] = {
            "n_units": len(recs),
            "floor_eval": mean_std([r["floor_eval"] for r in recs]),
            "families": {
                f: {
                    m: mean_std(vals)
                    for m in METRICS
                    if (
                        vals := [
                            r["families"][f][m] for r in recs if r["families"][f][m] is not None
                        ]
                    )
                }
                for f in recs[0]["families"]
            },
        }
    return out


def negative_control(agg: dict) -> dict:
    """Random damage must stay below shared-PGD damage; raise only if most families fail."""
    res = {}
    for protocol, a in agg.items():
        per = {}
        for f, m in a["families"].items():
            rd, sd = m["random_damage"]["mean"], m["shared_damage"]["mean"]
            per[f] = {"random_damage": rd, "shared_damage": sd, "passed": bool(rd < sd)}
        n_fail = sum(not v["passed"] for v in per.values())
        res[protocol] = {"families": per, "n_failed": n_fail, "n_families": len(per)}
    return res


def rank_and_decide(agg: dict) -> dict:
    """Retention ranking under the shared attack per protocol, Spearman rho, branch."""
    fams = [f for f in RANKED if f in agg["row"]["families"]]
    ret = {
        p: {f: agg[p]["families"][f]["shared_retention"]["mean"] for f in fams}
        for p in ("row", "whole_file")
    }
    rank = {p: sorted(fams, key=lambda f: -ret[p][f]) for p in ret}  # 1st = most retained
    rho = float(
        spearmanr([ret["row"][f] for f in fams], [ret["whole_file"][f] for f in fams]).statistic
    )
    top_bottom_swap = (
        rank["row"][0] == rank["whole_file"][-1] and rank["row"][-1] == rank["whole_file"][0]
    )
    if rho < KILL_RHO or top_bottom_swap:
        branch = "KILL"
    elif rho >= HYPOTHESIS_RHO:
        branch = "HYPOTHESIS"
    else:
        branch = "INTERMEDIATE"
    trees = [f for f in ("rf", "xgb", "lgbm") if f in agg["whole_file"]["families"]]
    skill = {f: agg["whole_file"]["families"][f]["clean_minus_floor"]["mean"] for f in trees}
    return {
        "families_ranked": fams,
        "mean_shared_retention": ret,
        "ranking_most_to_least_retained": rank,
        "spearman_rho": rho,
        "top_bottom_swap": top_bottom_swap,
        "branch": branch,
        "hypothesis_rho": HYPOTHESIS_RHO,
        "kill_rho": KILL_RHO,
        "whole_file_tree_clean_minus_floor": skill,
        "whole_file_trees_keep_skill": all(v >= TREE_SKILL_BAR for v in skill.values()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=THREADS, help="BLAS/OpenMP threads and n_jobs")
    ap.add_argument("--smoke", action="store_true", help="tiny end-to-end pass, not a result")
    ap.add_argument("--anchor-only", action="store_true", help="run only the Table I anchor")
    ap.add_argument("--resume", default=None, help="results_partial.json of an interrupted run")
    args = ap.parse_args()

    cfg_adv = load_yaml(REPO_ROOT / "configs" / "adversarial.yaml")
    cfg_c = load_yaml(REPO_ROOT / "configs" / "classical.yaml")
    pgd_cfg = dict(cfg_adv["evasion"]["pgd"])
    n_eval = cfg_adv["eval_subset_adv"]
    families, row_seeds, n_part, train_cap = ALL_FAMILIES, ROW_SEEDS, N_PARTITIONS, None
    if args.smoke:
        families, row_seeds, n_part = SMOKE_FAMILIES, ROW_SEEDS[:1], 2
        n_eval, pgd_cfg["n_steps"], train_cap = 40, 3, SMOKE_TRAIN_CAP

    params = full_regime_params()
    X, y, gid = reconstruct_groups()
    Xp, yp, splits_row = load_processed(VARIANT)
    assert X.shape == Xp.shape, f"group reconstruction misaligned: {X.shape} vs {Xp.shape}"
    assert np.array_equal(y, yp), "label vector diverged from the shipped artifact"
    rng_part = np.random.default_rng(0)  # same stream as Table I: partition sets match it
    partitions = []
    for p in range(n_part):
        sp, held = group_splits(gid, y, rng_part)
        partitions.append((p, sp, held[2]))

    name = f"whole_file_robustness_{VARIANT}_full"
    name += "_anchor" if args.anchor_only else "_smoke" if args.smoke else ""
    config = {
        **vars(args),
        "eps": EPS,
        "families": families,
        "row_seeds": row_seeds,
        "whole_file_seed": WHOLE_FILE_SEED,
        "n_partitions": n_part,
        "n_eval": n_eval,
        "pgd": pgd_cfg,
        "svm_train_cap": cfg_c["svm_train_cap"],
        "train_cap": train_cap,
        "hyperparameters": {m: params[m] for m in RANKED},
        "hyperparameter_source": f"optuna studies cls_<model>_{VARIANT}_full (best_params)",
        "table_i_artifact": TABLE_I_RUN,
    }
    with RunDir(name, config=config, seeds=row_seeds) as run:
        anchor_res = anchor(X, y, partitions, params["lgbm"], args.threads)
        run.write_json("anchor.json", anchor_res)
        if args.anchor_only:
            print(f"-> {run.path}")
            return

        ctx = {
            "params": params,
            "families": families,
            "threads": args.threads,
            "train_cap": train_cap,
            "svm_cap": cfg_c["svm_train_cap"],
            "n_eval": n_eval,
            "pgd": pgd_cfg,
        }
        records = json.loads(open(args.resume).read())["records"] if args.resume else []
        done = {(r["protocol"], r["split"], r["seed"]) for r in records}

        def do(unit_key, **kw):
            if unit_key in done:
                log.info("resume: skipping %s", unit_key)
                return
            records.append(run_unit(ctx=ctx, **kw))
            run.write_json("results_partial.json", {"records": records})

        reg = prepare_regime(VARIANT, "full", seed=0)  # the published row-level split
        for s in row_seeds:
            do(
                ("row", "row", s),
                tag="row",
                protocol="row",
                split="row",
                seed=s,
                held=None,
                Ztr=reg.X["train"],
                ytr=reg.y["train"],
                Zte=reg.X["test"],
                yte=reg.y["test"],
            )
        for p, sp, held in partitions:
            Z = fit_transform_splits(build_preprocessor("full", n_components=None, seed=0), X, sp)
            do(
                ("whole_file", p, WHOLE_FILE_SEED),
                tag=f"wf{p}",
                protocol="whole_file",
                split=p,
                seed=WHOLE_FILE_SEED,
                held=held,
                Ztr=Z["train"],
                ytr=y[sp["train"]],
                Zte=Z["test"],
                yte=y[sp["test"]],
            )

        agg = aggregate(records)
        neg = negative_control(agg)
        decision = rank_and_decide(agg)
        out = {
            "smoke": args.smoke,
            "anchor": anchor_res,
            "records": records,
            "aggregate": agg,
            "negative_control": neg,
            "decision": decision,
        }
        run.write_json("results.json", out)
        log.info(
            "NEGATIVE CONTROL failures %s",
            {p: f"{v['n_failed']}/{v['n_families']}" for p, v in neg.items()},
        )
        log.info("DECISION\n%s", json.dumps(decision, indent=2))
        for p, v in neg.items():
            if 2 * v["n_failed"] > v["n_families"]:
                msg = f"negative control failed for most families under {p}: {v}"
                if args.smoke:
                    log.warning("smoke (not blocking): %s", msg)
                else:
                    raise AssertionError(msg)
        print(f"-> {run.path}")


if __name__ == "__main__":
    main()
