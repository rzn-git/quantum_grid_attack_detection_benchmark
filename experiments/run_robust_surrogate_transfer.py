"""P28: does adversarially training the surrogate re-open the closed transfer asymmetry?

THE GAP THIS PROBES. Every model in the study is undefended, and the one prior quantum
machine learning result on this dataset ships WITH a defense — the most predictable
reviewer question the paper cannot answer. The full adversarial-training study is
journal-revision scope (the VQC arm costs ~20x a training run); this probe buys the cheap
arm first: harden only the SURROGATE and ask whether the corrected transfer picture
(P19d: class weighting collapses the 10x asymmetry to ~1x) survives a hardened attacker.

Two surrogates per seed, identical in architecture, hyperparameters, class weighting and
early-stopping rule; the training examples are the only difference:
  std — the class-weighted torch MLP exactly as in P19d (the corrected arm);
  at  — the same model trained Madry-style: each minibatch replaced by PGD-10 examples at
        the evaluation budget (epsilon 0.5 sigma, step fraction 0.25), weighted loss
        retained, early stopping on CLEAN validation macro-F1.

Pre-registered (decision log 2026-08-20T18:27Z): P1 — the AT surrogate's attacks transfer
MORE (mean target damage strictly higher). P2 — the closed asymmetry STAYS closed
(forward/reverse <= 1.5x). KILL — asymmetry >= 2x under the AT surrogate means the
"mis-fitted surrogate manufactures the asymmetry" instance needs a scope note. Positive
control gating interpretation: the std arm must reproduce P19d's corrected picture
(asymmetry near 1x) before the AT arm is read.

Run:  python -m experiments.run_robust_surrogate_transfer [seeds ...]

R4-18 (decision log 2026-11-16T11:34Z, round-4 part-B amendment): `--rng-order p19d` re-runs P28
with the bounded statistic B = (F - R)/(|F| + |R|), F the surrogate -> VQC drop and R the
VQC -> surrogate drop, pooled over seeds (B from the seed means of F and R), and the paired
effect D = B(AT) - B(std) with a 10,000-draw paired seed-bootstrap 95% CI. Each seed draws every
PGD random start from ONE `spawn_rng(s, "adv")` stream in P19d's order (pgd_mlp, discarded;
pgd_vqc, the reverse attack; then the std surrogate's attack, and the AT surrogate's attack from
the same generator state, so the two arms are paired).
  Blocking anchor: seeds 0-9 reproduce P19d's per-seed F, R and clean F1s (VQC and weighted
  surrogate) to 1e-6 (`20260815T093853Z_probe_review_claims_binary_q8`).
  Validity controls (all must hold, else CONTROL_FAILED and D is not read): pooled |B(std)| <=
  0.2; the AT surrogate's own white-box macro-F1 exceeds the std surrogate's by >= 0.05 (seed
  mean); the AT surrogate's clean macro-F1 is above the 0.5024 stratified-random floor (seed mean).
  Branches: HYPOTHESIS D >= 0.2 and CI lower > 0; REVERSED D <= -0.2 and CI upper < 0; KILL
  |D| < 0.1 and CI inside (-0.2, 0.2); INCONCLUSIVE otherwise.
Ruling 8 (2026-11-16T13:05Z): the runner guard applies to the VQC target and both surrogates;
a degenerate non-read target (the unweighted sklearn MLP) is dropped from P1 and listed per seed.
The validity controls are read first; when they fail, B and D are written under "not_read".
Seeds run in a process pool (`--workers`, `--threads` pinned before numpy and torch import, a
memory guard on the smoke-measured peak RSS), one JSON per seed; `--resume-from` reuses the seed
files of an earlier run.

Run:  python -m experiments.run_robust_surrogate_transfer --rng-order p19d --seeds 0 1 2 ... 19
        --workers 2 --threads 3 --rss-gb <smoke peak>
"""

from __future__ import annotations

import argparse
import os
import sys

_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--threads", type=int, default=None)
THREADS = _pre.parse_known_args(sys.argv[1:])[0].threads
if THREADS is not None:
    for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[_var] = str(THREADS)

import copy  # noqa: E402
import json  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
from sklearn.utils.class_weight import compute_class_weight  # noqa: E402

from experiments.run_adversarial import CLASSICAL, FittedZoo  # noqa: E402

# run_review_claims parses sys.argv as integer seeds at import: hide this runner's flags from it
_argv, sys.argv = sys.argv, sys.argv[:1]
from experiments.run_review_claims import WeightedMLP  # noqa: E402

sys.argv = _argv
from experiments.runner_guard import (  # noqa: E402
    DegeneratePredictorError,
    assert_clean_predictor,
)
from qgridbench.attacks.evasion import pgd  # noqa: E402
from qgridbench.attacks.gradients import GradProvider  # noqa: E402
from qgridbench.models.classical.zoo import stratified_cap  # noqa: E402
from qgridbench.protocol import prepare_regime  # noqa: E402
from qgridbench.utils.run_tracking import REPO_ROOT, RunDir, get_logger, load_yaml  # noqa: E402
from qgridbench.utils.seeding import set_all_seeds, spawn_rng  # noqa: E402

log = get_logger(__name__)

DEFAULT_SEEDS = [0, 1, 2]
EPS = 0.5
ENCODING = "zz"
AT_STEPS = 10
AT_STEP_FRAC = 0.25
TARGETS = CLASSICAL + ["qsvm", "vqc"]


class AdversarialWeightedMLP(WeightedMLP):
    """WeightedMLP trained Madry-style; everything else is inherited unchanged.

    The inner attack ascends the WEIGHTED training loss (the loss being minimized, the
    standard adversarial-training recipe); the outer evaluation attack stays the
    published unweighted-CE path via `loss_input_grad`, so the surrogate swap changes
    the model, never the attack.
    """

    def __init__(self, *args, feature_scale=None, eps=EPS, **kwargs):
        super().__init__(*args, **kwargs)
        self.feature_scale = np.asarray(feature_scale, dtype=np.float32)
        self.eps = eps

    def fit(self, X, y, X_val, y_val):
        torch.manual_seed(self.seed)
        d = X.shape[1]
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d, self.h1),
            torch.nn.ReLU(),
            torch.nn.Linear(self.h1, self.h2),
            torch.nn.ReLU(),
            torch.nn.Linear(self.h2, 2),
        )
        w = torch.tensor(
            compute_class_weight("balanced", classes=np.unique(y), y=y), dtype=torch.float32
        )
        loss_fn = torch.nn.CrossEntropyLoss(weight=w)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.alpha)
        Xt = torch.tensor(X, dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.long)
        radius = torch.tensor(self.eps * self.feature_scale)  # sigma-unit L-inf ball
        step = AT_STEP_FRAC * radius
        best, best_state, bad = -1.0, None, 0
        g = torch.Generator().manual_seed(self.seed)
        for _ in range(self.max_epochs):
            perm = torch.randperm(len(Xt), generator=g)
            self.net.train()
            for i in range(0, len(Xt), self.batch_size):
                idx = perm[i : i + self.batch_size]
                xb, yb = Xt[idx], yt[idx]
                # PGD-10 on the weighted training loss, then train on the adversarial batch
                delta = torch.zeros_like(xb)
                for _k in range(AT_STEPS):
                    delta.requires_grad_(True)
                    loss = loss_fn(self.net(xb + delta), yb)
                    (grad,) = torch.autograd.grad(loss, delta)
                    delta = (delta + step * grad.sign()).clamp(-radius, radius).detach()
                opt.zero_grad()
                loss_fn(self.net(xb + delta), yb).backward()
                opt.step()
            # early stopping on CLEAN val macro-F1, mirroring the std arm exactly
            f1 = f1_score(y_val, self.predict_proba(X_val).argmax(1), average="macro")
            if f1 > best + 1e-5:
                best, bad = f1, 0
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        self.net.load_state_dict(best_state)
        self.best_val_f1 = float(best)
        return self


# ---- R4-18: bounded statistic, P19d random-number order -----------------------------------

P19D_RUN = "results/runs/20260815T093853Z_probe_review_claims_binary_q8"
ANCHOR_TOL = 1e-6
ANCHOR_SEEDS = range(10)  # P19d's seeds
B_BAND = 0.2
ROBUST_MIN = 0.05  # E12b's "defense is real" bar
FLOOR = 0.5024  # stratified-random macro-F1 at n = 500 (P19d floors.evasion_n500)
D_HYP, D_KILL, D_KILL_CI = 0.2, 0.1, 0.2
N_BOOT = 10_000
MEM_FRACTION = 0.8


def bounded_b(f, r):
    """B = (F - R)/(|F| + |R|) in [-1, 1]; 0 when both drops are zero."""
    den = abs(f) + abs(r)
    return 0.0 if den == 0 else float((f - r) / den)


def pooled_b(F, R) -> float:
    return bounded_b(float(np.mean(F)), float(np.mean(R)))


def paired_delta_ci(Fs, Rs, Fa, Ra, n_boot: int, seed: int = 0) -> dict:
    """D = B(AT) - B(std), pooled, with a paired seed-bootstrap 95% CI."""
    Fs, Rs, Fa, Ra = (np.asarray(x, dtype=float) for x in (Fs, Rs, Fa, Ra))
    idx = np.random.default_rng(seed).integers(0, len(Fs), size=(n_boot, len(Fs)))

    def pb(F, R):
        f, r = F[idx].mean(1), R[idx].mean(1)
        den = np.abs(f) + np.abs(r)
        return np.where(den == 0, 0.0, (f - r) / np.where(den == 0, 1.0, den))

    d = pb(Fa, Ra) - pb(Fs, Rs)
    return {
        "delta": pooled_b(Fa, Ra) - pooled_b(Fs, Rs),
        "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
    }


def r4_18_branch(delta: float, ci: list) -> str:
    lo, hi = ci
    if delta >= D_HYP and lo > 0:
        return "HYPOTHESIS"
    if delta <= -D_HYP and hi < 0:
        return "REVERSED"
    if abs(delta) < D_KILL and lo > -D_KILL_CI and hi < D_KILL_CI:
        return "KILL"
    return "INCONCLUSIVE"


def _peak_rss_gb() -> float:
    import psutil

    mi = psutil.Process().memory_info()
    return (mi.peak_wset if sys.platform == "win32" else mi.rss) / 2**30


def run_seed_p19d(s: int) -> dict:
    """One seed of R4-18, consuming the PGD stream exactly as run_review_claims (P19d) does."""
    if THREADS is not None:
        if os.environ.get("OMP_NUM_THREADS") != str(THREADS):
            raise AssertionError(f"thread cap {os.environ.get('OMP_NUM_THREADS')} != {THREADS}")
        torch.set_num_threads(THREADS)
    cfg_adv = load_yaml(REPO_ROOT / "configs" / "adversarial.yaml")
    cfg_q = load_yaml(REPO_ROOT / "configs" / "quantum.yaml")
    bp = json.loads((REPO_ROOT / "results" / "adv_best_params_binary_q8.json").read_text())
    mp = bp["mlp"]
    pgd_kw = {
        "n_steps": cfg_adv["evasion"]["pgd"]["n_steps"],
        "step_frac": cfg_adv["evasion"]["pgd"]["step_frac"],
    }
    reg = prepare_regime("binary", "pca", n_components=8, seed=0)
    scale = reg.feature_scale
    ev = stratified_cap(reg.X["test"], reg.y["test"], cfg_adv["eval_subset_adv"], seed=0)
    va = stratified_cap(reg.X["val"], reg.y["val"], 2000, seed=0)
    Xte, yte = reg.X["test"][ev], reg.y["test"][ev]
    Xva, yva = reg.X["val"][va], reg.y["val"][va]

    rng = spawn_rng(s, "adv")  # ONE stream per seed, as P19d
    zoo = FittedZoo(reg, ENCODING, 8, s, cfg_q["qkernel"]["subset_cap"], bp)
    common = (mp["h1"], mp["h2"], mp["alpha"], mp["learning_rate_init"], mp["batch_size"])
    set_all_seeds(s)
    std = WeightedMLP(*common, s).fit(zoo.Xtr, zoo.ytr, Xva, yva)
    set_all_seeds(s)
    at = AdversarialWeightedMLP(*common, s, feature_scale=scale).fit(zoo.Xtr, zoo.ytr, Xva, yva)

    # Ruling 8 (2026-11-16T13:05Z): the guard applies to the models B reads (the VQC target and
    # both surrogates). A degenerate non-read target is recorded and left out of P1, never used.
    clean, dropped = {"vqc": zoo.clean_proba("vqc", Xte, yte)}, []
    for t in TARGETS:
        if t == "vqc":
            continue
        clean[t] = zoo.clean_proba(t, Xte, yte)
        try:
            assert_clean_predictor(t, yte, clean[t], clean[t].argmax(1))
        except DegeneratePredictorError as e:
            dropped.append({"target": t, "reason": str(e)})
    p1_targets = [t for t in TARGETS if t not in {d["target"] for d in dropped}]
    clean_f1 = {t: float(f1_score(yte, p.argmax(1), average="macro")) for t, p in clean.items()}

    # P19d's order: pgd_mlp (discarded, it only advances the stream), then pgd_vqc
    pgd(zoo.grad_provider("mlp"), Xte, yte, EPS, scale, rng=rng, **pgd_kw)
    Xadv_vqc = pgd(zoo.grad_provider("vqc"), Xte, yte, EPS, scale, rng=rng, **pgd_kw)
    state = copy.deepcopy(rng.bit_generator.state)

    rec = {"seed": s, "clean": clean_f1, "dropped_targets": dropped, "p1_targets": p1_targets}
    for arm, sur in (("std", std), ("at", at)):
        rng.bit_generator.state = copy.deepcopy(state)  # paired random start for both arms
        p_clean = sur.predict_proba(Xte)
        assert_clean_predictor(f"surrogate_{arm}", yte, p_clean, p_clean.argmax(1))
        sur_clean = float(f1_score(yte, p_clean.argmax(1), average="macro"))
        Xadv = pgd(GradProvider(sur, "vqc"), Xte, yte, EPS, scale, rng=rng, **pgd_kw)
        adv = {
            t: float(f1_score(yte, zoo.predict_proba(t, Xadv).argmax(1), average="macro"))
            for t in TARGETS
        }
        own = float(f1_score(yte, sur.predict_proba(Xadv).argmax(1), average="macro"))
        rev = float(f1_score(yte, sur.predict_proba(Xadv_vqc).argmax(1), average="macro"))
        F, R = clean_f1["vqc"] - adv["vqc"], sur_clean - rev
        rec[arm] = {
            "surrogate_val_f1": sur.best_val_f1,
            "surrogate_clean_f1": sur_clean,
            "surrogate_own_adv_f1": own,
            "adv": adv,
            "F": F,
            "R": R,
            "B": bounded_b(F, R),
            "mean_target_damage": float(np.mean([clean_f1[t] - adv[t] for t in p1_targets])),
        }
    rec["peak_rss_gb"] = _peak_rss_gb()
    return rec


def p19d_reference() -> dict:
    per = json.loads((REPO_ROOT / P19D_RUN / "results.json").read_text())["per_seed"]
    out = {}
    for s, p in enumerate(per):
        f = p["f1"]
        out[s] = {
            "F": f["clean"]["vqc"] - f["pgd_mlp_weighted"]["vqc"],
            "R": f["clean"]["mlp_weighted"] - f["pgd_vqc"]["mlp_weighted"],
            "vqc_clean": f["clean"]["vqc"],
            "sur_clean": f["clean"]["mlp_weighted"],
        }
    return out


def summarize_r4_18(per_seed: list[dict]) -> dict:
    ref = p19d_reference()
    anchor = {}
    for p in per_seed:
        s = p["seed"]
        if s in ANCHOR_SEEDS:
            got = {
                "F": p["std"]["F"],
                "R": p["std"]["R"],
                "vqc_clean": p["clean"]["vqc"],
                "sur_clean": p["std"]["surrogate_clean_f1"],
            }
            anchor[str(s)] = {k: abs(got[k] - ref[s][k]) for k in got}
    worst = max((max(v.values()) for v in anchor.values()), default=None)
    Fs = [p["std"]["F"] for p in per_seed]
    Rs = [p["std"]["R"] for p in per_seed]
    Fa = [p["at"]["F"] for p in per_seed]
    Ra = [p["at"]["R"] for p in per_seed]
    b_std, b_at = pooled_b(Fs, Rs), pooled_b(Fa, Ra)
    gains = [p["at"]["surrogate_own_adv_f1"] - p["std"]["surrogate_own_adv_f1"] for p in per_seed]
    at_clean = float(np.mean([p["at"]["surrogate_clean_f1"] for p in per_seed]))
    controls = {
        "anchor_worst_abs_dev_seeds_0_9": worst,
        "anchor_pass": worst is not None and worst <= ANCHOR_TOL,
        "anchor_per_seed": anchor,
        "b_std_pooled": b_std,
        "b_std_in_band": abs(b_std) <= B_BAND,
        "at_robustness_gain": float(np.mean(gains)),
        "at_is_robust": float(np.mean(gains)) >= ROBUST_MIN,
        "at_clean_f1_mean": at_clean,
        "at_above_floor": at_clean > FLOOR,
        "denominator_small_seeds": {
            arm: int(sum(abs(p[arm]["F"]) + abs(p[arm]["R"]) < 0.01 for p in per_seed))
            for arm in ("std", "at")
        },
    }
    controls["validity_pass"] = bool(
        controls["b_std_in_band"] and controls["at_is_robust"] and controls["at_above_floor"]
    )
    # Ruling 8: the validity controls are read on all seeds BEFORE B. When they fail, B, D and
    # the branch are kept apart under "not_read" so the first reading is the control.
    d = paired_delta_ci(Fs, Rs, Fa, Ra, N_BOOT)
    stats = {
        "branch": r4_18_branch(d["delta"], d["ci95"]),
        "b_at_pooled": b_at,
        "delta": d["delta"],
        "delta_ci95": d["ci95"],
        "b_mean_of_seeds": {
            arm: float(np.mean([p[arm]["B"] for p in per_seed])) for arm in ("std", "at")
        },
    }
    out = {
        "controls": controls,
        "branch": stats["branch"] if controls["validity_pass"] else "CONTROL_FAILED",
        "p1_mean_target_damage_non_degenerate": {
            arm: float(np.mean([p[arm]["mean_target_damage"] for p in per_seed]))
            for arm in ("std", "at")
        },
        "dropped_targets_per_seed": {
            str(p["seed"]): [x["target"] for x in p.get("dropped_targets", [])] for p in per_seed
        },
        "n_seeds": len(per_seed),
    }
    if controls["validity_pass"]:
        out.update(stats)
    else:
        out["not_read"] = stats
    return out


def main_r4_18(args) -> None:
    import psutil

    avail = psutil.virtual_memory().available / 2**30
    if args.workers * args.rss_gb > MEM_FRACTION * avail:
        raise MemoryError(f"{args.workers} x {args.rss_gb} GB > {MEM_FRACTION:.0%} of {avail:.2f}")
    done = {}
    if args.resume_from:
        for f in (REPO_ROOT / args.resume_from).glob("seed_*.json"):
            rec = json.loads(f.read_text())
            done[rec["seed"]] = rec
    todo = [s for s in args.seeds if s not in done]
    cfg = {
        "mode": "r4_18_p19d_order",
        "eps": EPS,
        "at_steps": AT_STEPS,
        "at_step_frac": AT_STEP_FRAC,
        "workers": args.workers,
        "threads": THREADS,
        "resume_from": args.resume_from,
        "amendment": "decision log 2026-11-16T11:34Z",
    }
    with RunDir("robust_surrogate_bounded_binary_q8", config=cfg, seeds=args.seeds) as run:
        for rec in done.values():
            run.write_json(f"seed_{rec['seed']:02d}.json", rec)

        def _keep(rec):
            run.write_json(f"seed_{rec['seed']:02d}.json", rec)
            done[rec["seed"]] = rec
            log.info("seed %d done | peak %.2f GB", rec["seed"], rec["peak_rss_gb"])

        if args.workers > 1:
            with ProcessPoolExecutor(args.workers) as ex:
                for rec in ex.map(run_seed_p19d, todo):
                    _keep(rec)
        else:
            for s in todo:
                _keep(run_seed_p19d(s))
        per_seed = [done[s] for s in sorted(done) if s in args.seeds]
        summ = summarize_r4_18(per_seed)
        summ["peak_rss_gb_max"] = max(p["peak_rss_gb"] for p in per_seed)
        run.write_json("results.json", {"summary": summ, "per_seed": per_seed})
        run.metrics.update(
            {
                "branch": summ["branch"],
                "validity_pass": summ["controls"]["validity_pass"],
                "at_robustness_gain": summ["controls"]["at_robustness_gain"],
                "b_std_pooled": summ["controls"]["b_std_pooled"],
            }
        )
        log.info("controls %s", json.dumps(summ["controls"]))
        log.info(
            "R4-18 %s",
            json.dumps({k: v for k, v in summ.items() if k not in ("controls", "not_read")}),
        )
        if not summ["controls"]["anchor_pass"] and any(s in ANCHOR_SEEDS for s in args.seeds):
            raise RuntimeError(f"R4-18 anchor FAILED: {summ['controls']['anchor_per_seed']}")
        print(f"-> {run.path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("seeds_pos", nargs="*", type=int, default=[])
    ap.add_argument("--rng-order", choices=["p28", "p19d"], default="p28")
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--threads", type=int, default=None, help="pinned before numpy/torch import")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--rss-gb", type=float, default=None, help="smoke-measured peak RSS per seed")
    ap.add_argument("--resume-from", default=None, help="run dir whose seed_*.json are reused")
    args = ap.parse_args()
    if args.rng_order == "p19d":
        if args.seeds is None or args.rss_gb is None:
            raise SystemExit("--rng-order p19d needs --seeds and --rss-gb (from the smoke)")
        main_r4_18(args)
        return
    seeds = args.seeds_pos or DEFAULT_SEEDS

    cfg_adv = load_yaml(REPO_ROOT / "configs" / "adversarial.yaml")
    cfg_q = load_yaml(REPO_ROOT / "configs" / "quantum.yaml")
    best_params = json.loads((REPO_ROOT / "results" / "adv_best_params_binary_q8.json").read_text())
    mp = best_params["mlp"]
    n_steps = cfg_adv["evasion"]["pgd"]["n_steps"]
    step_frac = cfg_adv["evasion"]["pgd"]["step_frac"]

    reg = prepare_regime("binary", "pca", n_components=8, seed=0)
    scale = reg.feature_scale
    ev_idx = stratified_cap(reg.X["test"], reg.y["test"], cfg_adv["eval_subset_adv"], seed=0)
    va_idx = stratified_cap(reg.X["val"], reg.y["val"], 2000, seed=0)
    Xte, yte = reg.X["test"][ev_idx], reg.y["test"][ev_idx]
    Xva, yva = reg.X["val"][va_idx], reg.y["val"][va_idx]

    cfg = {
        "eps": EPS,
        "at_steps": AT_STEPS,
        "at_step_frac": AT_STEP_FRAC,
        "seeds": seeds,
        "preregistration": "decision log 2026-08-20T18:27Z (P28)",
    }
    with RunDir("robust_surrogate_transfer_binary_q8", config=cfg, seeds=seeds) as run:
        per_seed = []
        for s in seeds:
            zoo = FittedZoo(reg, ENCODING, 8, s, cfg_q["qkernel"]["subset_cap"], best_params)
            common = (mp["h1"], mp["h2"], mp["alpha"], mp["learning_rate_init"], mp["batch_size"])
            set_all_seeds(s)
            std = WeightedMLP(*common, s).fit(zoo.Xtr, zoo.ytr, Xva, yva)
            set_all_seeds(s)
            at = AdversarialWeightedMLP(*common, s, feature_scale=scale).fit(
                zoo.Xtr, zoo.ytr, Xva, yva
            )

            clean = {
                t: float(f1_score(yte, zoo.clean_proba(t, Xte, yte).argmax(1), average="macro"))
                for t in TARGETS
            }
            rec = {"seed": s, "clean": clean, "surrogate_val_f1": {}}

            # reverse direction once per seed: the VQC-crafted attack, applied to each surrogate
            rng = spawn_rng(s, "adv")
            Xadv_vqc = pgd(
                zoo.grad_provider("vqc"),
                Xte,
                yte,
                EPS,
                scale,
                n_steps=n_steps,
                step_frac=step_frac,
                rng=rng,
            )

            for arm, sur in (("std", std), ("at", at)):
                rec["surrogate_val_f1"][arm] = sur.best_val_f1
                sur_clean = float(f1_score(yte, sur.predict_proba(Xte).argmax(1), average="macro"))
                rng = spawn_rng(s, "adv")
                Xadv = pgd(
                    GradProvider(sur, "vqc"),
                    Xte,
                    yte,
                    EPS,
                    scale,
                    n_steps=n_steps,
                    step_frac=step_frac,
                    rng=rng,
                )
                adv = {
                    t: float(f1_score(yte, zoo.predict_proba(t, Xadv).argmax(1), average="macro"))
                    for t in TARGETS
                }
                sur_adv_own = float(
                    f1_score(yte, sur.predict_proba(Xadv).argmax(1), average="macro")
                )
                sur_adv_vqc = float(
                    f1_score(yte, sur.predict_proba(Xadv_vqc).argmax(1), average="macro")
                )
                fwd = clean["vqc"] - adv["vqc"]  # surrogate -> VQC
                rev = sur_clean - sur_adv_vqc  # VQC -> surrogate
                cls_drop = float(np.mean([clean[t] - adv[t] for t in CLASSICAL if t != "mlp"]))
                rec[arm] = {
                    "surrogate_clean_f1": sur_clean,
                    "surrogate_own_adv_f1": sur_adv_own,
                    "adv": adv,
                    "mean_target_damage": float(np.mean([clean[t] - adv[t] for t in TARGETS])),
                    "vqc_drop_forward": fwd,
                    "surrogate_drop_reverse": rev,
                    "asymmetry": fwd / max(rev, 1e-9),
                    "classical_mean_drop": cls_drop,
                }
                log.info(
                    "seed %d %s | val %.4f own-adv %.4f | dmg %.4f | fwd %.4f rev %.4f asym %.2fx",
                    s,
                    arm,
                    sur.best_val_f1,
                    sur_adv_own,
                    rec[arm]["mean_target_damage"],
                    fwd,
                    rev,
                    rec[arm]["asymmetry"],
                )
            per_seed.append(rec)
            run.write_json("results_partial.json", {"per_seed": per_seed})

        def arm_mean(arm, key):
            return float(np.mean([p[arm][key] for p in per_seed]))

        std_asym, at_asym = arm_mean("std", "asymmetry"), arm_mean("at", "asymmetry")
        std_dmg, at_dmg = (
            arm_mean("std", "mean_target_damage"),
            arm_mean("at", "mean_target_damage"),
        )
        control_ok = 0.5 <= std_asym <= 1.5
        summary = {
            "std_asymmetry_mean": std_asym,
            "at_asymmetry_mean": at_asym,
            "std_mean_damage": std_dmg,
            "at_mean_damage": at_dmg,
            "P1_at_transfers_more": bool(at_dmg > std_dmg),
            "P2_asymmetry_stays_closed": bool(at_asym <= 1.5),
            "control_std_arm_near_1x": bool(control_ok),
        }
        if not control_ok:
            summary["verdict"] = (
                "CONTROL FAILED — std arm does not reproduce P19d's ~1x; AT arm not interpreted"
            )
        elif at_asym >= 2.0:
            summary["verdict"] = (
                "KILL — a hardened surrogate re-opens the asymmetry; scope note owed"
            )
        else:
            summary["verdict"] = (
                f"P2 HOLDS — asymmetry stays closed under the AT surrogate ({at_asym:.2f}x); "
                f"P1 {'holds' if at_dmg > std_dmg else 'FAILS (null, reported)'}"
            )
        run.metrics = summary
        run.write_json("results.json", {"per_seed": per_seed, "summary": summary})
        log.info("VERDICT: %s", summary["verdict"])
        print(f"-> {run.path}")


if __name__ == "__main__":
    main()
