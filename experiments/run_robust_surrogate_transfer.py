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
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight

from experiments.run_adversarial import CLASSICAL, FittedZoo
from experiments.run_review_claims import WeightedMLP
from qgridbench.attacks.evasion import pgd
from qgridbench.attacks.gradients import GradProvider
from qgridbench.models.classical.zoo import stratified_cap
from qgridbench.protocol import prepare_regime
from qgridbench.utils.run_tracking import REPO_ROOT, RunDir, get_logger, load_yaml
from qgridbench.utils.seeding import set_all_seeds, spawn_rng

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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("seeds", nargs="*", type=int, default=DEFAULT_SEEDS)
    args = ap.parse_args()
    seeds = args.seeds or DEFAULT_SEEDS

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
                t: float(f1_score(yte, zoo.predict_proba(t, Xte).argmax(1), average="macro"))
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
