"""P19: adjudicate four external-review claims that turn on untested facts.

One zoo build per seed serves all four sub-probes. Everything is measured on the
paper's own subsets (evasion n=500, black-box n=100, both stratified, seed 0) so
the numbers are directly comparable to Table II.

P19a  ENSEMBLE (review gap 1). The appendix states as a DEDUCTION that a VQC added
      to a classical ensemble "would inherit classical vulnerability without
      contributing classical coverage".
      H0 (the deduction): under MLP-crafted PGD at 0.5 sigma, a soft-vote ensemble
      WITH the VQC does not beat the same ensemble without it.
      KILL: if adding the VQC raises adversarial macro-F1 by > 0.02 and the gain
      exceeds the 3-seed spread, the deduction is falsified and the appendix
      paragraph must be rewritten.

P19b  OPERATING POINT (review gap 5). The paper reports argmax-threshold and
      threshold-free metrics only. A deployed detector runs at a fixed FPR.
      H: at FPR = 1% (threshold fitted on VAL, never on test), the PCA-8 models
      have negligible detection rate, i.e. the regime's near-floor macro-F1 is not
      an artifact of the operating point.
      KILL: any model reaching TPR >= 0.5 at 1% FPR would show the macro-F1 view
      understates the regime, and the accuracy section would need an operating-
      point caveat.

P19c  FLOOR-RELATIVE ROBUSTNESS (blocking item 1). Retention = adv/clean rewards a
      model whose clean score is already at the stratified-random floor.
      H: recomputing Table II as (F1 - floor) reorders the robustness ranking and
      removes the QSVM from the top.
      KILL: if the ranking is unchanged, retention is a safe primary metric.

P19d  MLP SURROGATE (blocking item 2). The MLP scores 0.4565 macro-F1 at PCA-8,
      BELOW the stratified-random floor (0.5005), and its best value over all 50
      Optuna trials was 0.4690 -- also below floor. sklearn's MLPClassifier accepts
      neither class_weight nor sample_weight (zoo.py docstring), so it is the only
      family in the study fitted without the class-weight policy on a 71%-positive
      task.
      H: the below-floor score is caused by the missing class weighting, not by
      regime difficulty.
      KILL: if a class-weighted MLP of identical architecture still lands below the
      0.5005 floor, the cause is the regime and the surrogate needs no fix.
      Follow-on (the decisive one for the headline): re-craft the transfer attack
      from the FIXED surrogate. If MLP -> VQC no longer damages the VQC more than
      any classical target, the C1 asymmetry rests on a mis-specified surrogate.

Run:  python -m experiments.run_review_claims  (from repo root)

Graduated from tmp/ because the appendix reports its numbers (the ensemble result, the
surrogate-fix baseline) and because run_constant_shift and run_operating_point_full_features import
`floors`, `macro_f1` and `tpr_at_fpr` from here -- a shipped experiment cannot import
dev-only scratch. The RunDir name keeps its original `probe_` prefix so already-recorded
results still resolve.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_curve
from sklearn.utils.class_weight import compute_class_weight

from experiments.run_adversarial import CLASSICAL, FittedZoo
from qgridbench.attacks.evasion import pgd
from qgridbench.attacks.gradients import GradProvider
from qgridbench.models.classical.zoo import stratified_cap
from qgridbench.protocol import prepare_regime
from qgridbench.utils.run_tracking import REPO_ROOT, RunDir, get_logger, load_yaml
from qgridbench.utils.seeding import set_all_seeds, spawn_rng

log = get_logger(__name__)

SEEDS = [int(a) for a in sys.argv[1:]] or [0, 1, 2]  # published evasion run used 0..9
ENCODING = "zz"  # the published Table II configuration (run params.yaml), NOT the CLI default
HEADLINE_EPS = 0.5  # sigma; the Table II operating budget
FPR_TARGETS = [0.01, 0.05]


# --------------------------------------------------------------------------- #
# class-weighted MLP: the fix for the only unweighted family in the zoo
# --------------------------------------------------------------------------- #
class WeightedMLP:
    """2-hidden-layer ReLU MLP with class-weighted cross-entropy.

    Same architecture and hyperparameters as the tuned sklearn MLP; the ONLY
    difference under test is the class weighting sklearn cannot express. Early
    stopping on val macro-F1 mirrors the sklearn model's early_stopping=True.
    """

    def __init__(
        self, h1, h2, alpha, lr, batch_size, seed, max_epochs=300, patience=15, weighted=True
    ):
        self.h1, self.h2, self.alpha, self.lr = h1, h2, alpha, lr
        self.batch_size, self.seed = batch_size, seed
        self.max_epochs, self.patience = max_epochs, patience
        self.weighted = weighted  # False = the CONTROL isolating weighting from re-implementation
        self.net = None

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
        w = (
            torch.tensor(
                compute_class_weight("balanced", classes=np.unique(y), y=y), dtype=torch.float32
            )
            if self.weighted
            else None
        )
        loss_fn = torch.nn.CrossEntropyLoss(weight=w)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.alpha)
        Xt = torch.tensor(X, dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.long)
        best, best_state, bad = -1.0, None, 0
        g = torch.Generator().manual_seed(self.seed)
        for _ in range(self.max_epochs):
            perm = torch.randperm(len(Xt), generator=g)
            self.net.train()
            for i in range(0, len(Xt), self.batch_size):
                idx = perm[i : i + self.batch_size]
                opt.zero_grad()
                loss_fn(self.net(Xt[idx]), yt[idx]).backward()
                opt.step()
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

    def predict_proba(self, X):
        self.net.eval()
        with torch.no_grad():
            logits = self.net(torch.tensor(np.asarray(X), dtype=torch.float32))
            return torch.softmax(logits, dim=1).numpy().astype(np.float64)

    def loss_input_grad(self, X, y):
        """d(unweighted CE)/dX -- the SAME attack objective the sklearn MLP path
        ascends, so a surrogate swap changes the model, not the attack."""
        self.net.eval()
        Xt = torch.tensor(np.asarray(X), dtype=torch.float32, requires_grad=True)
        yt = torch.tensor(np.asarray(y), dtype=torch.long)
        loss = torch.nn.functional.cross_entropy(self.net(Xt), yt, reduction="sum")
        (grad,) = torch.autograd.grad(loss, Xt)
        return grad.detach().numpy().astype(np.float64)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def macro_f1(y, proba):
    return float(f1_score(y, proba.argmax(1), average="macro"))


def floors(y_train, y_eval, n_draws=200, seed=0):
    """Majority and stratified-random macro-F1 on THIS evaluation subset."""
    classes, counts = np.unique(y_train, return_counts=True)
    prior = counts / counts.sum()
    maj = classes[counts.argmax()]
    rng = np.random.default_rng(seed)
    f1s = [
        f1_score(y_eval, rng.choice(classes, size=len(y_eval), p=prior), average="macro")
        for _ in range(n_draws)
    ]
    return {
        "majority": float(f1_score(y_eval, np.full(len(y_eval), maj), average="macro")),
        "stratified_random": float(np.mean(f1s)),
        "stratified_random_std": float(np.std(f1s, ddof=1)),
    }


def tpr_at_fpr(scores_val, y_val, scores_test, y_test, target_fpr):
    """Threshold chosen on VAL at the target FPR; TPR/FPR reported on the test set.

    Test-set discipline: the operating point never sees test data.
    """
    fpr, _tpr, thr = roc_curve(y_val, scores_val)
    ok = np.where(fpr <= target_fpr)[0]
    tau = float(thr[ok[-1]]) if len(ok) else float(thr[0])
    pred = (scores_test >= tau).astype(int)
    pos, neg = y_test == 1, y_test == 0
    return {
        "threshold": tau,
        "tpr": float(pred[pos].mean()) if pos.any() else float("nan"),
        "fpr": float(pred[neg].mean()) if neg.any() else float("nan"),
    }


def main():
    cfg_adv = load_yaml(REPO_ROOT / "configs" / "adversarial.yaml")
    cfg_q = load_yaml(REPO_ROOT / "configs" / "quantum.yaml")
    subset_cap = cfg_q["qkernel"]["subset_cap"]
    best_params = json.loads((REPO_ROOT / "results" / "adv_best_params_binary_q8.json").read_text())

    reg = prepare_regime("binary", "pca", n_components=8, seed=0)
    scale = reg.feature_scale
    ev_idx = stratified_cap(reg.X["test"], reg.y["test"], cfg_adv["eval_subset_adv"], seed=0)
    bb_idx = stratified_cap(
        reg.X["test"], reg.y["test"], cfg_adv["blackbox"]["eval_subset"], seed=0
    )
    va_idx = stratified_cap(reg.X["val"], reg.y["val"], 2000, seed=0)
    Xte, yte = reg.X["test"][ev_idx], reg.y["test"][ev_idx]
    Xva, yva = reg.X["val"][va_idx], reg.y["val"][va_idx]

    fl_ev = floors(reg.y["train"], yte)
    fl_bb = floors(reg.y["train"], reg.y["test"][bb_idx])
    log.info("floors | evasion n=%d %s | blackbox n=%d %s", len(yte), fl_ev, len(bb_idx), fl_bb)

    families = CLASSICAL + ["qsvm", "vqc"]
    ENSEMBLES = {
        "cls3": ["rf", "xgb", "lgbm"],
        "cls3+vqc": ["rf", "xgb", "lgbm", "vqc"],
        "cls3+qsvm": ["rf", "xgb", "lgbm", "qsvm"],
        "cls5": ["logreg", "rbf_svm", "rf", "xgb", "lgbm"],
        "cls5+vqc": ["logreg", "rbf_svm", "rf", "xgb", "lgbm", "vqc"],
        "cls5+qsvm": ["logreg", "rbf_svm", "rf", "xgb", "lgbm", "qsvm"],
    }

    with RunDir("probe_review_claims_binary_q8", config={"seeds": SEEDS}, seeds=SEEDS) as run:
        per_seed = []
        for s in SEEDS:
            rng = spawn_rng(s, "adv")  # same stream the published run used
            zoo = FittedZoo(reg, ENCODING, 8, s, subset_cap, best_params)

            # --- P19d: the class-weighted surrogate ---------------------------
            set_all_seeds(s)
            mp = best_params["mlp"]
            wmlp = WeightedMLP(
                mp["h1"], mp["h2"], mp["alpha"], mp["learning_rate_init"], mp["batch_size"], s
            ).fit(zoo.Xtr, zoo.ytr, Xva, yva)
            # CONTROL: identical torch model WITHOUT the class weighting. If this one
            # also lands below the floor, the weighting is the cause and not the
            # re-implementation (boring causes first).
            set_all_seeds(s)
            umlp = WeightedMLP(
                mp["h1"],
                mp["h2"],
                mp["alpha"],
                mp["learning_rate_init"],
                mp["batch_size"],
                s,
                weighted=False,
            ).fit(zoo.Xtr, zoo.ytr, Xva, yva)
            log.info(
                "seed %d | val macro-F1: weighted %.4f | unweighted-control %.4f",
                s,
                wmlp.best_val_f1,
                umlp.best_val_f1,
            )

            extra = {"mlp_weighted": wmlp, "mlp_torch_unweighted": umlp}

            def proba(name, X, _e=extra):
                return _e[name].predict_proba(X) if name in _e else zoo.predict_proba(name, X)

            all_models = families + list(extra)

            # --- attack sets at the headline budget ---------------------------
            sets = {"clean": Xte}
            for src, gp in [
                ("mlp", zoo.grad_provider("mlp")),
                ("vqc", zoo.grad_provider("vqc")),
                ("mlp_weighted", GradProvider(wmlp, "vqc")),  # kind="vqc" -> .loss_input_grad
                ("mlp_torch_unweighted", GradProvider(umlp, "vqc")),
            ]:
                sets[f"pgd_{src}"] = pgd(
                    gp,
                    Xte,
                    yte,
                    HEADLINE_EPS,
                    scale,
                    n_steps=cfg_adv["evasion"]["pgd"]["n_steps"],
                    step_frac=cfg_adv["evasion"]["pgd"]["step_frac"],
                    rng=rng,
                )

            probas = {k: {m: proba(m, X) for m in all_models} for k, X in sets.items()}
            f1s = {k: {m: macro_f1(yte, p) for m, p in d.items()} for k, d in probas.items()}

            # --- P19a: soft-vote ensembles ------------------------------------
            ens = {}
            for k, d in probas.items():
                ens[k] = {
                    name: macro_f1(yte, np.mean([d[m] for m in members], axis=0))
                    for name, members in ENSEMBLES.items()
                }

            # --- P19b: operating point ----------------------------------------
            op = {}
            for m in all_models:
                sv = proba(m, Xva)[:, 1]
                op[m] = {}
                for t in FPR_TARGETS:
                    op[m][str(t)] = {
                        k: tpr_at_fpr(sv, yva, probas[k][m][:, 1], yte, t) for k in sets
                    }

            per_seed.append(
                {
                    "f1": f1s,
                    "ensembles": ens,
                    "operating_point": op,
                    "weighted_mlp_val_f1": wmlp.best_val_f1,
                }
            )
            run.write_json("results_partial.json", {"per_seed": per_seed})
            log.info(
                "seed %d done | clean %s", s, {m: round(f1s["clean"][m], 3) for m in all_models}
            )

        out = {
            "per_seed": per_seed,
            "floors": {"evasion_n500": fl_ev, "blackbox_n100": fl_bb},
            "ensemble_members": ENSEMBLES,
            "eps": HEADLINE_EPS,
        }
        run.write_json("results.json", out)
        print(f"-> {run.path}")


if __name__ == "__main__":
    main()
