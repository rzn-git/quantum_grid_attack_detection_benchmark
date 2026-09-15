"""N2b — does a NONLINEAR 8-dimensional embedding clear the floor where linear ones do not?

PRE-REGISTRATION (BACKLOG `Spike` row, opened by the N2 finding 2026-08-22T19:52Z; frozen before
the run). N2 tested three LINEAR 8-dimensional projections on whole-file splits — maximize pooled
variance, remove file offsets, and a generalized eigenproblem minimizing between-file scatter — and
all three landed in the same near-floor band as random projections (best invariant arm +0.0276,
invariance-optimized arm +0.0183, random +0.0202, bar floor + 0.05). That bounds LINEAR maps only.
A nonlinear 8-dimensional embedding is reachable through the same angle encoding the paper uses, so
if one cleared the floor the matched-dimensionality comparison would become testable and W1's
verdict WOULD narrow to "under linear projection" after all — which is exactly the escalation N2
was run to test for and did not find.

This is a SPIKE: the output is a finding, never an implementation. Nothing here is adopted.

FOUR ARMS, all 8-dimensional, all fit on TRAINING FILES ONLY — the same non-transductive discipline
N2 used, because an embedding fit with sight of the held-out files answers an easier question:
  pca8              N2's baseline, carried over as the linear reference
  kpca_rbf8         kernel PCA with an RBF kernel — the standard nonlinear analogue of PCA
  autoenc8          an 8-unit bottleneck MLP autoencoder (single hidden layer either side)
  file_adv8         the domain-adversarial arm: an 8-dimensional encoder trained so a classifier
                    CANNOT recover which TRAINING FILE a row came from, while a label head stays
                    accurate. This is the nonlinear form of N2's invariance objective and the one
                    with a real prior of working.

DECISION RULE, fixed now — identical to N2's so the two are directly comparable:
  best nonlinear arm's classical excess >= floor + 0.05  -> ESCALATE: W1's verdict narrows to
                                                            linear projections and the matched
                                                            comparison becomes testable
  otherwise                                              -> the dimensionality statement CLOSES for
                                                            good; no further projection family is
                                                            worth trying, linear or nonlinear

CONTROLS.
  positive (blocking)  the pca8 arm must reproduce N2's PCA-8 XGBoost figure (+0.0280 excess over
                       floor, whole-file partitions 0-2, seeds 0-2) within 0.005 — same partitions,
                       same stream, same cap
  negative             a random 8-dimensional projection bounds what any 8-dimensional map buys for
                       free; a nonlinear embedding that merely matches random has learned nothing
                       that its extra capacity paid for
  reported             each arm's reconstruction or adversarial diagnostic, so an arm that failed to
                       TRAIN is distinguishable from an arm that trained and did not help — the two
                       have opposite implications and collapsing them would be the easy mistake

DETERMINISM NOTE (found by this spike, 2026-08-23). `qgridbench.utils.seeding.set_all_seeds` seeds
python `random`, numpy legacy and PYTHONHASHSEED — but NOT torch. The first two runs of this file
proved it: every non-torch arm (pca8, kpca_rbf8, random8) was BIT-IDENTICAL across runs while the
autoencoder arm moved by up to 0.034. Both torch arms therefore call `torch.manual_seed` explicitly
here. Shipped code is deliberately left alone: the shipped VQC already seeds torch itself
(`vqc.py:111` and `:169`), so no published number is affected, and widening `set_all_seeds` is a
behaviour change that owes the eval-gate.

Run:  python -m experiments.run_nonlinear_embedding
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import optuna
import torch
from sklearn.decomposition import PCA, KernelPCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from experiments.run_group_split_control import group_splits, reconstruct_groups
from qgridbench.eval.baselines import trivial_baselines
from qgridbench.features.reduce import build_preprocessor, fit_transform_splits
from qgridbench.models.classical.zoo import build, fit_model, stratified_cap
from qgridbench.utils.run_tracking import REPO_ROOT, RunDir, get_logger
from qgridbench.utils.seeding import set_all_seeds

log = get_logger(__name__)

VARIANT = "binary"
DIM = 8
CAP = 2000
N2_PCA8_XGB_EXCESS = 0.0280  # decision log 2026-08-22T19:52Z, summary.pca8.xgb_excess
N2_TOL = 0.005
QUALIFY = 0.05
KPCA_FIT_N = 3000  # kernel PCA is O(N^2) in memory; a fixed stratified subsample bounds it
EPOCHS = 60
HIDDEN = 64


def _tensor(a: np.ndarray) -> torch.Tensor:
    return torch.tensor(a, dtype=torch.float32)


class Encoder(torch.nn.Module):
    def __init__(self, d_in: int, d_out: int = DIM, hidden: int = HIDDEN):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d_in, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, d_out)
        )

    def forward(self, x):
        return self.net(x)


def fit_autoencoder(Z: np.ndarray, seed: int) -> tuple[Encoder, float]:
    """8-unit bottleneck; returns the encoder and its final reconstruction MSE."""
    set_all_seeds(seed)
    torch.manual_seed(seed)  # set_all_seeds does NOT cover torch — see the module docstring
    enc = Encoder(Z.shape[1])
    dec = torch.nn.Sequential(
        torch.nn.Linear(DIM, HIDDEN), torch.nn.ReLU(), torch.nn.Linear(HIDDEN, Z.shape[1])
    )
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=1e-3)
    X = _tensor(Z)
    loss = torch.tensor(float("nan"))
    for _ in range(EPOCHS):
        for i in range(0, len(X), 256):
            xb = X[i : i + 256]
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(dec(enc(xb)), xb)
            loss.backward()
            opt.step()
    return enc, float(loss.item())


def fit_file_adversarial(
    Z: np.ndarray, y: np.ndarray, gid: np.ndarray, seed: int, lam_max: float = 0.3
):
    """8-dimensional encoder trained so a file-discriminator cannot tell the files apart.

    Gradient reversal, the standard DANN construction: the label head and the encoder minimize
    label loss, the discriminator minimizes file loss, and the encoder receives the NEGATED
    discriminator gradient. Returns the encoder plus the discriminator's final accuracy, which is
    the diagnostic that says whether the adversarial game actually worked.
    """
    set_all_seeds(seed)
    torch.manual_seed(seed)  # set_all_seeds does NOT cover torch — see the module docstring
    files = {f: i for i, f in enumerate(sorted(np.unique(gid).tolist()))}
    enc = Encoder(Z.shape[1])
    label_head = torch.nn.Linear(DIM, 2)
    disc = torch.nn.Sequential(
        torch.nn.Linear(DIM, HIDDEN), torch.nn.ReLU(), torch.nn.Linear(HIDDEN, len(files))
    )
    opt_e = torch.optim.Adam(list(enc.parameters()) + list(label_head.parameters()), lr=1e-3)
    opt_d = torch.optim.Adam(disc.parameters(), lr=1e-3)

    X, Y = _tensor(Z), torch.tensor(y, dtype=torch.long)
    G = torch.tensor([files[g] for g in gid], dtype=torch.long)
    disc_acc = float("nan")
    for ep in range(EPOCHS):
        lam = lam_max * min(1.0, ep / max(1, EPOCHS // 2))  # ramp, so the label head trains first
        correct = total = 0
        for i in range(0, len(X), 256):
            xb, yb, gb = X[i : i + 256], Y[i : i + 256], G[i : i + 256]
            # discriminator step on a detached code
            opt_d.zero_grad()
            code = enc(xb).detach()
            dl = torch.nn.functional.cross_entropy(disc(code), gb)
            dl.backward()
            opt_d.step()
            # encoder step: label loss MINUS lambda * file loss (gradient reversal)
            opt_e.zero_grad()
            code = enc(xb)
            logits = disc(code)
            loss = torch.nn.functional.cross_entropy(
                label_head(code), yb
            ) - lam * torch.nn.functional.cross_entropy(logits, gb)
            loss.backward()
            opt_e.step()
            correct += int((logits.argmax(1) == gb).sum())
            total += len(gb)
        disc_acc = correct / total
    return enc, disc_acc, 1.0 / len(files)


def encode(enc: Encoder, A: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return enc(_tensor(A)).numpy()


def domain_auc(Ztr: np.ndarray, gid: np.ndarray, seed: int) -> float:
    """How separable are the TRAINING files in this embedding? A diagnostic, not a target."""
    a, b = np.unique(gid)[:2]
    m = np.isin(gid, [a, b])
    if m.sum() < 100:
        return float("nan")
    Xa, Xb, ya, yb = train_test_split(
        Ztr[m], (gid[m] == b).astype(int), test_size=0.3, random_state=seed, stratify=gid[m]
    )
    clf = LogisticRegression(max_iter=500).fit(Xa, ya)
    return float(roc_auc_score(yb, clf.predict_proba(Xb)[:, 1]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--partitions", type=int, default=3)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument(
        "--lambda-adv",
        type=float,
        default=0.3,
        help="adversarial weight; 0.0 is the SUPERVISION-ONLY control that separates "
        "file-invariance from plain supervised representation learning",
    )
    args = ap.parse_args()

    db = (REPO_ROOT / "results" / "optuna_cls_xgb_binary_pca8.db").as_posix()
    xgb_params = optuna.load_study(
        study_name="cls_xgb_binary_pca8", storage=f"sqlite:///{db}"
    ).best_params

    X, y, gid = reconstruct_groups()
    rng_part = np.random.default_rng(0)  # shipped stream — partitions must match every prior run
    parts = []
    for p in range(args.partitions):
        sp, _held = group_splits(gid, y, rng_part)
        parts.append((p, sp))

    cfg = {
        "variant": VARIANT,
        "dim": DIM,
        "cap": CAP,
        "arms": ["pca8", "kpca_rbf8", "autoenc8", "file_adv8", "random8"],
        "epochs": EPOCHS,
        "hidden": HIDDEN,
        "qualify": QUALIFY,
        "lambda_adv": args.lambda_adv,
        "n2_reference": N2_PCA8_XGB_EXCESS,
        "transductive": False,
        "note": "every embedding is fit on TRAINING files only; held-out files are never seen",
    }

    tag = "" if args.lambda_adv == 0.3 else f"_lam{args.lambda_adv:g}"
    with RunDir(f"nonlinear_embedding_{VARIANT}{tag}", config=cfg, seeds=args.seeds) as run:
        rows: list[dict] = []
        diagnostics: list[dict] = []

        for p, sp in parts:
            pipe = build_preprocessor("full", n_components=None, seed=0)
            Z = fit_transform_splits(pipe, X, sp)
            ytr_all, yte = y[sp["train"]], y[sp["test"]]
            gtr = gid[sp["train"]]
            floor = trivial_baselines(ytr_all, yte)["baseline_stratified_random"]["aggregate"][
                "macro_f1"
            ]["mean"]

            # fit every embedding on the TRAINING split only
            emb = {}
            pca = PCA(n_components=DIM, random_state=0).fit(Z["train"])
            emb["pca8"] = lambda A, _p=pca: _p.transform(A)

            sub = stratified_cap(Z["train"], ytr_all, min(KPCA_FIT_N, len(ytr_all)), seed=0)
            kpca = KernelPCA(n_components=DIM, kernel="rbf", random_state=0).fit(Z["train"][sub])
            emb["kpca_rbf8"] = lambda A, _k=kpca: _k.transform(A)

            ae, recon = fit_autoencoder(Z["train"][sub], seed=0)
            emb["autoenc8"] = lambda A, _e=ae: encode(_e, A)

            adv, disc_acc, chance = fit_file_adversarial(
                Z["train"][sub], ytr_all[sub], gtr[sub], seed=0, lam_max=args.lambda_adv
            )
            emb["file_adv8"] = lambda A, _e=adv: encode(_e, A)

            r = np.random.default_rng(7000)
            R = r.standard_normal((Z["train"].shape[1], DIM)) / np.sqrt(Z["train"].shape[1])
            emb["random8"] = lambda A, _R=R: A @ _R

            diagnostics.append(
                {
                    "partition": p,
                    "autoencoder_final_mse": recon,
                    "file_discriminator_acc": disc_acc,
                    "file_discriminator_chance": chance,
                    "adversary_beat_chance": bool(disc_acc > chance * 1.5),
                }
            )

            for arm, proj in emb.items():
                Ztr_full, Zte = proj(Z["train"]), proj(Z["test"])
                idx = stratified_cap(Ztr_full, ytr_all, CAP, seed=0)
                Ztr, ytr = Ztr_full[idx], ytr_all[idx]
                auc = domain_auc(Ztr_full, gtr, seed=0)
                for s in args.seeds:
                    set_all_seeds(s)
                    m = build("xgb", xgb_params, s)
                    fit_model("xgb", m, Ztr, ytr)
                    f1 = float(f1_score(yte, m.predict(Zte), average="macro"))
                    rows.append(
                        {
                            "partition": p,
                            "arm": arm,
                            "seed": s,
                            "floor": floor,
                            "xgb_f1": f1,
                            "xgb_excess": f1 - floor,
                            "train_file_domain_auc": auc,
                        }
                    )
                log.info(
                    "p%d %-12s xgb %+0.4f (floor %.4f, train-file AUC %.3f)",
                    p,
                    arm,
                    float(np.mean([x["xgb_excess"] for x in rows[-len(args.seeds) :]])),
                    floor,
                    auc,
                )
                run.write_json("results_partial.json", {"rows": rows})

        # ---- aggregate + controls -------------------------------------------------------
        arms = cfg["arms"]
        summary = {
            a: {
                "xgb_excess": float(np.mean([r["xgb_excess"] for r in rows if r["arm"] == a])),
                "train_file_domain_auc": float(
                    np.nanmean([r["train_file_domain_auc"] for r in rows if r["arm"] == a])
                ),
            }
            for a in arms
        }
        pos_delta = summary["pca8"]["xgb_excess"] - N2_PCA8_XGB_EXCESS
        controls = {
            "positive_pca8_xgb_excess": summary["pca8"]["xgb_excess"],
            "positive_delta_vs_n2": float(pos_delta),
            "positive_pass": bool(abs(pos_delta) <= N2_TOL),
            "negative_random8_xgb_excess": summary["random8"]["xgb_excess"],
            "negative_note": (
                "a nonlinear embedding that merely matches a random 8-dimensional projection has "
                "learned nothing its extra capacity paid for"
            ),
            "training_diagnostics": diagnostics,
            "diagnostic_note": (
                "reported so an arm that failed to TRAIN is distinguishable from an arm that "
                "trained and did not help — the two have opposite implications"
            ),
        }
        controls["pass"] = controls["positive_pass"]
        log.info("CONTROLS\n%s", json.dumps(controls, indent=2))

        nonlinear = ["kpca_rbf8", "autoenc8", "file_adv8"]
        best = max(nonlinear, key=lambda a: summary[a]["xgb_excess"])
        qualifies = summary[best]["xgb_excess"] >= QUALIFY
        verdict = {
            "controls": controls,
            "summary": summary,
            "best_nonlinear_arm": best,
            "best_nonlinear_xgb_excess": summary[best]["xgb_excess"],
            "beats_pca_by": summary[best]["xgb_excess"] - summary["pca8"]["xgb_excess"],
            "beats_random_by": summary[best]["xgb_excess"] - summary["random8"]["xgb_excess"],
            "qualify_threshold": QUALIFY,
            "branch": (
                "ESCALATE_w1_narrows_to_linear_projections"
                if qualifies
                else "dimensionality_statement_CLOSES_linear_and_nonlinear"
            ),
            "rule": (
                f"best nonlinear arm's classical excess >= {QUALIFY} -> escalate; else the "
                "dimensionality statement closes for good"
            ),
            "rows": rows,
        }
        run.write_json("results.json", verdict)
        run.metrics = {"best_nonlinear_xgb_excess": summary[best]["xgb_excess"]}
        log.info(
            "VERDICT %s | best nonlinear %s %+0.4f (bar %.2f) | pca8 %+0.4f | random %+0.4f",
            verdict["branch"],
            best,
            summary[best]["xgb_excess"],
            QUALIFY,
            summary["pca8"]["xgb_excess"],
            summary["random8"]["xgb_excess"],
        )
        print(f"-> {run.path}")


if __name__ == "__main__":
    main()
