"""N2e — the control N2c never ran: what does the supervised encoder's OWN label head score?

PRE-REGISTRATION (BACKLOG `Validate` row, opened by the winds review 2026-08-23T15:13Z; frozen
before the run). N2b's supervised encoder is a DANN construction whose label head — a
`torch.nn.Linear(DIM, 2)` at `experiments/run_nonlinear_embedding.py:138` — is trained JOINTLY
with the encoder and then DISCARDED; only the 8-dimensional code is passed downstream. Neither
N2b (finding 2026-08-23T01:20Z) nor N2c (finding 2026-08-23T08:41Z) ever scored it.

WHY IT MATTERS. N2c's headline is that all three model families clear the floor at 8 dimensions in
this representation (XGBoost +0.0632, fidelity kernel +0.0578, VQC +0.0524) where the VQC sat below
it on PCA-8, so the matched-dimensionality comparison becomes testable. If the encoder's own head
already reaches that level on HELD-OUT FILES, then all three families are readouts of a
representation that already contains the answer, and the unlock restates as "three readouts of a
supervised representation agree" rather than as a comparison between model classes. This is a
SHARPER question than N2c's pre-registered caveat: that caveat says the quantum feature map is
built on classically-learned features; it does not say there is nothing left for the map to do.

DECISION RULE, fixed now:
  head_excess >= xgb_excess - 0.005  -> the REPRESENTATION is doing the work; N2c's unlock is
                                        restated as a readout result and the model-class reading
                                        is withdrawn
  head_excess <  xgb_excess - 0.005  -> the readout adds measurable signal; N2c stands as written

CONTROLS.
  positive (blocking)  XGBoost on this run's supervised code must reproduce N2b/N2c's +0.0632
                       within 5e-4 at 12 threads (the D2b lesson: the thread count is pinned in the
                       pre-registration, not left to the day's scheduling). This is what certifies
                       that the encoder rebuilt here IS the encoder N2c measured — the head cannot
                       be compared to N2c's numbers otherwise.
  negative             the same linear head trained on a RANDOM 8-dimensional code pins what a
                       linear readout buys with no learned representation under it.
  reported (boring     XGBoost refit on the SAME 3,000-row subset the encoder saw, so the head's
  cause)               3,000-vs-2,000 training-set advantage is visible rather than silent. The
                       asymmetry favours the head, so a head that TRAILS trails conservatively.
  reported (spread)    the head is re-fit at encoder seeds 0-2, because the primary arm inherits
                       N2c's single seed-0 encoder and would otherwise carry no uncertainty.

The head is a LINEAR readout and XGBoost is not; part of any gap is readout capacity rather than
representation content. That is the question, not a confound: "does the readout add signal".

Run:  OMP_NUM_THREADS=12 python -m experiments.run_encoder_head_control
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import optuna
import torch
from sklearn.metrics import f1_score

from experiments.run_group_split_control import group_splits, reconstruct_groups
from experiments.run_nonlinear_embedding import DIM, EPOCHS, HIDDEN, Encoder, _tensor, encode
from qgridbench.eval.baselines import trivial_baselines
from qgridbench.features.reduce import build_preprocessor, fit_transform_splits
from qgridbench.models.classical.zoo import build, fit_model, stratified_cap
from qgridbench.utils.run_tracking import REPO_ROOT, RunDir, get_logger
from qgridbench.utils.seeding import set_all_seeds

log = get_logger(__name__)

VARIANT = "binary"
CAP = 2000
ENC_FIT_N = 3000  # identical to N2b's KPCA_FIT_N — the subset the encoder was trained on
N2C_XGB_SUPERVISED = (
    0.06324948105905161  # results.json, 20260823T075450Z_supervised_repr_quantum_binary
)
POS_TOL = 5e-4
MARGIN = 0.005  # the decision-rule band


def fit_encoder_and_head(Z, y, gid, seed: int, lam_max: float = 0.0):
    """EXACT mirror of `run_nonlinear_embedding.fit_file_adversarial`, returning the label head too.

    Every RNG-consuming step is reproduced in order — including the discriminator, whose updates
    advance the torch stream even at lam_max = 0 — because the positive control demands the encoder
    be the one N2c measured, not merely one built the same way.
    """
    set_all_seeds(seed)
    torch.manual_seed(seed)
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
    for ep in range(EPOCHS):
        lam = lam_max * min(1.0, ep / max(1, EPOCHS // 2))
        for i in range(0, len(X), 256):
            xb, yb, gb = X[i : i + 256], Y[i : i + 256], G[i : i + 256]
            opt_d.zero_grad()
            code = enc(xb).detach()
            dl = torch.nn.functional.cross_entropy(disc(code), gb)
            dl.backward()
            opt_d.step()
            opt_e.zero_grad()
            code = enc(xb)
            logits = disc(code)
            loss = torch.nn.functional.cross_entropy(
                label_head(code), yb
            ) - lam * torch.nn.functional.cross_entropy(logits, gb)
            loss.backward()
            opt_e.step()
    return enc, label_head


def fit_head_only(C: np.ndarray, y: np.ndarray, seed: int) -> torch.nn.Linear:
    """The negative control: the same linear head, same schedule, over a code it did not shape."""
    set_all_seeds(seed)
    torch.manual_seed(seed)
    head = torch.nn.Linear(C.shape[1], 2)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    X, Y = _tensor(C), torch.tensor(y, dtype=torch.long)
    for _ in range(EPOCHS):
        for i in range(0, len(X), 256):
            opt.zero_grad()
            torch.nn.functional.cross_entropy(head(X[i : i + 256]), Y[i : i + 256]).backward()
            opt.step()
    return head


def head_f1(head: torch.nn.Linear, C: np.ndarray, y: np.ndarray) -> float:
    with torch.no_grad():
        pred = head(_tensor(C)).argmax(1).numpy()
    return float(f1_score(y, pred, average="macro"))


def xgb_excess(params, Ztr, ytr, Zte, yte, floor, seeds) -> list[float]:
    out = []
    for s in seeds:
        set_all_seeds(s)
        m = build("xgb", params, s)
        fit_model("xgb", m, Ztr, ytr)
        out.append(float(f1_score(yte, m.predict(Zte), average="macro")) - floor)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--partitions", type=int, default=3)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    args = ap.parse_args()

    db = (REPO_ROOT / "results" / "optuna_cls_xgb_binary_pca8.db").as_posix()
    xgb_params = optuna.load_study(
        study_name="cls_xgb_binary_pca8", storage=f"sqlite:///{db}"
    ).best_params

    X, y, gid = reconstruct_groups()
    rng_part = np.random.default_rng(0)  # shipped stream — partitions must match every prior run
    parts = [(p, group_splits(gid, y, rng_part)[0]) for p in range(args.partitions)]

    cfg = {
        "variant": VARIANT,
        "dim": DIM,
        "cap": CAP,
        "encoder_fit_n": ENC_FIT_N,
        "lambda_adv": 0.0,
        "epochs": EPOCHS,
        "seeds": args.seeds,
        "n2c_reference_xgb_supervised": N2C_XGB_SUPERVISED,
        "positive_tol": POS_TOL,
        "margin": MARGIN,
        "note": "the head is a LINEAR readout; the question is whether the readout adds signal",
    }

    with RunDir(f"encoder_head_control_{VARIANT}", config=cfg, seeds=args.seeds) as run:
        rows: list[dict] = []
        for p, sp in parts:
            pipe = build_preprocessor("full", n_components=None, seed=0)
            Z = fit_transform_splits(pipe, X, sp)
            ytr_all, yte = y[sp["train"]], y[sp["test"]]
            gtr = gid[sp["train"]]
            floor = trivial_baselines(ytr_all, yte)["baseline_stratified_random"]["aggregate"][
                "macro_f1"
            ]["mean"]
            sub = stratified_cap(Z["train"], ytr_all, min(ENC_FIT_N, len(ytr_all)), seed=0)

            # ---- the seed-0 encoder N2c used, plus its discarded head -----------------------
            enc, head = fit_encoder_and_head(Z["train"][sub], ytr_all[sub], gtr[sub], seed=0)
            Ctr_full, Cte = encode(enc, Z["train"]), encode(enc, Z["test"])
            head_ex = head_f1(head, Cte, yte) - floor

            # positive control: XGBoost on this code, 2,000 cap — must reproduce N2c
            idx = stratified_cap(Ctr_full, ytr_all, CAP, seed=0)
            xgb_ex = xgb_excess(
                xgb_params, Ctr_full[idx], ytr_all[idx], Cte, yte, floor, args.seeds
            )
            # boring cause: XGBoost on the SAME 3,000 rows the head saw
            xgb_ex_matched = xgb_excess(
                xgb_params, Ctr_full[sub], ytr_all[sub], Cte, yte, floor, args.seeds
            )

            # negative control: the same head over a random 8-dimensional code
            r = np.random.default_rng(7000)  # identical construction to N2b's random8 arm
            R = r.standard_normal((Z["train"].shape[1], DIM)) / np.sqrt(Z["train"].shape[1])
            rnd_head = fit_head_only(Z["train"][sub] @ R, ytr_all[sub], seed=0)
            rnd_head_ex = head_f1(rnd_head, Z["test"] @ R, yte) - floor

            # spread: the head at encoder seeds 0-2 (the primary arm inherits N2c's single seed)
            head_by_seed = []
            for s in args.seeds:
                e_s, h_s = fit_encoder_and_head(Z["train"][sub], ytr_all[sub], gtr[sub], seed=s)
                head_by_seed.append(head_f1(h_s, encode(e_s, Z["test"]), yte) - floor)

            rows.append(
                {
                    "partition": p,
                    "floor": floor,
                    "head_excess": head_ex,
                    "xgb_excess": float(np.mean(xgb_ex)),
                    "xgb_excess_by_seed": xgb_ex,
                    "xgb_excess_matched_3000": float(np.mean(xgb_ex_matched)),
                    "random_code_head_excess": rnd_head_ex,
                    "head_excess_by_encoder_seed": head_by_seed,
                }
            )
            log.info(
                "p%d head %+0.4f | xgb %+0.4f (matched %+0.4f) | random-code head %+0.4f",
                p,
                head_ex,
                float(np.mean(xgb_ex)),
                float(np.mean(xgb_ex_matched)),
                rnd_head_ex,
            )
            run.write_json("results_partial.json", {"rows": rows})

        head_mean = float(np.mean([r["head_excess"] for r in rows]))
        xgb_mean = float(np.mean([r["xgb_excess"] for r in rows]))
        spread = [v for r in rows for v in r["head_excess_by_encoder_seed"]]
        controls = {
            "positive_xgb_supervised": xgb_mean,
            "positive_reference": N2C_XGB_SUPERVISED,
            "positive_delta": abs(xgb_mean - N2C_XGB_SUPERVISED),
            "positive_pass": bool(abs(xgb_mean - N2C_XGB_SUPERVISED) <= POS_TOL),
            "positive_note": "certifies this IS N2c's encoder; the head is otherwise incomparable",
            "negative_random_code_head": float(
                np.mean([r["random_code_head_excess"] for r in rows])
            ),
            "boring_xgb_matched_3000": float(np.mean([r["xgb_excess_matched_3000"] for r in rows])),
            "boring_note": "the head saw 3,000 rows and the primary XGBoost arm 2,000 — the "
            "asymmetry favours the head, so a head that TRAILS trails conservatively",
        }
        controls["pass"] = controls["positive_pass"]
        representation_does_the_work = bool(head_mean >= xgb_mean - MARGIN)
        verdict = {
            "controls": controls,
            "head_excess_mean": head_mean,
            "head_excess_by_encoder_seed_all": spread,
            "head_excess_seed_std": float(np.std(spread)),
            "xgb_excess_mean": xgb_mean,
            "gap_xgb_minus_head": xgb_mean - head_mean,
            "margin": MARGIN,
            "branch": (
                "representation_does_the_work"
                if representation_does_the_work
                else "readout_adds_signal"
            ),
            "rule": f"head >= xgb - {MARGIN} -> representation does the work",
            "rows": rows,
        }
        run.write_json("results.json", verdict)
        run.metrics = {"head_excess_mean": head_mean, "gap_xgb_minus_head": xgb_mean - head_mean}
        log.info("CONTROLS\n%s", json.dumps(controls, indent=2))
        log.info(
            "VERDICT %s | head %+0.4f (sd %.4f) vs xgb %+0.4f | gap %+0.4f (margin %.3f)",
            verdict["branch"],
            head_mean,
            verdict["head_excess_seed_std"],
            xgb_mean,
            verdict["gap_xgb_minus_head"],
            MARGIN,
        )
        print(f"-> {run.path}")


if __name__ == "__main__":
    main()
