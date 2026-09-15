"""V2-E3/V2-E4: modern deep tabular arms under both split protocols.

PRE-REGISTERED 2026-09-01T11:05Z.

  --model ftt_default  (V2-E3)  FTTransformer at LITERATURE hyperparameters — QUPID's own
      baseline policy (untuned defaults, their strongest baseline family), full features,
      class-weighted loss, 10 seeds row-level + the five Table V whole-file partitions at
      seed 0. H1: row-level <= tuned LightGBM - 0.005 (an untuned deep baseline does not
      touch the tuned-tree headline). H2: whole-file mean lands in the 0.55-0.66 band.

  --model tabm         (V2-E4)  TabM (2026's strongest trained tabular net) under the FULL
      equal-budget contract: 50-trial Optuna (lr, weight decay, d_block, n_blocks, dropout;
      k = 32 fixed), objective = mean val macro-F1 over seeds {0,1,2} at reduced search
      epochs (<= 20, patience 5 — the VQC tuning-epoch precedent), winner retrained on 10
      seeds full schedule + the five partitions at seed 0. H1: row-level within 0.01 of
      tuned LightGBM. H2: whole-file mean in the 0.55-0.66 band; KILL >= 0.70.

Shared guards: Table V partition ids asserted; file-disjointness asserted; a constant
prediction FAILS the run (2026-08-24 runner-guard rule). Optuna store is sqlite-resumable.

Run:  python -m experiments.run_deep_tabular --model ftt_default
      python -m experiments.run_deep_tabular --model tabm [--trials 50]
"""

from __future__ import annotations

import argparse
import copy

import numpy as np
import optuna
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

from experiments.run_group_split_control import group_splits, reconstruct_groups, tuned_params
from experiments.run_whole_file_retune import EXPECTED_TEST_GROUPS, assert_file_disjoint
from qgridbench.data.preprocess import load_processed
from qgridbench.features.reduce import build_preprocessor, fit_transform_splits
from qgridbench.utils.run_tracking import REPO_ROOT, RunDir, get_logger
from qgridbench.utils.seeding import set_all_seeds

log = get_logger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

VARIANT = "binary"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH = 1024
FINAL_EPOCHS, FINAL_PATIENCE = 60, 10
SEARCH_EPOCHS, SEARCH_PATIENCE = 20, 5
TUNING_SEEDS = [0, 1, 2]
FINAL_SEEDS = list(range(10))
TABM_K = 32
PUBLISHED_LGBM_ROW = 0.9056  # MODEL_CARD 10-seed mean, the H1 reference


def build_model(name: str, n_features: int, params: dict):
    if name == "ftt_default":
        from rtdl_revisiting_models import FTTransformer

        return FTTransformer(
            n_cont_features=n_features,
            cat_cardinalities=[],
            d_out=2,
            **FTTransformer.get_default_kwargs(),
        )
    if name == "tabm":
        from tabm import TabM

        return TabM.make(
            n_num_features=n_features,
            d_out=2,
            n_blocks=params["n_blocks"],
            d_block=params["d_block"],
            dropout=params["dropout"],
            k=TABM_K,
        )
    raise ValueError(name)


def make_optimizer(name: str, model, params: dict):
    if name == "ftt_default":
        return model.make_default_optimizer()  # the literature default (AdamW, grouped decay)
    return torch.optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["wd"])


def forward_logits(name: str, model, xb):
    out = model(xb, None) if name == "ftt_default" else model(xb)
    return out  # ftt: (B, 2); tabm: (B, k, 2)


def loss_fn(logits, yb, class_w):
    if logits.ndim == 3:  # TabM: every ensemble member gets its own loss
        k = logits.shape[1]
        return F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), yb.repeat_interleave(k), weight=class_w
        )
    return F.cross_entropy(logits, yb, weight=class_w)


@torch.no_grad()
def predict(name: str, model, Z, batch=4096):
    model.eval()
    preds = []
    for i in range(0, len(Z), batch):
        xb = torch.as_tensor(Z[i : i + batch], dtype=torch.float32, device=DEVICE)
        logits = forward_logits(name, model, xb)
        if logits.ndim == 3:
            p = torch.softmax(logits, dim=-1).mean(dim=1)
        else:
            p = torch.softmax(logits, dim=-1)
        preds.append(p.argmax(dim=-1).cpu().numpy())
    return np.concatenate(preds)


def fit_and_score(name, params, Z, y, splits, seed, max_epochs, patience, score_split="test"):
    set_all_seeds(seed)
    torch.manual_seed(seed)
    Ztr, ytr = Z["train"], y[splits["train"]]
    Zva, yva = Z["val"], y[splits["val"]]
    model = build_model(name, Ztr.shape[1], params).to(DEVICE)
    opt = make_optimizer(name, model, params)
    counts = np.bincount(ytr, minlength=2)
    class_w = torch.as_tensor(len(ytr) / (2.0 * counts), dtype=torch.float32, device=DEVICE)

    Xt = torch.as_tensor(Ztr, dtype=torch.float32, device=DEVICE)
    yt = torch.as_tensor(ytr, dtype=torch.long, device=DEVICE)
    g = torch.Generator().manual_seed(seed)
    best_f1, best_state, bad = -1.0, None, 0
    for _epoch in range(max_epochs):
        model.train()
        for idx in torch.randperm(len(Xt), generator=g).split(BATCH):
            opt.zero_grad()
            loss = loss_fn(forward_logits(name, model, Xt[idx]), yt[idx], class_w)
            loss.backward()
            opt.step()
        va = float(f1_score(yva, predict(name, model, Zva), average="macro"))
        if va > best_f1:
            best_f1, best_state, bad = va, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    pred = predict(name, model, Z[score_split])
    if len(np.unique(pred)) < 2:
        raise SystemExit(f"DEGENERATE PREDICTOR: {name} seed {seed} predicts one class")
    return float(f1_score(y[splits[score_split]], pred, average="macro")), best_f1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, choices=["ftt_default", "tabm"])
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--smoke", action="store_true", help="tiny end-to-end pass, no artifacts")
    args = ap.parse_args()
    name = args.model

    X, y, gid = reconstruct_groups()
    Xp, yp, splits_row = load_processed(VARIANT)
    assert X.shape == Xp.shape and np.array_equal(y, yp), "group reconstruction misaligned"
    splits_row = {k: np.asarray(v) for k, v in splits_row.items()}

    if args.smoke:
        sub = {k: v[:3000] for k, v in splits_row.items()}
        pipe = build_preprocessor("full", n_components=None, seed=0)
        Z = fit_transform_splits(pipe, X, sub)
        params = {"lr": 1e-3, "wd": 1e-4, "n_blocks": 2, "d_block": 128, "dropout": 0.1}
        f1, va = fit_and_score(name, params, Z, y, sub, 0, max_epochs=2, patience=2)
        print(f"SMOKE OK {name}: test {f1:.4f} val {va:.4f} on 3k-row subset, device {DEVICE}")
        return

    lgbm_params, _ = tuned_params()
    cfg = {
        "variant": VARIANT,
        "model": name,
        "device": DEVICE,
        "batch": BATCH,
        "epochs": {"final": FINAL_EPOCHS, "search": SEARCH_EPOCHS},
        "tabm_k": TABM_K if name == "tabm" else None,
        "n_trials": args.trials if name == "tabm" else 0,
        "preregistration": "decision log 2026-09-01T11:05Z (V2-E3/V2-E4)",
    }
    with RunDir(f"deep_tabular_{name}_{VARIANT}", config=cfg, seeds=FINAL_SEEDS) as run:
        pipe = build_preprocessor("full", n_components=None, seed=0)
        Z_row = fit_transform_splits(pipe, X, splits_row)

        if name == "tabm":

            def objective(trial):
                params = {
                    "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
                    "wd": trial.suggest_float("wd", 1e-6, 1e-1, log=True),
                    "n_blocks": trial.suggest_int("n_blocks", 1, 4),
                    "d_block": trial.suggest_int("d_block", 128, 1024, step=64),
                    "dropout": trial.suggest_float("dropout", 0.0, 0.5),
                }
                vals = [
                    fit_and_score(
                        name,
                        params,
                        Z_row,
                        y,
                        splits_row,
                        s,
                        SEARCH_EPOCHS,
                        SEARCH_PATIENCE,
                        score_split="val",
                    )[0]
                    for s in TUNING_SEEDS
                ]
                return float(np.mean(vals))

            db = (REPO_ROOT / "results" / f"optuna_deep_{name}_{VARIANT}.db").as_posix()
            study = optuna.create_study(
                direction="maximize",
                study_name=f"deep_{name}_{VARIANT}",
                storage=f"sqlite:///{db}",
                load_if_exists=True,
                sampler=optuna.samplers.TPESampler(seed=0),
            )
            remaining = args.trials - len([t for t in study.trials if t.state.is_finished()])
            if remaining > 0:
                study.optimize(objective, n_trials=remaining, show_progress_bar=False)
            params = {**study.best_params}
            run.write_json("best_params.json", params)
            log.info("search done: best val %.4f, params %s", study.best_value, params)
        else:
            params = {}  # literature defaults are baked into build_model/make_optimizer

        # row-level finals, 10 seeds
        row_scores = []
        for s in FINAL_SEEDS:
            f1, _ = fit_and_score(
                name, params, Z_row, y, splits_row, s, FINAL_EPOCHS, FINAL_PATIENCE
            )
            row_scores.append(f1)
            log.info("row seed %d: %.4f", s, f1)
            run.write_json("results_partial.json", {"row_seed_scores": row_scores})

        # whole-file partitions at seed 0
        rng = np.random.default_rng(0)
        grp = []
        for p_idx in range(5):
            sp, held = group_splits(gid, y, rng)
            if held[2] != EXPECTED_TEST_GROUPS[p_idx]:
                raise SystemExit(f"PARTITION STREAM DIVERGED: p{p_idx} {held[2]}")
            assert_file_disjoint(sp, gid, p_idx)
            Zg = fit_transform_splits(build_preprocessor("full", n_components=None, seed=0), X, sp)
            f1, _ = fit_and_score(name, params, Zg, y, sp, 0, FINAL_EPOCHS, FINAL_PATIENCE)
            grp.append({"partition": p_idx, "test_groups": held[2], "f1": f1})
            log.info("group p%d: %.4f", p_idx, f1)
            run.write_json("results_partial.json", {"row_seed_scores": row_scores, "group": grp})

        row_m = float(np.mean(row_scores))
        grp_m = float(np.mean([g["f1"] for g in grp]))
        verdict = {
            "row_mean": row_m,
            "row_std": float(np.std(row_scores, ddof=1)),
            "row_vs_published_lgbm": row_m - PUBLISHED_LGBM_ROW,
            "group_mean": grp_m,
            "group_std": float(np.std([g["f1"] for g in grp], ddof=1)),
            "leak_gap": row_m - grp_m,
            "h1": (
                "pass: below tuned LightGBM by >= 0.005"
                if row_m <= PUBLISHED_LGBM_ROW - 0.005
                else "fail-direction: reaches/exceeds tuned LightGBM — report plainly"
            )
            if name == "ftt_default"
            else (
                "pass: within 0.01 of tuned LightGBM"
                if abs(row_m - PUBLISHED_LGBM_ROW) <= 0.01
                else f"outside 0.01: delta {row_m - PUBLISHED_LGBM_ROW:+.4f} — report as measured"
            ),
            "h2_in_band": bool(0.55 <= grp_m <= 0.66),
            "kill_fires": bool(grp_m >= 0.70),
            "params": params,
        }
        run.metrics = verdict
        run.write_json(
            "results.json", {"row_seed_scores": row_scores, "group": grp, "verdict": verdict}
        )
        log.info(
            "VERDICT %s | row %.4f±%.4f | group %.4f | leak gap %.4f | %s",
            name,
            row_m,
            verdict["row_std"],
            grp_m,
            verdict["leak_gap"],
            verdict["h1"],
        )
        print(f"-> {run.path}")


if __name__ == "__main__":
    main()
