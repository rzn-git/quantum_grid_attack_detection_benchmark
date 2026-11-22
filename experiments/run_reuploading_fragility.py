"""P21b: is the re-uploading VQC any less fragile than the deployed one?

P21 closed the strawman objection on ACCURACY only: a data re-uploading ansatz does not lift the
VQC (best val 0.5369 against the deployed 0.5404). The objection has a second half the probe did
not answer — the paper's fragility claims (retention 0.072 under white-box PGD) are claims about
the deployed ansatz, and a reviewer can still say a modern ansatz might be more robust.

H: the re-uploading VQC's white-box retention is no better than the deployed VQC's 0.072, so the
fragility result is ansatz-independent within this budget.
KILL: retention above ~0.3 for the re-uploading model would make the fragility finding
architecture-specific and it would have to be scoped to `AngleEmbedding` +
`StronglyEntanglingLayers` everywhere it appears.

Uses the SAME attack path as the published number: `loss_input_grad` through the simulator,
chain-ruled through the angle scaler, PGD at the headline budget.

Run:  python -m experiments.run_reuploading_fragility [--best-params FILE] [--run-tag TAG]
The deployed arm reads its depth and learning rate from the tuned file; the re-uploading arm
stays at its hard-coded P21 configuration.

Graduated from tmp/ because the appendix reports its numbers: white-box retention at 0.5
sigma is 0.053 +- 0.034 for the re-uploading model against 0.053 +- 0.021 for the deployed
one. The RunDir name keeps its original `probe_` prefix so already-recorded results still
resolve.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from sklearn.metrics import f1_score

from experiments.exports import Exports
from experiments.run_reuploading_vqc import ReuploadVQC
from experiments.tuned_file import add_tuned_file_args, load_tuned_file, run_name
from qgridbench.attacks.evasion import pgd
from qgridbench.models.classical.zoo import stratified_cap
from qgridbench.models.quantum.vqc import VQClassifier
from qgridbench.protocol import kernel_subset, prepare_regime
from qgridbench.utils.run_tracking import REPO_ROOT, RunDir, get_logger, load_yaml
from qgridbench.utils.seeding import set_all_seeds, spawn_rng

log = get_logger(__name__)

SEEDS = [0, 1, 2]
DEPLOYED_RETENTION = 0.0719  # paper Table II, VQC white-box PGD at 0.5 sigma


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_tuned_file_args(ap)
    args = ap.parse_args()
    cfg_adv = load_yaml(REPO_ROOT / "configs" / "adversarial.yaml")
    cfg_q = load_yaml(REPO_ROOT / "configs" / "quantum.yaml")
    bp_rel, best = load_tuned_file(args.best_params, "binary", 8)

    reg = prepare_regime("binary", "pca", n_components=8, seed=0)
    scale = reg.feature_scale
    sub = kernel_subset(reg, cfg_q["qkernel"]["subset_cap"], seed=0)
    A_tr, y_tr = reg.angle_scaler.transform(reg.X["train"][sub]), reg.y["train"][sub]
    va = stratified_cap(reg.X["val"], reg.y["val"], 2000, seed=0)
    A_va, y_va = reg.angle_scaler.transform(reg.X["val"][va]), reg.y["val"][va]
    ev = stratified_cap(reg.X["test"], reg.y["test"], cfg_adv["eval_subset_adv"], seed=0)
    Xte, yte = reg.X["test"][ev], reg.y["test"][ev]
    eps_grid = cfg_adv["evasion"]["epsilons"]

    # P21's best re-uploading configuration, and the deployed one for a matched control
    arms = {
        "reupload_d4_lr0.05": (ReuploadVQC, 4, 0.05),
        f"deployed_d{best['vqc_depth']}_lr{round(best['vqc_lr'], 4)}": (
            VQClassifier,
            best["vqc_depth"],
            best["vqc_lr"],
        ),
    }

    with RunDir(
        run_name("probe_reupload_fragility_binary_q8", args.run_tag),
        config={"arms": list(arms), "best_params": bp_rel},
        seeds=SEEDS,
    ) as run:
        per_seed = []
        ex = Exports("binary", bp_rel)
        for s in SEEDS:
            rec = {}
            for name, (cls, depth, lr) in arms.items():
                set_all_seeds(s)
                m = cls(
                    8,
                    depth=depth,
                    n_classes=2,
                    lr=lr,
                    batch_size=128,
                    max_epochs=cfg_q["vqc"]["max_epochs"],
                    patience=cfg_q["vqc"]["patience"],
                    seed=s,
                )
                m.fit(A_tr, y_tr, A_va, y_va)

                m_scale = reg.angle_scaler._mm.scale_

                def grad(X, y, _m=m, _sc=m_scale):
                    return _m.loss_input_grad(reg.angle_scaler.transform(X), y) * _sc

                def f1_of(X, cond, _m=m, _name=name, _s=s):
                    proba = _m.predict_proba(reg.angle_scaler.transform(X))
                    ex.predictions(_s, cond, _name, proba, yte, ev)
                    f1 = float(f1_score(yte, proba.argmax(1), average="macro"))
                    ex.metric(_s, _name, cond, "macro_f1", f1)
                    return f1

                clean = f1_of(Xte, "clean")
                rng = spawn_rng(s, ex.stream("reupfrag"))
                curve = {}
                for e in eps_grid:
                    Xa = pgd(
                        grad,
                        Xte,
                        yte,
                        e,
                        scale,
                        n_steps=cfg_adv["evasion"]["pgd"]["n_steps"],
                        step_frac=cfg_adv["evasion"]["pgd"]["step_frac"],
                        rng=rng,
                    )
                    ex.crafted_input(s, f"pgd@{e}", name, Xa)
                    curve[str(e)] = f1_of(Xa, f"pgd@{e}")
                rec[name] = {
                    "clean": clean,
                    "curve": curve,
                    "retention_0.5": curve["0.5"] / clean if clean else float("nan"),
                }
                log.info(
                    "seed %d | %s clean %.3f -> %.3f at 0.5 sigma (retention %.3f)",
                    s,
                    name,
                    clean,
                    curve["0.5"],
                    rec[name]["retention_0.5"],
                )
            per_seed.append(rec)
            run.write_json("results_partial.json", {"per_seed": per_seed})

        summary = {}
        for name in arms:
            r = np.array([p[name]["retention_0.5"] for p in per_seed])
            c = np.array([p[name]["clean"] for p in per_seed])
            a = np.array([p[name]["curve"]["0.5"] for p in per_seed])
            summary[name] = {
                "clean_mean": float(c.mean()),
                "adv_mean": float(a.mean()),
                "retention_mean": float(r.mean()),
                "retention_std": float(r.std(ddof=1)),
            }
        verdict = (
            "ANSATZ-INDEPENDENT — re-uploading is no less fragile"
            if summary["reupload_d4_lr0.05"]["retention_mean"] <= 0.3
            else "SCOPE REQUIRED — re-uploading is materially more robust"
        )
        ex.write(run.path)
        run.write_json(
            "results.json",
            {
                "per_seed": per_seed,
                "summary": summary,
                "deployed_published_retention": DEPLOYED_RETENTION,
                "verdict": verdict,
            },
        )
        log.info("VERDICT: %s | %s", verdict, json.dumps(summary, indent=2))
        print(f"-> {run.path}")


if __name__ == "__main__":
    main()
