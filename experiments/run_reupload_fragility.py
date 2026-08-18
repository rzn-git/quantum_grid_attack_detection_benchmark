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

Run:  python -m experiments.run_reupload_fragility

Graduated from tmp/ because the appendix reports its numbers: white-box retention at 0.5
sigma is 0.053 +- 0.034 for the re-uploading model against 0.053 +- 0.021 for the deployed
one. The RunDir name keeps its original `probe_` prefix so already-recorded results still
resolve.
"""

from __future__ import annotations

import json

import numpy as np
from sklearn.metrics import f1_score

from experiments.run_reuploading_vqc import ReuploadVQC
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
    cfg_adv = load_yaml(REPO_ROOT / "configs" / "adversarial.yaml")
    cfg_q = load_yaml(REPO_ROOT / "configs" / "quantum.yaml")
    best = json.loads((REPO_ROOT / "results" / "adv_best_params_binary_q8.json").read_text())

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
        "deployed_d4_lr0.0397": (VQClassifier, best["vqc_depth"], best["vqc_lr"]),
    }

    with RunDir(
        "probe_reupload_fragility_binary_q8", config={"arms": list(arms)}, seeds=SEEDS
    ) as run:
        per_seed = []
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

                def f1_of(X, _m=m):
                    return float(
                        f1_score(
                            yte,
                            _m.predict_proba(reg.angle_scaler.transform(X)).argmax(1),
                            average="macro",
                        )
                    )

                clean = f1_of(Xte)
                rng = spawn_rng(s, "reupfrag")
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
                    curve[str(e)] = f1_of(Xa)
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
