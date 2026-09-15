"""P20: how much does the epsilon = 0.5 sigma PCA-space attack demand in RAW sensor units?

The paper states physical realizability as a limitation in prose but never quantifies it.
This probe turns the caveat into numbers, and adds the one test that can actually falsify
it: the pipeline (median-impute -> standardize -> PCA) is affine and therefore exactly
invertible, so every adversarial PCA point has a unique raw-space preimage.

H1 (spread): the attack is not a sparse sensor spoof. Mapping the perturbation back, a
    large majority of the 128 raw channels move by a non-trivial fraction of their own
    train std -- i.e. the threat model grants the attacker simultaneous control of the
    whole measurement vector, not of one PMU.

H2 (discrete channels): the preimage demands non-integer changes to the discrete
    event/log counters and relay status flags (DATA_DICTIONARY: control_panel_log*,
    relay*_log, snort_log*, R*:S), which no attacker can physically produce.
    KILL/decision test: round those channels to their nearest attainable (integer) value,
    push the rounded raw vector back through the FROZEN pipeline, and re-measure. If
    macro-F1 under the rounded attack is essentially unchanged, discreteness is not a
    defence and the limitation is about coupling, not about integrality. If it recovers,
    the reported degradation is optimistic and must be scoped.

Run: python -m experiments.run_physical_realizability

Graduated from tmp/ because the Discussion reports its numbers: the raw-space footprint
(109 of 128 channels moved) and the rounding test's 0.016 macro-F1 recovery are the
quantified form of the physical-realizability limitation, so the producer has to be a file
the repository carries. The RunDir name is unchanged, so the existing run directories still
match.
"""

from __future__ import annotations

import json

import numpy as np
from sklearn.metrics import f1_score

from experiments.run_adversarial import CLASSICAL, FittedZoo
from qgridbench.attacks.evasion import pgd
from qgridbench.data.preprocess import load_processed
from qgridbench.features.reduce import build_preprocessor
from qgridbench.models.classical.zoo import stratified_cap
from qgridbench.protocol import prepare_regime
from qgridbench.utils.run_tracking import REPO_ROOT, RunDir, get_logger, load_yaml
from qgridbench.utils.seeding import spawn_rng

log = get_logger(__name__)

SEEDS = [0, 1, 2]
EPS = 0.5
ENCODING = "zz"


def main():
    cfg_adv = load_yaml(REPO_ROOT / "configs" / "adversarial.yaml")
    cfg_q = load_yaml(REPO_ROOT / "configs" / "quantum.yaml")
    subset_cap = cfg_q["qkernel"]["subset_cap"]
    best_params = json.loads((REPO_ROOT / "results" / "adv_best_params_binary_q8.json").read_text())

    # raw table + the SAME frozen pipeline the study uses (fit on train only)
    Xraw, y, splits = load_processed("binary")
    pipe = build_preprocessor("pca", n_components=8, seed=0)
    pipe.fit(Xraw[splits["train"]])
    Z = {s: pipe.transform(Xraw[splits[s]]) for s in ("train", "val", "test")}
    ys = {s: y[splits[s]] for s in ("train", "val", "test")}

    def to_raw(Zp):
        """Exact affine preimage: PCA^-1 then StandardScaler^-1 (imputation is identity
        on the observed cells, so this is the raw-unit measurement vector)."""
        return pipe.named_steps["scale"].inverse_transform(
            pipe.named_steps["pca"].inverse_transform(Zp)
        )

    raw_std = np.nanstd(Xraw[splits["train"]], axis=0, ddof=0)
    raw_std_safe = np.where(raw_std > 0, raw_std, 1.0)
    # a channel is discrete if every observed train value is an integer
    train_raw = Xraw[splits["train"]]
    finite = np.isfinite(train_raw)
    is_discrete = np.array(
        [
            bool(np.all(np.equal(np.mod(train_raw[finite[:, j], j], 1), 0)))
            for j in range(train_raw.shape[1])
        ]
    )
    log.info(
        "discrete (integer-valued) raw channels: %d of %d",
        is_discrete.sum(),
        len(is_discrete),
    )

    ev_idx = stratified_cap(Z["test"], ys["test"], cfg_adv["eval_subset_adv"], seed=0)
    Xte, yte = Z["test"][ev_idx], ys["test"][ev_idx]
    raw_te = Xraw[splits["test"]][ev_idx]
    scale = Z["train"].std(axis=0, ddof=0)

    families = CLASSICAL + ["qsvm", "vqc"]

    def macro_f1(zoo, X):
        """Macro-F1 of every family on one feature matrix, scored against the same labels."""
        return {
            f: float(f1_score(yte, zoo.predict_proba(f, X).argmax(1), average="macro"))
            for f in families
        }

    with RunDir("probe_physical_realizability_binary_q8", config={"eps": EPS}, seeds=SEEDS) as run:
        per_seed = []
        for s in SEEDS:
            rng = spawn_rng(s, "adv")
            reg = prepare_regime("binary", "pca", n_components=8, seed=0)
            zoo = FittedZoo(reg, ENCODING, 8, s, subset_cap, best_params)
            gp = zoo.grad_provider("mlp")
            Xadv = pgd(
                gp,
                Xte,
                yte,
                EPS,
                scale,
                n_steps=cfg_adv["evasion"]["pgd"]["n_steps"],
                step_frac=cfg_adv["evasion"]["pgd"]["step_frac"],
                rng=rng,
            )

            # --- H1: raw-space footprint -----------------------------------
            # The pipeline is affine, so the DELTA is exact even though an 8-component
            # reconstruction of the point is not: reconstruction error cancels here.
            d_raw = np.abs(to_raw(Xadv) - to_raw(Xte))  # native units
            per_channel = (d_raw / raw_std_safe).mean(axis=0)  # per channel, in its own std

            # --- H2: discreteness + the rounding test ----------------------
            # Apply the exact raw-space delta to the TRUE measurement vector, so the
            # rounding acts on real sensor values rather than on a reconstruction.
            raw_adv_true = raw_te + (to_raw(Xadv) - to_raw(Xte))
            disc = raw_adv_true[:, is_discrete]
            frac_noninteger = float(np.mean((np.abs(np.mod(disc, 1)) > 1e-9)[np.isfinite(disc)]))
            raw_round = raw_adv_true.copy()
            raw_round[:, is_discrete] = np.round(raw_round[:, is_discrete])
            # CONTROL: the same round trip WITHOUT rounding. Cells that are non-finite in
            # the raw table are imputed to a train median, so the attacker's perturbation
            # on them is lost in the round trip; this arm isolates that from rounding.
            Z_trip = pipe.transform(raw_adv_true)
            Z_round = pipe.transform(raw_round)  # impute -> scale -> PCA, all frozen
            # what the rounding costs the attacker in budget terms
            eps_after = np.abs(Z_round - Xte).max(axis=1) / scale.max()

            f1 = {
                "clean": macro_f1(zoo, Xte),
                "pgd": macro_f1(zoo, Xadv),
                "pgd_roundtrip": macro_f1(zoo, Z_trip),
                "pgd_rounded": macro_f1(zoo, Z_round),
            }
            rec = {
                "f1": f1,
                "raw_footprint": {
                    "channels_moved_gt_0.01sigma": int((per_channel > 0.01).sum()),
                    "channels_moved_gt_0.10sigma": int((per_channel > 0.10).sum()),
                    "n_channels": int(len(per_channel)),
                    "mean_abs_delta_sigma": float(per_channel.mean()),
                    "max_abs_delta_sigma": float(per_channel.max()),
                    "median_abs_delta_sigma": float(np.median(per_channel)),
                },
                "discrete": {
                    "n_discrete_channels": int(is_discrete.sum()),
                    "frac_noninteger_after_attack": frac_noninteger,
                    "mean_abs_delta_sigma_discrete": float(per_channel[is_discrete].mean())
                    if is_discrete.any()
                    else float("nan"),
                    "mean_abs_delta_sigma_continuous": float(per_channel[~is_discrete].mean()),
                },
                "rounding": {
                    "mean_eps_after_rounding": float(eps_after.mean()),
                    "median_eps_after_rounding": float(np.median(eps_after)),
                },
            }
            per_seed.append(rec)
            run.write_json("results_partial.json", {"per_seed": per_seed})
            log.info("seed %d | %s", s, json.dumps(rec["raw_footprint"]))
            log.info("seed %d | discrete %s", s, json.dumps(rec["discrete"]))
            rounded_triples = {
                f: (
                    round(f1["pgd"][f], 3),
                    round(f1["pgd_roundtrip"][f], 3),
                    round(f1["pgd_rounded"][f], 3),
                )
                for f in families
            }
            log.info("seed %d | f1 pgd vs rounded: %s", s, rounded_triples)
        run.write_json("results.json", {"per_seed": per_seed, "eps": EPS})
        print(f"-> {run.path}")


if __name__ == "__main__":
    main()
