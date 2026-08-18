"""P21: is the VQC's near-chance accuracy an artefact of a 2018-era ansatz?

External review: "angle embedding + strongly-entangling layers is 2020-era; data
re-uploading is the modern standard and cheap at 8 qubits. One re-uploading VQC either
lifts the model into relevance or shows the weakness is architecture-independent."

DESIGN (the variable is re-uploading and nothing else). The probe subclasses the
deployed VQClassifier and overrides only the circuit: instead of one AngleEmbedding
followed by `depth` trainable layers, it interleaves — AngleEmbedding, one trainable
layer, AngleEmbedding, one trainable layer, ... (Perez-Salinas et al. 2020). The weight
tensor keeps the SAME shape, so parameter count, depth, optimiser, class-weighted loss,
early stopping, seeds and data subset are all identical to the published model.

PRE-REGISTERED KILL CRITERION. The deployed VQC's tuned validation macro-F1 at 8 qubits
is 0.5404 (best of 50 Optuna trials; median 0.5214, min 0.4979 — a flat, near-floor
distribution). The stratified-random floor is 0.5005.
  - If the best re-uploading configuration reaches val macro-F1 >= 0.57, the ansatz
    matters, the strawman objection has teeth, and the VQC family results must be re-run
    on the better architecture before the fragility claims are restated.
  - If it does not clear 0.5404, the weakness is architecture-independent within this
    budget, and the paper can say so with a citation to this probe.
Anything between is reported as inconclusive, not spun either way.

BUDGET HONESTY: this is a 6-configuration grid at one seed, i.e. LESS budget than the
50-trial Optuna search the deployed VQC received. A win here would therefore be decisive;
a loss is evidence, not proof, and is logged as such.

Run:  python -m experiments.run_reuploading_vqc

Graduated from tmp/ because the appendix reports its numbers (six configurations spanning
0.513-0.537 validation macro-F1) and because run_reupload_fragility imports `ReuploadVQC`
from here -- a shipped experiment cannot import dev-only scratch. The RunDir name keeps its
original `probe_` prefix so already-recorded results still resolve.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pennylane as qml
from sklearn.metrics import f1_score

from qgridbench.models.classical.zoo import stratified_cap
from qgridbench.models.quantum.vqc import VQClassifier
from qgridbench.protocol import kernel_subset, prepare_regime
from qgridbench.utils.run_tracking import REPO_ROOT, RunDir, get_logger, load_yaml
from qgridbench.utils.seeding import set_all_seeds

log = get_logger(__name__)

DEPLOYED_VAL_BEST = 0.5404  # optuna_vqc_binary_q8.db, best of 50 trials
FLOOR = 0.5005
LIFT_THRESHOLD = 0.57  # pre-registered "the ansatz matters" line
SEED = 0


class ReuploadVQC(VQClassifier):
    """Identical to VQClassifier except the data is re-uploaded before every layer."""

    def _build_qnodes(self) -> None:
        def circuit(x, weights):
            for layer in range(self.depth):
                qml.AngleEmbedding(x, wires=range(self.n_qubits), rotation="Y")
                qml.StronglyEntanglingLayers(weights[layer : layer + 1], wires=range(self.n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_out)]

        self._circuit = circuit
        dev = qml.device(self.device_name, wires=self.n_qubits)
        self._qnode = qml.QNode(circuit, dev, interface="torch", diff_method=self.diff_method)
        self._qnode_grad = self._qnode


def main():
    cfg_q = load_yaml(REPO_ROOT / "configs" / "quantum.yaml")
    subset_cap = cfg_q["qkernel"]["subset_cap"]
    max_epochs = cfg_q["vqc"]["tuning_max_epochs"]
    patience = cfg_q["vqc"]["patience"]

    reg = prepare_regime("binary", "pca", n_components=8, seed=0)
    sub = kernel_subset(reg, subset_cap, seed=0)
    A_tr = reg.angle_scaler.transform(reg.X["train"][sub])
    y_tr = reg.y["train"][sub]
    va = stratified_cap(reg.X["val"], reg.y["val"], 2000, seed=0)
    A_va = reg.angle_scaler.transform(reg.X["val"][va])
    y_va = reg.y["val"][va]
    A_te = reg.angle_scaler.transform(reg.X["test"])
    y_te = reg.y["test"]

    grid = [(d, lr) for d in (2, 4, 6) for lr in (0.0175, 0.05)]
    with RunDir("probe_reuploading_vqc_binary_q8", config={"grid": grid}, seeds=[SEED]) as run:
        results = []
        for depth, lr in grid:
            for kind, cls in (("reupload", ReuploadVQC), ("deployed_ansatz", VQClassifier)):
                if kind == "deployed_ansatz" and lr != 0.0175:
                    continue  # the matched control only needs one lr per depth
                set_all_seeds(SEED)
                t0 = time.perf_counter()
                m = cls(
                    8,
                    depth=depth,
                    n_classes=2,
                    lr=lr,
                    batch_size=128,
                    max_epochs=max_epochs,
                    patience=patience,
                    seed=SEED,
                )
                m.fit(A_tr, y_tr, A_va, y_va)
                val = float(f1_score(y_va, m.predict(A_va), average="macro"))
                test = float(f1_score(y_te, m.predict(A_te), average="macro"))
                rec = {
                    "ansatz": kind,
                    "depth": depth,
                    "lr": lr,
                    "val_macro_f1": val,
                    "test_macro_f1": test,
                    "n_params": int(np.prod(m.weights.shape)),
                    "seconds": round(time.perf_counter() - t0, 1),
                }
                results.append(rec)
                run.write_json("results_partial.json", {"results": results})
                log.info("%s", json.dumps(rec))

        ru = [r for r in results if r["ansatz"] == "reupload"]
        best = max(ru, key=lambda r: r["val_macro_f1"])
        verdict = (
            "LIFT — ansatz matters, VQC family results must be re-run"
            if best["val_macro_f1"] >= LIFT_THRESHOLD
            else "NO LIFT — weakness is architecture-independent within this budget"
            if best["val_macro_f1"] <= DEPLOYED_VAL_BEST
            else "INCONCLUSIVE — between the deployed best and the pre-registered lift line"
        )
        out = {
            "results": results,
            "best_reupload": best,
            "verdict": verdict,
            "deployed_val_best": DEPLOYED_VAL_BEST,
            "floor": FLOOR,
            "lift_threshold": LIFT_THRESHOLD,
        }
        run.write_json("results.json", out)
        log.info("VERDICT: %s (best re-upload val %.4f)", verdict, best["val_macro_f1"])
        print(f"-> {run.path}")


if __name__ == "__main__":
    main()
