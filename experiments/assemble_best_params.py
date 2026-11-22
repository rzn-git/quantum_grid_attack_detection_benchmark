"""Assemble a variant's tuned-parameter file for the robustness runners from its Optuna studies.

Reads the best trial of each PCA-regime study of one variant and qubit count:
  classical  cls_<model>_<variant>_pca<q>      -> {<model>: best_params}
  QSVM       qk_<encoding>_<variant>_q<q>      -> qk_bw, qk_C
  VQC        vqc_<variant>_q<q>_q<q>           -> vqc_depth, vqc_lr, vqc_batch
(each name with _<suffix> after the variant's tag when --study-suffix is given)
and writes results/adv_best_params_<variant>_q<q>.json (the file protocol.best_params_path
resolves), with a provenance file beside it. `vqc_epochs` is not a tuned parameter, so it is
an explicit argument. `--check` compares against an existing file and writes nothing; any
difference fails loud with the per-key values.

Run:  uv run python -m experiments.assemble_best_params --variant gas_binary --vqc-epochs 40
      uv run python -m experiments.assemble_best_params --variant binary --vqc-epochs 40 \
          --check results/adv_best_params_binary_q8.json
      uv run python -m experiments.assemble_best_params --variant binary --vqc-epochs 40 \
          --out results/adv_best_params_binary_q8_table1.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import optuna

from qgridbench.models.classical.zoo import MODEL_NAMES
from qgridbench.utils.run_tracking import REPO_ROOT, compute_env, git_describe, utc_stamp

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _study(name: str, db_stem: str) -> optuna.Study:
    db = REPO_ROOT / "results" / f"optuna_{db_stem}.db"
    if not db.exists():
        raise FileNotFoundError(f"missing Optuna storage {db}")
    return optuna.load_study(study_name=name, storage=f"sqlite:///{db.as_posix()}")


def _best(name: str, db_stem: str, n_trials: int) -> tuple[dict, dict]:
    st = _study(name, db_stem)
    done = [t for t in st.trials if t.state.is_finished()]
    if len(done) < n_trials:
        raise RuntimeError(f"study {name} has {len(done)} finished trials, budget is {n_trials}")
    return st.best_params, {
        "study": name,
        "n_finished": len(done),
        "best_trial": st.best_trial.number,
        "best_value": st.best_value,
    }


def assemble(
    variant: str, qubits: int, encoding: str, vqc_epochs: int, n_trials: int, suffix: str = ""
):
    sfx = f"_{suffix}" if suffix else ""
    out: dict = {}
    prov: dict = {}
    for m in MODEL_NAMES:
        name = f"cls_{m}_{variant}_pca{qubits}{sfx}"
        out[m], prov[m] = _best(name, name, n_trials)
    qk_name = f"qk_{encoding}_{variant}_q{qubits}{sfx}"
    qk, prov["qsvm"] = _best(qk_name, qk_name, n_trials)
    out["qk_bw"], out["qk_C"] = qk["bandwidth"], qk["C"]
    vqc, prov["vqc"] = _best(
        f"vqc_{variant}_q{qubits}{sfx}_q{qubits}", f"vqc_{variant}_q{qubits}{sfx}", n_trials
    )
    out["vqc_depth"], out["vqc_lr"], out["vqc_batch"] = vqc["depth"], vqc["lr"], vqc["batch_size"]
    out["vqc_epochs"] = vqc_epochs
    return out, prov


def diff(a: dict, b: dict, prefix: str = "") -> list[str]:
    lines = []
    for k in sorted(set(a) | set(b)):
        va, vb = a.get(k, "<absent>"), b.get(k, "<absent>")
        if isinstance(va, dict) and isinstance(vb, dict):
            lines += diff(va, vb, f"{prefix}{k}.")
        elif va != vb:
            lines.append(f"{prefix}{k}: assembled {va!r} != file {vb!r}")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--variant", required=True)
    ap.add_argument("--qubits", type=int, default=8)
    ap.add_argument("--encoding", default="zz", help="QSVM encoding of the robustness table")
    ap.add_argument("--vqc-epochs", type=int, required=True)
    ap.add_argument(
        "--n-trials", type=int, default=50, help="the equal budget every study must hold"
    )
    ap.add_argument("--check", default=None, help="compare against this file; write nothing")
    ap.add_argument(
        "--study-suffix",
        default="",
        help="the --study-suffix the run_classical and run_quantum searches used",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="path relative to the repo root (default: the variant's own file under results/)",
    )
    args = ap.parse_args()

    out, prov = assemble(
        args.variant, args.qubits, args.encoding, args.vqc_epochs, args.n_trials, args.study_suffix
    )
    if args.check:
        ref = json.loads((REPO_ROOT / args.check).read_text())
        lines = diff(out, ref)
        if lines:
            sys.exit(
                "assembled parameters differ from " + args.check + ":\n  " + "\n  ".join(lines)
            )
        print(f"identical to {args.check}")
        return
    path = REPO_ROOT / (args.out or f"results/adv_best_params_{args.variant}_q{args.qubits}.json")
    if path.exists():
        raise FileExistsError(f"{path} exists; compare with --check instead of overwriting")
    path.write_text(json.dumps(out, indent=2))
    Path(str(path).replace(".json", "_provenance.json")).write_text(
        json.dumps(
            {
                "written_at": utc_stamp(),
                "git": git_describe(),
                "args": vars(args),
                "studies": prov,
                "env": compute_env(),
            },
            indent=2,
            default=str,
        )
    )
    print(f"-> {path}")


if __name__ == "__main__":
    main()
