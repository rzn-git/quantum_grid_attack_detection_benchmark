# RUN — QuantumGridBench runbook

Commands only, clustered by cadence. One command per block.

## One-time setup

1. Create env and install (editable) with locked deps.
```bash
uv venv --python 3.11 && uv pip install -r requirements.txt -e .
```

2. Download, verify (SHA256), and extract the dataset.
```bash
python -m qgridbench.data.download
```

3. Validate schema + class balance → `results/runs/…/data_report.json`.
```bash
python -m qgridbench.data.validate
```

4. Pool, clean, and split (fixed 60/20/20 indices).
```bash
python -m qgridbench.data.preprocess binary triple
```

## Sanity (run before any scaling)

5. MVP smoke pass — entire pipeline in miniature.
```bash
python -m experiments.run_smoke_test
```

## Definition of done (gates — each exits non-zero on failure)

6. Tests, lint, format.
```bash
python -m pytest -q
```
```bash
python -m ruff check src experiments tests
```
```bash
python -m ruff format --check src experiments tests
```

## Full experiments (Phases 2–4)

7. Classical baselines — full-feature regime, 6 models, 10 seeds.
```bash
python -m experiments.run_classical --variant binary --regime full
```

8. Classical baselines — PCA regime (quantum-comparison set), per qubit count.
```bash
python -m experiments.run_classical --variant binary --regime pca --pca 8 --subset-cap 2000
```

9. Quantum kernels (QSVM) with diagnostics, per qubit count.
```bash
python -m experiments.run_quantum --variant binary --qubits 8 --model qkernel
```

10. Variational quantum classifier.
```bash
python -m experiments.run_quantum --variant binary --qubits 8 --model vqc
```

11. Ablations + data-efficiency curves (shots, feature-map/qubit sweep, VQC depth).
```bash
python -m experiments.run_ablations --variant binary --qubits 8
```

12. Adversarial robustness — evasion curves, transfer matrix, sensor noise, poisoning
    (10 seeds). Excludes the black-box stage, which is run separately in step 13.
```bash
python -m experiments.run_adversarial --variant binary --qubits 8 --only evasion noise poison
```

13. Black-box control (HopSkipJump). Cost is set by the DEFENDER's per-query latency, and
    random forest is ~1000x slower per query than logistic regression, so families are run
    as concurrent single-family processes. Each finished family checkpoints to
    `results/blackbox_binary_q8/<family>.json`; a kill costs at most one family. Run these
    four concurrently.
```bash
python -m experiments.run_adversarial --variant binary --qubits 8 --seeds 0 --only blackbox --bb-families rf
```
```bash
python -m experiments.run_adversarial --variant binary --qubits 8 --seeds 0 --only blackbox --bb-families vqc qsvm
```
```bash
python -m experiments.run_adversarial --variant binary --qubits 8 --seeds 0 --only blackbox --bb-families xgb lgbm rbf_svm logreg mlp
```

14. Merge the per-family black-box results into one run dir.
```bash
python -m experiments.run_adversarial --variant binary --qubits 8 --merge-blackbox
```

14b. White-box anchors and the kernel mechanism. Three stages: `p5a` targets the fidelity
    kernel directly (gradients through the statevector simulator, numerically gated, with
    the tuned RBF-SVM as the matched classical-kernel control), `p6a` anchors robustness at
    full-feature accuracy, `p7b` measures why the two kernels diverge (local sensitivity,
    then kernel exhaustion out to 2σ). Run stages separately (`--only p5a`) on a busy box;
    `p6a` refits ten 128-feature MLPs and dominates the wall clock.
```bash
python -m experiments.run_whitebox_qsvm --variant binary --qubits 8
```

14c. Split-protocol controls behind the paper's leakage section: the paired whole-file
    (group) split control, and the white-box kernel replication under whole-file training.
```bash
python -m experiments.run_group_split_control
```
```bash
python -m experiments.run_group_whitebox_control
```

## Reporting artifacts

15. Generate every table, figure, and statistic from `results/runs/` — the only writer of
    the generated exhibits; never hand-edit them.
```bash
python -m experiments.make_report
```

## Optional — quantum hardware (gated)

18. Estimate cost first; a paid Braket submission is blocked unless `QGB_APPROVED_BUDGET_USD`
    ≥ estimate is set by the operator (see `utils/cost_guard.py`). Never submit without approval.
