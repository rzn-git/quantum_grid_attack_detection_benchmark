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

6. Tests, lint, format. Run from the activated `.venv` (or prefix each with `uv run`):
   a system interpreter of a different minor version fails at collection.
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
    XGBoost's fit depends on the thread count; the published accuracy rows used
    `OMP_NUM_THREADS=6` (full features) and 4 (PCA-8), and across 1–24 threads they move by
    less than one seed standard deviation.
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
    `--encoding zz` selects the ZZ-map QSVM of the paper's robustness results in steps 12
    to 14; the runner's default is the RY-angle map.
```bash
python -m experiments.run_adversarial --variant binary --qubits 8 --encoding zz --only evasion noise poison
```

13. Black-box control (HopSkipJump). Cost is set by the DEFENDER's per-query latency, and
    random forest is ~1000x slower per query than logistic regression, so families are run
    as concurrent single-family processes. Each finished family checkpoints to
    `results/blackbox_binary_q8/<family>.json`; a kill costs at most one family. Run these
    four concurrently.
```bash
python -m experiments.run_adversarial --variant binary --qubits 8 --encoding zz --seeds 0 --only blackbox --bb-families rf
```
```bash
python -m experiments.run_adversarial --variant binary --qubits 8 --encoding zz --seeds 0 --only blackbox --bb-families vqc qsvm
```
```bash
python -m experiments.run_adversarial --variant binary --qubits 8 --encoding zz --seeds 0 --only blackbox --bb-families xgb lgbm rbf_svm logreg mlp
```

14. Merge the per-family black-box results into one run dir.
```bash
python -m experiments.run_adversarial --variant binary --qubits 8 --encoding zz --merge-blackbox
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
    The second attacks at the six grid radii up to 0.5σ on partitions 0 and 2 and takes about
    eight minutes per partition at ten threads; `--partitions 0` and `--partitions 2` with
    `--run-tag` run the two side by side. Its runs reach `make_report` through
    `read_worst_radius` (step 14g).
```bash
python -m experiments.run_group_split_control
```
```bash
python -m experiments.run_group_whitebox_control
```

14d. The split-mechanism chain behind the paper's leak-rate law and per-family slopes, and
    the capacity control. Each is standalone, takes `--smoke` for a tiny end-to-end pass,
    and none feeds `make_report`. `run_capacity_matched_control` walks a fixed LightGBM
    capacity ladder to show the whole-file ceiling is not an architecture effect;
    `run_file_row_order` measures lag-distance ratios and label-run structure to establish
    that row order within a released file is temporal; `run_temporal_split` builds the
    within-(file, scenario)-block ordered, embargoed, shuffled, middle-band and reversed
    splits and reads the leak-rate law from them; `run_leak_rate_families` re-runs those six
    splits across five classical families (it imports the split constructions from
    `run_temporal_split`, so run that one first and keep its artifact for the positive
    control). All four need steps 2–4 and the tuned studies from step 7.
```bash
python -m experiments.run_capacity_matched_control
```
```bash
python -m experiments.run_file_row_order
```
```bash
python -m experiments.run_temporal_split
```
```bash
python -m experiments.run_leak_rate_families
```

14e. The leak-decay curve behind the paper's "the leak has a length" sentence, and its
    replication on a second ICS release. `run_leak_decay_curve` needs steps 2–4 and the
    tuned LightGBM and XGBoost studies from step 7 (about two minutes: seven fits, a radius
    graph, and a permutation null). `run_leak_decay_gas_pipeline` needs the Morris
    gas-pipeline release (`IanArffDataset.arff` from the same ICS datasets page as step 2,
    placed at `data/raw/gas_pipeline/`) and nothing else; it runs library-default models.
    Both take `--smoke`.
```bash
python -m experiments.run_leak_decay_curve
```
```bash
python -m experiments.run_leak_decay_gas_pipeline
```

14f. File-tail split (order-preserving 60/25/15 per file, binary and three-class): shows that a
    positional split of whole files on the released order is a class holdout, and that the
    same split after a within-file shuffle returns to the row-level band. Needs steps 2–4 and
    the tuned LightGBM studies from step 7. Standalone; takes `--smoke`; does not feed
    `make_report`.
```bash
python -m experiments.run_file_tail_split
```

14g. Probes behind numbers the paper reports in prose rather than in a generated table.
    Each is standalone; none feeds `make_report` except `run_hopskipjump_restarts`, whose runs
    reach it through `pool_hsj_restarts`, and the white-box runs, which reach it through
    `read_worst_radius` (both below). `run_kernel_cap_cost` produces the cap cost,
    `run_hopskipjump_restarts` the black-box restart cells, `run_physical_realizability` the
    raw-channel footprint, and `run_surrogate_mechanism` the mechanism behind the corrected
    transfer asymmetry. `run_hopskipjump_restarts` takes an optional restart count and family
    list.
```bash
python -m experiments.run_kernel_cap_cost
```
```bash
python -m experiments.run_hopskipjump_restarts 5
```
    `pool_hsj_restarts` then pools the restart runs named in `experiments/hsj_restart_pool.yaml`
    into the one run `make_report` pins (`HSJ_POOLED_RUN`). After a new restart run: add it to
    the pool file, run this, and set the pin to the new pooled run; `make_report` fails if the
    pool file and the pinned run disagree.
```bash
python -m experiments.pool_hsj_restarts
```
    `read_worst_radius` reads the stored white-box curves named in
    `experiments/worst_radius_sources.yaml` at the worst grid radius inside the budget into the
    one run `make_report` pins (`WORST_RADIUS_RUN`). It attacks nothing. After a new source
    run: name it in the sources file, run this, and set the pin to the new run; `make_report`
    fails if the sources file and the pinned run disagree.
```bash
python -m experiments.read_worst_radius
```
```bash
python -m experiments.run_physical_realizability
```
```bash
python -m experiments.run_surrogate_mechanism
```

14h. Replication on the gas-pipeline data with pressure log-scaled: the gas column of the paper's
    replication figure. Needs the Morris gas-pipeline release (`IanArffDataset.arff` at
    `data/raw/gas_pipeline/`), the variant `gas_binary_slog` preprocessed, and its tuned
    models (`results/adv_best_params_gas_binary_slog_q8.json`). Those earlier steps, the
    main-study gas pipeline and the two simulated grids are not listed in this runbook yet:
    each runner's module docstring gives its command, and `experiments/run_labels.csv` names
    the runs the exhibits read. The nine jobs are independent except that the Square and
    per-class jobs name the first job's run; the white-box job scores the QSVM on the exact
    double-precision path.
```bash
python -m experiments.run_adversarial --variant gas_binary_slog --qubits 8 --encoding zz --only evasion noise poison
```
```bash
python -m experiments.run_adversarial --variant gas_binary_slog --qubits 8 --encoding zz --seeds 0 --only random
```
```bash
python -m experiments.run_whitebox_qsvm --variant gas_binary_slog --qubits 8 --only p5a --exact-eval
```
```bash
python -m experiments.run_surrogate_control --variant gas_binary_slog --qubits 8
```
```bash
python -m experiments.run_hopskipjump_restarts 5 qsvm vqc --variant gas_binary_slog
```
```bash
python -m experiments.run_hopskipjump_restarts 5 xgb lgbm rbf_svm logreg mlp --variant gas_binary_slog
```
```bash
python -m experiments.run_hopskipjump_restarts 5 rf --variant gas_binary_slog
```
```bash
python -m experiments.run_square_blackbox --variant gas_binary_slog --seeds 0 1 2 --reference-run results/runs/<stamp>_adversarial_gas_binary_slog_q8
```
```bash
python -m experiments.run_evasion_per_class --variant gas_binary_slog --anchor-run results/runs/<stamp>_adversarial_gas_binary_slog_q8 --workers 1 --threads 4
```
    The verdict reads the nine runs and applies the registered rules; it attacks nothing.
    Then name the white-box, verdict and shared-protocol runs under the source `gas_slog` in
    `experiments/worst_radius_sources.yaml` and run `read_worst_radius` (step 14g).
```bash
python -m experiments.run_gas_robustness_verdict --variant gas_binary_slog --adv-run <run> --random-run <run> --whitebox-run <run> --surrogate-run <run> --hsj-runs <run> <run> <run> --square-run <run> --per-class-run <run>
```

14i. The matched contest over redraws of the 2000 training rows (a row of the paper's
    further-controls table). One run per dimensionality; draw 0 is the published subset and
    must reproduce the published quantum-kernel scores. The 16-dimension run holds about
    2.5 GB per worker.
```bash
python -m experiments.run_training_draws --pca 8 --workers 3 --threads 3
```
```bash
python -m experiments.run_training_draws --pca 12 --workers 3 --threads 3
```
```bash
python -m experiments.run_training_draws --pca 16 --workers 2 --threads 4
```

14j. HopSkipJump at three query budgets (the published settings, then about 2780 and about
    10 870 queries per point), five restarts each for the QSVM, the MLP and LightGBM. The
    library draws its starting points unseeded, so a restart is a sample, not a repeatable
    draw; one QSVM restart at the highest budget takes about 110 minutes, so run restarts as
    separate processes (`--restarts <index>`). Name the runs in
    `experiments/hsj_query_sweep.yaml`; the reader checks them and applies the registered
    rules. It also reads the sweep by the model's clean label and, row by row, against the
    random-noise run the same file names (`run_adversarial --only random`, the run
    `make_report` pins as `random`, on the same 100 rows and fitted models); that
    reading was made after the result and is stored apart from the registered one. Pin the
    reader's run in `experiments/make_report.py` (`HSJ_SWEEP_VERDICT_RUN`): the paper's
    sweep table reads it.
```bash
python -m experiments.run_hopskipjump_restarts 5 qsvm --max-iter 10 --init-eval 20 --max-eval 200 --run-tag sweep_low
```
```bash
python -m experiments.run_hopskipjump_restarts 5 mlp lgbm --max-iter 10 --init-eval 20 --max-eval 200 --run-tag sweep_low
```
```bash
python -m experiments.run_hopskipjump_restarts 5 qsvm --restarts 0 --max-iter 20 --init-eval 40 --max-eval 400 --run-tag sweep_mid
```
```bash
python -m experiments.run_hopskipjump_restarts 5 mlp lgbm --max-iter 20 --init-eval 40 --max-eval 400 --run-tag sweep_mid
```
```bash
python -m experiments.run_hopskipjump_restarts 5 qsvm --restarts 0 --max-iter 40 --init-eval 60 --max-eval 400 --run-tag sweep_high
```
```bash
python -m experiments.run_hopskipjump_restarts 5 mlp lgbm --max-iter 40 --init-eval 60 --max-eval 400 --run-tag sweep_high
```
    Repeat the two single-restart commands for restarts 1 to 4. Then:
```bash
python -m experiments.read_hsj_query_sweep
```

## Reporting artifacts

15. Generate every table, figure, and statistic from `results/runs/` — the only writer of
    the generated exhibits; never hand-edit them. It needs the pooled and worst-radius runs
    of step 14g, the replication runs of step 14h, the training-draw runs of step 14i and
    the sweep reader's run of step 14j, and stops with the name of a run it cannot find.
```bash
python -m experiments.make_report
```

## Optional — quantum hardware (gated)

18. Estimate cost first; a paid Braket submission is blocked unless `QGB_APPROVED_BUDGET_USD`
    ≥ estimate is set by the operator (see `utils/cost_guard.py`). Never submit without approval.
