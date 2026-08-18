# QuantumGridBench

Hybrid quantum-classical machine learning benchmark for **cyberattack detection on public
power-system data** (MSU/ORNL Power System Attack Dataset). It measures **accuracy** and
**adversarial robustness / attack transferability** across classical and quantum model
families under one comparability contract — and its headline finding is methodological:
**evaluation choices (the data split, the attack protocol, the tuning) decide the
benchmark's outcome before the models do.** Each choice ships with the control that
catches it.

This is an **honest benchmark**, not advocacy: a null or negative quantum result is a valid
outcome and is reported as-is. No quantum-advantage claim is made unless the data shows one.

**Author:** Md Rezwanul Islam ([ORCID 0009-0002-4100-3796](https://orcid.org/0009-0002-4100-3796)),
rznislam@gmail.com

## Paper

This repository is the reproducibility package for:

> Md Rezwanul Islam, *Benchmarking Quantum Machine Learning for Power-System Attack
> Detection: Evaluation Choices Decide the Outcome Before the Models Do*, 2026.
> [arXiv:2608.15617](https://arxiv.org/abs/2608.15617) \[cs.LG\],
> doi:[10.48550/arXiv.2608.15617](https://doi.org/10.48550/arXiv.2608.15617).

Every number in the paper traces to a versioned run directory produced by this code
(resolved config, git SHA, seeds, wall time recorded per run).

## What it produces

- Classical baselines (LogReg, RBF-SVM, RF, XGBoost, LightGBM, MLP) and quantum models
  (fidelity-kernel QSVM, variational quantum classifier) evaluated on identical splits over
  10 seeds, with AUPRC / macro-F1 / balanced-accuracy / ROC-AUC / Brier / ECE.
- A paired split-protocol control: the pooled row-level split against whole-source-file
  held-out partitions, with a nearest-neighbour leakage audit.
- A source×target adversarial **transferability matrix**, evasion degradation curves
  (FGSM/PGD), a white-box attack on the fidelity kernel (numerically gated against the
  deployed path), a query-based black-box attack (HopSkipJump) with a random-perturbation
  control, benign sensor-noise robustness, and label-noise poisoning — all crafted in a
  shared standardized feature space.
- Quantum-kernel diagnostics (kernel-target alignment, geometric difference, spectral
  shape) and VQC trainability (gradient-variance / barren-plateau) logging.
- Auto-generated LaTeX tables and PDF figures — no number is ever hand-typed into the paper.

## Quickstart

```bash
uv venv --python 3.11 && uv pip install -r requirements.txt -e .
python -m qgridbench.data.download          # fetch + verify (SHA-256) + extract the dataset
python -m qgridbench.data.preprocess binary triple
python -m experiments.smoke                 # end-to-end miniature sanity run
```

The smoke pass exercises every stage in miniature (tiny subset, 2 seeds, 4 qubits, one ε)
through to a compiled dummy PDF — it is the integration check, not a result. To reproduce
the paper's numbers, run the full sweep in [RUN.md](RUN.md), then:

```bash
python -m experiments.make_report           # writes every table and figure from results/runs/
```

`make_report` reads exclusively from `results/runs/`, so no reported number can exist
without an artifact behind it. Expect the full sweep to take roughly a day on a 24-core
workstation — the quantum-kernel and black-box stages dominate, and both checkpoint so
they can be resumed.

## License & data terms

Code is MIT ([LICENSE](LICENSE)). The MSU/ORNL dataset is **not** redistributed here and is
governed by its authors' terms — cite Pan, Morris & Adhikari (IEEE Trans. Smart Grid) as
requested on the dataset page. Dataset provenance (source, retrieval date, SHA-256 hashes)
is recorded in `configs/data.yaml`.
