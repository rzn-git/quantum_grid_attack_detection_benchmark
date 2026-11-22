# MODEL_CARD — QuantumGridBench

Current measured state of the benchmarked systems. These numbers are the ones reported in
Md Rezwanul Islam, *Benchmarking Quantum Machine Learning for Power-System Attack Detection:
Evaluation Choices Decide the Outcome Before the Models Do*,
[arXiv:2608.15617](https://arxiv.org/abs/2608.15617) (2026).

**Every number below is generated, not typed.** The tables live in marker-fenced blocks
rewritten by `qgridbench.eval.model_card` from `paper/tables/*.csv`, which are themselves
written only from `results/runs/`. Editing a figure by hand will be overwritten on the next
`python -m experiments.make_report`. Prose outside the markers is authored and preserved.

## Intended use

Research benchmark comparing classical vs hybrid-quantum detectors for power-system cyberattack
detection on the public MSU/ORNL dataset, on accuracy **and** adversarial robustness/transfer.

## Out-of-scope use

- Not a production grid IDS; no deployment claims.
- Not a quantum-hardware-advantage claim (results are classically simulated statevectors).
- The PCA-regime comparison is *matched-dimensionality*, not "replace full-feature XGBoost."
- Sensor-noise robustness is a PMU-realism proxy, **not** a DER/renewables scenario.

## Headline accuracy (binary, test split, mean over 10 seeds)

**Every figure in this section is measured under the row-level split protocol, and that
protocol is itself the study's largest measured evaluator effect** (P12a + the shipped
control, decision log 2026-08-15T22:06Z / 2026-08-16T02:44Z, artifact
`results/runs/20260816T023620Z_group_split_control_binary/`). Rows within a source file
are ordered samples of one session and therefore near-duplicates of their temporal
neighbours (45.8% of test rows have a train neighbour within 0.1 under this protocol, and
for 99% of them that neighbour is the adjacent row in file order; 5.0 ± 0.4% with whole
files held out — decision log 2026-09-01T23:09Z, artifact
`results/runs/20260901T230730Z_file_row_order_binary/`). Holding whole files out drops
tuned LightGBM from 0.905 to 0.594 ± 0.008 macro-F1, and a temporal split that keeps every
file and scenario on both sides lands at the same place (0.6005,
`results/runs/20260902T022541Z_temporal_split_binary/`). The leak has a measured length:
macro-F1 falls from 0.876 for test rows adjacent to a training row to its floor within nine
rows, decay length 2.5 released rows for LightGBM and 2.3 for XGBoost (N-DECAY-2b, decision
log 2026-09-15T19:14Z, `results/runs/20260915T190906Z_leak_decay_curve_binary/`). At matched dimensionality under the 2,000
cap, tuned XGBoost falls to 0.523 ± 0.013 and the tuned ZZ QSVM to 0.508 ± 0.012 against
a 0.499 floor, with their contrast at +0.015 ± 0.015 and sign-unstable across partitions.
Within-protocol comparisons in the tables below stand; absolute levels are
protocol-dependent.

Best classical model per regime, plus every quantum model at matched dimensionality.

<!-- BEGIN GENERATED: headline -->
| Model | regime | macro-F1 | AUPRC | ROC-AUC | Brier | ECE |
|---|---|---|---|---|---|---|
| **lgbm** (best) | full (128) | **0.9056** | 0.9856 | 0.9696 | 0.1257 | 0.0510 |
| **xgb** (best) | PCA-8 | **0.5968** | 0.7843 | 0.6303 | 0.4383 | 0.0685 |
| qsvm_angle_ry | PCA-8 | 0.5265 | 0.7509 | 0.5651 | 0.5191 | 0.1117 |
| qsvm_zz | PCA-8 | 0.5644 | 0.7518 | 0.5837 | 0.5145 | 0.1173 |
| vqc | PCA-8 | 0.5141 | 0.7400 | 0.5362 | 0.4838 | 0.0592 |
| _baseline_stratified_random_ | PCA-8 | 0.5005 | 0.7103 | 0.5005 | 0.4116 | 0.0000 |
| **xgb** (best) | PCA-12 | **0.6063** | 0.7944 | 0.6483 | 0.4378 | 0.1126 |
| qsvm_angle_ry | PCA-12 | 0.5405 | 0.7474 | 0.5618 | 0.4918 | 0.0544 |
| qsvm_zz | PCA-12 | 0.5640 | 0.7545 | 0.5859 | 0.4733 | 0.0861 |
| vqc | PCA-12 | 0.5096 | 0.7267 | 0.5220 | 0.4888 | 0.0617 |
| _baseline_stratified_random_ | PCA-12 | 0.5005 | 0.7103 | 0.5005 | 0.4116 | 0.0000 |
| **xgb** (best) | PCA-16 | **0.6071** | 0.8005 | 0.6538 | 0.4323 | 0.1103 |
| qsvm_angle_ry | PCA-16 | 0.5742 | 0.7556 | 0.5893 | 0.4991 | 0.0945 |
| qsvm_zz | PCA-16 | 0.5670 | 0.7512 | 0.5855 | 0.4635 | 0.0745 |
| _baseline_stratified_random_ | PCA-16 | 0.5005 | 0.7103 | 0.5005 | 0.4116 | 0.0000 |
<!-- END GENERATED: headline -->

**The PCA rows are depressed by two handicaps, and the larger one is not PCA.** Measured, on
identical features and hyperparameters (XGBoost and LightGBM at their matched-contest settings,
`results/runs/20260813T202645Z_ablations_binary_q8/kernel_cap_cost.json`; the learning curve
and the uncapped 0.787 below are XGBoost at those settings, so they are not the tuned
full-feature LightGBM, 0.906, that the paper's accuracy-bridge figure starts from):

| handicap | cost in macro-F1 |
|---|---|
| **2,000-sample kernel cap** (O(N²) fidelity-kernel ceiling) | **+0.17 to +0.20** |
| PCA projection, 8 → 16 dims | +0.02 |
| best quantum vs best classical *at* the cap | −0.032 to −0.042 |

The cap costs about **six times the quantum–classical gap** (0.197 against 0.033 at 16 dimensions, the paper's accuracy-bridge figure), and the learning curve is still
climbing at the full split (0.5947 at n=2,000 → 0.7631 at n=47,020), so it sits on the steep
part, not a plateau. Uncapped, the same PCA-16 feature space reaches 0.787 against 0.602
under the cap, so **dimensionality is a mild constraint on this data; sample size is
the damaging one** (in the paper's accuracy-bridge figure PCA to 16 components costs 0.063
and the cap 0.197).

**The comparison remains fair** — every family, classical and quantum, trains on the identical
2,000-sample stratified subset, so the ordering in the table stands. What the cap changes is
the reading of the absolute values: ~0.60 is not what this task supports, and the correct
conclusion is that quantum kernels are competitive only inside the regime their own sample
complexity forces. These rows answer "is the quantum feature map competitive at matched
dimensionality," never "should a utility replace full-feature LightGBM."

## Robustness under attack (binary, PCA-8)

**The QSVM's high retention is an artifact of attack availability, now measured rather than
suspected** (P5a, decision log 2026-08-14T07:55Z). Under a targeted white-box attack —
differentiating the fidelity kernel through the simulator — the QSVM falls from 0.917
retention under the transferred attack in the same run (0.886 under the ten-seed shared
protocol) to **0.023**, read at its worst grid radius inside the 0.5σ budget (0.3σ; 0.064 at
0.5σ itself), with the collapse already underway at ε = 0.01σ. The matched control: the
identical attack on the tuned RBF-SVM leaves **0.243** at its worst radius (0.1σ; 0.685 at
0.5σ, where its output has gone constant). An attacker bounded by a radius may use any
smaller perturbation, so a white-box result is read at the worst grid radius inside the
budget, not at the budget alone (decision log 2026-11-20T10:17Z;
`experiments/read_worst_radius.py`). The quantum kernel is the lower of the two at every ε
(paired over 10 attack starts, each model at its worst radius: p = 0.002, rank-biserial
−1.0). Both attacked kernels land far below the constant-predictor floor of 0.415 (macro-F1
0.013 and 0.143), so the attacker inverts these detectors rather than merely blinding them.
The worst-radius reading is still an upper bound on what the attacker can do: one who picks
the radius per input does better, and a score that rises with the radius means the 20-step
attack has not converged at the larger radii.

**The interleaving mechanism is now measured, not inferred (P17e, 2026-08-15T21:16Z).** Sampling
the decision function at 200 points along a fixed 0.5σ perturbation ray: at the deployed
bandwidth 26% of rays cross more than one decision boundary, against 17% for the tuned RBF-SVM
at its own accuracy optimum — real but modest, ~1.7×. The families diverge far more at wide
length scale, where the quantum count rises monotonically to 3.84 crossings per ray (one ray
crosses 19 times) while the classical count is non-monotone and falls to 0.03. Quote the
operating-point numbers, not the wide-bandwidth ones. **Still unexplained:** the accuracy/
fragility co-peak — the parity test pre-registered to explain it was vacuous by construction.

**Retention and the random-perturbation flip rate must be read together.** Retention alone
makes the fidelity-kernel QSVM look like the most robust model here; the flip rate shows
random noise breaks it *more* easily than the transferred attack does. That is
attack-direction misalignment, not robustness — and the white-box column above is the
direct confirmation.

**The fragility cannot be detuned away — there is no safe operating point on the bandwidth
axis** (`fragility` ablation stage). Rebuilding the QSVM across bandwidths, the flip rate
*co-peaks with accuracy*: 42% at the accuracy-optimal bandwidth (≈ the tuned value), falling
to 19–29% only at degenerate extremes where the model has collapsed to chance (macro-F1
0.499) and drifted off the true positive rate. Within this model family, whatever accuracy
exists is bought with fragility — unlike the MLP, which holds a 3% flip rate at a comparably
weak F1.

The evasion column is deliberately **not** apples-to-apples: differentiable models (mlp, vqc)
are attacked white-box, every other family by transfer from the MLP surrogate, which is
strictly weaker. The black-box table below and the white-box QSVM/RBF columns are the two
controls for that asymmetry. **The tree families still admit no gradient-based white-box attack, and no tree-specific one was run, so their
retention figures remain upper bounds — read them as attack availability, not robustness.**

<!-- BEGIN GENERATED: robustness -->
At ε = 0.5σ, 10 seeds. Retention = adversarial / clean macro-F1.

| Model | clean | under attack | retention | white-box retention | random-perturbation flip rate |
|---|---|---|---|---|---|
| qsvm | 0.563 | 0.499 | 0.886 | 0.023 at 0.3σ | 45% |
| logreg | 0.514 | 0.431 | 0.838 | 0.214 | 8% |
| rbf_svm | 0.587 | 0.415 | 0.707 | 0.243 at 0.1σ | 24% |
| mlp | 0.465 | 0.289 | 0.621 | 0.621 | 3% |
| rf | 0.616 | 0.367 | 0.596 | — | 18% |
| lgbm | 0.634 | 0.373 | 0.589 | — | 24% |
| xgb | 0.620 | 0.340 | 0.548 | — | 27% |
| mlp (class-weighted) | 0.521 | 0.259 | 0.498 | 0.076 at 0.5σ only | — |
| vqc | 0.476 | 0.029 | 0.060 | 0.060 | 25% |

Quantum-vs-classical paired comparisons significant after Holm correction: **12 of 12** — in *both* directions, which is why no family-level quantum robustness claim is made.

The **white-box retention** column is the model's own white-box attack, read at the grid radius inside the budget that leaves the lowest score: an attacker bounded by a radius may use any smaller perturbation. Where that radius is not the budget it is printed beside the figure. The class-weighted MLP's own attack ran at the budget only, so its figure is an upper bound on its worst case. A `—` there means the family received no white-box attack (the tree ensembles admit no gradient-based one, and no tree-specific attack was run), so its retention figure is an upper bound, not a robustness measurement. Logistic regression's figure is from its own run of ten attack starts. A `—` in the flip-rate column means it was not measured.
<!-- END GENERATED: robustness -->

## Black-box robustness (HopSkipJump, identical query budget per family)

Headline column is the median L∞ distortion **in units of train σ** needed to flip a
prediction — larger means the attacker must work harder. Unlike F1-under-attack it does not
reward a model for already being wrong; read the two together.

<!-- BEGIN GENERATED: blackbox -->
| Model | clean F1 | F1 under attack | median $\ell_\infty$ (σ), one draw | queries/sample | **median $\ell_\infty$ (σ) over restarts (n restarts)** |
|---|---|---|---|---|---|
| qsvm | 0.556 | 0.379 | 2.818 | 624 | 3.413 ± 1.386 (n = 10) |
| mlp | 0.463 | 0.272 | 0.867 | 597 | 0.877 ± 0.015 (n = 10) |
| lgbm | 0.586 | 0.329 | 0.548 | 610 | 0.859 ± 0.120 (n = 10) |
| logreg | 0.590 | 0.356 | 0.517 | 581 | 0.520 ± 0.009 (n = 5) |
| vqc | 0.466 | 0.440 | 0.452 | 590 | 0.481 ± 0.067 (n = 5) |
| xgb | 0.627 | 0.320 | 0.200 | 605 | 0.212 ± 0.048 (n = 10) |
| rbf_svm | 0.576 | 0.415 | 0.000 † | 216 | 0.013 ± 0.001 on 24% of points (n = 5) |
| rf | 0.573 | 0.453 | 0.000 † | 248 | 0.062 ± 0.042 on 28% of points (n = 5) |

† Not a distortion: on most of the 100 points the attack found no starting point and returned the input unchanged. See the note below the table.
<!-- END GENERATED: blackbox -->

A median distortion of exactly 0.000 is **not a distortion and not robustness**. For rbf_svm
and rf, on about three quarters of the 100 points none of the attack's 100 random starting
draws left the class the model already predicted, and the library returns such an input
unchanged (which is why these two rows show about 216 and 248 queries against about 600).
An earlier version of this card read the zero as adversarial points found at the unperturbed
inputs; that reading was wrong (decision log 2026-11-20T02:03Z and 02:54Z).

> **The "one draw" column is ONE unreplicated draw per model (found 2026-08-15, P22b).** ART's
> HopSkipJump seeds its initial adversarial point from an unseeded `np.random.RandomState()`
> (`hop_skip_jump.py:295`, ART 1.20.1), which our `np.random.seed(...)` cannot reach — so the
> attack is not seed-reproducible. Measured: three same-seed MLP runs give medians
> 0.661 / 0.636 / 0.703. **Read the restart column**: it is the mean ± std the paper reports,
> generated from the paper's robustness table, with the number of restarts beside each figure.
> For rbf_svm and rf it is read over the points the attack could start on,
> with their share beside it; the all-points median is zero in every restart. The qsvm figure
> moves by sigmas between restarts, because its per-point distortions fall into two groups and
> the median sits between them. Two single draws are known not to match their restarts: the
> lgbm draw lies outside its own restart range, and the vqc draw is an earlier VQC configuration
> than the regenerated one in the restart column. The section's conclusion does not depend on the
> values — it rests on the random-perturbation control and the white-box result.

## Compute cost (PCA-8, the matched-dimensionality regime)

Accuracy alone does not decide deployability: grid detection is real-time. Quantum rows are
**simulator** timings — they say what the model class costs to evaluate, not what a device
would deliver.

<!-- BEGIN GENERATED: cost -->
| Model | train (s) | inference (µs/sample) | vs fastest |
|---|---|---|---|
| logreg | 0.01 | 0.07 | 1× |
| mlp | 1.85 | 2.37 | 33× |
| xgb | 0.78 | 5.40 | 76× |
| lgbm | 3.76 | 12.18 | 171× |
| rf | 2.46 | 26.03 | 365× |
| vqc ‡ | 92.81 | 136.72 | 1918× |
| rbf_svm | 0.40 | 242.73 | 3405× |
| qsvm_angle_ry ‡ | 4.81 | 2233.24 | 31327× |
| qsvm_zz ‡ | 10.82 | 5352.39 | 75080× |

‡ simulated quantum model — statevector-simulator latency, not a device measurement and not a deployment claim.
<!-- END GENERATED: cost -->

## Quantum kernel diagnostics

Per Huang et al., "Power of data in quantum machine learning": a large geometric difference
*g* is a **necessary but not sufficient** condition for quantum advantage. This benchmark is a
case where *g* is large and advantage is nonetheless absent, because kernel-target alignment
(KTA) is low — the feature map is geometrically exotic but not label-relevant.

<!-- BEGIN GENERATED: diagnostics -->
| Encoding | qubits | KTA (quantum) | KTA (RBF) | geometric difference |
|---|---|---|---|---|
| angle_ry | 8 | 0.0023 | 0.0125 | 13.43 |
| zz | 8 | 0.0035 | 0.0125 | 6.34 |
| angle_ry | 12 | 0.0022 | 0.0187 | 17.21 |
| zz | 12 | 0.0145 | 0.0052 | 73.04 |
| angle_ry | 16 | 0.0031 | 0.0058 | 5.60 |
| zz | 16 | 0.0143 | 0.0058 | 70.12 |
<!-- END GENERATED: diagnostics -->

**The pipeline can detect a quantum advantage when one exists — measured, not assumed.** On
labels engineered from each Gram's own g eigenvector (Huang et al. construction; the
`relabel` ablation stage), the quantum kernel beats *freshly re-tuned* RBF-SVM and XGBoost
in **all 8 cells** — win +0.004 to +0.092 macro-F1, mean ± std over 5 train/test resamples,
7 of 8 clearing zero at mean − 2σ (`angle_ry` PCA-12 is marginal at +0.004 ± 0.003 and is
reported as such). **The win tracks the geometric difference**: Pearson r(log g, win) =
+0.85 (p = 0.007); mean win +0.076 for g > 50 versus +0.016 for g ≤ 50. The RQ1 null is
therefore a property of the grid attack labels — KTA ≤ 0.015, versus 0.046–0.50 for the
engineered labels on the identical Grams — not a pipeline artifact.

**On the real labels, KTA predicts accuracy and g does not — replicated on both label
distributions.** Sweeping encoding bandwidth across 54 configurations (the `bandwidth`
ablation stage, binary + triple variants), Spearman ρ(KTA, val macro-F1) is positive in all
6 cells (+0.65 to +0.97), while ρ(g, val macro-F1) **changes sign between cells of the same
study** (−0.82 to +0.63). Report KTA as the operative diagnostic; g is
necessary-but-not-sufficient *and* confounded.

**At the prediction level, the tuned quantum kernels are "just another model family" — not
classical clones, and not detectably special** (`dequant` ablation stage). Agreement between
each tuned quantum SVM and a best-effort classical RBF (Cohen's κ 0.18–0.64) straddles the
classical-vs-classical reference pair on the identical split (RBF vs XGBoost, κ 0.31–0.38),
with F1 differences inside noise. The two near-identity cells (zz PCA-12/16) fall *below*
the reference — degeneracy decouples their predictions from every classical family without
adding accuracy: different, but not better. Scope: this is prediction-level agreement on a
near-chance task; it neither confirms nor refutes kernel-level dequantization results.

**What confounds g is spectral degeneracy at both ends of the bandwidth axis** (pooled
ρ(degeneracy, g) = +0.80). At small bandwidth the Gram is near-*constant* (median
off-diagonal fidelity 0.95–0.99; top eigenvalue 96–99% of the spectrum); at large bandwidth
it is near-*identity* (up to 98% of pairs below 0.01 fidelity) — which is exactly the regime
the tuned zz PCA-12/16 kernels occupy, and the source of their headline g = 73/70. Both
extremes sit far from any classical kernel for reasons unrelated to useful structure; where
the spectrum is healthiest, g falls to its *minimum* (4.8–6.2). Exponential concentration
(Thanasilp et al.) is **not** the mechanism: fixed-bandwidth off-diagonal variance falls only
~0.05–0.09 log₂ per qubit (2⁻ⁿ would be −1.0), and shot noise is 2–6% of the informative
kernel spread. The same near-orthogonality (zz PCA-8: 41% of pairs < 0.01) is the geometry
behind the fragmented decision surface and the 45% random-flip rate reported above.

**No bandwidth rescues the result, including outside the tuning box.** Best val macro-F1
anywhere on the axis is 0.5886 (zz PCA-8), and a bandwidth of 3.0 — deliberately beyond the
[0.05, 2.0] Optuna range — loses to the best in-box point in every cell. The tuned optima
landing near the box edge (≈1.92–1.94) was not a clipped search. **This feature map has no
regime on this data that is simultaneously non-degenerate and label-aligned.**

## Known failure modes / limitations

- **The robustness sections describe models trained under the row-level protocol** —
  detectors that partly memorize near-duplicate rows (see the split-protocol note above).
  The comparisons are within-split at fixed models and stand arithmetically; the operating
  regime they describe is the published protocol's, not a leakage-free one. **Exception,
  measured (P24, 2026-08-16; at six radii, 2026-11-20):** the white-box kernel contrast
  replicates on whole-file-trained kernels, each read at its worst radius inside 0.5σ — QSVM
  0.47–0.49 below the stratified-random floor vs RBF 0.35–0.36 below, the QSVM lower in 20/20
  attack-seed pairs (`results/runs/20261120T112410Z_group_whitebox_control_binary_q8_radii_gate_p0/`
  and `..._radii_gate_p2/`, reproducible via `experiments/run_group_whitebox_control.py`). Read at
  0.5σ alone, as the first run did, the RBF reads only ≈0.10 below the floor and the gap
  looks about three times larger.
- **No collapse of the gradients with depth at 8 qubits** (gradient-norm variance flat across
  depth 2/4/6), so the VQC's weak accuracy is model-class fit, not trainability collapse. The
  plateau is on the qubit axis: the paper's appendix measures the gradient variance falling with
  the qubit count.
- **No quantum data-efficiency advantage:** full-feature XGBoost leads the VQC by 0.125 macro-F1
  at 250 training samples and 0.118 at 2,000, and the RY-angle QSVM by 0.116 and 0.121 (not a
  matched comparison).
- **VQC at 16 qubits is unrun** (293 s/epoch projects past 24 h); reducing its budget would
  break the equal-budget fairness contract. The quantum kernel covers 16 qubits.
- Quantum-kernel seed spread is exactly 0.0000 — deterministic given a fixed subset — so it is
  compared by margin against the classical seed distribution, not by a paired p-value.
- **The MLP row is mis-specified, and it is the transfer surrogate (found 2026-08-15, P19d;
  cause corrected by P19f).** The MLP is the only family here fitted without the study's
  imbalance policy — not a library limitation but a project defect: the zoo expresses the policy
  as a `class_weight=` **constructor** argument, and `MLPClassifier` is the one estimator that
  does not take one, so it is skipped silently. (`MLPClassifier.fit` *does* accept
  `sample_weight` on scikit-learn 1.9.0; the remedy is one argument.) Consequences, measured: it
  predicts the majority class 96.6% of the time against a true 71% positive rate, its clean
  PCA-8 macro-F1 (0.465) sits **below** the 0.5005 stratified-random floor, and no configuration
  in its 50-trial search cleared that floor (best val 0.4690). Fitted with balanced sample
  weights it reaches 0.532; an identical-architecture torch MLP with class-weighted loss reaches
  0.552. Every MLP figure in this card, and every transfer figure using it as the surrogate,
  is a *lower* bound on that family and carries a surrogate artefact whose mechanism is open
  (the class-prior account was proposed and killed — P23). Decision log 2026-08-15T10:15Z /
  11:20Z / 12:20Z; correction is a Checkpoint-5 decision.
  **Scope of the artefact, settled 2026-08-15T19:52Z (P11a):** it is not specific to this
  defect or to neural-net surrogates. Across three surrogates ordered by how far each clears
  the stratified-random floor (MLP as tuned −0.031, MLP balanced +0.035, RBF-SVM +0.090), the
  forward/reverse asymmetry falls monotonically — 10.16, 0.78, 0.15 — and still does after
  normalizing for attack strength (1.65 → 0.56 → 0.20). Read transfer numbers here as a
  function of surrogate competence, and note that every corrected arm sits *below* unity,
  which is the expected within-family transfer effect and **not** a quantum robustness result.
- Feature-space ε-balls are not guaranteed physically realizable grid measurements. **Quantified
  2026-08-15 (P20):** at ε = 0.5σ the attack displaces 109 of 128 raw channels by more than
  0.1σ each, so it presumes far broader sensor control than a few-meter FDIA adversary; but
  rounding all 15 integer-valued log/status channels to attainable values costs the attacker at
  most 0.016 macro-F1, so discreteness is not a defence.
- **No model is deployable at a realistic operating point (2026-08-15, P19b).** At a threshold
  fitted on validation for a 1% false-positive budget, PCA-8 detection rates run 0.3%–4.1%
  (5% budget: 5.4%–12.8%). Under attack at that fixed threshold the realised false-alarm rate
  rises to 0.465 (MLP) and 0.603 (VQC) — the failure is a false-alarm flood, not silence.
- The ceiling contingency (study protocol §8.6) does **not** trigger: best binary macro-F1 is 0.906,
  well below the ~0.99 saturation risk.

## Provenance

Every figure resolves to a `results/runs/<UTC>_<name>/` dir carrying resolved config, git SHA
(+dirty flag), seed list, wall time, and dependency lock hash. Eval-set version =
`data/processed/splits_<variant>.npz` (split seed 1337).

<!-- BEGIN GENERATED: provenance -->
- Generated: **2026-11-22T17:02:33Z**
- Code version: `2e62932`
- Regenerate with `python -m experiments.make_report` — every figure above is read
  from `paper/tables/*.csv`, which is itself written only from `results/runs/`.
<!-- END GENERATED: provenance -->
