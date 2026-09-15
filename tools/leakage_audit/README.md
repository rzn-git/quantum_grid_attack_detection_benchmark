# leakage_audit — is your row-level split measuring generalization or memorization?

Many public tabular benchmarks ship as a small number of source files (captures, sessions,
scenarios) whose rows are near-duplicates within a file. A row-level train/test split then
puts near-copies of test rows into training, and the reported score measures memorization of
those near-duplicates. On the MSU/ORNL power system attack dataset this is worth **0.31
macro-F1**: tuned LightGBM scores 0.905 under a row-level split and 0.594 when whole source
files are held out (QuantumGridBench, arXiv:2608.15617). The audit applies to every published
result on that benchmark, and to any other dataset with group structure.

This tool packages the paper's control so it can be pointed at your own dataset and model
without reading the paper. It depends only on numpy, pandas and scikit-learn.

## The two functions

```python
from tools.leakage_audit import neighbour_audit, paired_protocol
```

**1. Neighbour audit** — the mechanism number. What fraction of test rows have a train
neighbour closer than a threshold (default 0.1) in standardized feature space (median-impute +
z-score, fit on train only)?

```python
report = neighbour_audit(X_train, X_test, threshold=0.1)
# {"frac_within_threshold": 0.458, "nn_distance_median": ..., ...}
```

Reading it: under an honest split of independent rows this fraction is small; 45.8% (row-level)
versus 5.0% (whole-file holdout) is what the 0.31 gap above looks like at the feature level.

**2. Paired-protocol runner** — the score number. Fit any scikit-learn-compatible estimator
under both protocols with identical train-fitted transforms and report both scores plus the
delta:

```python
from lightgbm import LGBMClassifier

result = paired_protocol(
    LGBMClassifier(random_state=0),
    X,
    y,
    groups,  # one group id per row: source file, session, patient...
    n_partitions=5,  # whole-group partitions (mean ± std reported)
    seed=0,
)
# result["row_level"]["score"], result["whole_group"]["score_mean"],
# result["delta_row_minus_group"], plus the neighbour audit under both splits
```

Protocol details (deliberate, and asserted in code): every fitted transform is fit on the
train fold only; group splits hold out whole groups and their disjointness is asserted; a
validation fold (default 20%) is excluded from training in both arms, mirroring a tuning
holdout (`val_size=0` disables it); the metric defaults to macro-F1 (`metric=` accepts any
`f(y_true, y_pred)`).

## CLI

From the repository root:

```
python -m tools.leakage_audit neighbours --train train.csv --test test.csv --threshold 0.1
```

```
python -m tools.leakage_audit paired --data data.csv --label-col label --group-col file_id \
    --model lightgbm.LGBMClassifier --model-params '{"random_state": 0}' --partitions 5
```

Both print a JSON report to stdout. CSV feature input uses the numeric columns; `.npy`/`.npz`
matrices are accepted for `neighbours`.

## Verify the tool first: the worked example

Reproduces the published 0.905 / 0.594 result end-to-end through this tool (requires the
project's processed data and the tuned LightGBM study; run the pipeline's preprocessing and
classical tuning stages first if missing):

```
python -m tools.leakage_audit.examples.reproduce_msu_ornl
```

It checks the row-level score, all five whole-file partition scores and their held-out file
ids, and both neighbour-audit fractions against the archived full-precision artifact values,
and fails loud on any drift. Expected closing lines:

```
row-level 0.9053  vs  whole-file 0.5945 ± 0.0088  ->  delta 0.3109
PASS: the tool reproduces the published row-level and whole-file results.
```

## Tests

```
pytest tools/leakage_audit/tests
```

(Synthetic-data tests with known answers; not part of the project's default `pytest` run.)

## Caveats

* A large delta means the row-level number is not a deployment estimate; it does not by itself
  say which protocol matches your threat model — that is a modelling question.
* The neighbour audit is scale-sensitive by design: it standardizes on the train fold. If your
  features are already standardized, pass `standardize=False`.
* With few groups (this dataset has 15 files) the whole-group score has high variance; report
  the per-partition spread, never a single draw.
