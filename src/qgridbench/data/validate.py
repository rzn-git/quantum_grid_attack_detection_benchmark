"""Schema + integrity validation of the raw dataset. Fails loud on violations.

Produces a machine-readable data report (row counts, class distribution,
inf/NaN incidence, per-file consistency) consumed by checkpoint 1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from qgridbench.utils.run_tracking import REPO_ROOT, get_logger, load_yaml

log = get_logger(__name__)


_INF_TOKENS = {
    "infinity": np.inf,
    "inf": np.inf,
    "+infinity": np.inf,
    "-infinity": -np.inf,
    "-inf": -np.inf,
}


def _coerce_numeric(s: pd.Series) -> pd.Series:
    """Coerce a feature column to float, mapping Infinity tokens to +/-inf.

    ARFF nominal-typed numeric columns arrive as strings; the CSV reader already
    maps 'Infinity' to inf. Normalizing here keeps both paths identical. A token
    that is neither numeric nor an infinity literal fails loud (errors='raise').
    """
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    lowered = s.astype(str).str.strip().str.lower()
    mapped = lowered.map(lambda v: _INF_TOKENS.get(v, v))
    return pd.to_numeric(mapped, errors="raise").astype(float)


def _read_arff(path) -> pd.DataFrame:
    from scipy.io import arff

    data, _meta = arff.loadarff(path)
    df = pd.DataFrame(data)
    for c in df.columns:  # nominal attributes come back as bytes
        if df[c].dtype == object:
            df[c] = df[c].str.decode("utf-8")
    return df


def load_variant_frames(variant: str) -> list[pd.DataFrame]:
    """Load the ~15 file groups of one variant (CSV or ARFF), in sorted name order.

    Feature columns are coerced to float at load (Infinity tokens -> +/-inf); the
    label column is left as-is.
    """
    cfg = load_yaml(REPO_ROOT / "configs" / "data.yaml")
    label_col = cfg["schema"]["label_column"]
    vdir = REPO_ROOT / cfg["paths"]["raw"] / variant
    files = sorted([*vdir.rglob("*.csv"), *vdir.rglob("*.arff")])
    if not files:
        raise FileNotFoundError(f"no data files under {vdir} — run qgridbench.data.download first")
    frames = []
    for f in files:
        df = _read_arff(f) if f.suffix == ".arff" else pd.read_csv(f)
        for c in df.columns:
            if c != label_col:
                df[c] = _coerce_numeric(df[c])
        df.attrs["source_file"] = f.name
        frames.append(df)
    return frames


def validate_variant(variant: str) -> dict:
    cfg = load_yaml(REPO_ROOT / "configs" / "data.yaml")
    label_col = cfg["schema"]["label_column"]
    n_expected = cfg["schema"]["n_feature_columns"]

    frames = load_variant_frames(variant)
    ref_cols = list(frames[0].columns)
    report: dict = {"variant": variant, "n_files": len(frames), "files": []}

    for df in frames:
        if list(df.columns) != ref_cols:
            raise ValueError(
                f"{variant}/{df.attrs['source_file']}: column set differs from first file"
            )
        feat = df.drop(columns=[label_col])
        numeric = feat.select_dtypes(include=[np.number])
        report["files"].append(
            {
                "file": df.attrs["source_file"],
                "rows": int(len(df)),
                "n_inf": int(np.isinf(numeric.to_numpy(dtype=float, na_value=np.nan)).sum()),
                "n_nan": int(feat.isna().sum().sum()),
            }
        )

    if label_col not in ref_cols:
        raise ValueError(f"{variant}: label column '{label_col}' missing")
    n_features = len(ref_cols) - 1
    if n_features != n_expected:
        log.warning(
            "%s: %d feature columns (expected ~%d) — recording actual count",
            variant,
            n_features,
            n_expected,
        )

    pooled = pd.concat(frames, ignore_index=True)
    class_counts = pooled[label_col].value_counts().sort_index()
    non_numeric = [
        c for c in pooled.columns if c != label_col and not pd.api.types.is_numeric_dtype(pooled[c])
    ]
    report.update(
        {
            "n_feature_columns": n_features,
            "total_rows": int(len(pooled)),
            "duplicate_feature_rows": int(pooled.duplicated().sum()),
            "class_distribution": {str(k): int(v) for k, v in class_counts.items()},
            "non_numeric_feature_columns": non_numeric,
            "feature_stats": {
                "n_cells_inf": int(
                    np.isinf(
                        pooled.drop(columns=[label_col])
                        .select_dtypes(include=[np.number])
                        .to_numpy(dtype=float, na_value=np.nan)
                    ).sum()
                ),
                "n_cells_nan": int(pooled.drop(columns=[label_col]).isna().sum().sum()),
            },
        }
    )
    if non_numeric:
        raise ValueError(f"{variant}: non-numeric feature columns found: {non_numeric}")
    return report


if __name__ == "__main__":
    import json
    import sys

    from qgridbench.utils.run_tracking import RunDir

    variants = sys.argv[1:] or ["binary", "triple", "multiclass"]
    with RunDir("data_validation", config={"variants": variants}) as run:
        reports = {v: validate_variant(v) for v in variants}
        run.write_json("data_report.json", reports)
        for v, r in reports.items():
            run.metrics[f"{v}_rows"] = r["total_rows"]
        print(
            json.dumps(
                {
                    v: {
                        k: r[k]
                        for k in (
                            "n_files",
                            "total_rows",
                            "n_feature_columns",
                            "duplicate_feature_rows",
                            "class_distribution",
                        )
                    }
                    for v, r in reports.items()
                },
                indent=2,
            )
        )
        print(f"report -> {run.path}")
