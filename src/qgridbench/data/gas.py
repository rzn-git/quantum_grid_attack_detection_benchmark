"""Load, validate, deduplicate and split the Morris gas-pipeline SCADA dataset (R4-12).

The second ICS benchmark of the cross-domain replication. One flat ARFF capture
(`IanArffDataset.arff`, configs/data_gas.yaml) becomes two variants, `gas_binary` and
`gas_three`, written with the same file names and keys as the grid variants
(data/processed/<variant>.npz, splits_<variant>.npz), so `load_processed`, `prepare_regime`
and every runner that takes `--variant` read them unchanged. Auxiliary per-row arrays
(raw position, time, category, attack type, time block) go to <variant>_aux.npz for the
grouped split arms.

Pipeline, mirroring qgridbench.data.preprocess:
  1. verify SHA256 and the schema anchor (row and class counts of the source paper)
  2. map the variant's label (fail loud on unknown values)
  3. assign each raw row its equal-duration time block (G-time arm), before dedupe
  4. drop exact duplicates on the 16 features + label, first occurrence kept, BEFORE
     splitting (`time` is not a feature: the capture repeats every four packets)
  5. R arm: stratified 60/20/20 with the fixed split seed
No imputation here: missing fields stay NaN and are imputed by the train-fitted transform.
Derived variants (configs/data_gas.yaml, `derived_variants`) reuse a base variant's rows,
labels and splits with a fixed feature transform on X; they are written from the base
variant's processed files.

Run:  uv run python -m qgridbench.data.gas
      uv run python -m qgridbench.data.gas --derived
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from qgridbench.data.download import sha256_file
from qgridbench.utils.run_tracking import REPO_ROOT, get_logger, load_yaml

log = get_logger(__name__)

VARIANTS = ("gas_binary", "gas_three")
SIDES = ("train", "val", "test")


def gas_config() -> dict:
    return load_yaml(REPO_ROOT / "configs" / "data_gas.yaml")


def gas_features(cfg: dict | None = None) -> list[str]:
    cfg = cfg or gas_config()
    return list(cfg["columns"][: cfg["n_features"]])


def read_arff(path: Path, columns: list[str], missing_token: str) -> pd.DataFrame:
    """Parse the ARFF body; the attribute count must equal `columns` (fail loud)."""
    lines = path.read_text(encoding="latin-1").splitlines()
    n_attr = sum(1 for line in lines if line.strip().lower().startswith("@attribute"))
    if n_attr != len(columns):
        raise ValueError(f"{path.name}: {n_attr} attributes, config lists {len(columns)}")
    start = next(i for i, line in enumerate(lines) if line.strip().lower() == "@data") + 1
    body = "\n".join(line for line in lines[start:] if line.strip() and not line.startswith("%"))
    return pd.read_csv(
        io.StringIO(body), header=None, names=columns, na_values=missing_token, quotechar="'"
    )


def validate_frame(df: pd.DataFrame, cfg: dict) -> None:
    """Schema anchor: row count, class counts, monotone time, label consistency."""
    schema = cfg["schema"]
    if len(df) != schema["n_rows"]:
        raise ValueError(f"{len(df)} rows, schema expects {schema['n_rows']}")
    for col in ("binary", "categorized"):
        got = {int(k): int(v) for k, v in df[col].value_counts().items()}
        want = {int(k): int(v) for k, v in schema[f"{col}_counts"].items()}
        if got != want:
            raise ValueError(f"{col} counts {got} != schema {want}")
    if not np.all(np.diff(df["time"].to_numpy(dtype=np.float64)) >= 0):
        raise ValueError("time column is not monotone")
    normal = df["binary"].to_numpy() == 0
    for col in ("categorized", "specific"):
        if not np.array_equal(df[col].to_numpy() == 0, normal):
            raise ValueError(f"{col} == 0 does not coincide with binary == 0")
    feats = df[gas_features(cfg)]
    non_numeric = [c for c in feats.columns if not pd.api.types.is_numeric_dtype(feats[c])]
    if non_numeric:
        raise ValueError(f"non-numeric feature columns: {non_numeric}")


def load_gas_frame(cfg: dict | None = None) -> pd.DataFrame:
    """The raw capture in time order, SHA256- and schema-verified."""
    cfg = cfg or gas_config()
    src = cfg["source"]
    path = REPO_ROOT / src["raw_dir"] / src["file"]
    digest = sha256_file(path)
    if digest != src["sha256"]:
        raise RuntimeError(
            f"SHA256 mismatch for {path.name}: got {digest}, expected {src['sha256']}"
        )
    df = read_arff(path, cfg["columns"], cfg["missing_token"])
    validate_frame(df, cfg)
    return df


def label_vector(df: pd.DataFrame, cfg: dict, variant: str) -> np.ndarray:
    spec = cfg["variants"][variant]
    mapping = {int(k): int(v) for k, v in spec["map"].items()}
    raw = df[spec["label"]].astype(int)
    unknown = set(raw.unique()) - set(mapping)
    if unknown:
        raise ValueError(f"{variant}: unmapped label values {sorted(unknown)}")
    return raw.map(mapping).to_numpy(dtype=np.int64)


def time_blocks(t: np.ndarray, n_blocks: int) -> np.ndarray:
    """Equal-duration contiguous blocks of the capture, ids 0..n_blocks-1."""
    edges = np.linspace(t[0], t[-1], n_blocks + 1)
    return np.clip(np.searchsorted(edges, t, side="right") - 1, 0, n_blocks - 1)


def dedupe_mask(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """True for the first occurrence of each (features, label) row; NaN equals NaN."""
    df = pd.DataFrame(X)
    df["__y"] = y
    return (~df.duplicated()).to_numpy()


def row_split(y: np.ndarray, ratios: list[float], seed: int) -> dict[str, np.ndarray]:
    """The grid's stratified 60/20/20 split (qgridbench.data.preprocess), sorted indices."""
    idx = np.arange(len(y))
    tr, tmp = train_test_split(idx, train_size=ratios[0], stratify=y, random_state=seed)
    rel_val = ratios[1] / (ratios[1] + ratios[2])
    va, te = train_test_split(tmp, train_size=rel_val, stratify=y[tmp], random_state=seed)
    return {"train": np.sort(tr), "val": np.sort(va), "test": np.sort(te)}


def time_block_splits(block: np.ndarray, partition: int, n_blocks: int) -> dict[str, np.ndarray]:
    """G-time: shuffle block ids with default_rng(partition); 60/20/20 by BLOCK.

    The same rule as the grid's whole-file split (run_group_split_control.group_splits).
    """
    g = np.arange(n_blocks)
    np.random.default_rng(partition).shuffle(g)
    cut = {"train": g[: int(0.6 * n_blocks)], "val": g[int(0.6 * n_blocks) : int(0.8 * n_blocks)]}
    cut["test"] = g[int(0.8 * n_blocks) :]
    return {s: np.flatnonzero(np.isin(block, cut[s])) for s in SIDES}


def attack_name_splits(
    row: dict[str, np.ndarray],
    normal: np.ndarray,
    specific: np.ndarray,
    categorized: np.ndarray,
    partition: int,
    name_groups: list[list[int]],
    cycle: list[str],
) -> dict[str, np.ndarray]:
    """G-type: attack rows held out by attack NAME, category-stratified; normal rows keep R.

    Within each category (ascending), its groups are permuted with one default_rng(partition)
    stream. A category with >= 3 groups follows `cycle`; with 2 groups, group `partition % 2`
    goes to test and the other to train; a single group stays in train.
    """
    side_of = np.full(len(specific), "", dtype=object)
    for s in SIDES:
        side_of[row[s]] = s
    out = {s: [np.flatnonzero(normal & (side_of == s))] for s in SIDES}
    attack_types = set(np.unique(specific[~normal]).tolist())
    covered = {t for grp in name_groups for t in grp}
    if attack_types - covered:
        raise ValueError(f"attack types without a name group: {sorted(attack_types - covered)}")
    by_cat: dict[int, list[list[int]]] = {}
    for grp in name_groups:
        rows = np.isin(specific, grp) & ~normal
        cats = np.unique(categorized[rows])
        if len(cats) != 1:
            raise ValueError(f"name group {grp} spans categories {cats.tolist()}")
        by_cat.setdefault(int(cats[0]), []).append(grp)
    rng = np.random.default_rng(partition)
    for _cat, groups in sorted(by_cat.items()):
        order = rng.permutation(len(groups))
        for k, gi in enumerate(order):
            if len(groups) >= 3:
                side = cycle[k % len(cycle)]
            elif len(groups) == 2:
                side = "test" if gi == partition % 2 else "train"
            else:
                side = "train"
            out[side].append(np.flatnonzero(np.isin(specific, groups[gi]) & ~normal))
    return {s: np.sort(np.concatenate(out[s])) for s in SIDES}


def preprocess_gas(variant: str, cfg: dict | None = None) -> dict:
    cfg = cfg or gas_config()
    if variant not in cfg["variants"]:
        raise ValueError(f"unknown gas variant '{variant}'")
    out_dir = REPO_ROOT / load_yaml(REPO_ROOT / "configs" / "data.yaml")["paths"]["processed"]
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_gas_frame(cfg)
    feats = gas_features(cfg)
    X = df[feats].to_numpy(dtype=np.float64)
    y = label_vector(df, cfg, variant)
    t = df["time"].to_numpy(dtype=np.float64)
    block = time_blocks(t, cfg["groups"]["time_blocks"])
    keep = dedupe_mask(X, y)
    n_raw = len(y)
    X, y = X[keep], y[keep]
    raw_pos = np.flatnonzero(keep)
    sp = cfg["preprocess"]["split"]
    splits = row_split(y, sp["ratios"], sp["seed"])

    np.savez_compressed(out_dir / f"{variant}.npz", X=X.astype(np.float32), y=y)
    np.savez_compressed(
        out_dir / f"splits_{variant}.npz", **splits, split_seed=np.array([sp["seed"]])
    )
    np.savez_compressed(
        out_dir / f"{variant}_aux.npz",
        raw_pos=raw_pos,
        time=t[keep],
        categorized=df["categorized"].to_numpy(dtype=np.int64)[keep],
        specific=df["specific"].to_numpy(dtype=np.int64)[keep],
        block=block[keep],
    )
    meta = {
        "variant": variant,
        "source_sha256": cfg["source"]["sha256"],
        "n_raw_rows": n_raw,
        "n_duplicate_rows_dropped": int(n_raw - keep.sum()),
        "n_rows": int(len(y)),
        "n_features": len(feats),
        "feature_names": feats,
        "label": cfg["variants"][variant],
        "class_counts": {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))},
        "missing_fraction": {f: float(np.isnan(X[:, i]).mean()) for i, f in enumerate(feats)},
        "split_sizes": {s: int(len(v)) for s, v in splits.items()},
        "split_seed": sp["seed"],
    }
    (out_dir / f"meta_{variant}.json").write_text(json.dumps(meta, indent=2))
    log.info("%s", {k: meta[k] for k in ("variant", "n_rows", "class_counts", "split_sizes")})
    return meta


def load_aux(variant: str) -> dict[str, np.ndarray]:
    """Per-row auxiliary arrays of a processed gas variant, aligned with its X and y."""
    proc = REPO_ROOT / load_yaml(REPO_ROOT / "configs" / "data.yaml")["paths"]["processed"]
    d = np.load(proc / f"{variant}_aux.npz")
    return {k: d[k] for k in d.files}


# ---------------------------------------------------------------- derived variants ----------
# A derived variant keeps its base variant's rows, labels, splits and aux arrays and applies
# fixed, unfitted, row-wise feature transforms to X (configs/data_gas.yaml, derived_variants).
# Nothing is fit, so applying the transform before the split-fitted imputer, scaler and PCA
# cannot carry information across splits.

DERIVED_TRANSFORMS = ("signed_log1p",)


def derived_specs(cfg: dict | None = None) -> dict:
    cfg = cfg or gas_config()
    specs = dict(cfg.get("derived_variants") or {})
    clash = set(specs) & set(cfg["variants"])
    if clash:
        raise ValueError(f"derived variant names clash with base variants: {sorted(clash)}")
    return specs


def base_variant(variant: str, cfg: dict | None = None) -> str:
    """The base variant whose rows, labels and splits `variant` uses (itself for a base)."""
    cfg = cfg or gas_config()
    if variant in cfg["variants"]:
        return variant
    spec = derived_specs(cfg).get(variant)
    if spec is None:
        raise ValueError(f"unknown gas variant '{variant}'")
    if spec["base"] not in cfg["variants"]:
        raise ValueError(f"{variant}: base '{spec['base']}' is not a base variant")
    return spec["base"]


def signed_log1p(X: np.ndarray, columns: list[int]) -> np.ndarray:
    """sign(x) * log1p(|x|) on `columns` of a copy of X, computed in float64 and stored at X's
    dtype. NaN stays NaN; every other column is returned unchanged."""
    X = np.array(X, copy=True)
    for j in columns:
        p = X[:, j].astype(np.float64)
        X[:, j] = (np.sign(p) * np.log1p(np.abs(p))).astype(X.dtype)
    return X


def apply_variant_transform(X: np.ndarray, variant: str, cfg: dict | None = None) -> np.ndarray:
    """The variant's fixed feature transforms on X (a base variant returns X unchanged)."""
    cfg = cfg or gas_config()
    if base_variant(variant, cfg) == variant:
        return X
    spec = derived_specs(cfg)[variant]
    unknown = set(spec) - {"base", *DERIVED_TRANSFORMS}
    if unknown:
        raise ValueError(f"{variant}: unknown transforms {sorted(unknown)}")
    feats = gas_features(cfg)
    cols = []
    for name in spec.get("signed_log1p", []):
        if feats.count(name) != 1:
            raise ValueError(f"{variant}: feature '{name}' not found once in {feats}")
        cols.append(feats.index(name))
    return signed_log1p(X, cols)


def derive_variant(variant: str, cfg: dict | None = None) -> dict:
    """Write a derived variant's files from its base variant's processed files."""
    cfg = cfg or gas_config()
    base = base_variant(variant, cfg)
    if base == variant:
        raise ValueError(f"{variant} is a base variant")
    proc = REPO_ROOT / load_yaml(REPO_ROOT / "configs" / "data.yaml")["paths"]["processed"]
    d = np.load(proc / f"{base}.npz")
    X = apply_variant_transform(d["X"], variant, cfg)
    np.savez_compressed(proc / f"{variant}.npz", X=X, y=d["y"])
    s = np.load(proc / f"splits_{base}.npz")
    np.savez_compressed(proc / f"splits_{variant}.npz", **{k: s[k] for k in s.files})
    a = np.load(proc / f"{base}_aux.npz")
    np.savez_compressed(proc / f"{variant}_aux.npz", **{k: a[k] for k in a.files})
    meta = json.loads((proc / f"meta_{base}.json").read_text())
    spec = derived_specs(cfg)[variant]
    meta.update(
        variant=variant,
        base_variant=base,
        transforms={k: v for k, v in spec.items() if k != "base"},
        base_npz_sha256=sha256_file(proc / f"{base}.npz"),
    )
    (proc / f"meta_{variant}.json").write_text(json.dumps(meta, indent=2))
    log.info("%s", {k: meta[k] for k in ("variant", "base_variant", "transforms", "n_rows")})
    return meta


if __name__ == "__main__":
    import sys

    from qgridbench.utils.run_tracking import RunDir

    if sys.argv[1:] == ["--derived"]:
        names = list(derived_specs())
        with RunDir("preprocess_gas_derived", config={"variants": names}) as run:
            results = {v: derive_variant(v) for v in names}
            run.write_json("preprocess_summary.json", results)
            run.metrics.update({f"{v}_rows": r["n_rows"] for v, r in results.items()})
            print(f"-> {run.path}")
        raise SystemExit(0)
    with RunDir("preprocess_gas", config={"variants": list(VARIANTS)}) as run:
        results = {v: preprocess_gas(v) for v in VARIANTS}
        run.write_json("preprocess_summary.json", results)
        run.metrics.update({f"{v}_rows": r["n_rows"] for v, r in results.items()})
        print(json.dumps({v: r["class_counts"] for v, r in results.items()}, indent=2))
        print(f"-> {run.path}")
