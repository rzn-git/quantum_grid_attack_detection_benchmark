"""Gas-pipeline variant (R4-12): schema, dedupe, split determinism, grouped arms, leakage.

All on a tiny synthetic ARFF written to tmp_path with the release's 20 attributes, `?` for
missing fields and quoted nominal labels.
"""

import hashlib

import numpy as np
import pytest

import qgridbench.data.gas as gas
from qgridbench.features.reduce import (
    ANGLE_RANGE,
    AngleScaler,
    build_preprocessor,
    fit_transform_splits,
)

COLUMNS = [
    "address", "function", "length", "setpoint", "gain", "reset_rate", "deadband",
    "cycle_time", "rate", "system_mode", "control_scheme", "pump", "solenoid", "pressure",
    "crc_rate", "command_response", "time", "binary", "categorized", "specific",
]  # fmt: skip
# category 4 has four single-type groups, category 1 two, category 6 one (the DoS shape)
NAME_GROUPS = [[1], [2], [3], [4], [5], [6], [7]]
TYPE_CAT = {1: 4, 2: 4, 3: 4, 4: 4, 5: 1, 6: 1, 7: 6}


def _rows():
    rows, t = [], 1000.0
    for i in range(60):  # normal traffic: 30 distinct packets, each sent twice
        k = i % 30
        rows.append([4, 3, 46, "?", "?", "?", "?", "?", "?", "?", "?", "?", "?", 5.0 + k / 10,
                     12000 + k, 0, t, 0, 0, 0])  # fmt: skip
        t += 1.0
    for spec, cat in TYPE_CAT.items():
        for j in range(4):  # 4 distinct attack packets per type
            rows.append([4, 16, 90, 10 + spec, 115, 0.2, 0.5, 1, 0, 0, 1, j % 2, 0, "?",
                         17000 + 10 * spec + j, 1, t, 1, cat, spec])  # fmt: skip
            t += 1.0
    return rows


def _write_arff(path, rows, n_attr=20):
    head = ["@relation gas"] + [f"@attribute 'a{i}' real" for i in range(n_attr)] + ["@data"]
    body = []
    for r in rows:
        r = list(r)
        r[17:] = [f"'{v}'" for v in r[17:]]
        body.append(",".join(str(v) for v in r))
    path.write_text("\n".join(head + body) + "\n", encoding="latin-1")


def _cfg(rows, sha="x"):
    binary = [r[17] for r in rows]
    cat = [r[18] for r in rows]
    return {
        "source": {"raw_dir": "raw", "file": "gas.arff", "sha256": sha},
        "columns": COLUMNS,
        "n_features": 16,
        "missing_token": "?",
        "schema": {
            "n_rows": len(rows),
            "binary_counts": {v: binary.count(v) for v in set(binary)},
            "categorized_counts": {v: cat.count(v) for v in set(cat)},
        },
        "variants": {
            "gas_binary": {"label": "binary", "map": {0: 0, 1: 1}},
            "gas_three": {"label": "categorized", "map": {0: 0, 1: 1, 4: 2, 6: 2}},
        },
        "preprocess": {"split": {"ratios": [0.6, 0.2, 0.2], "seed": 1337}},
        "groups": {"partitions": 5, "time_blocks": 5, "attack_names": NAME_GROUPS,
                   "name_cycle": ["test", "val", "train", "train", "train"]},
    }  # fmt: skip


@pytest.fixture
def repo(tmp_path, monkeypatch):
    rows = _rows()
    (tmp_path / "raw").mkdir()
    arff = tmp_path / "raw" / "gas.arff"
    _write_arff(arff, rows)
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "data.yaml").write_text("paths:\n  processed: proc\n")
    monkeypatch.setattr(gas, "REPO_ROOT", tmp_path)
    cfg = _cfg(rows, hashlib.sha256(arff.read_bytes()).hexdigest())
    return tmp_path, cfg, rows


def test_schema_reads_missing_and_labels(repo):
    _, cfg, rows = repo
    df = gas.load_gas_frame(cfg)
    assert len(df) == len(rows) and list(df.columns) == COLUMNS
    assert df["setpoint"].isna().sum() == 60 and df["pressure"].isna().sum() == 28
    assert df["specific"].dtype.kind == "i"


def test_schema_fails_loud(repo, tmp_path):
    root, cfg, rows = repo
    bad = dict(cfg, schema=dict(cfg["schema"], n_rows=len(rows) + 1))
    with pytest.raises(ValueError, match="rows"):
        gas.load_gas_frame(bad)
    with pytest.raises(RuntimeError, match="SHA256"):
        gas.load_gas_frame(dict(cfg, source=dict(cfg["source"], sha256="0" * 64)))
    _write_arff(tmp_path / "wide.arff", rows, n_attr=21)
    with pytest.raises(ValueError, match="attributes"):
        gas.read_arff(tmp_path / "wide.arff", COLUMNS, "?")
    shuffled = gas.read_arff(root / "raw" / "gas.arff", COLUMNS, "?").iloc[::-1]
    with pytest.raises(ValueError, match="monotone"):
        gas.validate_frame(shuffled.reset_index(drop=True), cfg)


def test_unknown_label_fails_loud(repo):
    _, cfg, _ = repo
    df = gas.load_gas_frame(cfg)
    bad = dict(cfg, variants={"gas_three": {"label": "categorized", "map": {0: 0, 1: 1}}})
    with pytest.raises(ValueError, match="unmapped"):
        gas.label_vector(df, bad, "gas_three")
    y3 = gas.label_vector(df, cfg, "gas_three")
    assert set(y3.tolist()) == {0, 1, 2}


def test_dedupe_excludes_time_keeps_first_and_label_twins():
    X = np.array([[1.0, np.nan], [1.0, np.nan], [2.0, 3.0], [2.0, 3.0]])
    y = np.array([0, 0, 0, 1])  # rows 0/1 differ only in time (not a column here)
    assert gas.dedupe_mask(X, y).tolist() == [True, False, True, True]


def test_row_split_deterministic_disjoint_stratified():
    y = np.repeat([0, 1], [300, 100])
    a, b = gas.row_split(y, [0.6, 0.2, 0.2], 1337), gas.row_split(y, [0.6, 0.2, 0.2], 1337)
    assert all(np.array_equal(a[s], b[s]) for s in gas.SIDES)
    allidx = np.concatenate([a[s] for s in gas.SIDES])
    assert len(set(allidx)) == len(y) == len(allidx)
    assert len(a["train"]) == 240 and y[a["test"]].mean() == pytest.approx(0.25)


def test_time_blocks_split_by_block_only():
    t = np.linspace(0.0, 150.0, 301)
    block = gas.time_blocks(t, 15)
    assert block.min() == 0 and block.max() == 14
    for p in range(5):
        a, b = gas.time_block_splits(block, p, 15), gas.time_block_splits(block, p, 15)
        assert all(np.array_equal(a[s], b[s]) for s in gas.SIDES)
        sides = [set(block[a[s]].tolist()) for s in gas.SIDES]
        assert [len(x) for x in sides] == [9, 3, 3]
        assert not (sides[0] & sides[2]) and not (sides[0] & sides[1])
        assert sum(len(a[s]) for s in gas.SIDES) == len(t)


def test_attack_name_split_holds_groups_out_whole(repo):
    _, cfg, _ = repo
    df = gas.load_gas_frame(cfg)
    y = gas.label_vector(df, cfg, "gas_binary")
    spec, cat = df["specific"].to_numpy(), df["categorized"].to_numpy()
    row = gas.row_split(y, [0.6, 0.2, 0.2], 1337)
    cycle = cfg["groups"]["name_cycle"]
    for p in range(5):
        sp = gas.attack_name_splits(row, y == 0, spec, cat, p, NAME_GROUPS, cycle)
        again = gas.attack_name_splits(row, y == 0, spec, cat, p, NAME_GROUPS, cycle)
        assert all(np.array_equal(sp[s], again[s]) for s in gas.SIDES)
        side = {s: set(spec[sp[s]][y[sp[s]] == 1].tolist()) for s in gas.SIDES}
        assert not (side["train"] & side["test"]) and not (side["train"] & side["val"])
        assert all(side[s] & {1, 2, 3, 4} for s in gas.SIDES)  # >= 3 groups: every side
        assert 7 in side["train"]  # single-group category never held out
        assert len(side["test"] & {5, 6}) == 1  # two groups: one tested, rotating
        for s in gas.SIDES:  # normal rows keep their R-arm side
            assert set(sp[s][y[sp[s]] == 0].tolist()) == set(row[s][y[row[s]] == 0].tolist())
    with pytest.raises(ValueError, match="without a name group"):
        gas.attack_name_splits(row, y == 0, spec, cat, 0, NAME_GROUPS[:-1], cycle)


def test_transforms_fit_on_train_only_and_angles_in_range(repo):
    _, cfg, _ = repo
    df = gas.load_gas_frame(cfg)
    X = df[gas.gas_features(cfg)].to_numpy(dtype=np.float64)
    y = gas.label_vector(df, cfg, "gas_binary")
    sp = gas.row_split(y, [0.6, 0.2, 0.2], 1337)
    pipe = build_preprocessor("pca", n_components=4, seed=0)
    Z = fit_transform_splits(pipe, X, sp)
    ref = build_preprocessor("pca", n_components=4, seed=0).fit(X[sp["train"]])
    np.testing.assert_array_equal(
        pipe.named_steps["impute"].statistics_, ref.named_steps["impute"].statistics_
    )
    np.testing.assert_allclose(
        pipe.named_steps["pca"].components_, ref.named_steps["pca"].components_
    )
    X2 = X.copy()
    X2[sp["test"]] = 1e6  # test rows must not move any fitted statistic
    pipe2 = build_preprocessor("pca", n_components=4, seed=0)
    fit_transform_splits(pipe2, X2, sp)
    np.testing.assert_array_equal(pipe.named_steps["scale"].mean_, pipe2.named_steps["scale"].mean_)
    ang = AngleScaler().fit(Z["train"])
    for s in gas.SIDES:
        a = ang.transform(Z[s])
        assert a.min() >= ANGLE_RANGE[0] and a.max() <= ANGLE_RANGE[1]


def test_preprocess_writes_aligned_variant_files(repo):
    root, cfg, rows = repo
    meta = gas.preprocess_gas("gas_binary", cfg)
    d = np.load(root / "proc" / "gas_binary.npz")
    s = np.load(root / "proc" / "splits_gas_binary.npz")
    aux = np.load(root / "proc" / "gas_binary_aux.npz")
    assert meta["n_duplicate_rows_dropped"] == 30 and meta["n_rows"] == len(rows) - 30
    assert d["X"].dtype == np.float32 and d["X"].shape == (meta["n_rows"], 16)
    assert len(aux["specific"]) == len(d["y"]) and np.all(np.diff(aux["raw_pos"]) > 0)
    assert np.array_equal(aux["specific"] > 0, d["y"] == 1)
    assert sorted(np.concatenate([s[k] for k in gas.SIDES]).tolist()) == list(range(meta["n_rows"]))


# ---------------------------------------------------------------- derived variants ----------


def _derived_cfg(cfg):
    spec = {"base": "gas_binary", "signed_log1p": ["pressure"]}
    return {**cfg, "derived_variants": {"gas_binary_slog": spec}}


def test_signed_log1p_changes_only_named_columns_and_keeps_dtype():
    X = np.array([[1.0, -3.0, np.nan], [2.0, 3.36e38, 0.0], [5.0, -0.5, 7.0]], dtype=np.float32)
    before = X.copy()
    out = gas.signed_log1p(X, [1])
    assert out.dtype == np.float32 and np.array_equal(X, before, equal_nan=True)
    p = X[:, 1].astype(np.float64)
    assert np.array_equal(out[:, 1], (np.sign(p) * np.log1p(np.abs(p))).astype(np.float32))
    assert np.array_equal(out[:, [0, 2]], X[:, [0, 2]], equal_nan=True)
    assert np.isnan(gas.signed_log1p(np.array([[np.nan]], dtype=np.float32), [0])[0, 0])


def test_variant_transform_base_is_identity_and_unknowns_fail(repo):
    _, cfg, _ = repo
    dcfg = _derived_cfg(cfg)
    X = np.arange(32, dtype=np.float32).reshape(2, 16)
    assert gas.apply_variant_transform(X, "gas_binary", dcfg) is X
    assert gas.base_variant("gas_binary_slog", dcfg) == "gas_binary"
    with pytest.raises(ValueError, match="unknown gas variant"):
        gas.base_variant("gas_nope", dcfg)
    bad = {**cfg, "derived_variants": {"gas_x": {"base": "gas_binary", "rank": ["pressure"]}}}
    with pytest.raises(ValueError, match="unknown transforms"):
        gas.apply_variant_transform(X, "gas_x", bad)
    clash = {**cfg, "derived_variants": {"gas_binary": {"base": "gas_three"}}}
    with pytest.raises(ValueError, match="clash"):
        gas.derived_specs(clash)


def test_derived_variant_keeps_rows_labels_splits_and_changes_only_pressure(repo):
    root, cfg, _ = repo
    dcfg = _derived_cfg(cfg)
    gas.preprocess_gas("gas_binary", dcfg)
    meta = gas.derive_variant("gas_binary_slog", dcfg)
    proc = root / "proc"
    base, der = np.load(proc / "gas_binary.npz"), np.load(proc / "gas_binary_slog.npz")
    j = COLUMNS.index("pressure")
    assert der["X"].dtype == base["X"].dtype == np.float32
    assert np.array_equal(der["y"], base["y"])
    keep = [i for i in range(16) if i != j]
    assert np.array_equal(der["X"][:, keep], base["X"][:, keep], equal_nan=True)
    assert np.array_equal(der["X"][:, j], gas.signed_log1p(base["X"], [j])[:, j], equal_nan=True)
    for stem in ("splits_{}.npz", "{}_aux.npz"):
        a = np.load(proc / stem.format("gas_binary"))
        b = np.load(proc / stem.format("gas_binary_slog"))
        assert a.files == b.files and all(np.array_equal(a[k], b[k]) for k in a.files)
    assert meta["base_variant"] == "gas_binary"
    assert meta["transforms"] == {"signed_log1p": ["pressure"]}
    with pytest.raises(ValueError, match="is a base variant"):
        gas.derive_variant("gas_binary", dcfg)
