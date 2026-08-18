"""Statistical testing: paired Wilcoxon signed-rank across seeds, Holm-Bonferroni
correction across the headline comparison family, and effect sizes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon


def paired_wilcoxon(a: np.ndarray, b: np.ndarray) -> dict:
    """Paired Wilcoxon over per-seed metric values a vs b (same seed order).

    Returns p-value, rank-biserial effect size, and paired Cohen's d.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape:
        raise ValueError("paired arrays must share shape")
    d = a - b
    if np.allclose(d, 0):
        return {
            "p_value": 1.0,
            "rank_biserial": 0.0,
            "cohens_d": 0.0,
            "mean_diff": 0.0,
            "n": len(d),
        }
    _stat, p = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided", method="exact")
    nz = d[d != 0]
    ranks = np.argsort(np.argsort(np.abs(nz))) + 1.0
    w_plus = float(ranks[nz > 0].sum())
    w_minus = float(ranks[nz < 0].sum())
    rank_biserial = (w_plus - w_minus) / (w_plus + w_minus)
    sd = d.std(ddof=1)
    cohens_d = float(d.mean() / sd) if sd > 0 else 0.0
    return {
        "p_value": float(p),
        "rank_biserial": rank_biserial,
        "cohens_d": cohens_d,
        "mean_diff": float(d.mean()),
        "n": int(len(d)),
        "w_plus": w_plus,
        "w_minus": w_minus,
    }


def holm_bonferroni(pvals: dict[str, float]) -> dict[str, float]:
    """Holm step-down corrected p-values, keyed like the input."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    corrected, running_max = {}, 0.0
    for rank, (key, p) in enumerate(items):
        adj = min(1.0, (m - rank) * p)
        running_max = max(running_max, adj)  # enforce monotonicity
        corrected[key] = running_max
    return corrected


QUANTUM_FAMILIES = ("qsvm", "vqc")


def robustness_stats(adv_dir: Path) -> dict:
    """Test RQ2 as the PRE-REGISTERED directional hypothesis, not a post-hoc pick.

    study protocol §8.3 frames RQ2 as: prior evidence for a quantum robustness
    advantage exists only on image data — does it hold on tabular grid data? The
    honest instantiation is therefore EVERY quantum family against EVERY classical
    family, Holm-corrected over that whole set. Testing the quantum kernel against
    only the tree ensembles (an earlier version of this function) selects the
    comparison after seeing the data and inflates the apparent effect.

    Retention (adversarial / clean) is reported beside raw degradation because a
    model that was already near-chance cannot lose much: degradation alone rewards
    a weak model, which is the confound that makes robustness claims cheap.
    """
    d = json.loads((adv_dir / "results.json").read_text())
    ps = d["per_seed"]
    if not ps or "evasion" not in ps[0]:
        return {}
    families = list(ps[0]["evasion"]["curves"])
    top = sorted(ps[0]["evasion"]["curves"][families[0]]["pgd"], key=float)[-1]

    def series(fam: str, kind: str) -> np.ndarray:
        if kind == "clean":
            return np.array([s["evasion"]["clean"][fam] for s in ps])
        return np.array([s["evasion"]["curves"][fam]["pgd"][top] for s in ps])

    out: dict = {
        "epsilon": float(top),
        "epsilon_unit": "fraction of train per-feature sigma",
        "n_seeds": len(ps),
        "per_family": {},
        "quantum_vs_classical": {},
    }
    for fam in families:
        c, a = series(fam, "clean"), series(fam, "adv")
        out["per_family"][fam] = {
            "clean_mean": float(c.mean()),
            "clean_std": float(c.std(ddof=1)) if len(c) > 1 else 0.0,
            "adv_mean": float(a.mean()),
            "adv_std": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
            "degradation": float(c.mean() - a.mean()),
            "retention": float(a.mean() / c.mean()) if c.mean() else float("nan"),
        }

    quantum = [f for f in families if f in QUANTUM_FAMILIES]
    classical = [f for f in families if f not in QUANTUM_FAMILIES]
    raw, detail = {}, {}
    for q in quantum:
        for c in classical:
            st = paired_wilcoxon(series(q, "adv"), series(c, "adv"))
            key = f"{q}_vs_{c}"
            raw[key], detail[key] = st["p_value"], st
    corrected = holm_bonferroni(raw) if raw else {}
    for key, st in detail.items():
        st["p_holm"] = corrected[key]
        st["significant_holm_0.05"] = bool(corrected[key] < 0.05)
        out["quantum_vs_classical"][key] = st
    out["n_comparisons"] = len(raw)
    out["n_significant_holm"] = sum(
        1 for v in out["quantum_vs_classical"].values() if v["significant_holm_0.05"]
    )
    return out
