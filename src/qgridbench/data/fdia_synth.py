"""Generate the synthetic false-data-injection (FDIA) variants on IEEE 14-bus and 118-bus (R4-22).

Run:  uv run python -m qgridbench.data.fdia_synth --workers 8 --threads 1

A public, regenerable FDIA benchmark built from code and one config (configs/fdia_synth.yaml;
design frozen in the decision log, 2026-11-16T12:17Z, "PRE-RUN AMENDMENT (R4-22"). Per case:
  1. load: the IEEE RTS-96 load-factor tables (weekly x daily x hourly), interpolated to 15
     minutes, times a per-bus multiplicative AR(1) deviation; one calendar day is one scenario
     group (90 days stratified over the 13 four-week blocks, 96 steps each);
  2. an AC power flow per snapshot (pandapower), warm-started within a day, a fresh network
     per day so every day is a deterministic unit whatever worker runs it;
  3. the SCADA measurement vector z = [Vm, P_inj, Q_inj, P_from, Q_from] plus Gaussian noise
     (sigma = 1% of the channel's base-case magnitude, with a per-block floor);
  4. episodes of 2 hours: normal, stealthy (AC-exact a = h(Vm, theta + c) - h(Vm, theta) at the
     true state, k targeted angles, a three-scale magnitude ladder) and non-stealthy (the same
     injection permuted onto random meters of the same block, random signs);
  5. an AC weighted-least-squares estimate (Gauss-Newton from measured Vm and DC angles) and
     a chi-square bad-data decision for every row, and for every attacked row also for the
     clean measurement it was built from (the paired generator control).
Written with the grid's file names and keys (data/processed/<variant>.npz, splits_<variant>.npz,
<variant>_aux.npz, meta_<variant>.json), so `load_processed`, `prepare_regime` and every
`--variant` runner read the variants unchanged. The binary variants keep normal and stealthy
rows (1:1); the non-stealthy rows enter only the generator's controls.

Blocking controls (the run raises after writing meta): h(x) reproduces pandapower's injections
and flows; every clean estimate converges; each stealthy row's decision equals its paired clean
row's on at least 99% of stealthy rows.
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.stats import chi2

from qgridbench.data.gas import load_aux, row_split
from qgridbench.utils.run_tracking import REPO_ROOT, get_logger, git_describe, load_yaml

log = get_logger(__name__)

CFG_PATH = REPO_ROOT / "configs" / "fdia_synth.yaml"
NORMAL, STEALTHY, NONSTEALTHY = 0, 1, 2
KIND_CODE = {"normal": NORMAL, "stealthy": STEALTHY, "nonstealthy": NONSTEALTHY}
SCALES = ("small", "medium", "large")
BLOCKS = ("vm", "p_inj", "q_inj", "p_from", "q_from")
CASE_ID = {"case14": 14, "case118": 118}
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
DAYS_PER_YEAR = 364
SIDES = ("train", "val", "test")

__all__ = ["Case", "load_aux", "fdia_config", "day_fold_splits", "generate_variant"]


def fdia_config() -> dict:
    return load_yaml(CFG_PATH)


# ------------------------------------------------------------------------------- load model


def _season(week1: int, seasons: dict) -> str:
    for name, spans in seasons.items():
        if any(lo <= week1 <= hi for lo, hi in spans):
            return name
    raise ValueError(f"week {week1} is in no season")


def step_factors(cfg: dict) -> np.ndarray:
    """(364, 24 * steps_per_hour) load factor in fractions of the annual peak."""
    lf = cfg["load_factors"]
    hourly = np.empty(DAYS_PER_YEAR * 24)
    for d in range(DAYS_PER_YEAR):
        week = d // 7
        dow = d % 7
        kind = "weekday" if dow < 5 else "weekend"
        prof = lf["hourly"][f"{_season(week + 1, lf['seasons'])}_{kind}"]
        hourly[d * 24 : (d + 1) * 24] = (
            np.asarray(prof, float) * lf["daily"][dow] * lf["weekly"][week] / 1e6
        )
    sph = lf["steps_per_hour"]
    t_hour = np.arange(hourly.size) + 0.5
    t_step = (np.arange(hourly.size * sph) + 0.5) / sph
    return np.interp(t_step, t_hour, hourly).reshape(DAYS_PER_YEAR, 24 * sph)


def sample_days(cfg: dict) -> np.ndarray:
    """n_days day ids, stratified over equal four-week blocks, seeded."""
    dc = cfg["days"]
    n, nb = dc["n_days"], dc["blocks"]
    per = DAYS_PER_YEAR // nb
    rng = np.random.default_rng(cfg["seed"])
    counts = np.full(nb, n // nb)
    counts[rng.choice(nb, size=n % nb, replace=False)] += 1
    days = [b * per + rng.choice(per, size=c, replace=False) for b, c in enumerate(counts)]
    return np.sort(np.concatenate(days)).astype(int)


# --------------------------------------------------------------------------------- network


class Case:
    """One IEEE case: measurement function h, AC Jacobian, WLS estimator, chi-square test.

    The state vector is x = (non-reference angles, all voltage magnitudes), internal bus
    order, in radians and per unit. Measurements are per unit on the case's MVA base.
    """

    def __init__(self, name: str, cfg: dict):
        import pandapower as pp
        import pandapower.networks as pn

        self.name, self.cfg, self.pp = name, cfg, pp
        self.net = getattr(pn, name)()
        pp.runpp(self.net, numba=False)
        i = self.net._ppc["internal"]
        self.Ybus, self.Yf, self.Yt = i["Ybus"].tocsr(), i["Yf"].tocsr(), i["Yt"].tocsr()
        self.branch = i["branch"]
        self.Bbus, self.Bf = i["Bbus"].tocsr(), i["Bf"].tocsr()
        self.Pbusinj = np.asarray(i["Pbusinj"]).ravel()
        self.Pfinj = np.asarray(i["Pfinj"]).ravel()
        self.base_mva = float(i["baseMVA"])
        self.lookup = self.net._pd2ppc_lookups["bus"]
        self.n = self.Ybus.shape[0]
        self.nbr = self.Yf.shape[0]
        self.ref = int(np.asarray(i["ref"]).ravel()[0])
        self.nonref = np.array([k for k in range(self.n) if k != self.ref])
        self.f = self.branch[:, 0].real.astype(int)
        self.base_p = self.net.load.p_mw.to_numpy().copy()
        self.base_q = self.net.load.q_mvar.to_numpy().copy()
        self.base_gen = self.net.gen.p_mw.to_numpy().copy()
        self.vm0, self.va0 = self.state_from_net()
        self.theta_ref = float(self.va0[self.ref])
        z0 = self.h(self.vm0, self.va0)
        self.m = z0.size
        self.n_state = self.nonref.size + self.n
        mc = cfg["measurement"]
        sig = np.empty(self.m)
        for sl in self.blocks().values():
            b = np.abs(z0[sl])
            floor = mc["floor_rel"] * np.median(b[b > 1e-9])
            sig[sl] = np.maximum(mc["sigma_rel"] * b, floor)
        self.sigma = sig
        self.w = 1.0 / sig**2
        self.dof = self.m - self.n_state

    # ---- layout
    def blocks(self) -> dict[str, slice]:
        n, b = self.n, self.nbr
        edges = np.cumsum([0, n, n, n, b, b])
        return {k: slice(int(edges[j]), int(edges[j + 1])) for j, k in enumerate(BLOCKS)}

    def feature_names(self) -> list[str]:
        out = []
        for k in BLOCKS:
            unit = "bus" if k in ("vm", "p_inj", "q_inj") else "br"
            size = self.n if unit == "bus" else self.nbr
            out += [f"{k}_{unit}{j}" for j in range(size)]
        return out

    def pack(self, vm: np.ndarray, va: np.ndarray) -> np.ndarray:
        return np.concatenate([va[self.nonref], vm])

    def unpack(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        va = np.full(self.n, self.theta_ref)
        va[self.nonref] = x[: self.nonref.size]
        return x[self.nonref.size :].copy(), va

    def state_from_net(self) -> tuple[np.ndarray, np.ndarray]:
        rb = self.net.res_bus.loc[self.net.bus.index]
        pos = self.lookup[rb.index.to_numpy()]
        vm, va = np.empty(self.n), np.empty(self.n)
        vm[pos] = rb.vm_pu.to_numpy()
        va[pos] = np.deg2rad(rb.va_degree.to_numpy())
        return vm, va

    # ---- physics
    def h(self, vm: np.ndarray, va: np.ndarray) -> np.ndarray:
        V = vm * np.exp(1j * va)
        S = V * np.conj(self.Ybus @ V)
        Sf = V[self.f] * np.conj(self.Yf @ V)
        return np.concatenate([vm, S.real, S.imag, Sf.real, Sf.imag])

    def h_x(self, x: np.ndarray) -> np.ndarray:
        return self.h(*self.unpack(x))

    def h_rows(self, xs: np.ndarray) -> np.ndarray:
        """h for many states at once: xs (rows, n_state) -> (rows, m)."""
        nr = self.nonref.size
        va = np.full((xs.shape[0], self.n), self.theta_ref)
        va[:, self.nonref] = xs[:, :nr]
        vm = xs[:, nr:]
        V = vm * np.exp(1j * va)
        S = V * np.conj((self.Ybus @ V.T).T)
        Sf = V[:, self.f] * np.conj((self.Yf @ V.T).T)
        return np.hstack([vm, S.real, S.imag, Sf.real, Sf.imag])

    def jac(self, vm: np.ndarray, va: np.ndarray) -> np.ndarray:
        """dh/dx, (m, n_state), x = (non-reference angles, voltage magnitudes)."""
        from pandapower.pypower.dSbr_dV import dSbr_dV
        from pandapower.pypower.dSbus_dV import dSbus_dV

        V = vm * np.exp(1j * va)
        dS_dVm, dS_dVa = dSbus_dV(self.Ybus, V)
        dSf_dVa, dSf_dVm, *_ = dSbr_dV(self.branch, self.Yf, self.Yt, V)
        dS_dVm, dS_dVa = dS_dVm.toarray(), dS_dVa.toarray()
        dSf_dVa, dSf_dVm = np.asarray(dSf_dVa.todense()), np.asarray(dSf_dVm.todense())
        nr = self.nonref
        return np.vstack(
            [
                np.hstack([np.zeros((self.n, nr.size)), np.eye(self.n)]),
                np.hstack([dS_dVa.real[:, nr], dS_dVm.real]),
                np.hstack([dS_dVa.imag[:, nr], dS_dVm.imag]),
                np.hstack([dSf_dVa.real[:, nr], dSf_dVm.real]),
                np.hstack([dSf_dVa.imag[:, nr], dSf_dVm.imag]),
            ]
        )

    def jac_x(self, x: np.ndarray) -> np.ndarray:
        return self.jac(*self.unpack(x))

    def threshold(self, alpha: float) -> float:
        return float(chi2.ppf(1.0 - alpha, self.dof))

    def dc_start(self, z: np.ndarray) -> np.ndarray:
        """Measured Vm and DC weighted-least-squares angles from the P channels."""
        bl = self.blocks()
        nr = self.nonref
        A = np.vstack([self.Bbus[:, nr].toarray(), self.Bf[:, nr].toarray()])
        ref_b = self.Bbus[:, [self.ref]].toarray()[:, 0] * self.theta_ref
        ref_f = self.Bf[:, [self.ref]].toarray()[:, 0] * self.theta_ref
        rhs = np.concatenate(
            [z[bl["p_inj"]] - ref_b - self.Pbusinj, z[bl["p_from"]] - ref_f - self.Pfinj]
        )
        wp = np.concatenate([self.w[bl["p_inj"]], self.w[bl["p_from"]]])
        th = np.linalg.solve(A.T @ (wp[:, None] * A), A.T @ (wp * rhs))
        return np.concatenate([th, z[bl["vm"]]])

    def wls(self, z: np.ndarray) -> tuple[float, bool, np.ndarray]:
        """(J, converged, x_hat). A non-converged estimate returns J = inf (an alarm)."""
        est = self.cfg["estimator"]
        x = self.dc_start(z)
        for _ in range(est["max_iter"]):
            H = self.jac_x(x)
            r = z - self.h_x(x)
            dx = np.linalg.solve(H.T @ (self.w[:, None] * H), H.T @ (self.w * r))
            x = x + dx
            if np.max(np.abs(dx)) < est["tol"]:
                r = z - self.h_x(x)
                return float(np.sum(self.w * r * r)), True, x
        return float("inf"), False, x

    def control_h(self) -> float:
        """Largest |h(x) - pandapower| over injections and from-end flows, MW / Mvar."""
        net = self.net
        pp_res = self.h(self.vm0, self.va0) * self.base_mva
        bl = self.blocks()
        rb = net.res_bus.loc[net.bus.index]
        pos = self.lookup[rb.index.to_numpy()]
        p_inj, q_inj = np.empty(self.n), np.empty(self.n)
        p_inj[pos] = -rb.p_mw.to_numpy()
        q_inj[pos] = -rb.q_mvar.to_numpy()
        if len(net.shunt):  # shunts sit inside Ybus; pandapower books them as bus consumption
            sh = net.res_shunt.groupby(net.shunt.bus).q_mvar.sum()
            q_inj[self.lookup[sh.index.to_numpy()]] += sh.to_numpy()
        pf = np.concatenate([net.res_line.p_from_mw.to_numpy(), net.res_trafo.p_hv_mw.to_numpy()])
        qf = np.concatenate(
            [net.res_line.q_from_mvar.to_numpy(), net.res_trafo.q_hv_mvar.to_numpy()]
        )
        ref = {"p_inj": p_inj, "q_inj": q_inj, "p_from": pf, "q_from": qf}
        return float(max(np.max(np.abs(pp_res[bl[k]] - v)) for k, v in ref.items()))

    def set_operating_point(self, mult: np.ndarray) -> None:
        self.net.load.p_mw = self.base_p * mult
        self.net.load.q_mvar = self.base_q * mult
        self.net.gen.p_mw = self.base_gen * (mult @ self.base_p) / self.base_p.sum()


# ------------------------------------------------------------------------------ generation


def _draw_c(rng: np.random.Generator, n_angles: int, scale: dict) -> np.ndarray:
    k = int(rng.integers(1, n_angles // 2 + 1))
    idx = rng.choice(n_angles, size=k, replace=False)
    if scale["dist"] == "uniform":
        v = rng.uniform(scale["low"], scale["high"], size=k) * rng.choice([-1.0, 1.0], size=k)
    elif scale["dist"] == "normal":
        v = rng.normal(0.0, scale["std"], size=k)
    else:
        raise ValueError(f"unknown scale distribution {scale['dist']}")
    c = np.zeros(n_angles)
    c[idx] = v
    return c


def generate_day(case_name: str, day: int, factors: np.ndarray, cfg: dict) -> dict:
    """All snapshots of one day; deterministic in (cfg seed, case, day)."""
    case = Case(case_name, cfg)
    rng = np.random.default_rng(np.random.SeedSequence([cfg["seed"], CASE_ID[case_name], day]))
    ep, lm, est = cfg["episodes"], cfg["load_model"], cfg["estimator"]
    steps = cfg["days"]["steps_per_day"]
    if ep["length"] * ep["per_day"] != steps or factors.shape[0] != steps:
        raise ValueError("episode layout does not tile the day")
    kinds = rng.permutation([k for k, c in ep["composition"].items() for _ in range(c)])
    offset = {k: int(rng.integers(len(SCALES))) for k in ("stealthy", "nonstealthy")}
    n_load = case.base_p.size
    phi, sd = lm["ar1_phi"], lm["ar1_sigma"]
    eta = rng.normal(0.0, sd, size=n_load)
    thr = case.threshold(est["alpha"])
    n_ang = case.nonref.size
    bl = case.blocks()
    out = {
        k: []
        for k in (
            "X", "cls", "scale", "k", "episode", "step", "J_clean", "J_row",
            "conv_clean", "conv_row", "x_hat", "x_true",
        )
    }  # fmt: skip
    seen = dict.fromkeys(("stealthy", "nonstealthy"), 0)
    for e, kind in enumerate(kinds):
        kind = str(kind)
        scale_i, c, perm = -1, None, None
        if kind != "normal":
            scale_i = (offset[kind] + seen[kind]) % len(SCALES)
            seen[kind] += 1
            c = _draw_c(rng, n_ang, cfg["attack"]["scales"][SCALES[scale_i]])
        if kind == "nonstealthy":
            perm = {
                b: (
                    rng.permutation(sl.stop - sl.start),
                    rng.choice([-1.0, 1.0], sl.stop - sl.start),
                )
                for b, sl in bl.items()
            }
        for t in range(e * ep["length"], (e + 1) * ep["length"]):
            eta = phi * eta + rng.normal(0.0, sd * np.sqrt(1.0 - phi**2), size=n_load)
            case.set_operating_point(factors[t] * (1.0 + eta))
            case.pp.runpp(case.net, numba=False, init="results")
            if not case.net.converged:
                raise RuntimeError(f"{case_name} day {day} step {t}: power flow did not converge")
            vm, va = case.state_from_net()
            z_true = case.h(vm, va)
            z_clean = z_true + rng.normal(0.0, case.sigma)
            a = np.zeros(case.m)
            if kind != "normal":
                va_c = va.copy()
                va_c[case.nonref] += c
                a = case.h(vm, va_c) - z_true
            if kind == "nonstealthy":
                a_ns = np.empty_like(a)
                for b, sl in bl.items():
                    p, s = perm[b]
                    a_ns[sl] = a[sl][p] * s
                a = a_ns
            z = z_clean + a
            J0, ok0, xh0 = case.wls(z_clean)
            if kind == "normal":
                J1, ok1, xh = J0, ok0, xh0
            else:
                J1, ok1, xh = case.wls(z)
            out["X"].append(z)
            out["cls"].append(KIND_CODE[kind])
            out["scale"].append(scale_i)
            out["k"].append(0 if c is None else int(np.count_nonzero(c)))
            out["episode"].append(e)
            out["step"].append(t)
            out["J_clean"].append(J0)
            out["J_row"].append(J1)
            out["conv_clean"].append(ok0)
            out["conv_row"].append(ok1)
            out["x_hat"].append(xh)
            out["x_true"].append(case.pack(vm, va))
    res = {k: np.asarray(v) for k, v in out.items()}
    res["day"] = np.full(steps, day)
    res["threshold"] = thr
    return res


def _day_job(args):
    return generate_day(*args)


def day_fold_splits(day: np.ndarray, fold: int, n_folds: int, seed: int) -> dict[str, np.ndarray]:
    """Disjoint day folds (E-DF's rule): days permuted once with default_rng(seed) into
    n_folds folds; fold k is test, fold k+1 (mod n_folds) validation, the rest train."""
    days = np.unique(day)
    if days.size % n_folds:
        raise ValueError(f"{days.size} days do not divide into {n_folds} folds")
    folds = np.random.default_rng(seed).permutation(days).reshape(n_folds, -1)
    cut = {"test": folds[fold % n_folds], "val": folds[(fold + 1) % n_folds]}
    cut["train"] = np.setdiff1d(days, np.concatenate([cut["test"], cut["val"]]))
    return {s: np.flatnonzero(np.isin(day, cut[s])) for s in SIDES}


def _rates(dec: np.ndarray, mask: np.ndarray) -> float | None:
    return float(dec[mask].mean()) if mask.any() else None


def generate_variant(
    variant: str,
    cfg: dict | None = None,
    out_dir: Path | None = None,
    workers: int = 1,
    days: np.ndarray | None = None,
    git: str | None = None,
) -> dict:
    """Generate one variant, write its four files, and return its meta (controls included).

    `git` is the code state the data come from; the caller captures it at launch (a run can
    outlive later commits), and it defaults to the state when this variant starts.
    """
    import pandapower

    git = git or git_describe()
    cfg = cfg or fdia_config()
    vcfg = cfg["variants"][variant]
    case_name = vcfg["case"]
    out_dir = (
        out_dir or REPO_ROOT / load_yaml(REPO_ROOT / "configs" / "data.yaml")["paths"]["processed"]
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    days = sample_days(cfg) if days is None else np.asarray(days, int)
    fac = step_factors(cfg)[:, : cfg["days"]["steps_per_day"]]
    jobs = [(case_name, int(d), fac[d], cfg) for d in days]
    if workers == 1:
        parts = [_day_job(j) for j in jobs]
    else:
        with ProcessPoolExecutor(workers) as ex:
            parts = list(ex.map(_day_job, jobs))
    keys = [k for k in parts[0] if k != "threshold"]
    allr = {k: np.concatenate([p[k] for p in parts]) for k in keys}
    case = Case(case_name, cfg)
    est = cfg["estimator"]
    alphas = [est["alpha"], *est["alpha_reported"]]

    cls = allr["cls"]
    if not allr["conv_clean"].all():
        bad = np.flatnonzero(~allr["conv_clean"])
        raise RuntimeError(f"{variant}: clean estimate did not converge on rows {bad[:10]}")
    controls = {"h_max_abs_err_mw": case.control_h()}
    bdd = {}
    for a in alphas:
        th = case.threshold(a)
        d_clean = allr["J_clean"] <= th
        d_row = allr["conv_row"] & (allr["J_row"] <= th)
        row = {"threshold": th}
        for name, code in KIND_CODE.items():
            m = cls == code
            row[f"pass_{name}"] = _rates(d_row, m)
            row[f"pass_{name}_paired_clean"] = _rates(d_clean, m)
            row[f"agree_{name}"] = _rates(d_row == d_clean, m)
            for si, s in enumerate(SCALES):
                row[f"pass_{name}_{s}"] = _rates(d_row, m & (allr["scale"] == si))
        row["nonconverged_attacked"] = int((~allr["conv_row"] & (cls != NORMAL)).sum())
        bdd[str(a)] = row
    controls["paired_decision_agreement_stealthy"] = bdd[str(est["alpha"])]["agree_stealthy"]

    keep = np.isin(cls, vcfg["classes"])
    y = (cls[keep] == STEALTHY).astype(np.int64)
    X = allr["X"][keep]
    sp = cfg["split"]
    splits = row_split(y, sp["ratios"], sp["seed"])
    np.savez_compressed(out_dir / f"{variant}.npz", X=X.astype(np.float32), y=y)
    np.savez_compressed(
        out_dir / f"splits_{variant}.npz", **splits, split_seed=np.array([sp["seed"]])
    )
    np.savez_compressed(
        out_dir / f"{variant}_aux.npz",
        **{k: v[keep] for k, v in allr.items() if k != "X"},
        X_float64=X,
        sigma=case.sigma,
    )
    meta = {
        "variant": variant,
        "case": case_name,
        "generator": "qgridbench.data.fdia_synth",
        "git": git,
        "pandapower": pandapower.__version__,
        "config_sha256": _sha256_text(json.dumps(cfg, sort_keys=True)),
        "config_snapshot": cfg,
        "seed": cfg["seed"],
        "days": days.tolist(),
        "n_rows_generated": int(cls.size),
        "class_counts_generated": {
            int(k): int(v) for k, v in zip(*np.unique(cls, return_counts=True))
        },
        "n_rows": int(y.size),
        "class_counts": {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))},
        "n_features": int(X.shape[1]),
        "n_state": case.n_state,
        "chi2_dof": case.dof,
        "feature_names": case.feature_names(),
        "split_sizes": {s: int(len(v)) for s, v in splits.items()},
        "split_seed": sp["seed"],
        "bdd": bdd,
        "controls": controls,
    }
    (out_dir / f"meta_{variant}.json").write_text(json.dumps(meta, indent=2))
    return meta


def _sha256_text(s: str) -> str:
    import hashlib

    return hashlib.sha256(s.encode()).hexdigest()


def check_controls(meta: dict, cfg: dict) -> None:
    """Raise if a blocking generator control fails (meta is already on disk)."""
    c, lim = meta["controls"], cfg["controls"]
    if c["h_max_abs_err_mw"] > lim["h_max_abs_err_mw"]:
        raise RuntimeError(f"h control failed: {c['h_max_abs_err_mw']}")
    agree = c["paired_decision_agreement_stealthy"]
    if agree is None or agree < lim["paired_decision_agreement_min"]:
        raise RuntimeError(f"paired decision agreement {agree} below the registered minimum")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="*", default=None)
    ap.add_argument("--workers", type=int, required=True)
    ap.add_argument("--threads", type=int, required=True, help="BLAS threads per worker")
    args = ap.parse_args()
    for v in THREAD_VARS:  # inherited by spawned workers before they import numpy
        os.environ[v] = str(args.threads)

    import psutil

    from qgridbench.utils.run_tracking import RunDir

    cfg = fdia_config()
    variants = args.variants or list(cfg["variants"])
    avail = psutil.virtual_memory().available / 2**30
    need = args.workers * cfg["compute"]["worker_peak_rss_gb"]
    if need > 0.8 * avail:
        raise MemoryError(
            f"{args.workers} workers need {need:.2f} GB; 80% of free is {0.8 * avail:.2f}"
        )
    git = git_describe()  # the launch state, recorded in every variant's meta
    with RunDir("fdia_synth_generate", config={"variants": variants, **vars(args)}) as run:
        metas = {}
        for v in variants:
            meta = generate_variant(v, cfg, workers=args.workers, git=git)
            metas[v] = {k: meta[k] for k in ("n_rows", "class_counts", "bdd", "controls")}
            run.metrics.update(
                {f"{v}_paired_agreement": meta["controls"]["paired_decision_agreement_stealthy"]}
            )
            check_controls(meta, cfg)
        run.write_json("generate_summary.json", metas)
        print(json.dumps({v: m["controls"] for v, m in metas.items()}, indent=2))
        print(f"-> {run.path}")


if __name__ == "__main__":
    main()
