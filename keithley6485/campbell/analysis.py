"""Analyses across Campbell runs (one HDF5 file per run), following Will's plan.

- ``fit_charge``: in each (range, NPLC) group of photocurrent runs, fit
  ``T_int * V = a + e <I> + b <I>^2`` (V = Var, sdiff or the band periodogram).
  ``a`` is the floor, the slope is e, ``b`` the laser's classical noise.
- ``resistor_control``: pair each photocurrent run with a battery + resistor run
  at about the same current, range and NPLC:
  ``e = T_int (V_pd - V_res + 2kT/(R T_int)) / <I_pd>``.
- ``johnson_teff``: zero-volt resistor runs; ``V = c + d/R + f/R^2`` and
  ``T_eff = 2 k T_R / d``, which gives T_eff without e.
- ``nplc_table``: T_int * V / <I> per NPLC at one range, for the check that it
  does not depend on NPLC.

Every number is recomputed from the raw arrays in the files with
``campbell_stats``, so a different band or estimator is one argument away.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np

from .stats import E_CHARGE, campbell_stats

K_B = 1.380649e-23        # J/K, exact in SI
ESTIMATORS = {"naive": ("var_a2", "var_rel_err"),
              "sdiff": ("sdiff_var_a2", "sdiff_rel_err"),
              "band": ("psd_band_a2", "psd_band_rel_err")}
KINDS = ("photocurrent", "dark", "zero-check floor", "resistor control", "johnson")


def _num(v) -> float:
    try:
        return float(str(v).strip().split()[0])
    except (ValueError, IndexError):
        return float("nan")


@dataclass
class Run:
    path: str
    run_id: str
    kind: str
    range_a: float
    nplc: float
    f_line: float
    zero_check: bool
    stats: dict
    notes: dict = field(default_factory=dict)

    @property
    def t_int(self) -> float:
        return self.nplc / self.f_line

    @property
    def mean(self) -> float:
        return self.stats.get("mean_a", float("nan"))

    def v(self, est: str) -> tuple[float, float]:
        k, r = ESTIMATORS[est]
        v = self.stats.get(k, float("nan"))
        return v, abs(v) * self.stats.get(r, float("nan"))

    @property
    def resistor_ohm(self) -> float:
        return _num(self.notes.get("resistor_ohm", ""))

    @property
    def temperature_k(self) -> float:
        t = _num(self.notes.get("temperature_k", ""))
        return t if np.isfinite(t) else 295.0

    @property
    def nd(self) -> str:
        return str(self.notes.get("nd", ""))


def load_runs(paths: Iterable[str], band_hz=(1.0, 10.0)) -> list[Run]:
    from .store import read_run

    runs = []
    for p in paths:
        r = read_run(p)
        s = r["settings"] or {}
        a = r["attrs"]
        nplc = float(s.get("nplc") or a.get("set_nplc"))
        flf = float(s.get("line_freq_hz") or a.get("set_line_freq_hz") or 60.0)
        zc = bool(s.get("zero_check", a.get("set_zero_check", False)))
        notes = r["notes"] or {}
        kind = notes.get("kind") or ("zero-check floor" if zc else "photocurrent")
        st = campbell_stats(r["i_a"], r["t_s"], nplc, flf, burst=r["burst"], status=r["status"], band_hz=band_hz)
        runs.append(Run(str(p), r["run_id"], kind, float(s.get("range_a") or a.get("set_range_a") or float("nan")),
                        nplc, flf, zc, st, notes))
    return runs


def _group_key(r: Run) -> tuple:
    return (round(math.log10(r.range_a), 3) if r.range_a > 0 else None, round(r.nplc, 6))


def _poly_fit(x: np.ndarray, y: np.ndarray, sy: np.ndarray, deg: int):
    """Weighted polynomial fit y = c0 + c1 x (+ c2 x^2) on scaled variables (x ~ 1e-9 A would make the
    normal equations singular). Returns coefficients, their errors (scaled by sqrt(chi2/dof) when > 1),
    chi2 and dof."""
    xs = float(np.max(np.abs(x))) or 1.0
    ys = float(np.max(np.abs(y))) or 1.0
    u, v, sv = x / xs, y / ys, sy / ys
    X = np.column_stack([u ** k for k in range(deg + 1)])
    w = 1.0 / sv
    A = X * w[:, None]
    coef, *_ = np.linalg.lstsq(A, v * w, rcond=None)
    cov = np.linalg.inv(A.T @ A)
    resid = (v - X @ coef) / sv
    dof = len(v) - (deg + 1)
    chi2 = float(resid @ resid)
    scale = math.sqrt(chi2 / dof) if dof > 0 and chi2 > dof else 1.0
    err = np.sqrt(np.diag(cov)) * scale
    conv = np.array([ys / xs ** k for k in range(deg + 1)])
    return coef * conv, err * conv, chi2, dof, scale


def fit_charge(runs: list[Run], est: str = "naive", quadratic: Optional[bool] = None) -> list[dict]:
    """Fit T_int V = a + e I (+ b I^2) per (range, NPLC) group of photocurrent runs.

    ``quadratic`` None: use the I^2 term when the group has at least 4 runs.
    Errors are scaled by sqrt(chi2/dof) when that is above 1.
    """
    groups: dict = {}
    for r in runs:
        if r.kind == "photocurrent" and not r.zero_check and np.isfinite(r.mean) and r.mean > 0:
            groups.setdefault(_group_key(r), []).append(r)
    out = []
    for key, rs in sorted(groups.items(), key=lambda kv: (kv[0][0] or 0, kv[0][1])):
        rs = sorted(rs, key=lambda r: r.mean)
        I = np.array([r.mean for r in rs])
        vv = np.array([r.v(est) for r in rs])
        t = rs[0].t_int
        y, sy = t * vv[:, 0], t * vv[:, 1]
        res = {"range_a": rs[0].range_a, "nplc": rs[0].nplc, "t_int_s": t, "estimator": est,
               "n_runs": len(rs), "runs": [r.run_id for r in rs], "current_a": I.tolist(),
               "t_v": y.tolist(), "t_v_err": sy.tolist()}
        use_q = (len(rs) >= 4) if quadratic is None else quadratic
        if len(rs) < (3 if use_q else 2) or not np.all(np.isfinite(sy)) or np.any(sy <= 0):
            res["note"] = "too few runs (or no errors) for a fit"
            out.append(res)
            continue
        coef, err, chi2, dof, scale = _poly_fit(I, y, sy, 2 if use_q else 1)
        res.update(a_c_a=float(coef[0]), a_err=float(err[0]), e_c=float(coef[1]), e_err_c=float(err[1]),
                   e_over_e=float(coef[1] / E_CHARGE), e_over_e_err=float(err[1] / E_CHARGE),
                   b_s=float(coef[2]) if use_q else 0.0, b_err=float(err[2]) if use_q else 0.0,
                   quadratic=use_q, chi2=chi2, dof=dof, err_scale=scale)
        out.append(res)
    return out


def resistor_control(runs: list[Run], est: str = "naive", max_rel_diff: float = 0.1) -> list[dict]:
    """Pair photocurrent runs with resistor-control runs (same range and NPLC, mean within 10 %)."""
    pds = [r for r in runs if r.kind == "photocurrent" and not r.zero_check and r.mean > 0]
    ctl = [r for r in runs if r.kind == "resistor control" and not r.zero_check and r.mean > 0]
    out = []
    for p in pds:
        cands = [c for c in ctl if _group_key(c) == _group_key(p) and abs(c.mean / p.mean - 1) <= max_rel_diff]
        if not cands:
            continue
        c = min(cands, key=lambda c: abs(c.mean / p.mean - 1))
        vp, sp = p.v(est)
        vc, sc = c.v(est)
        R = c.resistor_ohm
        johnson = 2 * K_B * c.temperature_k / (R * p.t_int) if np.isfinite(R) and R > 0 else float("nan")
        shot = vp - vc + (johnson if np.isfinite(johnson) else 0.0)
        e = p.t_int * shot / p.mean
        err = p.t_int * math.hypot(sp, sc) / p.mean
        out.append({"photocurrent_run": p.run_id, "control_run": c.run_id, "range_a": p.range_a, "nplc": p.nplc,
                    "current_a": p.mean, "control_current_a": c.mean, "resistor_ohm": R,
                    "johnson_var_a2": johnson, "var_pd_a2": vp, "var_ctl_a2": vc,
                    "e_c": e, "e_err_c": err, "e_over_e": e / E_CHARGE, "e_over_e_err": err / E_CHARGE,
                    "note": "" if np.isfinite(johnson) else "no resistor_ohm in the control run's notes: Johnson term left out"})
    return out


def johnson_teff(runs: list[Run], est: str = "naive") -> list[dict]:
    """T_eff from zero-volt resistor runs: V = c + d/R (+ f/R^2), T_eff = 2 k T_R / d, per (range, NPLC)."""
    groups: dict = {}
    for r in runs:
        if r.kind == "johnson" and not r.zero_check and np.isfinite(r.resistor_ohm) and r.resistor_ohm > 0:
            groups.setdefault(_group_key(r), []).append(r)
    out = []
    for key, rs in groups.items():
        G = np.array([1.0 / r.resistor_ohm for r in rs])
        vv = np.array([r.v(est) for r in rs])
        T_R = float(np.mean([r.temperature_k for r in rs]))
        res = {"range_a": rs[0].range_a, "nplc": rs[0].nplc, "t_int_s": rs[0].t_int, "n_runs": len(rs),
               "resistor_ohm": [r.resistor_ohm for r in rs], "var_a2": vv[:, 0].tolist(), "temperature_k": T_R}
        quad = len(rs) >= 4
        if len(rs) < 3 or np.any(vv[:, 1] <= 0):
            res["note"] = "need at least 3 resistors"
            out.append(res)
            continue
        coef, err, chi2, dof, scale = _poly_fit(G, vv[:, 0], vv[:, 1], 2 if quad else 1)
        d, de = float(coef[1]), float(err[1])
        t_eff = 2 * K_B * T_R / d if d > 0 else float("nan")
        res.update(c_a2=float(coef[0]), d_a2_ohm=d, d_err=de, quadratic=quad, chi2=chi2, dof=dof,
                   t_eff_s=t_eff, t_eff_err_s=abs(t_eff) * de / abs(d) if d else float("nan"),
                   t_eff_over_t_int=t_eff / rs[0].t_int if d > 0 else float("nan"))
        out.append(res)
    return out


def nplc_table(runs: list[Run], est: str = "naive") -> list[dict]:
    """T_int V / <I> (in units of e) for every photocurrent run, sorted by range, ND/current and NPLC."""
    rows = []
    for r in runs:
        if r.kind != "photocurrent" or r.zero_check or not r.mean > 0:
            continue
        v, sv = r.v(est)
        rows.append({"run": r.run_id, "range_a": r.range_a, "nplc": r.nplc, "nd": r.nd, "current_a": r.mean,
                     "t_v_over_i_e": r.t_int * v / r.mean / E_CHARGE,
                     "err_e": r.t_int * sv / r.mean / E_CHARGE, "acf1": r.stats.get("acf", [float("nan")])[0]})
    return sorted(rows, key=lambda x: (x["range_a"], x["current_a"], x["nplc"]))
