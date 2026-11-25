"""Campbell (shot-noise) statistics of a series of ammeter readings.

Pure numpy: this module must not import the instrument driver, so an analysis
can recompute everything from a saved file on any machine.

Two times, never to be confused:

- ``T_int = NPLC / f_line``, the integration aperture. Campbell's theorem for
  an ideal boxcar of length T_int gives ``Var(I) = e <I> / T_int``, so the
  naive charge is ``T_int * Var / <I>``.
- ``dt``, the reading interval from the instrument's time stamps. It sets the
  frequency axis of the spectra.

The readings come in contiguous segments (one per buffer burst). Every
estimator that looks at neighbouring readings (successive differences,
autocorrelation, periodogram, Allan variance) works inside segments only, so
the gaps between bursts never enter it.
"""

from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence

import numpy as np

E_CHARGE = 1.602176634e-19   # C, exact in SI

# The 6485 returns +9.9E37 for an overflow or overvoltage reading and
# +9.91E37 for NAN (6485-901-01 Rev. A, FORMat:ELEMents, p. 13-6).
OVERFLOW_A = 9.9e37

# Status word bits (same page).
STATUS_OVERFLOW = 1 << 0
STATUS_FILTER = 1 << 1
STATUS_MATH = 1 << 2
STATUS_NULL = 1 << 3
STATUS_LIMITS = 1 << 4
STATUS_OVERVOLTAGE = 1 << 7
STATUS_ZERO_CHECK = 1 << 9
STATUS_ZERO_CORRECT = 1 << 10

DEFAULT_BAND_HZ = (1.0, 10.0)
DEFAULT_SHAPE_EDGES_HZ = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0)


def invalid_mask(i_a: np.ndarray, status: Optional[np.ndarray] = None) -> np.ndarray:
    """True for readings that are not measurements: overflow, overvoltage, NAN, non-finite."""
    i_a = np.asarray(i_a, dtype=float)
    bad = ~np.isfinite(i_a) | (np.abs(i_a) >= 0.99 * OVERFLOW_A)
    if status is not None:
        st = np.asarray(status).astype(np.int64)
        bad |= (st & (STATUS_OVERFLOW | STATUS_OVERVOLTAGE)) != 0
    return bad


def segment_bounds(t_s: np.ndarray, burst: Optional[np.ndarray] = None,
                   valid: Optional[np.ndarray] = None,
                   gap_factor: float = 5.0) -> list[tuple[int, int]]:
    """Half-open index ranges [a, b) of contiguous, valid readings.

    A new segment starts where the burst index changes, where the time stamp
    goes back (each buffer's absolute time stamps start at 0), where a step is
    longer than ``gap_factor`` times the median step, and around every invalid
    reading.
    """
    t = np.asarray(t_s, dtype=float)
    n = len(t)
    if n == 0:
        return []
    valid = np.ones(n, bool) if valid is None else np.asarray(valid, bool)
    brk = np.zeros(n, bool)
    brk[0] = True
    if n > 1:
        d = np.diff(t)
        pos = d[d > 0]
        med = float(np.median(pos)) if len(pos) else 0.0
        cut = d <= 0
        if med > 0:
            cut |= d > gap_factor * med
        brk[1:] |= cut
        if burst is not None:
            b = np.asarray(burst)
            brk[1:] |= b[1:] != b[:-1]
    seg = []
    start = None
    for k in range(n):
        if brk[k] and start is not None:
            seg.append((start, k))
            start = None
        if not valid[k]:
            if start is not None:
                seg.append((start, k))
                start = None
            continue
        if start is None:
            start = k
    if start is not None:
        seg.append((start, n))
    return [(a, b) for a, b in seg if b > a]


def periodogram(x: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Periodogram of the linearly detrended series in variance-per-reading units.

    White readings of variance s^2 give P = s^2 on average (``psd_band`` of the
    205 K Campbell analysis uses the same normalisation).
    """
    x = np.asarray(x, dtype=float)
    k = np.arange(len(x))
    y = x - np.polyval(np.polyfit(k, x, 1), k)
    return np.fft.rfftfreq(len(y), dt), np.abs(np.fft.rfft(y)) ** 2 / len(y)


def segment_spectra(i_a, t_s, burst=None, status=None, min_len: int = 16):
    """Periodograms of every contiguous segment: list of (f_hz, P_a2) pairs."""
    i_a = np.asarray(i_a, dtype=float)
    t_s = np.asarray(t_s, dtype=float)
    valid = ~invalid_mask(i_a, status)
    out = []
    for a, b in segment_bounds(t_s, burst, valid):
        if b - a < min_len:
            continue
        # mean step over the segment: the 6485's stamps tick in 1/1024 s, so single steps are quantized
        dt = float((t_s[b - 1] - t_s[a]) / (b - a - 1))
        if dt > 0:
            out.append(periodogram(i_a[a:b], dt))
    return out


def binned_spectrum(i_a, t_s, burst=None, status=None, n_bins: int = 60):
    """Mean periodogram of all segments in log-spaced frequency bins (for display).

    Returns (f_centre_hz, P_a2, count); empty arrays when there is too little data.
    """
    spectra = segment_spectra(i_a, t_s, burst, status)
    if not spectra:
        return np.array([]), np.array([]), np.array([], int)
    f = np.concatenate([s[0][1:] for s in spectra])
    p = np.concatenate([s[1][1:] for s in spectra])
    if len(f) == 0:
        return np.array([]), np.array([]), np.array([], int)
    edges = np.geomspace(f.min(), f.max() * 1.0001, n_bins + 1)
    idx = np.clip(np.digitize(f, edges) - 1, 0, n_bins - 1)
    cnt = np.bincount(idx, minlength=n_bins)
    tot = np.bincount(idx, weights=p, minlength=n_bins)
    keep = cnt > 0
    centre = np.sqrt(edges[:-1] * edges[1:])
    return centre[keep], tot[keep] / cnt[keep], cnt[keep]


def _f(x) -> float:
    return float(x) if x is not None and np.isfinite(x) else float("nan")


def campbell_stats(i_a: Sequence[float], t_s: Sequence[float], nplc: float, f_line: float,
                   *, burst: Optional[Sequence[int]] = None,
                   status: Optional[Sequence[int]] = None,
                   band_hz: tuple[float, float] = DEFAULT_BAND_HZ,
                   max_lag: int = 4,
                   shape_edges_hz: Iterable[float] = DEFAULT_SHAPE_EDGES_HZ) -> dict:
    """Statistics for a Campbell run, from the raw arrays alone.

    Parameters
    ----------
    i_a, t_s : readings (A) and the instrument's time stamps (s). Each buffer
        burst's time stamps start at 0, which is how segments are found when
        ``burst`` is not given.
    nplc, f_line : integration rate and line frequency, as read back from the
        meter. ``T_int = nplc / f_line``.
    burst : optional burst index per reading.
    status : optional status word per reading; overflow and overvoltage
        readings are left out of every statistic and counted.
    band_hz : band of the mean periodogram ``psd_band``.

    Returns a flat dict of floats, ints and lists (JSON and HDF5 friendly).
    Errors (``*_err``) are the white-noise expectations: they are right only
    when the readings are uncorrelated, which the autocorrelation shows.
    """
    i_a = np.asarray(i_a, dtype=float)
    t_s = np.asarray(t_s, dtype=float)
    if i_a.shape != t_s.shape:
        raise ValueError(f"i_a and t_s differ in shape: {i_a.shape} vs {t_s.shape}")
    st = None if status is None else np.asarray(status).astype(np.int64)
    bad = invalid_mask(i_a, st)
    valid = ~bad
    segs = segment_bounds(t_s, burst, valid)
    t_int = float(nplc) / float(f_line)
    out: dict = {
        "n_total": int(len(i_a)),
        "n_invalid": int(bad.sum()),
        "n_overflow": int(np.sum(np.abs(i_a) >= 0.99 * OVERFLOW_A)
                          if st is None else np.sum(((st & STATUS_OVERFLOW) != 0)
                                                    | (np.abs(i_a) >= 0.99 * OVERFLOW_A))),
        "n_overvoltage": 0 if st is None else int(np.sum((st & STATUS_OVERVOLTAGE) != 0)),
        "n_zero_check": 0 if st is None else int(np.sum((st & STATUS_ZERO_CHECK) != 0)),
        "n_filter_flag": 0 if st is None else int(np.sum((st & STATUS_FILTER) != 0)),
        "n_segments": len(segs),
        "nplc": float(nplc),
        "f_line_hz": float(f_line),
        "t_int_s": t_int,
    }
    x = i_a[valid]
    n = int(len(x))
    out["n"] = n
    nan = float("nan")

    # time stamps
    steps = [np.diff(t_s[a:b]) for a, b in segs if b - a > 1]
    steps = np.concatenate(steps) if steps else np.array([])
    dt_med = float(np.median(steps)) if len(steps) else nan
    out["dt_median_s"] = dt_med
    out["dt_mean_s"] = float(steps.mean()) if len(steps) else nan
    out["dt_sd_s"] = float(steps.std()) if len(steps) > 1 else nan
    dt_mean = out["dt_mean_s"]
    out["readings_per_s"] = 1.0 / dt_mean if dt_mean > 0 else nan
    out["duty_t_int_over_dt"] = t_int / dt_mean if dt_mean > 0 else nan

    if n < 2:
        out.update(mean_a=_f(x.mean()) if n else nan, var_a2=nan, sd_a=nan, sem_a=nan)
        return out

    mean = float(x.mean())
    var = float(x.var(ddof=1))
    out["mean_a"] = mean
    out["var_a2"] = var
    out["sd_a"] = math.sqrt(var)
    out["sem_a"] = math.sqrt(var / n)
    out["var_rel_err"] = math.sqrt(2.0 / (n - 1))
    out["shot_sd_nominal_a"] = math.sqrt(E_CHARGE * abs(mean) / t_int)
    uniq = np.unique(x)
    steps_x = np.diff(uniq)
    out["min_step_a"] = float(steps_x.min()) if len(steps_x) else nan

    # per-segment quantities
    sd2 = 0.0
    nd = 0
    num = np.zeros(max_lag)
    den = 0.0
    slopes = []
    for a, b in segs:
        xs = i_a[a:b]
        if len(xs) > 1:
            d = np.diff(xs)
            sd2 += float(np.sum(d * d))
            nd += len(d)
        y = xs - xs.mean()
        den += float(np.sum(y * y))
        for j in range(1, max_lag + 1):
            if len(y) > j:
                num[j - 1] += float(np.sum(y[:-j] * y[j:]))
        if len(xs) > 2:
            tt = t_s[a:b]
            slopes.append((np.polyfit(tt - tt[0], xs, 1)[0], len(xs)))
    sdiff = sd2 / (2.0 * nd) if nd else nan
    out["sdiff_var_a2"] = sdiff
    out["sdiff_rel_err"] = math.sqrt(3.0 / nd) if nd else nan
    # +n_seg/n undoes the mean subtraction's bias (-1/len per segment), as acf() in the 205 K analysis
    acf = (num / den + len(segs) / n) if den > 0 else np.full(max_lag, nan)
    out["acf"] = [float(v) for v in acf]
    out["acf_err"] = 1.0 / math.sqrt(n)
    out["acf_lags"] = list(range(1, max_lag + 1))
    if slopes:
        w = np.array([s[1] for s in slopes], float)
        out["drift_a_per_s"] = float(np.sum(w * np.array([s[0] for s in slopes])) / w.sum())
    else:
        out["drift_a_per_s"] = nan

    # spectra
    lo, hi = float(band_hz[0]), float(band_hz[1])
    spectra = segment_spectra(i_a, t_s, burst, st)
    f_nyq = 0.5 / dt_mean if dt_mean > 0 else nan
    hi_used = min(hi, f_nyq) if np.isfinite(f_nyq) else hi
    band_p = [p[(f >= lo) & (f < hi_used)] for f, p in spectra]
    band_p = np.concatenate(band_p) if band_p else np.array([])
    out["psd_band_hz"] = [lo, float(hi_used)]
    out["psd_band_a2"] = float(band_p.mean()) if len(band_p) else nan
    out["psd_band_bins"] = int(len(band_p))
    out["psd_band_rel_err"] = 1.0 / math.sqrt(len(band_p)) if len(band_p) else nan
    out["f_nyquist_hz"] = float(f_nyq)
    edges = [float(e) for e in shape_edges_hz]
    shape = []
    for e0, e1 in zip(edges[:-1], edges[1:]):
        sel = [p[(f >= e0) & (f < e1)] for f, p in spectra]
        sel = np.concatenate(sel) if sel else np.array([])
        shape.append(float(sel.mean()) if len(sel) else nan)
    out["psd_shape_edges_hz"] = edges
    out["psd_shape_a2"] = shape

    # Allan variance at m = 1, 2, 4, ... readings, inside segments
    ms, av, cnts = [], [], []
    m = 1
    longest = max((b - a for a, b in segs), default=0)
    while 2 * m <= longest:
        s2 = 0.0
        c = 0
        for a, b in segs:
            k = (b - a) // m
            if k < 2:
                continue
            blk = i_a[a:a + k * m].reshape(k, m).mean(axis=1)
            d = np.diff(blk)
            s2 += float(np.sum(d * d))
            c += len(d)
        if c < 4:
            break
        ms.append(m)
        av.append(s2 / (2.0 * c))
        cnts.append(c)
        m *= 2
    out["allan_m"] = ms
    out["allan_var_a2"] = av
    out["allan_m_var_a2"] = [mm * v for mm, v in zip(ms, av)]   # flat (= var) for white readings
    out["allan_count"] = cnts

    # charge estimates: T_int * (variance estimate) / <I>
    def charge(v, rel):
        if mean == 0 or not np.isfinite(v):
            return nan, nan
        q = t_int * v / mean
        return q, abs(q) * rel

    for key, v, rel in (("naive", var, out["var_rel_err"]),
                        ("sdiff", sdiff, out["sdiff_rel_err"]),
                        ("band", out["psd_band_a2"], out["psd_band_rel_err"])):
        q, qe = charge(v, rel)
        out[f"e_{key}_c"] = q
        out[f"e_{key}_err_c"] = qe
        out[f"e_{key}_over_e"] = q / E_CHARGE if np.isfinite(q) else nan
        out[f"e_{key}_over_e_err"] = qe / E_CHARGE if np.isfinite(qe) else nan
    return out


def floor_subtracted_charge(light: dict, dark: dict, key: str = "naive") -> tuple[float, float]:
    """e from a light run minus a dark (floor) run at the same range and NPLC.

    ``key`` picks the variance estimate: "naive" (var), "sdiff" or "band".
    Returns (e_c, err_c) with the white-noise errors of both runs combined.
    """
    vkey, rkey = {"naive": ("var_a2", "var_rel_err"),
                  "sdiff": ("sdiff_var_a2", "sdiff_rel_err"),
                  "band": ("psd_band_a2", "psd_band_rel_err")}[key]
    t_int = light["t_int_s"]
    dv = light[vkey] - dark[vkey]
    err = math.hypot(light[vkey] * light[rkey], dark[vkey] * dark[rkey])
    mean = light["mean_a"] - dark["mean_a"]
    return t_int * dv / mean, abs(t_int * err / mean)
