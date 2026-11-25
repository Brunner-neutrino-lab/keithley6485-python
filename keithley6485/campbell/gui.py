"""Web GUI (NiceGUI) for Keithley 6485 Campbell runs.

    k6485-campbell-gui            # http://127.0.0.1:8771/
    k6485-campbell-gui --host 0.0.0.0 --port 8771

One meter per process: every browser tab shows and drives the same session.
"""

from __future__ import annotations

import argparse
import math
import os
from typing import Optional

import numpy as np

from . import __version__
from .guard import list_ports
from .meter import BUFFER_MAX, RANGES_A
from .session import NOTE_FIELDS, SIM_PORT, Connection, RunSettings, Session
from .sim import SimPhysics
from .stats import E_CHARGE, binned_spectrum, invalid_mask
from .store import DEFAULT_DIR
from .transport import TERMINATORS

SESSION = Session()

_UNITS = (("mA", 1e-3), ("µA", 1e-6), ("nA", 1e-9), ("pA", 1e-12), ("fA", 1e-15), ("aA", 1e-18))
_RANGE_LABELS = {f"{r / u:g} {n}": r for r in RANGES_A for n, u in _UNITS if 1 <= r / u < 1000}
_BAUDS = [300, 600, 1200, 2400, 4800, 9600, 19200, 38400, 57600]
_NOTE_LABELS = {"operator": "operator", "laser": "laser (model, drive, wavelength)", "nd": "ND filters",
                "photodiode": "photodiode (type, area)", "bias": "photodiode bias", "box": "box / shielding state",
                "text": "free text"}


def unit_for(v: float) -> tuple[str, float]:
    v = abs(v)
    for n, u in _UNITS:
        if v >= u:
            return n, u
    return "aA", 1e-18


def fmt_a(v: Optional[float], digits: int = 4) -> str:
    if v is None or not np.isfinite(v):
        return "—"
    if v == 0:
        return "0 A"
    n, u = unit_for(v)
    return f"{v / u:.{digits}g} {n}"


def _g(v, fmt="{:.4g}") -> str:
    if v is None:
        return "—"
    if isinstance(v, float) and not np.isfinite(v):
        return "—"
    return fmt.format(v)


def stats_rows(st: Optional[dict], thr: dict) -> list[dict]:
    if not st:
        return []
    rows = []

    def add(q, v, note=""):
        rows.append({"id": len(rows), "q": q, "v": v, "note": note})

    add("readings used", f"{st['n']} of {st['n_total']}",
        f"{st['n_invalid']} invalid, {st['n_overflow']} overflow, {st['n_segments']} segments")
    add("T_int = NPLC / f_line", f"{st['t_int_s'] * 1e3:.4g} ms", f"NPLC {st['nplc']:g} at {st['f_line_hz']:g} Hz")
    add("dt (mean, instrument)", _g(st.get("dt_mean_s", float("nan")) * 1e3, "{:.4g} ms"),
        f"{_g(st.get('readings_per_s'))} rdg/s; T_int/dt = {_g(st.get('duty_t_int_over_dt'), '{:.3f}')}")
    if "var_a2" not in st or not np.isfinite(st.get("var_a2", float("nan"))):
        return rows
    add("mean <I>", fmt_a(st["mean_a"], 7), f"sem {fmt_a(st['sem_a'])}")
    add("sd", fmt_a(st["sd_a"]), f"shot sqrt(e<I>/T_int) = {fmt_a(st['shot_sd_nominal_a'])}")
    add("Var (ddof 1)", f"{st['var_a2']:.4e} A²", f"rel. err {st['var_rel_err']:.2%}")
    add("successive-difference var", f"{st['sdiff_var_a2']:.4e} A²", "mean(dI²)/2, inside bursts")
    b = st["psd_band_hz"]
    add(f"periodogram {b[0]:g}-{b[1]:g} Hz", f"{st['psd_band_a2']:.4e} A²",
        f"{st['psd_band_bins']} bins, variance-per-reading units")
    add("autocorrelation lag 1..4", "  ".join(f"{a:+.3f}" for a in st["acf"]), f"± {st['acf_err']:.3f} (white)")
    if st.get("allan_m"):
        add("Allan m·σ²(m), m = " + ",".join(str(m) for m in st["allan_m"][:6]),
            "  ".join(f"{v / st['var_a2']:.2f}" for v in st["allan_m_var_a2"][:6]), "relative to Var; flat = white")
    for k, label in (("naive", "T_int·Var/<I>"), ("sdiff", "T_int·sdiff/<I>"), ("band", "T_int·P_band/<I>")):
        q, e = st[f"e_{k}_c"], st[f"e_{k}_err_c"]
        if np.isfinite(q):
            add(label, f"{q / E_CHARGE:.4f} ± {e / E_CHARGE:.4f} e", f"{q:.4e} C")
    if np.isfinite(st.get("e_naive_c", float("nan"))) and st["e_naive_c"] > 0:
        add("T_eff/T_int if pure shot noise", f"{E_CHARGE / st['e_naive_c']:.4f}", "= e / (T_int·Var/<I>)")
    add("smallest reading step", fmt_a(st["min_step_a"]), "resolution check")
    add("drift (in bursts)", f"{st['drift_a_per_s']:.3e} A/s", "")
    if thr:
        add("throughput", f"{_g(thr.get('readings_per_s'))} rdg/s",
            f"acquire {_g(thr.get('acquire_s_per_burst'), '{:.1f}')} s + transfer "
            f"{_g(thr.get('transfer_s_per_burst'), '{:.1f}')} s per burst; duty {_g(thr.get('duty_cycle'), '{:.2f}')}")
    return rows


def _decimate(t: np.ndarray, y: np.ndarray, max_pts: int = 4000):
    n = len(y)
    if n <= max_pts:
        return t, y
    k = n // (max_pts // 2)
    m = (n // k) * k
    tb = t[:m].reshape(-1, k)
    yb = y[:m].reshape(-1, k)
    imin, imax = yb.argmin(1), yb.argmax(1)
    rows = np.arange(len(yb))
    first = np.minimum(imin, imax)
    second = np.maximum(imin, imax)
    tt = np.column_stack([tb[rows, first], tb[rows, second]]).ravel()
    yy = np.column_stack([yb[rows, first], yb[rows, second]]).ravel()
    return tt, yy


def _chart(title: str, xname: str, yname: str, xlog=False, ylog=False) -> dict:
    return {
        "title": {"text": title, "left": "center", "textStyle": {"fontSize": 13}},
        "animation": False,
        "grid": {"left": 70, "right": 20, "top": 36, "bottom": 46},
        "tooltip": {"trigger": "axis"},
        "xAxis": {"type": "log" if xlog else "value", "name": xname, "nameLocation": "middle",
                  "nameGap": 28, "scale": True},
        "yAxis": {"type": "log" if ylog else "value", "name": yname, "nameLocation": "middle",
                  "nameGap": 52, "scale": True},
        "series": [],
    }


def index() -> None:
    from nicegui import run, ui

    s = SESSION
    ui.page_title("K6485 Campbell")
    ui.add_head_html("<style>.q-card{padding:10px 14px} .num input{font-variant-numeric:tabular-nums}"
                     " .kv td{padding:1px 8px;font-size:12px}</style>")

    with ui.row().classes("w-full items-center gap-4"):
        ui.label("Keithley 6485 · Campbell runs").classes("text-lg font-semibold")
        conn_badge = ui.badge("disconnected", color="grey")
        phase_badge = ui.badge("", color="grey")
        ui.space()
        ui.label(f"keithley6485.campbell {__version__}").classes("text-xs text-gray-500")

    with ui.row().classes("w-full items-start gap-3 no-wrap"):
        # ------------------------------------------------------------ connection
        with ui.card().classes("w-80"):
            ui.label("connection").classes("font-semibold")
            port_opts = [SIM_PORT] + list_ports()
            port_sel = ui.select(port_opts, label="port (/dev/serial/by-id or simulation)", value=SIM_PORT,
                                 with_input=True, new_value_mode="add-unique").classes("w-full")

            def refresh_ports():
                port_sel.options = [SIM_PORT] + list_ports()
                port_sel.update()

            with ui.row().classes("gap-2 items-center"):
                baud_sel = ui.select(_BAUDS, label="baud", value=9600).classes("w-20")
                rterm_sel = ui.select(list(TERMINATORS), label="rx term", value="CR").classes("w-24").tooltip(
                    "terminator the meter sends (its RS-232 menu)")
                wterm_sel = ui.select(list(TERMINATORS), label="tx term", value="CR").classes("w-24").tooltip(
                    "terminator sent after each command")
            with ui.row().classes("gap-2 items-center"):
                xon_sw = ui.checkbox("XON/XOFF", value=False)
                daq_sw = ui.checkbox("check ETS DAQ", value=True).tooltip(
                    "refuse if the DAQ web app has the K6485 connected or claimed")
                ui.button(icon="refresh", on_click=refresh_ports).props("flat dense").tooltip("rescan ports")
            with ui.expansion("simulation parameters").classes("w-full"):
                sim_cur = ui.number("photocurrent (A)", value=1e-9, format="%.4g").classes("w-full num")
                sim_floor = ui.number("floor sd per reading (A, empty = spec)", value=None,
                                      format="%.3g").classes("w-full num")
                sim_exc = ui.number("excess noise sd/<I>", value=0.0, format="%.3g").classes("w-full num")
                sim_filter = ui.select(["none", "ar1", "boxcar"], label="meter filter", value="none").classes("w-full")
                with ui.row().classes("gap-2"):
                    sim_phi = ui.number("AR(1) phi", value=-0.5, step=0.1).classes("w-24 num")
                    sim_m = ui.number("boxcar M", value=4, step=1, min=1).classes("w-20 num")
                    sim_l = ui.number("boxcar L", value=8, step=1, min=1).classes("w-20 num")
                with ui.row().classes("gap-2"):
                    sim_ts = ui.number("time scale", value=0.02, step=0.01, min=0).classes("w-24 num").tooltip(
                        "real seconds per instrument second: 1 = real time, 0 = instant")
                    sim_seed = ui.number("seed", value=None, step=1).classes("w-24 num")
            restore_disc = ui.checkbox("restore settings on disconnect", value=True)
            with ui.row().classes("gap-2"):
                connect_btn = ui.button("connect", color="primary")
                disconnect_btn = ui.button("disconnect", color="negative").props("outline")
            with ui.expansion("meter settings as found").classes("w-full"):
                found_html = ui.html("").classes("kv")

        # ------------------------------------------------------------ settings
        with ui.card().classes("w-80"):
            ui.label("Campbell settings").classes("font-semibold")
            range_sel = ui.select(list(_RANGE_LABELS), label="fixed range (autorange off)",
                                  value="20 nA").classes("w-full")
            with ui.row().classes("gap-2"):
                nplc_in = ui.number("NPLC", value=1, min=0.01, max=60, step=1).classes("w-24 num").tooltip(
                    "integer NPLC rejects line-frequency ripple")
                az_sw = ui.switch("autozero", value=False)
                disp_sw = ui.switch("display off", value=True)
            input_sel = ui.select({"live": "input live (zero check off)", "floor": "floor: zero check ON"},
                                  label="input", value="live").classes("w-full")
            with ui.row().classes("gap-2"):
                n_in = ui.number("readings N", value=20000, min=1, step=2500).classes("w-28 num")
                burst_in = ui.number("per burst", value=BUFFER_MAX, min=1, max=BUFFER_MAX, step=100).classes(
                    "w-24 num")
                settle_in = ui.number("settle s", value=2.0, min=0, step=1).classes("w-20 num")
            with ui.row().classes("gap-2"):
                band_lo = ui.number("band lo (Hz)", value=1.0, min=0, step=0.5).classes("w-28 num")
                band_hi = ui.number("band hi (Hz)", value=10.0, min=0, step=1).classes("w-28 num")
            name_in = ui.input("run name", value="campbell").classes("w-full")
            outdir_in = ui.input("output directory", value=str(DEFAULT_DIR)).classes("w-full")
            with ui.row().classes("gap-2"):
                restore_sw = ui.checkbox("restore after run", value=True)
                autosave_sw = ui.checkbox("save when done", value=True)
            with ui.row().classes("gap-2"):
                start_btn = ui.button("start", color="positive")
                stop_btn = ui.button("stop", color="warning")
                recompute_btn = ui.button("recompute").props("outline").tooltip("statistics with the band above")
            progress = ui.linear_progress(value=0, show_value=False).classes("w-full")
            progress_lbl = ui.label("").classes("text-xs")

        # ------------------------------------------------------------ notes
        with ui.card().classes("w-80"):
            ui.label("operator notes (saved in the file)").classes("font-semibold")
            for k in NOTE_FIELDS:
                def _bind(e, key=k):
                    s.notes[key] = e.value or ""
                widget = ui.textarea if k == "text" else ui.input
                widget(_NOTE_LABELS[k], value=s.notes.get(k, ""), on_change=_bind).classes("w-full")
            with ui.row().classes("gap-2 items-center w-full"):
                save_path = ui.input("save as (empty = automatic name)").classes("grow")
                save_btn = ui.button("save HDF5", color="primary")
            saved_lbl = ui.label("").classes("text-xs break-all")

    with ui.row().classes("w-full gap-3 no-wrap"):
        chart_t = ui.echart(_chart("I(t)", "t (s)", "I")).classes("grow h-72")
        chart_h = ui.echart(_chart("histogram of I - <I>", "I - <I>", "readings")).classes("grow h-72")
        chart_s = ui.echart(_chart("periodogram (variance per reading)", "f (Hz)", "P", True, True)).classes(
            "grow h-72")

    with ui.row().classes("w-full gap-3 no-wrap items-start"):
        with ui.card().classes("grow"):
            ui.label("statistics").classes("font-semibold")
            table = ui.table(rows=[], columns=[
                {"name": "q", "label": "quantity", "field": "q", "align": "left"},
                {"name": "v", "label": "value", "field": "v", "align": "left"},
                {"name": "note", "label": "", "field": "note", "align": "left"},
            ], row_key="id").props("dense flat").classes("w-full")
        with ui.card().classes("w-1/3"):
            ui.label("log").classes("font-semibold")
            log = ui.log(max_lines=400).classes("w-full h-64 text-xs")

    # ---------------------------------------------------------------- actions
    def conn_from_ui() -> Connection:
        port = (port_sel.value or "").strip()
        sim = None
        if port == SIM_PORT:
            sim = SimPhysics(current_a=float(sim_cur.value or 0.0),
                             floor_a=None if sim_floor.value in (None, "") else float(sim_floor.value),
                             excess_rel_sd=float(sim_exc.value or 0.0), filter=sim_filter.value,
                             ar1_phi=float(sim_phi.value or 0.0), boxcar_m=int(sim_m.value or 1),
                             boxcar_l=int(sim_l.value or 1),
                             seed=None if sim_seed.value in (None, "") else int(sim_seed.value))
        return Connection(port=port, baud=int(baud_sel.value), read_term=TERMINATORS[rterm_sel.value],
                          write_term=TERMINATORS[wterm_sel.value], xonxoff=bool(xon_sw.value), sim=sim,
                          time_scale=float(sim_ts.value or 0.0))

    async def do_connect():
        if s.connected:
            ui.notify("already connected", type="warning")
            return
        conn = conn_from_ui()
        if not conn.port:
            ui.notify("choose a port", type="warning")
            return
        try:
            idn = await run.io_bound(s.connect, conn, bool(daq_sw.value))
            ui.notify(f"connected: {idn}", type="positive")
        except Exception as e:
            s.log(f"connect FAILED: {e}")
            ui.notify(f"connect failed: {e}", type="negative", multi_line=True, timeout=10000)

    async def do_disconnect():
        try:
            await run.io_bound(s.disconnect, bool(restore_disc.value))
        except Exception as e:
            ui.notify(str(e), type="negative")

    def run_settings() -> RunSettings:
        return RunSettings(range_a=_RANGE_LABELS[range_sel.value], nplc=float(nplc_in.value),
                           autozero=bool(az_sw.value), display_off=bool(disp_sw.value),
                           zero_check=input_sel.value == "floor", n_total=int(n_in.value),
                           burst_size=int(burst_in.value), settle_s=float(settle_in.value or 0.0),
                           restore_after=bool(restore_sw.value), autosave=bool(autosave_sw.value),
                           out_dir=outdir_in.value or ".", name=name_in.value or "campbell",
                           band_lo_hz=float(band_lo.value), band_hi_hz=float(band_hi.value))

    def do_start():
        try:
            rs = run_settings()
            if not 1 <= rs.burst_size <= BUFFER_MAX:
                raise ValueError(f"burst size must be 1..{BUFFER_MAX}")
            if abs(rs.nplc - round(rs.nplc)) > 1e-9:
                ui.notify("NPLC is not an integer: line ripple will not be rejected", type="warning")
            s.start(rs)
        except Exception as e:
            ui.notify(f"cannot start: {e}", type="negative")

    def do_save():
        try:
            p = s.save(save_path.value.strip() or None)
            ui.notify(f"saved {p}", type="positive")
            s.log(f"saved {p}")
        except Exception as e:
            ui.notify(f"save failed: {e}", type="negative")

    def do_recompute():
        s.recompute(float(band_lo.value), float(band_hi.value))

    connect_btn.on_click(do_connect)
    disconnect_btn.on_click(do_disconnect)
    start_btn.on_click(do_start)
    stop_btn.on_click(s.stop)
    save_btn.on_click(do_save)
    recompute_btn.on_click(do_recompute)

    # ---------------------------------------------------------------- refresh
    seen = {"version": -1, "log": 0, "found": None}

    def update_plots():
        with s.lock:
            r = s.run
            st = s.stats
            if r is None or r.n == 0:
                for c in (chart_t, chart_h, chart_s):
                    c.options["series"] = []
                    c.update()
                table.rows = []
                table.update()
                return
            i_a, status, burst_ix = r.i_a, r.status, r.burst
            t = r.t_run_s()
            t_s = r.t_s
            thr = dict(s.throughput)
        good = ~invalid_mask(i_a, status)
        if not good.any():
            return
        x = i_a[good]
        mean = float(x.mean())
        un, uu = unit_for(mean if mean else float(np.std(x)) or 1e-12)
        pts = []
        per_burst = max(200, 4000 // max(1, len(r.bursts)))
        for k in np.unique(burst_ix):
            sel = good & (burst_ix == k)
            tt, yy = _decimate(t[sel], i_a[sel], per_burst)
            if pts:
                pts.append([float(tt[0]), None])       # break the line across the gap between bursts
            pts.extend([float(a), float(b / uu)] for a, b in zip(tt, yy))
        chart_t.options["yAxis"]["name"] = f"I ({un})"
        chart_t.options["series"] = [{"type": "line", "showSymbol": False, "sampling": "none",
                                      "connectNulls": False, "lineStyle": {"width": 1}, "data": pts}]
        chart_t.update()

        dev = x - mean
        sd = float(dev.std()) or 1e-15
        hn, hu = unit_for(sd)
        cnt, edges = np.histogram(dev, bins=60)
        ctr = 0.5 * (edges[:-1] + edges[1:])
        w = edges[1] - edges[0]
        series = [{"type": "bar", "barWidth": "95%", "name": "readings",
                   "data": [[float(c / hu), int(k)] for c, k in zip(ctr, cnt)]},
                  {"type": "line", "showSymbol": False, "name": f"Gaussian sd {fmt_a(sd)}",
                   "data": [[float(c / hu), float(len(x) * w / (sd * math.sqrt(2 * math.pi))
                                                  * math.exp(-0.5 * (c / sd) ** 2))] for c in ctr]}]
        if st and np.isfinite(st.get("shot_sd_nominal_a", float("nan"))) and st["shot_sd_nominal_a"] > 0:
            sh = st["shot_sd_nominal_a"]
            series.append({"type": "line", "showSymbol": False, "lineStyle": {"type": "dashed"},
                           "name": f"shot sd {fmt_a(sh)}",
                           "data": [[float(c / hu), float(len(x) * w / (sh * math.sqrt(2 * math.pi))
                                                          * math.exp(-0.5 * (c / sh) ** 2))] for c in ctr]})
        chart_h.options["xAxis"]["name"] = f"I - <I> ({hn})"
        chart_h.options["xAxis"]["type"] = "value"
        chart_h.options["legend"] = {"right": 22, "top": 40, "orient": "vertical", "textStyle": {"fontSize": 10}}
        chart_h.options["series"] = series
        chart_h.update()

        f, p, _c = binned_spectrum(i_a, t_s, burst_ix, status)
        if len(f):
            pn, pu = unit_for(math.sqrt(float(np.median(p))))
            ser = [{"type": "line", "name": "mean periodogram", "showSymbol": True, "symbolSize": 3,
                    "data": [[float(a), float(b / pu ** 2)] for a, b in zip(f, p) if b > 0]}]
            mark = []
            if st and np.isfinite(st.get("var_a2", float("nan"))):
                mark.append({"yAxis": st["var_a2"] / pu ** 2, "name": "Var",
                             "lineStyle": {"color": "#888"}, "label": {"formatter": "Var"}})
                if st.get("mean_a", 0) > 0:
                    lvl = E_CHARGE * st["mean_a"] / st["t_int_s"]
                    mark.append({"yAxis": lvl / pu ** 2, "name": "e<I>/T_int",
                                 "lineStyle": {"color": "#d33"}, "label": {"formatter": "e<I>/T_int"}})
                b = st.get("psd_band_hz", [1, 10])
                ser[0]["markArea"] = {"silent": True, "itemStyle": {"color": "rgba(80,140,255,0.08)"},
                                      "data": [[{"xAxis": b[0]}, {"xAxis": b[1]}]]}
            ser[0]["markLine"] = {"silent": True, "symbol": "none", "data": mark}
            levels = [float(p[p > 0].min()), float(p.max())] + [m["yAxis"] * pu ** 2 for m in mark]
            chart_s.options["yAxis"]["min"] = 10.0 ** math.floor(math.log10(min(levels) / pu ** 2))
            chart_s.options["yAxis"]["max"] = 10.0 ** math.ceil(math.log10(max(levels) / pu ** 2))
            chart_s.options["grid"]["right"] = 70
            chart_s.options["yAxis"]["name"] = f"P ({pn}²)"
            chart_s.options["series"] = ser
            chart_s.update()
        table.rows = stats_rows(st, thr)
        table.update()

    def refresh():
        connected = s.connected
        conn_badge.text = (f"connected · {s.conn.port}" if connected else "disconnected")
        conn_badge.props(f"color={'positive' if connected else 'grey'}")
        ph = s.phase
        phase_badge.text = ph + (" · stop requested" if s.stop_requested and s.running else "")
        phase_badge.props(f"color={'orange' if s.running else 'grey'}")
        connect_btn.set_enabled(not connected)
        disconnect_btn.set_enabled(connected and not s.running)
        start_btn.set_enabled(connected and not s.running)
        stop_btn.set_enabled(s.running)
        pr = s.progress
        if s.running and pr["n_bursts"]:
            frac = ((pr["burst"] - 1) + (0.5 * pr["frac"] if pr["phase"] == "acquire"
                                         else 0.5 + 0.5 * pr["frac"] if pr["phase"] == "transfer" else 1.0)) \
                / pr["n_bursts"]
            progress.value = max(0.0, min(1.0, frac))
            progress_lbl.text = f"burst {pr['burst']}/{pr['n_bursts']} · {pr['phase']} {pr['frac']:.0%}"
        elif s.running:
            progress_lbl.text = ph
        elif s.run is not None:
            progress.value = 1.0
            progress_lbl.text = (f"{s.run.n} readings" + (f" · error: {s.error}" if s.error else "")
                                 + (" · restored as found" if s.restore_diff == {} else
                                    f" · restore differences: {s.restore_diff}" if s.restore_diff else ""))
        if s.last_path is not None:
            saved_lbl.text = f"saved: {os.path.abspath(s.last_path)}"
        elif s.run is not None and s.run.n and s.saved_version != s.version:
            saved_lbl.text = "not saved"
        if s.found is not seen["found"]:
            seen["found"] = s.found
            f = s.found or {}
            found_html.content = "<table>" + "".join(
                f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in f.items()
                if not k.startswith("_")) + "</table>"
        msgs = list(s.messages)
        if len(msgs) < seen["log"]:
            seen["log"] = 0
        for m in msgs[seen["log"]:]:
            log.push(m)
        seen["log"] = len(msgs)
        if s.version != seen["version"]:
            seen["version"] = s.version
            update_plots()

    ui.timer(0.5, refresh)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="k6485-campbell-gui", description="Keithley 6485 Campbell GUI")
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to serve the lab network")
    ap.add_argument("--port", type=int, default=8771)
    ap.add_argument("--show", action="store_true", help="open a browser tab")
    a = ap.parse_args(argv)
    from nicegui import ui
    ui.page("/")(index)
    ui.run(host=a.host, port=a.port, title="K6485 Campbell", reload=False, show=a.show)


if __name__ in {"__main__", "__mp_main__"}:
    main()
