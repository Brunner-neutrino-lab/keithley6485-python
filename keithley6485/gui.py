"""
keithley6485/gui.py

NiceGUI control panel for the Keithley 6485 picoammeter.

Same two-mode pattern as the b2987b / vx2740 / pulse_mux /
phidget_stage GUIs:

  - Standalone (`python -m keithley6485.gui`): opens a browser
    with a Connection card that creates and owns its own
    K6485Driver. Useful for bench bring-up.

  - Embedded (`build_page(get_controller=..., show_connection=False)`):
    called from a parent NiceGUI app (the ETS DAQ web shell). The
    parent passes a getter that returns the shared driver; this
    panel hides its Connection card and drives the parent's driver.

Note: the K6485 package only ships a driver (no separate controller),
so the embedded mode shares a `K6485Driver` instance directly.
"""

from __future__ import annotations

import asyncio
import time
from typing import Callable, Optional

import numpy as np
from nicegui import ui

from .driver import K6485Driver, DEFAULT_VISA, BAUD_RATE, CURRENT_RANGES


# ---------------------------------------------------------------------------
# Style — xsphere/DAQ palette
# ---------------------------------------------------------------------------

_CSS = """
:root {
  --bg:#11151c; --panel:#1b2230; --panel2:#232c3d;
  --fg:#dde3ee; --mut:#8a93a6;
  --ok:#3fb950; --warn:#d29922; --bad:#f85149; --acc:#58a6ff;
  --line:#2d3648;
}
html, body, .nicegui-content { background:var(--bg) !important; color:var(--fg);
  font:14px/1.45 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif; margin:0; }
.pill { padding:.15rem .55rem; border-radius:999px; font-size:.78rem;
  font-weight:600; white-space:nowrap; display:inline-flex; align-items:center; gap:.3rem; }
.pill.ok   { background:rgba(63,185,80,.18);  color:var(--ok); }
.pill.bad  { background:rgba(248,81,73,.18);  color:var(--bad); }
.pill.warn { background:rgba(210,153,34,.18); color:var(--warn); }
.pill.mut  { background:rgba(138,147,166,.15);color:var(--mut); }
.q-card, .k6485-card {
  background:var(--panel) !important; color:var(--fg) !important;
  border:1px solid var(--line); border-radius:10px;
  box-shadow:none !important; padding:.55rem .85rem .7rem !important;
}
.k6485-card h2 { font-size:.92rem; margin:.05rem 0 .45rem; color:var(--acc);
  font-weight:600; letter-spacing:.3px; }
.q-btn { background:var(--panel2) !important; color:var(--fg) !important;
  border:1px solid var(--line) !important; border-radius:6px !important;
  box-shadow:none !important; padding:.18rem .65rem !important;
  min-height:32px !important; text-transform:none !important; }
.q-btn:hover { border-color:var(--acc) !important; }
.q-btn[data-q-color="primary"], .q-btn.bg-primary {
  background:var(--acc) !important; color:#08111f !important;
  border-color:var(--acc) !important; font-weight:600 !important; }
.q-btn[data-q-color="negative"], .q-btn.bg-negative {
  background:transparent !important; color:var(--bad) !important;
  border-color:var(--bad) !important; }
.q-field__control, .q-field--filled .q-field__control {
  background:var(--panel2) !important; border:1px solid var(--line) !important;
  border-radius:6px !important; min-height:32px !important; color:var(--fg) !important; }
.q-field__label, .q-field__native, .q-field input { color:var(--fg) !important; }
.q-field__label { color:var(--mut) !important; }
.q-field--filled .q-field__control:before,
.q-field--filled .q-field__control:after { display:none !important; }
.q-tab { color:var(--mut) !important; text-transform:none !important; }
.q-tab--active { color:var(--acc) !important; }
.q-tab__indicator { background:var(--acc) !important; }
.q-log, .nicegui-log { background:var(--panel2) !important; color:var(--fg) !important;
  border:1px solid var(--line); border-radius:6px;
  font-family:ui-monospace,Menlo,Consolas,monospace; font-size:.82rem; }
.num { font-variant-numeric:tabular-nums; }
"""


async def _in_thread(fn, *a, **kw):
    return await asyncio.to_thread(fn, *a, **kw)


# Range options for the dropdown — "AUTO" + each range from the driver
_RANGE_LABELS = ["AUTO"] + [f"{r:.0e} A" for r in CURRENT_RANGES]
_RANGE_VALUES = ["AUTO"] + [str(r) for r in CURRENT_RANGES]
_RANGE_MAP    = dict(zip(_RANGE_LABELS, _RANGE_VALUES))


# ===========================================================================
# build_page
# ===========================================================================

def build_page(get_controller: Optional[Callable[[], Optional[K6485Driver]]] = None,
               *, show_connection: Optional[bool] = None) -> None:
    """Render the K6485 control panel into the current container."""
    if show_connection is None:
        show_connection = (get_controller is None)

    _own = {"ctrl": None}
    if get_controller is None:
        def get_controller():
            return _own["ctrl"]

    log = ui.log(max_lines=120).classes("h-32 w-full")
    def log_msg(s: str): log.push(f"[{time.strftime('%H:%M:%S')}] {s}")

    def ensure_ctrl() -> Optional[K6485Driver]:
        c = get_controller()
        if c is None:
            log_msg("not connected" + (" — use the Connection card"
                                       if show_connection else
                                       " — connect on the DAQ's Connections tab"))
            return None
        return c

    with ui.row().classes("w-full gap-3 items-start"):

        # ----------- Connection (standalone only) -----------
        if show_connection:
            with ui.card().classes("k6485-card"):
                ui.html("<h2>connection</h2>")
                visa_in = ui.input(label="VISA / device path",
                                   value=DEFAULT_VISA).classes("w-72")
                mode_in = ui.select(["simulation", "hardware"],
                                    value="simulation",
                                    label="mode").classes("w-40")
                baud_in = ui.number(label="baud rate", value=BAUD_RATE,
                                    step=1200).classes("w-32 num")
                rterm   = ui.input(label='read term', value="\\r\\n").classes("w-32 num")
                wterm   = ui.input(label='write term', value="\\n").classes("w-32 num")
                conn_pill = ui.html('<span class="pill mut">disconnected</span>')

                def set_pill(text: str, cls: str):
                    conn_pill.content = f'<span class="pill {cls}">{text}</span>'

                def _decode_term(s: str) -> str:
                    return s.encode("ascii", errors="ignore").decode("unicode_escape")

                async def do_connect():
                    c = K6485Driver(visa=visa_in.value.strip(), mode=mode_in.value,
                                    baud_rate=int(baud_in.value),
                                    read_termination =_decode_term(rterm.value),
                                    write_termination=_decode_term(wterm.value))
                    set_pill("connecting…", "warn")
                    try:
                        await _in_thread(c.connect)
                        _own["ctrl"] = c
                        set_pill(f"OK — {visa_in.value.strip()}", "ok")
                        log_msg(f"connected ({mode_in.value} on {visa_in.value.strip()})")
                    except Exception as e:
                        set_pill(f"FAIL: {type(e).__name__}", "bad")
                        log_msg(f"connect FAIL: {type(e).__name__}: {e}")

                async def do_disconnect():
                    c = _own["ctrl"]
                    if c is None: return
                    try: await _in_thread(c.disconnect)
                    except Exception as e: log_msg(f"disconnect warn: {e}")
                    _own["ctrl"] = None
                    set_pill("disconnected", "mut")
                    log_msg("disconnected")

                with ui.row().classes("mt-1 gap-2"):
                    ui.button("connect",    on_click=do_connect).props("color=primary")
                    ui.button("disconnect", on_click=do_disconnect).props("color=negative flat")

        # ----------- Range / integration -----------
        with ui.card().classes("k6485-card"):
            ui.html("<h2>range &amp; integration</h2>")
            range_sel = ui.select(_RANGE_LABELS, value="AUTO",
                                  label="current range").classes("w-40")
            nplc_in   = ui.number(label="NPLC (1 = 16.7 ms @60 Hz)",
                                  value=1.0, step=0.1).classes("w-48 num")
            zero_pill = ui.html('<span class="pill mut">zero check: ?</span>')

            async def apply_range():
                c = ensure_ctrl()
                if c is None: return
                try:
                    val = _RANGE_MAP[str(range_sel.value)]
                    if val == "AUTO":
                        await _in_thread(c.set_range, "AUTO")
                    else:
                        await _in_thread(c.set_range, float(val))
                    log_msg(f"range → {range_sel.value}")
                except Exception as e:
                    log_msg(f"set_range FAIL: {type(e).__name__}: {e}")

            async def apply_nplc():
                c = ensure_ctrl()
                if c is None: return
                try:
                    await _in_thread(c.set_nplc, float(nplc_in.value))
                    log_msg(f"NPLC → {nplc_in.value}")
                except Exception as e:
                    log_msg(f"set_nplc FAIL: {type(e).__name__}: {e}")

            async def zc_on():
                c = ensure_ctrl()
                if c is None: return
                try:
                    await _in_thread(c.zero_check_on)
                    zero_pill.content = '<span class="pill warn">zero check: on</span>'
                    log_msg("zero check on (instrument input shorted)")
                except Exception as e:
                    log_msg(f"zero_check_on FAIL: {type(e).__name__}: {e}")

            async def zc_off():
                c = ensure_ctrl()
                if c is None: return
                try:
                    await _in_thread(c.zero_check_off)
                    zero_pill.content = '<span class="pill ok">zero check: off</span>'
                    log_msg("zero check off (input live)")
                except Exception as e:
                    log_msg(f"zero_check_off FAIL: {type(e).__name__}: {e}")

            async def zc_correct():
                c = ensure_ctrl()
                if c is None: return
                try:
                    await _in_thread(c.zero_correct)
                    log_msg("zero correct (offset captured)")
                except Exception as e:
                    log_msg(f"zero_correct FAIL: {type(e).__name__}: {e}")

            with ui.row().classes("gap-2 mt-1"):
                ui.button("apply range", on_click=apply_range).props("color=primary")
                ui.button("apply NPLC",  on_click=apply_nplc)
            with ui.row().classes("gap-2 mt-1 items-center"):
                ui.button("ZC on",       on_click=zc_on).props("color=warning")
                ui.button("ZC off",      on_click=zc_off).props("color=primary")
                ui.button("zero correct",on_click=zc_correct)
                zero_pill

        # ----------- Single + averaged read -----------
        with ui.card().classes("k6485-card"):
            ui.html("<h2>read current</h2>")
            n_samp = ui.number(label="n samples (avg)", value=10,
                               step=1, min=1).classes("w-40 num")
            delay_in = ui.number(label="delay between (s)", value=0.05,
                                 step=0.01, format="%.3f").classes("w-40 num")
            result_lbl = ui.label("I = —").classes("num text-2xl")
            stats_lbl  = ui.label("").classes("num text-sm text-gray-400")

            async def do_single():
                c = ensure_ctrl()
                if c is None: return
                try:
                    i = await _in_thread(c.read_single)
                    result_lbl.text = f"I = {i:+.4e} A"
                    stats_lbl.text  = ""
                    log_msg(f"single  I = {i:+.4e} A")
                except Exception as e:
                    log_msg(f"read_single FAIL: {type(e).__name__}: {e}")

            async def do_n():
                c = ensure_ctrl()
                if c is None: return
                n = int(n_samp.value)
                try:
                    arr, ts = await _in_thread(c.read_n, n, float(delay_in.value))
                    mean = float(np.mean(arr))
                    std  = float(np.std(arr, ddof=1)) if n > 1 else 0.0
                    sem  = std / np.sqrt(n) if n > 1 else 0.0
                    result_lbl.text = f"I = {mean:+.4e} A"
                    stats_lbl.text  = f"  σ = {std:.2e}   SEM = {sem:.2e}   (n = {n})"
                    log_msg(f"avg×{n}  I = {mean:+.4e} ± {sem:.2e} A")
                except Exception as e:
                    log_msg(f"read_n FAIL: {type(e).__name__}: {e}")

            with ui.row().classes("gap-2 mt-1"):
                ui.button("single read", on_click=do_single).props("color=primary")
                ui.button("averaged",    on_click=do_n).props("color=primary")


# ---------------------------------------------------------------------------
# Standalone entry — `python -m keithley6485.gui`
# ---------------------------------------------------------------------------

def main():
    import argparse
    p = argparse.ArgumentParser(description="Keithley 6485 web GUI")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8770)
    args = p.parse_args()

    @ui.page("/")
    def index():
        ui.add_head_html(f"<style>{_CSS}</style>")
        ui.dark_mode().enable()
        with ui.element("header").style(
            "display:flex;align-items:center;gap:.8rem;"
            "padding:.55rem 1rem;background:var(--panel);"
            "border-bottom:1px solid var(--line);position:sticky;top:0;z-index:5"
        ):
            ui.html("<h1 style='font-size:1.05rem;font-weight:600;margin:0'>"
                    "Keithley 6485 · picoammeter</h1>")
        build_page()

    ui.run(host=args.host, port=args.port, reload=False,
           title="K6485 Picoammeter", show=False)


if __name__ in {"__main__", "__mp_main__"}:
    main()
