"""Command line for Keithley 6485 Campbell runs.

    k6485-campbell ports
    k6485-campbell snapshot --port /dev/serial/by-id/...
    k6485-campbell run --port ... --range 20n --nplc 1 --n 20000 --name pd_nd2 --note-laser "..."
    k6485-campbell run --sim --sim-current 1e-9 --n 20000
    k6485-campbell stats data/campbell_k6485/pd_nd2_20261125_120000.h5
    k6485-campbell gui
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from typing import Optional

from .session import NOTE_FIELDS, SIM_PORT, Connection, RunSettings, Session
from .sim import SimPhysics
from .stats import E_CHARGE
from .store import DEFAULT_DIR

_PREFIX = {"f": 1e-15, "p": 1e-12, "n": 1e-9, "u": 1e-6, "µ": 1e-6, "m": 1e-3, "": 1.0}


def parse_amps(s: str) -> float:
    m = re.fullmatch(r"\s*([0-9.eE+-]+)\s*([fpnuµm]?)A?\s*", s)
    if not m:
        raise argparse.ArgumentTypeError(f"not a current: {s!r} (e.g. 20n, 2e-8, 200nA)")
    return float(m.group(1)) * _PREFIX[m.group(2)]


def fmt_si(v: float, unit: str = "A", digits: int = 4) -> str:
    if v is None or not math.isfinite(v):
        return "nan"
    if v == 0:
        return f"0 {unit}"
    for p, f in (("m", 1e-3), ("u", 1e-6), ("n", 1e-9), ("p", 1e-12), ("f", 1e-15), ("a", 1e-18)):
        if abs(v) >= f:
            return f"{v / f:.{digits}g} {p}{unit}"
    return f"{v:.{digits}g} {unit}"


def stats_lines(st: dict) -> list[str]:
    if not st:
        return ["no statistics"]
    L = [f"n = {st['n']} ({st['n_invalid']} invalid, {st['n_overflow']} overflow) in {st['n_segments']} segments",
         f"T_int = {st['t_int_s'] * 1e3:.4g} ms (NPLC {st['nplc']:g} / {st['f_line_hz']:g} Hz); "
         f"dt mean {st.get('dt_mean_s', float('nan')) * 1e3:.4g} ms -> {st.get('readings_per_s', float('nan')):.4g} rdg/s"]
    if "var_a2" in st and math.isfinite(st.get("var_a2", float("nan"))):
        L += [f"mean {fmt_si(st['mean_a'])}  sd {fmt_si(st['sd_a'])}  (shot e<I>/T_int: {fmt_si(st['shot_sd_nominal_a'])})",
              f"var {st['var_a2']:.4e} A^2  sdiff {st['sdiff_var_a2']:.4e}  psd[{st['psd_band_hz'][0]:g},{st['psd_band_hz'][1]:g}) Hz "
              f"{st['psd_band_a2']:.4e} ({st['psd_band_bins']} bins)",
              "acf lag1..4: " + "  ".join(f"{a:+.4f}" for a in st["acf"]) + f"  (+-{st['acf_err']:.4f})",
              f"min step {fmt_si(st['min_step_a'])}   drift {st['drift_a_per_s']:.3e} A/s"]
        for k, label in (("naive", "T_int*Var/<I>"), ("sdiff", "T_int*sdiff/<I>"), ("band", "T_int*psd_band/<I>")):
            q, e = st[f"e_{k}_c"], st[f"e_{k}_err_c"]
            L.append(f"{label:20s} = {q:.5e} +- {e:.2e} C  = {q / E_CHARGE:.4f} +- {e / E_CHARGE:.4f} e")
    return L


def _add_conn_args(p):
    p.add_argument("--port", help="serial device (e.g. /dev/serial/by-id/usb-Prolific...), or 'simulation'")
    p.add_argument("--sim", action="store_true", help="use the simulated meter")
    p.add_argument("--baud", type=int, default=9600, help="must match the meter's RS-232 menu (default 9600)")
    p.add_argument("--rterm", default="CR", help="meter's transmit terminator: CR, LF, CRLF, LFCR (default CR)")
    p.add_argument("--wterm", default="CR", help="terminator sent after each command (default CR)")
    p.add_argument("--xonxoff", action="store_true", help="XON/XOFF flow control (only if set on the meter)")
    p.add_argument("--no-daq-check", action="store_true",
                   help="skip the ETS DAQ web-app check (the other-process check still runs)")
    g = p.add_argument_group("simulation")
    g.add_argument("--sim-current", type=parse_amps, default=1e-9, help="photocurrent (default 1 nA)")
    g.add_argument("--sim-floor", type=parse_amps, default=None,
                   help="white floor sd per reading (default: spec RMS noise x sqrt(6/NPLC))")
    g.add_argument("--sim-excess", type=float, default=0.0, help="white excess noise, sd/<I>")
    g.add_argument("--sim-filter", choices=("none", "ar1", "boxcar"), default="none")
    g.add_argument("--sim-phi", type=float, default=-0.5, help="AR(1) coefficient")
    g.add_argument("--sim-m", type=int, default=4, help="boxcar: sub-conversions per aperture")
    g.add_argument("--sim-l", type=int, default=8, help="boxcar: sub-conversions averaged per reading")
    g.add_argument("--sim-seed", type=int, default=None)
    g.add_argument("--sim-time-scale", type=float, default=0.0,
                   help="real seconds per instrument second (0: as fast as possible, 1: real time)")


def _connection(a) -> Connection:
    from .transport import decode_terminator
    port = SIM_PORT if a.sim else a.port
    if not port:
        raise SystemExit("give --port DEVICE or --sim")
    sim = None
    if port == SIM_PORT:
        sim = SimPhysics(current_a=a.sim_current, floor_a=a.sim_floor, excess_rel_sd=a.sim_excess,
                         filter=a.sim_filter, ar1_phi=a.sim_phi, boxcar_m=a.sim_m, boxcar_l=a.sim_l,
                         seed=a.sim_seed)
    return Connection(port=port, baud=a.baud, read_term=decode_terminator(a.rterm),
                      write_term=decode_terminator(a.wterm), xonxoff=a.xonxoff, sim=sim,
                      time_scale=a.sim_time_scale)


def cmd_ports(a) -> int:
    from .guard import list_ports, processes_using
    ports = list_ports()
    if not ports:
        print("no serial ports found")
    for p in ports:
        users = processes_using(p)
        print(p + (f"   [open in: {', '.join(str(u[0]) for u in users)}]" if users else ""))
    return 0


def cmd_snapshot(a) -> int:
    s = Session(log=print)
    s.connect(_connection(a), check_daq=not a.no_daq_check)
    try:
        print(json.dumps({k: v for k, v in s.found.items()}, indent=1, default=str))
    finally:
        s.disconnect(restore=False)
    return 0


def cmd_run(a) -> int:
    s = Session(log=print)
    s.connect(_connection(a), check_daq=not a.no_daq_check)
    for k in NOTE_FIELDS:
        s.notes[k] = getattr(a, f"note_{k}") or ""
    rs = RunSettings(range_a=a.range, nplc=a.nplc, autozero=a.autozero, display_off=not a.display_on,
                     zero_check=a.zero_check, n_total=a.n, burst_size=a.burst, settle_s=a.settle,
                     restore_after=not a.no_restore, autosave=not a.no_save, out_dir=a.out_dir,
                     name=a.name, band_lo_hz=a.band[0], band_hi_hz=a.band[1],
                     read_abs_time=not a.no_abs_time)
    if abs(rs.nplc - round(rs.nplc)) > 1e-9:
        print(f"note: NPLC {rs.nplc:g} is not an integer: line-frequency ripple is not rejected")
    rc = 0
    try:
        s.start(rs)
        while s.running:
            try:
                s.wait(0.5)
            except KeyboardInterrupt:
                print("\nCtrl-C: stopping after this burst's data are read back (Ctrl-C again to quit hard)")
                s.stop()
                try:
                    s.wait()
                except KeyboardInterrupt:
                    return 130
        print("\n".join(stats_lines(s.stats or {})))
        if s.throughput:
            t = s.throughput
            print(f"throughput: {t['readings_per_s']:.4g} rdg/s, acquire {t['acquire_s_per_burst']:.1f} s/burst, "
                  f"transfer {t['transfer_s_per_burst']:.1f} s/burst ({t['transfer_s_per_reading'] * 1e3:.1f} ms/reading), "
                  f"duty {t['duty_cycle']:.2f}")
        if s.last_path:
            print(f"file: {s.last_path}")
        if s.error:
            rc = 1
    finally:
        s.disconnect(restore=not a.no_restore)
    return rc


def cmd_stats(a) -> int:
    from .stats import campbell_stats
    from .store import read_run
    r = read_run(a.file)
    st_set = r["settings"]
    nplc = float(st_set.get("nplc") or r["attrs"].get("set_nplc"))
    flf = float(st_set.get("line_freq_hz") or r["attrs"].get("set_line_freq_hz"))
    st = campbell_stats(r["i_a"], r["t_s"], nplc, flf, burst=r["burst"], status=r["status"],
                        band_hz=tuple(a.band))
    if a.json:
        print(json.dumps(st, indent=1))
    else:
        print(f"{a.file}: run {r['run_id']}")
        notes = {k: v for k, v in r["notes"].items() if v}
        if notes:
            print("notes: " + "; ".join(f"{k}: {v}" for k, v in notes.items()))
        print("\n".join(stats_lines(st)))
    return 0


def cmd_gui(a) -> int:
    from .gui import main as gui_main
    gui_main(["--host", a.host, "--port", str(a.port)] + (["--show"] if a.show else []))
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="k6485-campbell", description="Keithley 6485 Campbell (shot-noise) runs")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("ports", help="list serial ports and who has them open")
    p.set_defaults(fn=cmd_ports)

    p = sub.add_parser("snapshot", help="connect and print the meter's settings (changes nothing)")
    _add_conn_args(p)
    p.set_defaults(fn=cmd_snapshot)

    p = sub.add_parser("run", help="configure, take N raw readings in buffer bursts, restore, save")
    _add_conn_args(p)
    p.add_argument("--range", type=parse_amps, default=20e-9, help="fixed range: 2n, 20n, 200n, 2u, ... (default 20n)")
    p.add_argument("--nplc", type=float, default=1.0, help="integration rate in line cycles (default 1)")
    p.add_argument("--autozero", action=argparse.BooleanOptionalAction, default=False,
                   help="autozero on or off (default off)")
    p.add_argument("--display-on", action="store_true", help="leave the front panel on (slower)")
    p.add_argument("--zero-check", action="store_true", help="zero check ON: a floor run with the input disconnected")
    p.add_argument("--n", type=int, default=2500, help="readings in all (default 2500)")
    p.add_argument("--burst", type=int, default=2500, help="readings per buffer burst, <= 2500")
    p.add_argument("--settle", type=float, default=2.0, help="seconds between configuring and the first burst")
    p.add_argument("--band", type=float, nargs=2, default=(1.0, 10.0), metavar=("LO", "HI"),
                   help="periodogram band in Hz (default 1 10)")
    p.add_argument("--name", default="campbell", help="file name prefix")
    p.add_argument("--out-dir", default=str(DEFAULT_DIR), help=f"default {DEFAULT_DIR}")
    p.add_argument("--no-restore", action="store_true", help="leave the Campbell settings on the meter")
    p.add_argument("--no-save", action="store_true")
    p.add_argument("--no-abs-time", action="store_true", help="skip SENS:DATA? after each burst")
    for k in NOTE_FIELDS:
        p.add_argument(f"--note-{k}", default="", help=f"operator note: {k}")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("stats", help="recompute the statistics from a saved file")
    p.add_argument("file")
    p.add_argument("--band", type=float, nargs=2, default=(1.0, 10.0), metavar=("LO", "HI"))
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_stats)

    p = sub.add_parser("gui", help="start the web GUI")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8771)
    p.add_argument("--show", action="store_true", help="open a browser")
    p.set_defaults(fn=cmd_gui)
    return ap


def main(argv: Optional[list] = None) -> int:
    a = build_parser().parse_args(argv)
    try:
        return int(a.fn(a) or 0)
    except (RuntimeError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
