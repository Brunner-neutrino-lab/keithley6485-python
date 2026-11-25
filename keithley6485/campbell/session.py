"""One meter, one run at a time: the logic shared by the CLI and the GUI.

connect -> snapshot "as found" -> [configure -> settle -> bursts -> restore ->
stats -> save] per run -> disconnect (restoring first if needed).
"""

from __future__ import annotations

import dataclasses
import json
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from . import guard
from .meter import RANGES_A, CampbellMeter, MeterError, RunData
from .sim import SimPhysics, SimulatedK6485
from .stats import E_CHARGE, campbell_stats
from .store import DEFAULT_DIR, default_path, write_run
from .transport import SerialTransport, SimTransport

SIM_PORT = "simulation"
NOTE_FIELDS = ("operator", "kind", "laser", "nd", "photodiode", "bias", "box",
               "resistor_ohm", "battery_v", "temperature_k", "text")
RUN_KINDS = ("photocurrent", "dark", "zero-check floor", "resistor control", "johnson")


@dataclass
class RunSettings:
    range_a: float = 20e-9
    nplc: float = 1.0
    autozero: bool = False
    display_off: bool = True
    zero_check: bool = False      # True for a zero-check-on floor run
    n_total: int = 2500
    burst_size: int = 2500
    settle_s: float = 2.0         # wait after configuring (zero check off) before the first burst
    restore_after: bool = True
    autosave: bool = True
    out_dir: str = str(DEFAULT_DIR)
    name: str = "campbell"
    band_lo_hz: float = 1.0
    band_hi_hz: float = 10.0
    read_abs_time: bool = True


@dataclass
class Connection:
    port: str = ""
    baud: int = 9600
    read_term: str = "\r"
    write_term: str = "\r"
    xonxoff: bool = False
    sim: Optional[SimPhysics] = None
    time_scale: float = 0.0


class Session:
    def __init__(self, log: Optional[Callable[[str], None]] = None):
        self._log_cb = log
        self.lock = threading.RLock()
        self.messages: deque = deque(maxlen=1000)
        self.meter: Optional[CampbellMeter] = None
        self.simulator: Optional[SimulatedK6485] = None
        self.conn: Optional[Connection] = None
        self.found: Optional[dict] = None
        self.settings_used: Optional[dict] = None
        self.request: Optional[RunSettings] = None
        self.run: Optional[RunData] = None
        self.stats: Optional[dict] = None
        self.throughput: dict = {}
        self.restore_diff: Optional[dict] = None
        self.error: Optional[str] = None
        self.last_path: Optional[Path] = None
        self.saved_version = -1
        self.notes = {k: "" for k in NOTE_FIELDS}
        self.notes["kind"] = "photocurrent"
        self.run_notes: dict = {}
        self.history: list[dict] = []
        self.sequence: list = []
        self.seq_index = 0
        self.last_check: Optional[dict] = None
        self.phase = "disconnected"
        self.progress = {"burst": 0, "n_bursts": 0, "phase": "", "frac": 0.0}
        self.version = 0
        self.restored = True
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    # ------------------------------------------------------------------ log
    def log(self, s: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {s}"
        self.messages.append(line)
        if self._log_cb is not None:
            self._log_cb(line)

    @property
    def connected(self) -> bool:
        return self.meter is not None

    @property
    def stop_requested(self) -> bool:
        return self._stop.is_set()

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------ connection
    def connect(self, conn: Connection, check_daq: bool = True) -> str:
        if self.connected:
            raise RuntimeError("already connected")
        self.phase = "connecting"
        try:
            if conn.port == SIM_PORT:
                self.simulator = SimulatedK6485(conn.sim or SimPhysics(), time_scale=conn.time_scale)
                transport = SimTransport(self.simulator, baud=conn.baud)
            else:
                problems, notes = guard.port_problems(conn.port, check_daq=check_daq)
                for n in notes:
                    self.log(n)
                if problems:
                    raise RuntimeError("refusing to open the port: " + "; ".join(problems))
                transport = SerialTransport(conn.port, conn.baud, conn.read_term, conn.write_term,
                                            xonxoff=conn.xonxoff)
            return self._open(transport, conn)
        except Exception:
            self.phase = "disconnected"
            self.simulator = None
            raise

    def attach(self, transport, label: str = "external", baud: int = 9600) -> str:
        """Use a transport someone else owns (e.g. the ETS DAQ hub's open resource).

        No port guard: the owner has the port. ``transport`` needs ``write(cmd)``,
        ``readline(timeout, abort=None, on_bytes=None)``, ``flush_input()`` and
        ``close()`` (make it a no-op to keep the resource open); ``baud`` and
        ``describe()`` are optional.
        """
        if self.connected:
            raise RuntimeError("already connected")
        self.phase = "connecting"
        try:
            return self._open(transport, Connection(port=label, baud=int(getattr(transport, "baud", baud))))
        except Exception:
            self.phase = "disconnected"
            raise

    def _open(self, transport, conn: Connection) -> str:
        meter = CampbellMeter(transport)
        try:
            idn = meter.connect()
            desc = transport.describe() if hasattr(transport, "describe") else conn.port
            self.log(f"connected: {idn} ({desc})")
            if meter.found_errors:
                self.log(f"error queue at connect (cleared): {meter.found_errors}")
            t0 = time.monotonic()
            self.found = meter.settings_snapshot()
            self.log(f"settings as found read back in {time.monotonic() - t0:.1f} s"
                     + (f"; refused: {self.found['_refused']}" if self.found["_refused"] else ""))
        except Exception:
            transport.close()
            raise
        self.meter = meter
        self.conn = conn
        self.restored = True
        self.phase = "idle"
        return idn

    def disconnect(self, restore: bool = True) -> None:
        if self.running:
            raise RuntimeError("a run is in progress: stop it first")
        if self.meter is None:
            return
        try:
            if restore and not self.restored and self.found:
                self._restore()
        finally:
            self.meter.close()
            self.meter = None
            self.simulator = None
            self.phase = "disconnected"
            self.log("disconnected")

    def _restore(self) -> None:
        try:
            diff = self.meter.restore(self.found)
        except Exception as e:
            self.restore_diff = {"_errors": [str(e)]}
            self.log(f"restore FAILED: {e}")
            return
        self.restore_diff = diff
        self.restored = not diff
        self.log("meter settings restored as found" if not diff else f"restore left differences: {diff}")

    # ------------------------------------------------------------------ quick check
    def check(self, range_a: float, nplc: float, autozero: bool = False, n: int = 100,
              settle_s: float = 1.0) -> dict:
        """One short burst with zero check off: the current level, the noise and range advice.

        Saves nothing. The settings are restored on disconnect (or after the next run).
        """
        if not self.connected:
            raise RuntimeError("not connected")
        if self.running:
            raise RuntimeError("a run is in progress")
        m = self.meter
        self.restored = False
        cfg = m.configure_campbell(range_a, nplc, autozero, display_off=True, zero_check=False)
        time.sleep(settle_s)
        b = m.acquire_burst(int(n), read_abs_time=False)
        st = campbell_stats(b.i_a, b.t_s, cfg["nplc"], cfg["line_freq_hz"], status=b.status)
        res = {"range_a": range_a, "nplc": nplc, "n": b.n, "n_overflow": st["n_overflow"],
               "mean_a": st.get("mean_a", float("nan")), "sd_a": st.get("sd_a", float("nan")),
               "shot_sd_a": st.get("shot_sd_nominal_a", float("nan"))}
        res["advice"] = range_advice(range_a, res["mean_a"], res["sd_a"], st["n_overflow"], b.n)
        self.last_check = res
        self.log("check: " + res["advice"])
        return res

    # ------------------------------------------------------------------ runs
    def start(self, rs: RunSettings) -> None:
        self.start_sequence([rs])

    def start_sequence(self, seq: list) -> None:
        """Run the settings one after the other (one file each); restore after the last."""
        if not self.connected:
            raise RuntimeError("not connected")
        if self.running:
            raise RuntimeError("a run is already in progress")
        if not seq:
            raise ValueError("empty sequence")
        self._stop.clear()
        self.sequence = [dataclasses.replace(r) for r in seq]
        self.seq_index = 0
        self._reset_run(self.sequence[0])
        self._thread = threading.Thread(target=self._run_sequence, args=(self.sequence,), daemon=True,
                                        name="k6485-campbell-run")
        self._thread.start()

    def _reset_run(self, rs: RunSettings) -> None:
        with self.lock:
            self.request = rs
            self.run = None
            self.stats = None
            self.throughput = {}
            self.settings_used = None
            self.error = None
            self.restore_diff = None
            self.last_path = None
            self.progress = {"burst": 0, "n_bursts": 0, "phase": "", "frac": 0.0}
            self.run_notes = dict(self.notes)
            if rs.zero_check:
                self.run_notes["kind"] = "zero-check floor"
            self.version += 1

    def stop(self) -> None:
        if self.running:
            self._stop.set()
            self.log("stop requested: the meter finishes or aborts this burst, then the data are read back")

    def wait(self, timeout: Optional[float] = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    def _on_phase(self, k, nb, phase, frac):
        self.progress = {"burst": k + 1, "n_bursts": nb, "phase": phase, "frac": frac}
        self.phase = phase if phase in ("acquire", "transfer") else self.phase

    def _on_burst(self, run: RunData, b) -> None:
        with self.lock:
            self.run = run
            self._update_stats()
            self.version += 1
        n_over = int(((b.status & 1) != 0).sum()) if b.n else 0
        rate = (b.n - 1) / (b.t_s[-1] - b.t_s[0]) if b.n > 1 and b.t_s[-1] > b.t_s[0] else float("nan")
        self.log(f"burst {len(run.bursts)}: {b.n} readings"
                 + (" (aborted)" if b.aborted else "")
                 + f", acquire {b.acquire_s:.1f} s, transfer {b.transfer_s:.1f} s, {rate:.2f} rdg/s"
                 + (f"; {n_over} OVERFLOW readings: the input current is above 1.05 x the range" if n_over else "")
                 + ("; " + "; ".join(b.warnings) if b.warnings else ""))

    def _update_stats(self) -> None:
        if self.run is None or self.run.n == 0 or self.settings_used is None:
            return
        rs = self.request
        s = self.settings_used
        self.stats = campbell_stats(self.run.i_a, self.run.t_s, s["nplc"], s["line_freq_hz"],
                                    burst=self.run.burst, status=self.run.status,
                                    band_hz=(rs.band_lo_hz, rs.band_hi_hz))
        self.throughput = self.run.throughput()

    def recompute(self, band_lo_hz: float, band_hi_hz: float) -> None:
        with self.lock:
            if self.request is not None:
                self.request.band_lo_hz, self.request.band_hi_hz = band_lo_hz, band_hi_hz
            self._update_stats()
            self.version += 1

    def _run_sequence(self, seq: list) -> None:
        try:
            for i, rs in enumerate(seq):
                if self._stop.is_set():
                    break
                self.seq_index = i
                if i:
                    self._reset_run(rs)
                if len(seq) > 1:
                    self.log(f"sequence run {i + 1}/{len(seq)}: {rs.name}")
                self._run_one(rs, last=(i == len(seq) - 1) or self._stop.is_set())
        finally:
            if not self.restored and seq[-1].restore_after and self.found and self.meter is not None:
                self._restore()
            self.phase = "idle"

    def _run_one(self, rs: RunSettings, last: bool) -> None:
        m = self.meter
        try:
            self.phase = "configuring"
            self.restored = False
            t0 = time.monotonic()
            self.settings_used = m.configure_campbell(rs.range_a, rs.nplc, rs.autozero,
                                                      display_off=rs.display_off, zero_check=rs.zero_check)
            self.log(f"configured in {time.monotonic() - t0:.1f} s: range {rs.range_a:g} A, NPLC {rs.nplc:g}, "
                     f"autozero {'on' if rs.autozero else 'off'}, zero check {'on' if rs.zero_check else 'off'}, "
                     f"line {self.settings_used['line_freq_hz']:g} Hz")
            if rs.settle_s > 0:
                self.phase = "settling"
                self._stop.wait(rs.settle_s)
            if not self._stop.is_set():
                self.phase = "acquire"
                m.acquire(rs.n_total, rs.burst_size, abort=self._stop.is_set,
                          on_burst=self._on_burst, on_phase=self._on_phase,
                          read_abs_time=rs.read_abs_time)
        except Exception as e:
            self.error = f"{type(e).__name__}: {e}"
            self.log(f"run FAILED: {self.error}")
            if not isinstance(e, (MeterError, ValueError)):
                self.log(traceback.format_exc().strip().splitlines()[-1])
        finally:
            last = last or self._stop.is_set()
            if last and rs.restore_after and self.found:
                self.phase = "restoring"
                self._restore()
            with self.lock:
                self._update_stats()
                self.version += 1
            if rs.autosave and self.run is not None and self.run.n:
                try:
                    p = self.save()
                    self.log(f"saved {p}")
                except Exception as e:
                    self.log(f"save FAILED: {type(e).__name__}: {e}")
            self.log("run finished" + (f" with {self.run.n} readings" if self.run else " with no data"))

    # ------------------------------------------------------------------ saving
    def save(self, path=None) -> Path:
        with self.lock:
            if self.run is None or self.run.n == 0:
                raise RuntimeError("no data to save")
            rs = self.request
            path = Path(path) if path else default_path(rs.name, rs.out_dir)
            extra = {f"thr_{k}": v for k, v in self.throughput.items()}
            extra.update({
                "port": self.conn.port if self.conn else "",
                "baud": self.conn.baud if self.conn else 0,
                "simulated": self.simulator is not None,
                "stopped_early": self.stop_requested,
                "error": self.error or "",
                "restored_as_found": bool(self.restore_diff is not None and not self.restore_diff),
            })
            if self.restore_diff:
                extra["restore_diff_json"] = json.dumps(
                    {k: [str(x) for x in v] if isinstance(v, (tuple, list)) else str(v)
                     for k, v in self.restore_diff.items()})
            req = dataclasses.asdict(rs)
            notes = dict(self.run_notes or self.notes)
            p = write_run(path, i_a=self.run.i_a, t_s=self.run.t_s, status=self.run.status,
                          burst=self.run.burst, bursts=self.run.burst_table(),
                          settings=self.settings_used, found=self.found, request=req,
                          stats=self.stats, notes=notes,
                          sim=self.simulator.physics_dict() if self.simulator else None,
                          extra=extra)
            self.last_path = p
            self.saved_version = self.version
            st = self.stats or {}
            row = {"run_id": p.stem, "path": str(p), "kind": notes.get("kind", ""), "nd": notes.get("nd", ""),
                   "range_a": rs.range_a, "nplc": rs.nplc, "zero_check": rs.zero_check, "n": self.run.n,
                   "mean_a": st.get("mean_a"), "sd_a": st.get("sd_a"), "acf1": (st.get("acf") or [None])[0]}
            for k in ("naive", "sdiff", "band"):
                row[f"e_{k}_over_e"] = st.get(f"e_{k}_over_e")
            self.history = [h for h in self.history if h["path"] != str(p)] + [row]
            return p

    def fit_history(self, est: str = "naive", band_hz=None) -> dict:
        """Will's analyses over the runs saved in this connection."""
        from . import analysis
        paths = [h["path"] for h in self.history]
        if not paths:
            return {}
        band = band_hz or ((self.request.band_lo_hz, self.request.band_hi_hz) if self.request else (1.0, 10.0))
        runs = analysis.load_runs(paths, band_hz=band)
        return {"fit": analysis.fit_charge(runs, est), "control": analysis.resistor_control(runs, est),
                "johnson": analysis.johnson_teff(runs, est), "table": analysis.nplc_table(runs, est)}


def build_sequence(rs: RunSettings, nplc_list=None, zc_floor: bool = False) -> list:
    """One run per NPLC (names get _<nplc>plc), each optionally followed by a zero-check floor run."""
    nplcs = [float(x) for x in (nplc_list or [rs.nplc])]
    seq = []
    for x in nplcs:
        name = f"{rs.name}_{x:g}plc" if len(nplcs) > 1 else rs.name
        run = dataclasses.replace(rs, nplc=x, name=name)
        seq.append(run)
        if zc_floor and not rs.zero_check:
            seq.append(dataclasses.replace(run, zero_check=True, name=name + "_zcfloor"))
    return seq


def range_advice(range_a: float, mean: float, sd: float, n_over: int, n: int) -> str:
    """Will's brief: keep the photocurrent in the top half to two-thirds of the range, never overflowing."""
    if n_over:
        up = [r for r in RANGES_A if r > range_a]
        return (f"{n_over}/{n} readings OVERFLOW on the {range_a:g} A range: "
                + (f"use {up[0]:g} A or less light" if up else "less light"))
    if not (mean == mean):
        return "no valid readings"
    fill = abs(mean) / range_a
    fit = [r for r in RANGES_A if abs(mean) + 6 * sd < 1.05 * r]
    best = fit[0] if fit else None
    txt = f"<I> = {mean:.4g} A, sd {sd:.3g} A: {fill:.0%} of the {range_a:g} A range"
    if 0.5 <= fill <= 0.67:
        return txt + " (in the 50-67 % window)"
    if best is not None and best < range_a:
        return txt + f"; the {best:g} A range would be more sensitive"
    if fill > 0.67:
        return txt + "; above 2/3 of the range: overflow risk, less light or the next range"
    return txt + "; below half the range: more light, or accept the lower fill"
