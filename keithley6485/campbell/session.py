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
from .meter import CampbellMeter, MeterError, RunData
from .sim import SimPhysics, SimulatedK6485
from .stats import campbell_stats
from .store import DEFAULT_DIR, default_path, write_run
from .transport import SerialTransport, SimTransport

SIM_PORT = "simulation"
NOTE_FIELDS = ("operator", "laser", "nd", "photodiode", "bias", "box", "text")


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
            meter = CampbellMeter(transport)
            try:
                idn = meter.connect()
                self.log(f"connected: {idn} ({transport.describe()})")
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
        except Exception:
            self.phase = "disconnected"
            self.simulator = None
            raise

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

    # ------------------------------------------------------------------ runs
    def start(self, rs: RunSettings) -> None:
        if not self.connected:
            raise RuntimeError("not connected")
        if self.running:
            raise RuntimeError("a run is already in progress")
        self._stop.clear()
        with self.lock:
            self.request = dataclasses.replace(rs)
            self.run = None
            self.stats = None
            self.throughput = {}
            self.settings_used = None
            self.error = None
            self.restore_diff = None
            self.last_path = None
            self.progress = {"burst": 0, "n_bursts": 0, "phase": "", "frac": 0.0}
            self.version += 1
        self._thread = threading.Thread(target=self._run, args=(self.request,), daemon=True,
                                        name="k6485-campbell-run")
        self._thread.start()

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

    def _run(self, rs: RunSettings) -> None:
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
            self.phase = "restoring"
            if rs.restore_after and self.found:
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
            self.phase = "idle"
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
            p = write_run(path, i_a=self.run.i_a, t_s=self.run.t_s, status=self.run.status,
                          burst=self.run.burst, bursts=self.run.burst_table(),
                          settings=self.settings_used, found=self.found, request=req,
                          stats=self.stats, notes=dict(self.notes),
                          sim=self.simulator.physics_dict() if self.simulator else None,
                          extra=extra)
            self.last_path = p
            self.saved_version = self.version
            return p
