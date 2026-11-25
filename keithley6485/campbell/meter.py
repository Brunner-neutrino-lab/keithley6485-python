"""Keithley 6485 driver for Campbell (shot-noise) runs.

Every instrument fact here was checked against the Model 6485 Picoammeter
Instruction Manual, 6485-901-01 Rev. A (Nov 2001); page numbers refer to it.

- Buffer: 1 to 2500 readings, ``TRAC:POIN`` (Table 6-1). Each reading has a
  time stamp; with ``TRAC:TST:FORM ABS`` the first reading of a buffer is at 0
  and the others are referenced to it (p. 6-4).
- ``TRAC:DATA?`` sends one to four elements per reading, chosen and ordered by
  ``FORM:ELEM`` (p. 6-6). The default is all four (READ, UNIT, TIME, STAT);
  UNIT is appended to the reading ("+1.040564E-06A"), so the default string has
  three comma-separated fields per reading (Figure 13-1). This driver sets
  ``FORM:ELEM READ,TIME,STAT`` and parses in strides of three.
- Over RS-232 only ASCII data is allowed (p. 13-4).
- Overflow/overvoltage readings are +9.9E37, NAN is +9.91E37; the status word
  is 16 bits sent in decimal: bit 0 overflow, 1 averaging filter, 2 math,
  3 null, 4 limits, 7 overvoltage, 9 zero check, 10 zero correct (p. 13-6).
- While the meter is in the trigger model, commands other than ABORt, *RST,
  SYST:PRES, DCL/SDC/IFC wait until it is idle (p. 7-4, 11-3). So
  ``TRAC:POIN:ACT?`` cannot be polled during a burst: the driver sends
  ``*OPC?`` after INIT and waits for its "1".
- Filters: median (``MED``) and digital (``AVER``, with ``AVER:ADV``) only
  (Table 4-5). The 6485 has no damping filter command.
- NPLC: the SCPI tables (4-4, 14-4) give 0.01 to 6 (60 Hz) / 5 (50 Hz); the
  front-panel NPLC menu and the specifications give 0.01 to 60 / 50. This driver
  allows 0.01 to 60 / 50 and lets the error queue and the readback decide.
- Ranges 2 nA to 20 mA in decades of 2/20/200 (Table 4-3); full scale is 5 %
  over range (p. 4-2).
- RS-232: 300 to 57600 baud, no hardware handshake, optional XON/XOFF
  (p. 9-17, A-2).
- Buffer statistics ``CALC3:FORM MEAN|SDEV`` + ``CALC3:DATA?`` exist (p. 6-7);
  they are not used here because the raw readings are kept.

Nothing in this module sends *RST.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from .stats import OVERFLOW_A
from .transport import Aborted, TransportTimeout

IDN_MODEL = "MODEL 6485"
RANGES_A = (2e-9, 20e-9, 200e-9, 2e-6, 20e-6, 200e-6, 2e-3, 20e-3)
BUFFER_MAX = 2500
NPLC_MIN = 0.01
NPLC_MAX = {60.0: 60.0, 50.0: 50.0}
ELEMENTS = "READ,TIME,STAT"
CHARS_PER_READING = 42     # "+1.234567E-09,+1.234567E+01,+0.000000E+00," (Figure 13-1 format)


class MeterError(RuntimeError):
    """The meter reported an error, did not answer, or read back a wrong setting."""

    def __init__(self, what: str, errors=()):
        self.what = what
        self.errors = list(errors)
        msg = what if not self.errors else f"{what}: " + "; ".join(self.errors)
        super().__init__(msg)


# key, query, kind. Query forms are the short forms with optional nodes spelled out.
SNAPSHOT_QUERIES = (
    ("idn", "*IDN?", "str"),
    ("line_freq_hz", "SYST:LFR?", "float"),
    ("line_freq_auto", "SYST:LFR:AUTO:STAT?", "bool"),
    ("zero_check", "SYST:ZCH:STAT?", "bool"),
    ("zero_correct", "SYST:ZCOR:STAT?", "bool"),
    ("autozero", "SYST:AZER:STAT?", "bool"),
    ("range_a", "CURR:RANG?", "float"),
    ("autorange", "CURR:RANG:AUTO?", "bool"),
    ("nplc", "CURR:NPLC?", "float"),
    ("avg_state", "AVER:STAT?", "bool"),
    ("avg_count", "AVER:COUN?", "float"),
    ("avg_tcontrol", "AVER:TCON?", "name"),
    ("avg_adv_state", "AVER:ADV:STAT?", "bool"),
    ("avg_adv_ntol", "AVER:ADV:NTOL?", "float"),
    ("median_state", "MED:STAT?", "bool"),
    ("median_rank", "MED:RANK?", "float"),
    ("trig_delay_s", "TRIG:DEL?", "float"),
    ("trig_delay_auto", "TRIG:DEL:AUTO?", "bool"),
    ("trig_source", "TRIG:SOUR?", "name"),
    ("trig_count", "TRIG:COUN?", "float"),
    ("arm_source", "ARM:SOUR?", "name"),
    ("arm_count", "ARM:COUN?", "float"),
    ("format_data", "FORM:DATA?", "name"),
    ("format_elements", "FORM:ELEM?", "name"),
    ("trace_tstamp_format", "TRAC:TST:FORM?", "name"),
    ("trace_points", "TRAC:POIN?", "float"),
    ("trace_feed", "TRAC:FEED?", "name"),
    ("trace_feed_control", "TRAC:FEED:CONT?", "name"),
    ("display_enabled", "DISP:ENAB?", "bool"),
    ("display_digits", "DISP:DIG?", "float"),
    ("calc1_state", "CALC:STAT?", "bool"),
    ("null_state", "CALC2:NULL:STAT?", "bool"),
    ("limit1_state", "CALC2:LIM:STAT?", "bool"),
    ("limit2_state", "CALC2:LIM2:STAT?", "bool"),
)


def nominal_range(value_a: float) -> float:
    """The range (Table 4-3) nearest a readback in log space."""
    v = abs(float(value_a))
    if not v > 0:
        return float("nan")
    return min(RANGES_A, key=lambda r: abs(math.log(r / v)))


def _parse(kind: str, text: str):
    t = text.strip().strip('"')
    if kind == "bool":
        return t not in ("0", "OFF", "+0", "0.000000E+00", "+0.000000E+00")
    if kind == "float":
        return float(t)
    if kind == "name":
        return t.upper()
    return t


def _num_or_inf(v: float) -> str:
    return "INF" if v >= 0.99 * OVERFLOW_A else f"{int(round(v))}"


def parse_trace(raw: str, n_expected: Optional[int] = None, n_elements: int = 3):
    """Parse a TRAC:DATA? response taken with FORM:ELEM READ,TIME,STAT.

    Returns (i_a, t_s, status). A trailing unit letter on the reading is
    tolerated. Raises MeterError when the field count is not a multiple of the
    element count or does not match ``n_expected``.
    """
    fields = [f.strip() for f in raw.strip().split(",")]
    if fields == [""]:
        fields = []
    if len(fields) % n_elements:
        raise MeterError(f"TRAC:DATA? returned {len(fields)} fields, not a multiple of {n_elements}")
    n = len(fields) // n_elements
    if n_expected is not None and n != n_expected:
        raise MeterError(f"TRAC:DATA? returned {n} readings, expected {n_expected}")
    arr = np.array(fields, dtype=object).reshape(n, n_elements) if n else np.empty((0, n_elements), object)
    i_a = np.array([float(v.rstrip("Aa")) for v in arr[:, 0]], dtype=np.float64)
    t_s = np.array([float(v) for v in arr[:, 1]], dtype=np.float64)
    status = np.array([int(round(float(v))) for v in arr[:, 2]], dtype=np.int64)
    return i_a, t_s, status


@dataclass
class Burst:
    i_a: np.ndarray
    t_s: np.ndarray
    status: np.ndarray
    n_requested: int
    host_t_start: float          # time.time() before INIT was written
    host_t_init: float           # after INIT was written
    host_t_end: float            # *OPC? answered: the burst is over
    host_t_data: float           # TRAC:DATA? fully received
    t_abs_last_s: float          # SENS:DATA? time stamp of the last reading (instrument timer), NaN if unavailable
    aborted: bool = False
    warnings: list = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.i_a)

    @property
    def acquire_s(self) -> float:
        return self.host_t_end - self.host_t_init

    @property
    def transfer_s(self) -> float:
        return self.host_t_data - self.host_t_end


@dataclass
class RunData:
    bursts: list = field(default_factory=list)

    def _cat(self, attr, dtype):
        if not self.bursts:
            return np.array([], dtype)
        return np.concatenate([getattr(b, attr) for b in self.bursts]).astype(dtype)

    @property
    def i_a(self):
        return self._cat("i_a", np.float64)

    @property
    def t_s(self):
        return self._cat("t_s", np.float64)

    @property
    def status(self):
        return self._cat("status", np.int64)

    @property
    def burst(self):
        if not self.bursts:
            return np.array([], np.int32)
        return np.concatenate([np.full(b.n, k, np.int32) for k, b in enumerate(self.bursts)])

    @property
    def n(self) -> int:
        return sum(b.n for b in self.bursts)

    def burst_starts_s(self) -> tuple[np.ndarray, str]:
        """Start time of each burst's first reading relative to the first burst, and its source.

        "instrument": from the SENS:DATA? time stamp of each burst's last reading
        (the instrument timer), used when every burst has one and they increase;
        "host": the host clock just after INIT (carries a few ms of latency).
        """
        b = self.bursts
        if not b:
            return np.array([]), "none"
        abs_first = np.array([x.t_abs_last_s - (x.t_s[-1] - x.t_s[0]) if x.n else np.nan for x in b])
        if np.all(np.isfinite(abs_first)) and (len(b) == 1 or np.all(np.diff(abs_first) > 0)):
            ends_ok = all(abs_first[k] >= abs_first[k - 1] + (b[k - 1].t_s[-1] - b[k - 1].t_s[0])
                          for k in range(1, len(b)))
            if ends_ok:
                return abs_first - abs_first[0], "instrument"
        host = np.array([x.host_t_init for x in b])
        return host - host[0], "host"

    def t_run_s(self) -> np.ndarray:
        """One time axis for the whole run: burst start (see ``burst_starts_s``) + instrument stamp."""
        if not self.bursts:
            return np.array([])
        starts, _src = self.burst_starts_s()
        return np.concatenate([st + (x.t_s - (x.t_s[0] if x.n else 0.0)) for st, x in zip(starts, self.bursts)])

    def burst_table(self) -> dict:
        b = self.bursts
        dt = [float((x.t_s[-1] - x.t_s[0]) / (x.n - 1)) if x.n > 1 else float("nan") for x in b]
        gap_host = [0.0] + [b[k].host_t_init - (b[k - 1].host_t_init + (b[k - 1].t_s[-1] - b[k - 1].t_s[0]) + dt[k - 1])
                            if b[k - 1].n else float("nan") for k in range(1, len(b))]
        gap_instr = [float("nan")]
        for k in range(1, len(b)):
            prev, cur = b[k - 1], b[k]
            if cur.n and prev.n and np.isfinite(cur.t_abs_last_s) and np.isfinite(prev.t_abs_last_s):
                first = cur.t_abs_last_s - (cur.t_s[-1] - cur.t_s[0])
                gap_instr.append(first - prev.t_abs_last_s - dt[k - 1])
            else:
                gap_instr.append(float("nan"))
        starts, src = self.burst_starts_s()
        return {
            "start_s": starts,
            "start_from_instrument": np.full(len(b), src == "instrument"),
            "n": np.array([x.n for x in b], np.int32),
            "n_requested": np.array([x.n_requested for x in b], np.int32),
            "host_t_start": np.array([x.host_t_start for x in b]),
            "host_t_init": np.array([x.host_t_init for x in b]),
            "host_t_end": np.array([x.host_t_end for x in b]),
            "host_t_data": np.array([x.host_t_data for x in b]),
            "t_abs_last_s": np.array([x.t_abs_last_s for x in b]),
            "dt_mean_s": np.array(dt),
            "gap_host_s": np.array(gap_host),
            "gap_instr_s": np.array(gap_instr),
            "acquire_s": np.array([x.acquire_s for x in b]),
            "transfer_s": np.array([x.transfer_s for x in b]),
            "aborted": np.array([x.aborted for x in b], bool),
        }

    def throughput(self) -> dict:
        if not self.bursts:
            return {}
        tab = self.burst_table()
        full = tab["n"] > 1
        return {
            "readings_per_s": float(np.nanmedian(1.0 / tab["dt_mean_s"][full])) if full.any() else float("nan"),
            "acquire_s_per_burst": float(np.mean(tab["acquire_s"])),
            "transfer_s_per_burst": float(np.mean(tab["transfer_s"])),
            "transfer_s_per_reading": float(np.sum(tab["transfer_s"]) / max(1, self.n)),
            "gap_host_s_mean": float(np.nanmean(tab["gap_host_s"][1:])) if len(self.bursts) > 1 else float("nan"),
            "duty_cycle": float(np.sum(tab["acquire_s"]) / max(1e-9, (self.bursts[-1].host_t_data - self.bursts[0].host_t_start))),
        }


class CampbellMeter:
    """6485 over a line transport (``SerialTransport`` or ``SimTransport``).

    Parameters
    ----------
    transport : the line transport.
    query_timeout_s : how long to wait for an ordinary reply.
    drain_each : drain the error queue after every command and query.
    log : optional callable(str) for a command log.
    """

    def __init__(self, transport, query_timeout_s: float = 5.0, drain_each: bool = True,
                 log: Optional[Callable[[str], None]] = None):
        self.t = transport
        self.query_timeout_s = float(query_timeout_s)
        self.drain_each = drain_each
        self._log = log
        self.idn = ""
        self.found_errors: list[str] = []
        self.line_freq_hz = 60.0
        self.configured: Optional[dict] = None

    # ------------------------------------------------------------------ basics
    def log(self, s: str) -> None:
        if self._log is not None:
            self._log(s)

    def connect(self) -> str:
        """Identify the meter and empty its error queue. Sends no *RST."""
        self.t.flush_input()
        self.idn = ""
        seen = []
        for _attempt in range(2):
            self.t.write("*IDN?")
            try:
                # skip stale lines a previous program left in the meter's output queue
                for _ in range(5):
                    line = self.t.readline(3.0)
                    if IDN_MODEL in line.upper():
                        self.idn = line
                        break
                    seen.append(line)
            except TransportTimeout:
                pass
            if self.idn:
                break
        if not self.idn:
            if seen:
                raise MeterError(f"not a 6485, or not answering *IDN?: got {seen[:3]!r}")
            hint = ""
            lines = self.t.modem_lines() if hasattr(self.t, "modem_lines") else {}
            if lines and not any(lines.values()):
                # the 6485 ties RTS to CTS and DTR to DSR and DCD in its connector (Table 9-2),
                # so with a cable to the meter these follow our own RTS/DTR
                hint = ("; CTS, DSR and CD are all low, so no 6485 seems to be on this port "
                        "(the meter loops RTS->CTS and DTR->DSR/DCD), unless the adapter does not report them")
            raise MeterError("no reply to *IDN? within 3 s: check the port, baud rate, "
                             "terminator and that the meter is in SCPI/RS-232 mode" + hint)
        self.found_errors = self.drain_errors()
        if self.found_errors:
            self.log(f"errors already in the queue at connect: {self.found_errors}")
        try:
            self.line_freq_hz = float(self.query("SYST:LFR?"))
        except (MeterError, ValueError):
            pass
        return self.idn

    def close(self) -> None:
        try:
            self.t.write("SYST:LOC")
        except Exception:
            pass
        self.t.close()

    def _raw_query(self, q: str, timeout: float, **kw) -> str:
        self.t.write(q)
        return self.t.readline(timeout, **kw)

    def drain_errors(self, limit: int = 50) -> list[str]:
        """Read SYST:ERR? until "0,No error"; return the other messages."""
        errs = []
        for _ in range(limit):
            try:
                r = self._raw_query("SYST:ERR?", self.query_timeout_s)
            except TransportTimeout:
                raise MeterError("no reply to SYST:ERR?") from None
            try:
                code = int(r.split(",", 1)[0])
            except ValueError:
                raise MeterError(f"unexpected SYST:ERR? reply {r!r}") from None
            if code == 0:
                return errs
            errs.append(r)
        raise MeterError(f"error queue did not empty after {limit} reads", errs)

    def command(self, cmd: str) -> None:
        self.log(f"> {cmd}")
        self.t.write(cmd)
        if self.drain_each:
            errs = self.drain_errors()
            if errs:
                raise MeterError(cmd, errs)

    def query(self, q: str, timeout: Optional[float] = None, **kw) -> str:
        try:
            r = self._raw_query(q, self.query_timeout_s if timeout is None else timeout, **kw)
        except TransportTimeout:
            errs = self.drain_errors()
            raise MeterError(f"no reply to {q}", errs) from None
        if self.drain_each:
            errs = self.drain_errors()
            if errs:
                raise MeterError(q, errs)
        self.log(f"< {q} {r[:80]}")
        return r

    # ------------------------------------------------------------------ settings
    def settings_snapshot(self) -> dict:
        """Every setting this driver touches, read back by query, plus *IDN?,
        line frequency and the zero-correct state. A query the meter refuses is
        recorded as None and listed under "_refused"."""
        snap: dict = {}
        refused = []
        for key, q, kind in SNAPSHOT_QUERIES:
            try:
                snap[key] = _parse(kind, self.query(q))
            except (MeterError, ValueError) as e:
                snap[key] = None
                refused.append(f"{q}: {e}")
        snap["_refused"] = refused
        snap["_found_errors"] = list(self.found_errors)
        snap["_time"] = time.time()
        if snap.get("line_freq_hz"):
            self.line_freq_hz = float(snap["line_freq_hz"])
        return snap

    def configure_campbell(self, range_a: float, nplc: float, autozero: bool,
                           display_off: bool = True, zero_check: bool = False) -> dict:
        """Set the meter up for raw, unfiltered, fixed-range readings into the buffer.

        Returns the snapshot read back afterwards; raises MeterError on any
        error-queue entry or a readback that differs from the request.
        """
        r = nominal_range(range_a)
        if not (abs(r - range_a) / range_a < 0.01):
            raise ValueError(f"range {range_a:g} A is not one of {RANGES_A}")
        hi = NPLC_MAX.get(self.line_freq_hz, 60.0)
        if not (NPLC_MIN <= nplc <= hi):
            raise ValueError(f"NPLC {nplc:g} outside {NPLC_MIN}..{hi:g} at {self.line_freq_hz:g} Hz")
        cmds = [
            "CURR:RANG:AUTO OFF",
            f"CURR:RANG {r:.1E}",
            f"CURR:NPLC {nplc:g}",
            "AVER:ADV:STAT OFF",
            "AVER:STAT OFF",
            "MED:STAT OFF",
            f"SYST:AZER:STAT {'ON' if autozero else 'OFF'}",
            "TRIG:DEL:AUTO OFF",
            "TRIG:DEL 0",
            "TRIG:SOUR IMM",
            "ARM:SOUR IMM",
            "ARM:COUN 1",
            "TRIG:COUN 1",
            "FORM:DATA ASC",
            f"FORM:ELEM {ELEMENTS}",
            "TRAC:TST:FORM ABS",
            "TRAC:FEED:CONT NEV",
            "TRAC:FEED SENS",
            f"DISP:ENAB {'OFF' if display_off else 'ON'}",
            f"SYST:ZCH:STAT {'ON' if zero_check else 'OFF'}",
        ]
        for c in cmds:
            self.command(c)
        snap = self.settings_snapshot()
        want = {
            "autorange": False, "avg_state": False, "avg_adv_state": False, "median_state": False,
            "autozero": bool(autozero), "trig_delay_auto": False, "zero_check": bool(zero_check),
            "display_enabled": not display_off,
        }
        bad = [f"{k}={snap.get(k)!r} (want {v!r})" for k, v in want.items() if snap.get(k) != v]
        if snap.get("range_a") is None or nominal_range(snap["range_a"]) != r:
            bad.append(f"range_a={snap.get('range_a')!r} (want {r:g})")
        if snap.get("nplc") is None or abs(snap["nplc"] - nplc) > 1e-6 * max(1.0, nplc) + 1e-9:
            bad.append(f"nplc={snap.get('nplc')!r} (want {nplc:g})")
        if snap.get("trig_delay_s") != 0.0:
            bad.append(f"trig_delay_s={snap.get('trig_delay_s')!r} (want 0)")
        for k, v in (("trig_source", "IMM"), ("arm_source", "IMM"), ("trace_tstamp_format", "ABS"),
                     ("trace_feed", "SENS"), ("format_data", "ASC")):
            if not str(snap.get(k) or "").startswith(v):
                bad.append(f"{k}={snap.get(k)!r} (want {v})")
        if snap.get("arm_count") != 1.0:
            bad.append(f"arm_count={snap.get('arm_count')!r} (want 1)")
        elems = [e.strip()[:4] for e in str(snap.get("format_elements") or "").split(",")]
        if elems != ["READ", "TIME", "STAT"]:
            bad.append(f"format_elements={snap.get('format_elements')!r} (want {ELEMENTS})")
        if bad:
            raise MeterError("readback differs from the request", bad)
        snap["_request"] = {"range_a": r, "nplc": float(nplc), "autozero": bool(autozero),
                            "display_off": bool(display_off), "zero_check": bool(zero_check)}
        self.configured = snap
        return snap

    def restore(self, snapshot: dict) -> dict:
        """Put back every setting in ``snapshot`` (from ``settings_snapshot``).

        The buffer's contents cannot be restored. Returns {key: (wanted, now)}
        for settings that still differ afterwards (empty: restored as found).
        """
        s = snapshot

        def has(*keys):
            return all(s.get(k) is not None for k in keys)

        def onoff(v):
            return "ON" if v else "OFF"

        cmds = []
        if has("nplc"):
            cmds.append(f"CURR:NPLC {s['nplc']:g}")
        if has("range_a"):
            cmds.append(f"CURR:RANG {nominal_range(s['range_a']):.1E}")
        if has("autorange"):
            cmds.append(f"CURR:RANG:AUTO {onoff(s['autorange'])}")
        if has("avg_count"):
            cmds.append(f"AVER:COUN {int(round(s['avg_count']))}")
        if has("avg_tcontrol"):
            cmds.append(f"AVER:TCON {s['avg_tcontrol']}")
        if has("avg_adv_ntol"):
            cmds.append(f"AVER:ADV:NTOL {s['avg_adv_ntol']:g}")
        if has("avg_adv_state"):
            cmds.append(f"AVER:ADV:STAT {onoff(s['avg_adv_state'])}")
        if has("avg_state"):
            cmds.append(f"AVER:STAT {onoff(s['avg_state'])}")
        if has("median_rank"):
            cmds.append(f"MED:RANK {int(round(s['median_rank']))}")
        if has("median_state"):
            cmds.append(f"MED:STAT {onoff(s['median_state'])}")
        if has("autozero"):
            cmds.append(f"SYST:AZER:STAT {onoff(s['autozero'])}")
        if has("trig_delay_s"):
            cmds.append(f"TRIG:DEL {s['trig_delay_s']:g}")
        if has("trig_delay_auto"):
            cmds.append(f"TRIG:DEL:AUTO {onoff(s['trig_delay_auto'])}")
        if has("trig_source"):
            cmds.append(f"TRIG:SOUR {s['trig_source']}")
        if has("trig_count"):
            cmds.append(f"TRIG:COUN {_num_or_inf(s['trig_count'])}")
        if has("arm_source"):
            cmds.append(f"ARM:SOUR {s['arm_source']}")
        if has("arm_count"):
            cmds.append(f"ARM:COUN {_num_or_inf(s['arm_count'])}")
        if has("format_elements"):
            cmds.append(f"FORM:ELEM {s['format_elements']}")
        if has("trace_tstamp_format"):
            cmds.append(f"TRAC:TST:FORM {s['trace_tstamp_format']}")
        if has("trace_points"):
            cmds.append(f"TRAC:POIN {int(round(s['trace_points']))}")
        if has("trace_feed"):
            cmds.append(f"TRAC:FEED {s['trace_feed']}")
        if has("trace_feed_control"):
            cmds.append(f"TRAC:FEED:CONT {s['trace_feed_control']}")
        if has("display_enabled"):
            cmds.append(f"DISP:ENAB {onoff(s['display_enabled'])}")
        if has("zero_check"):
            cmds.append(f"SYST:ZCH:STAT {onoff(s['zero_check'])}")
        errors = []
        for c in cmds:
            try:
                self.command(c)
            except MeterError as e:
                errors.append(str(e))
        now = self.settings_snapshot()
        diff = {}
        for key, _q, _k in SNAPSHOT_QUERIES:
            want, got = s.get(key), now.get(key)
            if want is None or key == "idn":
                continue
            if key == "range_a":
                same = got is not None and nominal_range(got) == nominal_range(want)
            elif isinstance(want, float) and isinstance(got, float):
                same = math.isclose(want, got, rel_tol=1e-6, abs_tol=1e-12)
            else:
                same = want == got
            if not same:
                diff[key] = (want, got)
        if errors:
            diff["_errors"] = errors
        self.configured = None
        return diff

    # ------------------------------------------------------------------ acquisition
    def expected_dt_s(self, nplc: float, autozero: bool) -> float:
        """A generous guess of the reading interval, for timeouts only."""
        t_int = nplc / self.line_freq_hz
        return t_int * (3.0 if autozero else 1.5) + 2e-3

    def acquire_burst(self, n: int, *, abort: Optional[Callable[[], bool]] = None,
                      on_phase: Optional[Callable[[str, float], None]] = None,
                      timeout_s: Optional[float] = None, read_abs_time: bool = True) -> Burst:
        """Fill the buffer with n readings (one INIT) and read it back.

        ``abort`` is polled while the meter is measuring; when it returns True the
        burst is stopped with ABORt and the readings taken so far are kept. The
        transfer is never cut short. ``on_phase(phase, fraction)`` reports
        "acquire" and "transfer" progress.
        """
        if not 1 <= n <= BUFFER_MAX:
            raise ValueError(f"burst size {n} outside 1..{BUFFER_MAX}")
        cfg = self.configured or {}
        nplc = float(cfg.get("nplc") or 1.0)
        az = bool(cfg.get("autozero"))
        self.command("TRAC:FEED:CONT NEV")
        self.command("TRAC:CLE")
        self.command(f"TRAC:POIN {n}")
        self.command("TRAC:FEED SENS")
        self.command(f"TRIG:COUN {n}")
        self.command("TRAC:FEED:CONT NEXT")
        # real seconds per instrument second: 1 on the bench, the simulator's time scale otherwise
        scale = float(getattr(self.t, "time_scale", 1.0))
        expect = n * self.expected_dt_s(nplc, az) * scale
        wait = timeout_s if timeout_s is not None else 1.5 * expect + 10.0
        warnings = []
        host_t_start = time.time()
        self.t.write("INIT")
        host_t_init = time.time()
        self.t.write("*OPC?")
        aborted = False
        t_mono = time.monotonic()

        def acq_abort():
            if on_phase is not None:
                on_phase("acquire", min(1.0, (time.monotonic() - t_mono) / max(expect, 1e-3)))
            return abort is not None and abort()

        try:
            r = self.t.readline(wait, abort=acq_abort)
        except Aborted:
            aborted = True
            self.t.write("ABOR")
            r = self.t.readline(10.0)
        except TransportTimeout:
            self.t.write("ABOR")
            try:
                self.t.readline(10.0)
            except TransportTimeout:
                pass
            raise MeterError(f"burst of {n} did not finish within {wait:.0f} s", self.drain_errors()) from None
        host_t_end = time.time()
        if r.strip() != "1":
            warnings.append(f"*OPC? answered {r!r}")
        errs = self.drain_errors()
        if errs:
            raise MeterError("after INIT", errs)
        n_act = int(round(float(self.query("TRAC:POIN:ACT?"))))
        if n_act == 0:
            empty = np.array([])
            return Burst(empty, empty, np.array([], np.int64), n, host_t_start, host_t_init,
                         host_t_end, time.time(), float("nan"), aborted, warnings)
        baud = getattr(self.t, "baud", 9600)
        chars = n_act * (CHARS_PER_READING + 2) + 64
        t_xfer = chars * 10.0 / baud

        def xfer_progress(nbytes):
            if on_phase is not None:
                on_phase("transfer", min(1.0, nbytes / chars))

        raw = self.query("TRAC:DATA?", timeout=1.5 * t_xfer * scale + 10.0, on_bytes=xfer_progress)
        host_t_data = time.time()
        i_a, t_s, status = parse_trace(raw, n_act)
        if n_act > 1 and not np.all(np.diff(t_s) > 0):
            warnings.append("time stamps not strictly increasing")
        if not aborted and n_act != n:
            warnings.append(f"buffer holds {n_act} readings, {n} requested")
        t_abs = float("nan")
        if read_abs_time:
            try:
                last = self.query("SENS:DATA?").split(",")
                t_abs = float(last[1])
            except (MeterError, ValueError, IndexError) as e:
                warnings.append(f"SENS:DATA? time stamp unavailable: {e}")
        if on_phase is not None:
            on_phase("done", 1.0)
        return Burst(i_a, t_s, status, n, host_t_start, host_t_init, host_t_end, host_t_data,
                     t_abs, aborted, warnings)

    def acquire(self, n_total: int, burst_size: int = BUFFER_MAX, *,
                abort: Optional[Callable[[], bool]] = None,
                on_burst: Optional[Callable[[RunData, Burst], None]] = None,
                on_phase: Optional[Callable[[int, int, str, float], None]] = None,
                read_abs_time: bool = True) -> RunData:
        """Chain bursts until n_total readings (or ``abort``). Bursts are separated
        by the transfer time; their index and the gaps are in ``RunData``."""
        if n_total < 1:
            raise ValueError("n_total must be >= 1")
        burst_size = int(min(max(1, burst_size), BUFFER_MAX))
        n_bursts = math.ceil(n_total / burst_size)
        run = RunData()
        left = n_total
        for k in range(n_bursts):
            if abort is not None and abort():
                break
            n = min(burst_size, left)
            ph = (lambda phase, frac, k=k: on_phase(k, n_bursts, phase, frac)) if on_phase else None
            b = self.acquire_burst(n, abort=abort, on_phase=ph, read_abs_time=read_abs_time)
            run.bursts.append(b)
            left -= b.n
            if on_burst is not None:
                on_burst(run, b)
            if b.aborted:
                break
        return run
