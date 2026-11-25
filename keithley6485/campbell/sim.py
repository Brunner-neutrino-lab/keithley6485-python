"""A simulated Keithley 6485 at the SCPI level, for Campbell runs without a meter.

The simulator answers the same command lines the real meter gets over RS-232
(settings, queries, the error queue, the trigger model, the reading buffer and
``TRAC:DATA?`` in the ``FORM:ELEM`` order), so the driver, its parser, the
GUI and the tests all run the code path they run on the bench.

Readings model a photodiode current measured with an aperture T = NPLC/f_line:

    reading = e * Poisson(I T / e) / T  +  floor  (+ excess, offset)

optionally passed through a filter that mimics a correlated meter:

- ``ar1``: y_k = phi y_(k-1) + (1 - phi) x_k. Unit gain at DC; phi < 0 makes
  the readings anti-correlated with a spectrum rising toward Nyquist (the
  B2987 at 1 PLC looked like phi ~ -0.4).
- ``boxcar``: each aperture is M sub-conversions; a reading is the mean of the
  last L sub-conversions. L > M makes neighbouring readings overlap.

Both keep the low-frequency spectrum, so they change the plain variance but
not a periodogram taken in a band well below the filter's corner.

Timing is illustrative, not measured: dt = T * (autozero_factor if autozero
on else 1) + overhead_s, about 1000 readings/s at 0.01 PLC with autozero off
(the specification's buffer rate) and 57 readings/s at 1 PLC.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np

from .stats import E_CHARGE, OVERFLOW_A

IDN = "KEITHLEY INSTRUMENTS INC.,MODEL 6485,SIMULATED,SIM/SIM/A"

# Typical RMS noise, 6485 specifications (6485-901-01 p. A-2, note 2: 6 PLC,
# 1 sd, 100 readings, filter off, capped input).
SPEC_RMS_NOISE_A = {2e-9: 20e-15, 20e-9: 100e-15, 200e-9: 1e-12, 2e-6: 10e-12,
                    20e-6: 100e-12, 200e-6: 1e-9, 2e-3: 10e-9, 20e-3: 100e-9}
RANGES_A = tuple(sorted(SPEC_RMS_NOISE_A))

# Defaults from the "Default" columns of the 6485 SCPI tables (*RST values).
RST_STATE = {
    "zero_check": True, "zero_correct": False, "autozero": True,
    "range_a": 2e-4, "autorange": True, "nplc": 6.0,
    "avg_state": False, "avg_count": 10, "avg_tcontrol": "REP",
    "avg_adv_state": False, "avg_adv_ntol": 0.0,
    "median_state": False, "median_rank": 1,
    "trig_delay_s": 0.0, "trig_delay_auto": False, "trig_source": "IMM",
    "trig_count": 1.0, "arm_source": "IMM", "arm_count": 1.0, "arm_timer_s": 0.1,
    "format_data": "ASC", "format_elements": "READ,UNIT,TIME,STAT",
    "display_enabled": True, "display_digits": 6,
    "line_freq_hz": 60.0, "line_freq_auto": True,
    "calc1_state": False, "null_state": False, "limit1_state": False, "limit2_state": False,
}
TRACE_POWER_ON = {"trace_points": 100, "trace_feed": "SENS", "trace_feed_control": "NEV",
                  "trace_tstamp_format": "ABS"}

ERRORS = {
    -109: "Missing parameter", -113: "Undefined header", -221: "Settings conflict",
    -222: "Parameter data out of range", -224: "Illegal parameter value",
    -230: "Data corrupt or stale",
}

_LONG = {
    "SENSE": "SENS", "SENSE1": "SENS", "SENS1": "SENS", "CURRENT": "CURR", "RANGE": "RANG",
    "UPPER": "UPP", "NPLCYCLES": "NPLC", "AVERAGE": "AVER", "MEDIAN": "MED", "COUNT": "COUN",
    "TCONTROL": "TCON", "ADVANCED": "ADV", "NTOLERANCE": "NTOL", "STATE": "STAT",
    "SYSTEM": "SYST", "ZCHECK": "ZCH", "ZCORRECT": "ZCOR", "AZERO": "AZER",
    "LFREQUENCY": "LFR", "ERROR": "ERR", "ACQUIRE": "ACQ", "TRIGGER": "TRIG",
    "DELAY": "DEL", "SOURCE": "SOUR", "SEQUENCE": "SEQ", "SEQUENCE1": "SEQ", "SEQ1": "SEQ",
    "LAYER": "LAY", "LAYER1": "LAY", "LAY1": "LAY", "TIMER": "TIM", "FORMAT": "FORM",
    "ELEMENTS": "ELEM", "TRACE": "TRAC", "POINTS": "POIN", "ACTUAL": "ACT",
    "CONTROL": "CONT", "TSTAMP": "TST", "DISPLAY": "DISP", "ENABLE": "ENAB", "DIGITS": "DIG",
    "CALCULATE": "CALC", "CALCULATE1": "CALC", "CALC1": "CALC", "CALCULATE2": "CALC2",
    "CALCULATE3": "CALC3", "LIMIT": "LIM", "LIMIT1": "LIM", "LIM1": "LIM", "LIMIT2": "LIM2",
    "INITIATE": "INIT", "IMMEDIATE": "IMM", "ABORT": "ABOR", "FETCH": "FETC",
    "LATEST": "LAT", "CLEAR": "CLE", "PRESET": "PRES", "RESET": "RES", "LOCAL": "LOC",
    "REMOTE": "REM", "BORDER": "BORD", "DEFAULT": "DEF",
}
_BOOL_NODES = {"SYST:ZCH", "SYST:ZCOR", "SYST:AZER", "AVER", "AVER:ADV", "MED",
               "CURR:RANG:AUTO", "SYST:LFR:AUTO", "TRIG:DEL:AUTO"}
# Commands the 6485 executes while it is in the trigger model (section 7,
# "Idle and initiate"); everything else waits until it is idle again.
_WHILE_BUSY = {"ABOR", "*RST", "SYST:PRES"}


def canon(header: str) -> str:
    h = header.strip().upper().lstrip(":")
    if h.startswith("*"):
        return h
    nodes = [_LONG.get(n, n) for n in h.split(":") if n]
    if nodes and nodes[0] == "SENS":
        nodes = nodes[1:] or ["DATA"]
        if nodes[0] == "DATA":
            return "SENS:DATA"
    if nodes and nodes[0] == "DATA":
        if len(nodes) == 1 or nodes[1:] == ["LAT"]:
            return "SENS:DATA"
        nodes[0] = "TRAC"
    out = []
    for i, n in enumerate(nodes):
        prev = out[-1] if out else ""
        if n == "DC" and prev == "CURR":
            continue
        if n == "UPP" and prev == "RANG":
            continue
        if n == "IMM" and prev == "INIT":
            continue
        if n in ("SEQ", "LAY") and out and out[0] in ("TRIG", "ARM"):
            continue
        if n == "NEXT" and prev == "ERR":
            continue
        if n == "DATA" and prev == "FORM" and len(out) == 1:
            continue
        out.append(n)
    if out and out[0] in ("NPLC", "RANG"):
        out = ["CURR"] + out
    c = ":".join(out)
    if c.endswith(":STAT") and c[:-5] in _BOOL_NODES:
        c = c[:-5]
    return c


@dataclass
class SimPhysics:
    current_a: float = 1e-9          # photocurrent into the input (zero check off)
    dark_a: float = 0.0              # extra DC input current, also shot noise
    floor_a: Optional[float] = None  # white floor sd per reading; None: spec RMS noise x sqrt(6/NPLC)
    floor_scale: float = 1.0
    offset_a: float = 0.0
    excess_rel_sd: float = 0.0       # white classical (laser) noise, sd / <I>
    filter: str = "none"             # none | ar1 | boxcar
    ar1_phi: float = -0.5
    boxcar_m: int = 4                # sub-conversions per aperture
    boxcar_l: int = 8                # sub-conversions averaged per reading
    overhead_s: float = 0.83e-3
    autozero_factor: float = 2.0
    jitter_s: float = 2e-6
    quantum_a: float = 0.0           # round readings to this step (0: exponent format only)
    shot_noise: bool = True          # False: a noiseless DC current (battery + resistor control)
    johnson_r_ohm: float = 0.0       # > 0: add the Johnson noise of this source resistance, 2kT/(R T) per reading
    temperature_k: float = 295.0
    seed: Optional[int] = None

    def to_dict(self) -> dict:
        return asdict(self)


def _fmt(v: float) -> str:
    return f"{v:+.6E}"


class SimulatedK6485:
    """SCPI-level 6485. ``time_scale`` = real seconds per instrument second (0: instant)."""

    def __init__(self, physics: Optional[SimPhysics] = None, time_scale: float = 0.0,
                 state: Optional[dict] = None):
        self.p = physics or SimPhysics()
        self.rng = np.random.default_rng(self.p.seed)
        self.time_scale = float(time_scale)
        self.baud = 9600
        self.s = dict(RST_STATE)
        self.s.update(TRACE_POWER_ON)
        if state:
            self.s.update(state)
        self.errors: deque = deque()
        self.out: deque = deque()          # (text, t_start, t_end) on the real clock
        self.pending: deque = deque()
        self.commands: list[str] = []      # every line received, for tests
        self.vclock = 1000.0               # instrument timer, s since "power-on"
        self.busy_until = 0.0
        self._acq: Optional[dict] = None
        self.buf_i = np.array([])
        self.buf_t = np.array([])
        self.buf_st = np.array([], int)
        self.last_group: Optional[tuple] = None
        self._ar_y: Optional[float] = None
        self._subs: np.ndarray = np.array([])
        self._avg_hist: deque = deque()
        self._med_hist: deque = deque()
        self._out_end = 0.0
        self._h = self._handlers()

    # ------------------------------------------------------------------ line I/O
    def receive(self, line: str) -> None:
        self.vclock += (len(line) + 1) * 10.0 / self.baud
        for cmd in line.split(";"):
            cmd = cmd.strip()
            if cmd:
                self.commands.append(cmd)
                self.pending.append(cmd)
        self._process()

    def poll_output(self) -> tuple[Optional[str], int]:
        self._process()
        if not self.out:
            return None, 0
        text, t0, t1 = self.out[0]
        now = time.monotonic()
        if now >= t1:
            self.out.popleft()
            return text, len(text)
        frac = (now - t0) / (t1 - t0) if t1 > t0 and now > t0 else 0.0
        return None, int(len(text) * frac)

    def next_event_wait(self) -> float:
        now = time.monotonic()
        waits = [self.busy_until - now]
        if self.out:
            waits.append(self.out[0][2] - now)
        return max(0.0, min(waits))

    def clear_output(self) -> None:
        self.out.clear()

    def _busy(self) -> bool:
        return time.monotonic() < self.busy_until

    def _process(self) -> None:
        while self.pending:
            if self._busy():
                # ABORt and friends run at once, even behind a waiting *OPC?
                urgent = [c for c in self.pending if canon(c.split(" ", 1)[0].rstrip("?")) in _WHILE_BUSY]
                if not urgent:
                    return
                self.pending.remove(urgent[0])
                self._execute(urgent[0])
                continue
            self._execute(self.pending.popleft())

    def _reply(self, text: str) -> None:
        now = time.monotonic()
        t0 = max(now, self._out_end)
        dur = (len(text) + 1) * 10.0 / self.baud
        t1 = t0 + dur * self.time_scale
        self._out_end = t1
        self.vclock += dur
        self.out.append((text, t0, t1))

    def _error(self, code: int, extra: str = "") -> None:
        self.errors.append((code, ERRORS.get(code, "Error") + (f" ({extra})" if extra else "")))

    # ------------------------------------------------------------------ commands
    def _execute(self, cmd: str) -> None:
        parts = cmd.split(None, 1)
        raw_head = parts[0]
        arg = parts[1].strip() if len(parts) > 1 else ""
        is_q = raw_head.endswith("?")
        head = canon(raw_head.rstrip("?"))
        try:
            handler = self._h.get(head)
            if handler is None:
                self._error(-113, raw_head)
                return
            handler(is_q, arg)
        except _ParamError as e:
            self._error(e.code, f"{raw_head} {arg}".strip())

    def _handlers(self):
        s = self.s

        def setting(key, kind, lo=None, hi=None, names=None, post=None):
            def h(is_q, arg):
                if is_q:
                    v = s[key]
                    if kind == "bool":
                        self._reply("1" if v else "0")
                    elif kind == "name":
                        self._reply(str(v))
                    elif kind == "count":
                        self._reply(_fmt(OVERFLOW_A) if v >= OVERFLOW_A else str(int(v)))
                    else:
                        self._reply(_fmt(v))
                    return
                if not arg:
                    raise _ParamError(-109)
                if kind == "bool":
                    v = _parse_bool(arg)
                elif kind == "name":
                    v = arg.upper()
                    match = [n for n in names if v.startswith(n)]
                    if not match:
                        raise _ParamError(-224)
                    v = match[0]
                elif kind == "count":
                    if arg.upper().startswith("INF"):
                        v = OVERFLOW_A
                    else:
                        v = float(_parse_num(arg))
                        if not (lo <= v <= hi):
                            raise _ParamError(-222)
                        v = float(int(round(v)))
                else:
                    v = _parse_num(arg)
                    if lo is not None and not (lo <= v <= hi):
                        raise _ParamError(-222)
                s[key] = v
                if post:
                    post(v)
            return h

        def nplc_hi():
            return 60.0 if s["line_freq_hz"] == 60.0 else 50.0

        def nplc(is_q, arg):
            if is_q:
                self._reply(_fmt(s["nplc"]))
                return
            v = _parse_num(arg)
            if not (0.01 <= v <= nplc_hi()):
                raise _ParamError(-222)
            s["nplc"] = v

        def rang(is_q, arg):
            if is_q:
                self._reply(_fmt(s["range_a"]))
                return
            v = abs(_parse_num(arg))
            if v > 0.021:
                raise _ParamError(-222)
            s["range_a"] = next(r for r in RANGES_A if 1.05 * r >= v)
            s["autorange"] = False

        def form_data(is_q, arg):
            if is_q:
                self._reply(s["format_data"])
                return
            if not arg.upper().startswith("ASC"):
                raise _ParamError(-221)   # RS-232 carries ASCII only (p. 13-4)
            s["format_data"] = "ASC"

        def form_elem(is_q, arg):
            if is_q:
                self._reply(s["format_elements"])
                return
            items = [a.strip().upper() for a in arg.split(",") if a.strip()]
            if not items:
                raise _ParamError(-109)
            out = []
            for it in items:
                m = [n for n in ("READ", "UNIT", "TIME", "STAT") if it.startswith(n)]
                if not m:
                    raise _ParamError(-224)
                out.append(m[0])
            s["format_elements"] = ",".join(out)

        def lfr(is_q, arg):
            if is_q:
                self._reply(_fmt(s["line_freq_hz"]))
                return
            v = _parse_num(arg)
            if v not in (50.0, 60.0):
                raise _ParamError(-224)
            s["line_freq_hz"] = v

        def syst_err(is_q, arg):
            if not is_q:
                raise _ParamError(-113)
            if self.errors:
                code, msg = self.errors.popleft()
                self._reply(f'{code},"{msg}"')
            else:
                self._reply('0,"No error"')

        def syst_err_all(is_q, arg):
            items = [f'{c},"{m}"' for c, m in self.errors] or ['0,"No error"']
            self.errors.clear()
            self._reply(",".join(items))

        def syst_err_count(is_q, arg):
            self._reply(str(len(self.errors)))

        def clear_errs(is_q, arg):
            self.errors.clear()

        def noop(is_q, arg):
            if is_q:
                raise _ParamError(-113)

        def idn(is_q, arg):
            self._reply(IDN)

        def opc(is_q, arg):
            if is_q:
                self._reply("1")

        def rst(is_q, arg):
            self._abort()
            s.update(RST_STATE)

        def trac_cle(is_q, arg):
            self.buf_i, self.buf_t, self.buf_st = np.array([]), np.array([]), np.array([], int)

        def trac_act(is_q, arg):
            self._reply(str(len(self.buf_i)))

        def trac_data(is_q, arg):
            if not is_q:
                raise _ParamError(-113)
            if len(self.buf_i) == 0:
                raise _ParamError(-230)
            t = self.buf_t
            if s["trace_tstamp_format"] == "ABS":
                tt = t - t[0]
            else:
                tt = np.concatenate([[0.0], np.diff(t)])
            self._reply(self._format(self.buf_i, tt, self.buf_st))

        def sens_data(is_q, arg):
            if self.last_group is None:
                raise _ParamError(-230)
            i, t, st = self.last_group
            self._reply(self._format(i[-1:], t[-1:], st[-1:]))

        def fetch(is_q, arg):
            if self.last_group is None:
                raise _ParamError(-230)
            self._reply(self._format(*self.last_group))

        def init(is_q, arg):
            self._initiate()

        def read(is_q, arg):
            self._initiate()
            self.pending.appendleft("FETC?")

        def abort(is_q, arg):
            self._abort()

        def time_res(is_q, arg):
            self.vclock = 0.0

        h = {
            "*IDN": idn, "*OPC": opc, "*CLS": clear_errs, "*RST": rst,
            "SYST:ERR": syst_err, "SYST:ERR:ALL": syst_err_all, "SYST:ERR:COUN": syst_err_count,
            "SYST:CLE": clear_errs, "SYST:LOC": noop, "SYST:REM": noop, "SYST:TIME:RES": time_res,
            "SYST:ZCH": setting("zero_check", "bool"),
            "SYST:ZCOR": setting("zero_correct", "bool"),
            "SYST:ZCOR:ACQ": noop,
            "SYST:AZER": setting("autozero", "bool"),
            "SYST:LFR": lfr, "SYST:LFR:AUTO": setting("line_freq_auto", "bool"),
            "CURR:RANG": rang, "CURR:RANG:AUTO": setting("autorange", "bool"),
            "CURR:NPLC": nplc,
            "AVER": setting("avg_state", "bool"),
            "AVER:COUN": setting("avg_count", "num", 1, 100),
            "AVER:TCON": setting("avg_tcontrol", "name", names=("MOV", "REP")),
            "AVER:ADV": setting("avg_adv_state", "bool"),
            "AVER:ADV:NTOL": setting("avg_adv_ntol", "num", 0, 105),
            "MED": setting("median_state", "bool"),
            "MED:RANK": setting("median_rank", "num", 1, 5),
            "TRIG:DEL": setting("trig_delay_s", "num", 0, 999.9999),
            "TRIG:DEL:AUTO": setting("trig_delay_auto", "bool"),
            "TRIG:SOUR": setting("trig_source", "name", names=("IMM", "TLIN")),
            "TRIG:COUN": setting("trig_count", "count", 1, 2500),
            "ARM:SOUR": setting("arm_source", "name", names=("IMM", "TIM", "BUS", "TLIN", "MAN")),
            "ARM:COUN": setting("arm_count", "count", 1, 2500),
            "ARM:TIM": setting("arm_timer_s", "num", 0.001, 99999.999),
            "FORM": form_data, "FORM:ELEM": form_elem,
            "TRAC:CLE": trac_cle,
            "TRAC:POIN": setting("trace_points", "count", 1, 2500),
            "TRAC:POIN:ACT": trac_act,
            "TRAC:FEED": setting("trace_feed", "name", names=("SENS", "CALC2", "CALC")),
            "TRAC:FEED:CONT": setting("trace_feed_control", "name", names=("NEV", "NEXT")),
            "TRAC:TST:FORM": setting("trace_tstamp_format", "name", names=("ABS", "DELT")),
            "TRAC:DATA": trac_data,
            "SENS:DATA": sens_data, "FETC": fetch, "READ": read,
            "DISP:ENAB": setting("display_enabled", "bool"),
            "DISP:DIG": setting("display_digits", "num", 4, 7),
            "CALC:STAT": setting("calc1_state", "bool"),
            "CALC2:NULL:STAT": setting("null_state", "bool"),
            "CALC2:LIM:STAT": setting("limit1_state", "bool"),
            "CALC2:LIM2:STAT": setting("limit2_state", "bool"),
            "INIT": init, "ABOR": abort,
        }
        return h

    def _format(self, i, t, st) -> str:
        elems = self.s["format_elements"].split(",")
        unit = "UNIT" in elems
        fields = []
        for k in range(len(i)):
            for e in elems:
                if e == "READ":
                    fields.append(_fmt(i[k]) + ("A" if unit else ""))
                elif e == "TIME":
                    fields.append(_fmt(t[k]))
                elif e == "STAT":
                    fields.append(_fmt(float(st[k])))
        return ",".join(fields)

    # ------------------------------------------------------------------ trigger model
    def _initiate(self) -> None:
        s = self.s
        trig = s["trig_count"]
        arm = s["arm_count"]
        n = int(min(trig, 2500) * min(arm, 2500))
        if trig >= OVERFLOW_A or arm >= OVERFLOW_A:
            n = 2500
        i, t, st, dt = self._generate(n)
        self.last_group = (i, t, st)
        store = s["trace_feed_control"] == "NEXT"
        if store:
            m = min(n, int(s["trace_points"]))
            self.buf_i, self.buf_t, self.buf_st = i[:m].copy(), t[:m].copy(), st[:m].copy()
            if m >= int(s["trace_points"]):
                s["trace_feed_control"] = "NEV"
        now = time.monotonic()
        dur = n * dt + s["trig_delay_s"] * n
        self._acq = {"t0": now, "dt_real": dt * self.time_scale, "n": n, "t_start": t[0] - dt,
                     "store": store, "dt": dt}
        self.busy_until = now + dur * self.time_scale
        self.vclock = float(t[-1])

    def _abort(self) -> None:
        if self._acq is None or not self._busy():
            self.busy_until = 0.0
            self._acq = None
            return
        a = self._acq
        done = int((time.monotonic() - a["t0"]) / a["dt_real"]) if a["dt_real"] > 0 else a["n"]
        done = max(0, min(done, a["n"]))
        if a["store"]:
            self.buf_i, self.buf_t, self.buf_st = self.buf_i[:done], self.buf_t[:done], self.buf_st[:done]
        if self.last_group is not None and done > 0:
            self.last_group = tuple(x[:done] for x in self.last_group)
        self.vclock = a["t_start"] + done * a["dt"]
        self.busy_until = 0.0
        self._acq = None

    # ------------------------------------------------------------------ physics
    def floor_sd(self) -> float:
        p, s = self.p, self.s
        if p.floor_a is not None:
            return p.floor_a * p.floor_scale
        return SPEC_RMS_NOISE_A[self._range()] * math.sqrt(6.0 / s["nplc"]) * p.floor_scale

    def _range(self) -> float:
        s = self.s
        if s["autorange"]:
            cur = 0.0 if s["zero_check"] else abs(self.p.current_a + self.p.dark_a)
            return next((r for r in RANGES_A if 1.05 * r >= cur), RANGES_A[-1])
        return min(RANGES_A, key=lambda r: abs(math.log(r / s["range_a"])))

    def _generate(self, n: int):
        p, s, rng = self.p, self.s, self.rng
        T = s["nplc"] / s["line_freq_hz"]
        dt = T * (p.autozero_factor if s["autozero"] else 1.0) + p.overhead_s
        rep = int(s["avg_count"]) if (s["avg_state"] and s["avg_tcontrol"] == "REP") else 1
        dt *= rep
        n_conv = n * rep
        live = not s["zero_check"]
        i_dc = (p.current_a + p.dark_a) if live else 0.0
        floor = self.floor_sd()
        if p.filter == "boxcar":
            x = self._boxcar(n_conv, T, i_dc, floor)
        else:
            sig = self._shot(n_conv, T, i_dc) if p.shot_noise else np.full(n_conv, i_dc)
            x = sig + floor * rng.standard_normal(n_conv)
        if live and p.johnson_r_ohm > 0:
            x = x + math.sqrt(2 * 1.380649e-23 * p.temperature_k / (p.johnson_r_ohm * T)) * rng.standard_normal(n_conv)
        if live and p.excess_rel_sd:
            x = x + abs(p.current_a) * p.excess_rel_sd * rng.standard_normal(n_conv)
        x = x + p.offset_a
        if p.filter == "ar1":
            x = self._ar1(x)
        if s["median_state"]:
            x = self._median(x, int(s["median_rank"]))
        if rep > 1:
            x = x.reshape(n, rep).mean(axis=1)
        elif s["avg_state"]:
            x = self._moving_average(x, int(s["avg_count"]))
        if p.quantum_a > 0:
            x = np.round(x / p.quantum_a) * p.quantum_a
        bits = 0
        if s["avg_state"]:
            bits |= 1 << 1
        if s["calc1_state"]:
            bits |= 1 << 2
        if s["null_state"]:
            bits |= 1 << 3
        if s["limit1_state"] or s["limit2_state"]:
            bits |= 1 << 4
        if s["zero_check"]:
            bits |= 1 << 9
        if s["zero_correct"]:
            bits |= 1 << 10
        status = np.full(n, bits, int)
        over = np.abs(x) > 1.05 * self._range()
        x = np.where(over, OVERFLOW_A, x)
        status[over] |= 1
        t = self.vclock + dt * np.arange(1, n + 1) + p.jitter_s * rng.standard_normal(n)
        return x, t, status, dt

    def _shot(self, n: int, T: float, i_dc: float) -> np.ndarray:
        lam = abs(i_dc) * T / E_CHARGE
        if lam == 0:
            return np.zeros(n)
        counts = self.rng.poisson(lam, n) if lam < 1e15 else lam + math.sqrt(lam) * self.rng.standard_normal(n)
        return math.copysign(1.0, i_dc) * counts * E_CHARGE / T

    def _boxcar(self, n: int, T: float, i_dc: float, floor: float) -> np.ndarray:
        m, ell = max(1, int(self.p.boxcar_m)), max(1, int(self.p.boxcar_l))
        tau = T / m
        lam = abs(i_dc) * tau / E_CHARGE
        if lam > 0:
            c = self.rng.poisson(lam, n * m) if lam < 1e15 else lam + math.sqrt(lam) * self.rng.standard_normal(n * m)
            subs = math.copysign(1.0, i_dc) * c * E_CHARGE / tau
        else:
            subs = np.zeros(n * m)
        subs = subs + floor * math.sqrt(m) * self.rng.standard_normal(n * m)
        hist = self._subs
        if len(hist) < ell - 1:
            pad = np.full(ell - 1 - len(hist), subs[:m].mean())
            hist = np.concatenate([pad, hist])
        allsubs = np.concatenate([hist[len(hist) - (ell - 1):] if ell > 1 else hist[:0], subs])
        cs = np.concatenate([[0.0], np.cumsum(allsubs)])
        ends = (ell - 1) + m * np.arange(1, n + 1)
        y = (cs[ends] - cs[ends - ell]) / ell
        self._subs = allsubs[-(ell - 1):] if ell > 1 else np.array([])
        return y

    def _ar1(self, x: np.ndarray) -> np.ndarray:
        phi = float(self.p.ar1_phi)
        y = np.empty_like(x)
        prev = self._ar_y if self._ar_y is not None else float(x[0])
        for k, v in enumerate(x):
            prev = phi * prev + (1.0 - phi) * v
            y[k] = prev
        self._ar_y = prev
        return y

    def _median(self, x: np.ndarray, rank: int) -> np.ndarray:
        w = 2 * rank + 1
        out = np.empty_like(x)
        for k, v in enumerate(x):
            self._med_hist.append(v)
            while len(self._med_hist) > w:
                self._med_hist.popleft()
            out[k] = float(np.median(self._med_hist))
        return out

    def _moving_average(self, x: np.ndarray, count: int) -> np.ndarray:
        out = np.empty_like(x)
        for k, v in enumerate(x):
            self._avg_hist.append(v)
            while len(self._avg_hist) > count:
                self._avg_hist.popleft()
            out[k] = float(np.mean(self._avg_hist))
        return out

    def physics_dict(self) -> dict:
        return self.p.to_dict()


class _ParamError(Exception):
    def __init__(self, code: int):
        super().__init__(code)
        self.code = code


def _parse_bool(arg: str) -> bool:
    a = arg.strip().upper()
    if a in ("ON", "1"):
        return True
    if a in ("OFF", "0"):
        return False
    raise _ParamError(-224)


def _parse_num(arg: str) -> float:
    try:
        return float(arg.strip())
    except ValueError:
        raise _ParamError(-224) from None
