"""K6485Driver.read_buffer and connect against a fake 6485 (no hardware).

The fake answers the way the meter did on the bench (S/N 4456711, firmware
C01, 2026-11-25): three fields per reading in TRAC:DATA?, the unit glued to
the reading under the default FORM:ELEM, buffer time stamps from 0 in 1/1024 s
ticks, status words as floats, ARM:COUN found at 9.9E37. READ? is INIT then
FETCh?, which returns the whole trigger group. While a burst runs, every
command but ABORt and *RST waits (6485-901-01 Rev. A, p. 7-4).

Run from the repository root: python -m unittest discover -s tests -t .
"""

import time
import unittest
from collections import deque
from unittest import mock

import numpy as np

from keithley6485 import driver as drv_mod
from keithley6485.driver import K6485Driver, parse_trace

IDN = "KEITHLEY INSTRUMENTS INC.,MODEL 6485,4456711,C01   Sep 27 2004 12:22:00/A02  /E"
INF = 9.9e37
RST = {"TRIG:COUN": 1.0, "ARM:COUN": 1.0, "FORM:ELEM": "READ,UNIT,TIME,STAT"}


class VisaTimeout(Exception):
    pass


def _num(v: float) -> str:
    return f"{v:+.6E}"


class Fake6485:
    """A pyvisa resource that answers like a 6485 for the commands the driver sends."""

    def __init__(self, found=None, hang=False, dt_s=0.0833):
        self.timeout = drv_mod.TIMEOUT_MS
        self.chunk_size = 20 * 1024
        self.s = dict(RST)
        self.s.update(found or {})
        self.s.update({"TRAC:POIN": 100, "TRAC:FEED:CONT": "NEV", "TRAC:TST:FORM": "ABS"})
        self.hang = hang
        self.dt_s = dt_s
        self.log: list[str] = []
        self.read_timeouts: list[tuple[str, float]] = []   # (last command, timeout ms) per read
        self.out: deque[str] = deque()
        self.held: list[str] = []
        self.busy = False
        self.taken = 0
        self.buffer = None
        self.group = None
        self.closed = False
        self.flushes = 0
        # Lines from an earlier timeout that arrive after the input flush, and
        # write failures (the port gone), for the error paths.
        self.late_after_flush: list[str] = []
        self.late_after_init: list[str] = []
        self.fail_writes: set[str] = set()

    # pyvisa resource interface
    def flush(self, mask):
        self.flushes += 1
        self.out.clear()
        self.out.extend(self.late_after_flush)

    def write(self, cmd: str):
        if cmd in self.fail_writes:
            raise OSError(f"write failed: {cmd}")
        self.log.append(cmd)
        if cmd == "INIT":
            self.out.extend(self.late_after_init)
        if self.busy and cmd not in ("ABOR", "*RST"):
            self.held.append(cmd)
            return
        self._do(cmd)

    def read(self) -> str:
        self.read_timeouts.append((self.log[-1] if self.log else "", self.timeout))
        if not self.out:
            raise VisaTimeout("VI_ERROR_TMO")
        line = self.out.popleft()
        if len(line) + 1 > self.chunk_size:
            # pyvisa-py returns one chunk per read call; the line would need
            # several, each with a fresh timeout.
            raise AssertionError(f"reply of {len(line)} bytes over chunk_size {self.chunk_size}")
        return line

    def query(self, cmd: str) -> str:
        self.write(cmd)
        return self.read()

    def close(self):
        self.closed = True

    # the meter
    def _readings(self, n: int):
        k = np.arange(n)
        i = 1e-9 + 1e-12 * (self.taken + k)
        t = np.round(k * self.dt_s * 1024) / 1024
        st = np.zeros(n)
        self.taken += n
        return i, t, st

    def _format(self, group) -> str:
        elems = self.s["FORM:ELEM"].split(",")
        unit = "A" if "UNIT" in elems else ""
        fields = []
        for i, t, st in zip(*group):
            for e in elems:
                if e == "READ":
                    fields.append(_num(i) + unit)
                elif e == "TIME":
                    fields.append(_num(t))
                elif e == "STAT":
                    fields.append(_num(st))
        return ",".join(fields)

    def _init(self):
        if self.hang or self.s["ARM:COUN"] >= INF or self.s["TRIG:COUN"] >= INF:
            self.busy = True
            return
        n = int(self.s["TRIG:COUN"] * self.s["ARM:COUN"])
        self.group = self._readings(n)
        if self.s["TRAC:FEED:CONT"] == "NEXT":
            m = min(n, self.s["TRAC:POIN"])
            self.buffer = tuple(x[:m] for x in self.group)
            if m >= self.s["TRAC:POIN"]:
                self.s["TRAC:FEED:CONT"] = "NEV"

    def _do(self, cmd: str):
        head, _, arg = cmd.partition(" ")
        arg = arg.strip()
        if head == "*IDN?":
            self.out.append(IDN)
        elif head == "*OPC?":
            self.out.append("1")
        elif head == "*RST":
            self.busy = False
            self.held.clear()
            self.s.update(RST)
        elif head == "ABOR":
            self.busy = False
            held, self.held = self.held, []
            for c in held:
                self._do(c)
        elif head == "INIT":
            self._init()
        elif head == "READ?":
            self._init()
            if not self.busy:
                self.out.append(self._format(self.group))
        elif head in ("TRIG:COUN", "ARM:COUN"):
            self.s[head] = INF if arg.upper() == "INF" else float(arg)
        elif head in ("TRIG:COUN?", "ARM:COUN?"):
            self.out.append(_num(self.s[head[:-1]]))
        elif head == "FORM:ELEM":
            self.s["FORM:ELEM"] = arg.upper()
        elif head == "FORM:ELEM?":
            self.out.append(self.s["FORM:ELEM"])
        elif head == "TRAC:CLE":
            self.buffer = None
        elif head == "TRAC:POIN":
            self.s["TRAC:POIN"] = int(arg)
        elif head == "TRAC:FEED:CONT":
            self.s["TRAC:FEED:CONT"] = arg.upper()
        elif head == "TRAC:TST:FORM":
            self.s["TRAC:TST:FORM"] = arg.upper()
        elif head == "TRAC:POIN:ACT?":
            self.out.append(str(0 if self.buffer is None else len(self.buffer[0])))
        elif head == "TRAC:DATA?":
            self.out.append(self._format(self.buffer))
        # anything else (SYST:ZCH, TRAC:FEED SENS, ...) is accepted silently


def hw_driver(fake: Fake6485) -> K6485Driver:
    d = K6485Driver(visa="/dev/null", mode="hardware")
    d._inst = fake
    d._connected = True
    return d


class ParseTraceTest(unittest.TestCase):
    def test_reply_seen_on_the_meter(self):
        raw = ("+1.776357E-13,+0.000000E+00,+5.120000E+02,"
               "+1.811884E-13,+1.757813E-02,+5.120000E+02")
        i, t, st = parse_trace(raw, 2)
        np.testing.assert_array_equal(i, [1.776357e-13, 1.811884e-13])
        np.testing.assert_array_equal(t, [0.0, 0.01757813])
        np.testing.assert_array_equal(st, [512, 512])
        self.assertEqual(st.dtype, np.int64)

    def test_unit_glued_to_the_reading(self):
        i, t, st = parse_trace("+1.040564E-06A,+2.236299E+02,+1.380000E+02")
        self.assertEqual((i[0], t[0], st[0]), (1.040564e-06, 223.6299, 138))

    def test_wrong_shape_is_refused(self):
        with self.assertRaises(RuntimeError):
            parse_trace("+1.0E-09,+0.0E+00,+0.0E+00,+2.0E-09")
        with self.assertRaises(RuntimeError):
            parse_trace("+1.0E-09,+0.0E+00,+0.0E+00", n_expected=2)
        self.assertEqual(len(parse_trace("", n_expected=0)[0]), 0)


class ReadBufferTest(unittest.TestCase):
    def test_returns_the_meters_readings_and_time_stamps(self):
        fake = Fake6485(found={"ARM:COUN": INF})
        d = hw_driver(fake)
        before = time.time()
        currents, stamps = d.read_buffer(50)
        after = time.time()
        want_i, want_t, _ = fake.buffer
        # The meter sends 7 significant digits.
        np.testing.assert_allclose(currents, want_i, rtol=1e-6)
        # The old parser took one field per reading: time stamps and status
        # words came back as currents.
        self.assertTrue(np.all((currents > 0.9e-9) & (currents < 1.1e-9)))
        self.assertTrue(before <= stamps[0] <= after)
        # Epoch seconds in float64 resolve about 0.24 us, far below the
        # meter's 1/1024 s tick.
        np.testing.assert_allclose(stamps - stamps[0], want_t, atol=1e-6)
        self.assertAlmostEqual(np.median(np.diff(stamps)), 85 / 1024, places=6)

    def test_waits_on_opc_and_never_polls_the_count(self):
        fake = Fake6485()
        hw_driver(fake).read_buffer(20)
        burst = fake.log[fake.log.index("INIT"):]
        self.assertEqual(burst[:2], ["INIT", "*OPC?"])
        self.assertNotIn("TRAC:POIN:ACT?", fake.log)
        self.assertIn("TRAC:DATA?", burst)
        for c in ("FORM:ELEM READ,TIME,STAT", "TRAC:TST:FORM ABS", "ARM:COUN 1",
                  "TRIG:COUN 20", "TRAC:FEED:CONT NEXT", "TRAC:POIN 20"):
            self.assertIn(c, fake.log[:fake.log.index("INIT")])

    def test_counts_and_elements_go_back_as_found(self):
        fake = Fake6485(found={"ARM:COUN": INF, "TRIG:COUN": 7.0,
                               "FORM:ELEM": "READ,UNIT,TIME,STAT"})
        hw_driver(fake).read_buffer(30)
        self.assertEqual(fake.s["TRIG:COUN"], 7.0)
        self.assertEqual(fake.s["ARM:COUN"], INF)
        self.assertEqual(fake.s["FORM:ELEM"], "READ,UNIT,TIME,STAT")
        self.assertIn("ARM:COUN INF", fake.log)

    def test_read_single_after_a_burst_takes_one_reading(self):
        # The old driver left TRIG:COUN at n, so every READ? afterwards took
        # and sent n readings.
        fake = Fake6485()
        d = hw_driver(fake)
        d.read_buffer(200)
        taken = fake.taken
        v = d.read_single()
        self.assertEqual(fake.taken - taken, 1)
        self.assertAlmostEqual(v, 1e-9 + 1e-12 * taken, delta=1e-18)
        _i, _t = d.read_n(3)
        self.assertEqual(fake.taken - taken, 4)

    def test_visa_timeout_covers_the_burst_and_comes_back(self):
        n = 2500
        fake = Fake6485()
        hw_driver(fake).read_buffer(n)
        waits = dict(fake.read_timeouts)
        budget_ms = (2.0 * n + 5.0) * 1000.0
        # The transfer's allowance at 9600 baud: 1.5 x 56 ms a reading + 2 s.
        transfer_ms = (1.5 * n * 0.056 + 2.0) * 1000.0
        self.assertGreater(waits["*OPC?"], budget_ms - transfer_ms - 2000.0)
        self.assertLessEqual(waits["*OPC?"], budget_ms - transfer_ms)
        self.assertLessEqual(waits["TRAC:DATA?"], budget_ms)
        self.assertEqual(waits["TRIG:COUN?"], drv_mod.TIMEOUT_MS)
        self.assertEqual(fake.timeout, drv_mod.TIMEOUT_MS)
        # The whole line came in one read call, and chunk_size went back.
        self.assertEqual(fake.chunk_size, 20 * 1024)

    def test_a_slow_burst_still_leaves_the_transfer_its_time(self):
        # 2500 readings at 9600 baud take about 140 s to send; a burst that
        # uses every second the *OPC? wait allows must still leave that.
        n = 2500
        fake = Fake6485()
        hw_driver(fake).read_buffer(n)
        waits = dict(fake.read_timeouts)
        left_ms = (2.0 * n + 5.0) * 1000.0 - waits["*OPC?"]
        self.assertGreater(left_ms, 1.5 * 140_000.0)

    def test_a_burst_that_does_not_end_is_aborted(self):
        fake = Fake6485(found={"TRIG:COUN": 1.0, "ARM:COUN": 1.0}, hang=True)
        d = hw_driver(fake)
        with self.assertRaisesRegex(RuntimeError, "did not finish"):
            d.read_buffer(5)
        self.assertIn("ABOR", fake.log[fake.log.index("*OPC?"):])
        self.assertLessEqual(dict(fake.read_timeouts)["*OPC?"], 15_000.0)
        self.assertEqual(fake.timeout, drv_mod.TIMEOUT_MS)
        self.assertEqual((fake.s["TRIG:COUN"], fake.s["ARM:COUN"]), (1.0, 1.0))
        self.assertEqual(fake.s["FORM:ELEM"], "READ,UNIT,TIME,STAT")
        # The late "1" was read, so the next reply is the next query's own.
        self.assertEqual(d.query("TRIG:COUN?"), "+1.000000E+00")

    def test_a_stale_reply_on_the_line_is_discarded(self):
        fake = Fake6485(found={"ARM:COUN": INF})
        fake.out.extend(["1", "+1.000000E+00"])
        d = hw_driver(fake)
        currents, _ = d.read_buffer(10)
        self.assertEqual(fake.flushes, 1)
        self.assertEqual(len(currents), 10)
        self.assertEqual((fake.s["TRIG:COUN"], fake.s["ARM:COUN"]), (1.0, INF))
        self.assertEqual(fake.s["FORM:ELEM"], "READ,UNIT,TIME,STAT")

    def test_an_out_of_step_found_state_changes_nothing(self):
        # A late count reply after the flush shifts every answer by one: the
        # FORM:ELEM? answer is then a number, and nothing may be set.
        fake = Fake6485()
        fake.late_after_flush = ["+1.000000E+00"]
        with self.assertRaisesRegex(RuntimeError, "FORM:ELEM readback"):
            hw_driver(fake).read_buffer(10)
        self.assertEqual(fake.log, ["TRIG:COUN?", "ARM:COUN?", "FORM:ELEM?"])

    def test_an_out_of_step_opc_aborts_and_puts_back(self):
        fake = Fake6485()
        fake.late_after_init = ["+1.000000E+00"]
        d = hw_driver(fake)
        with self.assertRaisesRegex(RuntimeError, "answered"):
            d.read_buffer(10)
        self.assertIn("ABOR", fake.log[fake.log.index("*OPC?"):])
        self.assertEqual((fake.s["TRIG:COUN"], fake.s["ARM:COUN"]), (1.0, 1.0))
        self.assertEqual(fake.s["FORM:ELEM"], "READ,UNIT,TIME,STAT")
        self.assertEqual(d.query("FORM:ELEM?"), "READ,UNIT,TIME,STAT")

    def test_a_failed_put_back_is_reported_and_the_others_still_go(self):
        fake = Fake6485()
        fake.fail_writes = {"TRIG:COUN 1"}
        with self.assertRaisesRegex(RuntimeError, "not put back.*TRIG:COUN 1"):
            hw_driver(fake).read_buffer(10)
        self.assertEqual(fake.log[-2:], ["ARM:COUN 1", "FORM:ELEM READ,UNIT,TIME,STAT"])

    def test_a_failed_put_back_does_not_hide_the_bursts_error(self):
        fake = Fake6485(hang=True)
        fake.fail_writes = {"TRIG:COUN 1"}
        with self.assertRaisesRegex(RuntimeError, "did not finish"):
            hw_driver(fake).read_buffer(5)
        self.assertEqual(fake.s["FORM:ELEM"], "READ,UNIT,TIME,STAT")

    def test_n_outside_the_buffer_is_refused_before_anything_is_sent(self):
        fake = Fake6485()
        d = hw_driver(fake)
        for n in (0, 2501):
            with self.subTest(n=n), self.assertRaises(ValueError):
                d.read_buffer(n)
        self.assertEqual(fake.log, [])
        with self.assertRaises(ValueError):
            K6485Driver(mode="simulation").read_buffer(2501)

    def test_simulation_mode(self):
        d = K6485Driver(mode="simulation")
        d.connect()
        currents, stamps = d.read_buffer(10)
        self.assertEqual((len(currents), len(stamps)), (10, 10))
        self.assertTrue(np.all(np.diff(stamps) > 0))


class ConnectTest(unittest.TestCase):
    def _connect(self, fake):
        rm = mock.Mock()
        rm.open_resource.return_value = fake
        with mock.patch("pyvisa.ResourceManager", return_value=rm):
            d = K6485Driver(visa="/dev/ttyUSB9", mode="hardware")
            d.connect()
        return d, rm

    def test_connect_checks_idn_and_sends_no_rst(self):
        fake = Fake6485(found={"ARM:COUN": INF, "TRIG:COUN": 5.0})
        d, rm = self._connect(fake)
        self.assertTrue(d.is_connected)
        self.assertEqual(rm.open_resource.call_args.args[0], "ASRL/dev/ttyUSB9::INSTR")
        self.assertEqual(fake.log, ["*IDN?"])
        self.assertEqual((fake.s["TRIG:COUN"], fake.s["ARM:COUN"]), (5.0, INF))

    def test_reset_still_sends_rst(self):
        fake = Fake6485()
        d, _rm = self._connect(fake)
        with mock.patch.object(drv_mod.time, "sleep"):
            d.reset()
        self.assertEqual(fake.log[1:], ["*RST", "*CLS"])


if __name__ == "__main__":
    unittest.main()
