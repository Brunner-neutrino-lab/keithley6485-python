"""The driver against the simulated 6485: e recovery, meter filters, settings, error queue, bursts."""

import os
import sys
import threading
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from keithley6485.campbell.meter import CampbellMeter, MeterError, parse_trace  # noqa: E402
from keithley6485.campbell.sim import SimPhysics, SimulatedK6485                # noqa: E402
from keithley6485.campbell.stats import (campbell_stats, E_CHARGE,              # noqa: E402
                                          floor_subtracted_charge)
from keithley6485.campbell.transport import SimTransport                        # noqa: E402

LOW_BAND = (0.1, 1.5)     # well below the corner of the filters used here


def meter(physics=None, state=None, time_scale=0.0, drain_each=True):
    sim = SimulatedK6485(physics or SimPhysics(seed=1), time_scale=time_scale, state=state)
    m = CampbellMeter(SimTransport(sim), drain_each=drain_each)
    m.connect()
    return m, sim


def run_stats(physics, n, range_a=20e-9, nplc=1.0, band=(1.0, 10.0), zero_check=False):
    m, sim = meter(physics, drain_each=False)
    cfg = m.configure_campbell(range_a, nplc, autozero=False, zero_check=zero_check)
    run = m.acquire(n)
    return campbell_stats(run.i_a, run.t_s, cfg["nplc"], cfg["line_freq_hz"], burst=run.burst,
                          status=run.status, band_hz=band), run


def within(test, value, target, err, k=4.0, msg=""):
    test.assertLess(abs(value - target), k * err, f"{msg}: {value:.4f} vs {target:.4f} +- {err:.4f}")


class RecoverE(unittest.TestCase):
    def test_pure_shot_noise_gives_e(self):
        st, _ = run_stats(SimPhysics(current_a=1e-9, floor_a=0.0, seed=21), 20000)
        for k in ("naive", "sdiff", "band"):
            within(self, st[f"e_{k}_over_e"], 1.0, st[f"e_{k}_over_e_err"], msg=k)
        self.assertLess(st["e_naive_over_e_err"], 0.012)       # sqrt(2/N) at N = 20000

    def test_floor_run_subtracts(self):
        phys = dict(current_a=5e-9, floor_a=150e-15, seed=5)
        light, _ = run_stats(SimPhysics(**phys), 20000)
        dark, _ = run_stats(SimPhysics(**{**phys, "seed": 6}), 20000, zero_check=True)
        self.assertEqual(dark["n_zero_check"], dark["n"])
        self.assertLess(abs(dark["mean_a"]), 5 * dark["sem_a"])
        q, err = floor_subtracted_charge(light, dark)
        within(self, q / E_CHARGE, 1.0, err / E_CHARGE, msg="floor-subtracted e")
        self.assertGreater(light["e_naive_over_e"], 1.1)        # the floor alone biases it upward


class MeterFilters(unittest.TestCase):
    """Correlated meters move the plain variance; a band below the corner keeps e."""

    N = 50000

    def test_ar1_anticorrelated(self):
        phi = -0.5
        st, _ = run_stats(SimPhysics(current_a=1e-9, floor_a=0.0, filter="ar1", ar1_phi=phi, seed=8),
                          self.N, band=LOW_BAND)
        expect = (1 - phi) / (1 + phi)       # variance factor of y = phi y' + (1 - phi) x
        within(self, st["e_naive_over_e"], expect, st["e_naive_over_e_err"] * expect, msg="naive")
        within(self, st["e_band_over_e"], 1.0, st["e_band_over_e_err"], msg="band")
        within(self, st["acf"][0], phi, 0.02, msg="lag 1")
        hi = [p for p in st["psd_shape_a2"] if np.isfinite(p)]
        self.assertGreater(hi[-1] / hi[0], 4.0)     # spectrum rising toward Nyquist

    def test_boxcar_overlapping(self):
        st, _ = run_stats(SimPhysics(current_a=1e-9, floor_a=0.0, filter="boxcar", boxcar_m=4,
                                     boxcar_l=8, seed=9), self.N, band=LOW_BAND)
        within(self, st["e_naive_over_e"], 0.5, st["e_naive_over_e_err"] * 0.5, msg="naive")
        within(self, st["e_band_over_e"], 1.0, st["e_band_over_e_err"], msg="band")
        within(self, st["acf"][0], 0.5, 0.02, msg="lag 1")

    def test_no_filter_keeps_band_and_naive_equal(self):
        st, _ = run_stats(SimPhysics(current_a=1e-9, floor_a=0.0, seed=10), self.N, band=LOW_BAND)
        within(self, st["e_naive_over_e"], 1.0, st["e_naive_over_e_err"], msg="naive")
        within(self, st["e_band_over_e"], 1.0, st["e_band_over_e_err"], msg="band")


DIRTY = {"avg_state": True, "avg_count": 20, "avg_tcontrol": "MOV", "median_state": True,
         "median_rank": 3, "autorange": True, "nplc": 6.0, "zero_check": True, "autozero": True,
         "trig_count": 10.0, "trace_feed_control": "NEXT", "trace_points": 500,
         "format_elements": "READ,UNIT,TIME,STAT", "display_enabled": True, "trig_delay_s": 0.5}


class Settings(unittest.TestCase):
    def test_configure_reads_back_and_restore_returns_as_found(self):
        m, sim = meter(state=DIRTY)
        before = dict(sim.s)
        found = m.settings_snapshot()
        self.assertEqual(found["_refused"], [])
        cfg = m.configure_campbell(2e-9, 2.0, autozero=False, display_off=True)
        for k, v in {"avg_state": False, "median_state": False, "autorange": False, "autozero": False,
                     "zero_check": False, "display_enabled": False, "trig_delay_s": 0.0,
                     "trace_tstamp_format": "ABS", "format_elements": "READ,TIME,STAT", "nplc": 2.0}.items():
            self.assertEqual(cfg[k], v, k)
        self.assertEqual(cfg["range_a"], 2e-9)
        m.acquire(100)
        diff = m.restore(found)
        self.assertEqual(diff, {})
        after = dict(sim.s)
        for k in DIRTY:
            self.assertEqual(after[k], before[k], k)

    def test_never_sends_rst(self):
        m, sim = meter()
        found = m.settings_snapshot()
        m.configure_campbell(20e-9, 1.0, autozero=True)
        m.acquire(3000, 2500)
        m.restore(found)
        self.assertFalse([c for c in sim.commands if c.upper().startswith(("*RST", "SYST:PRES"))])

    def test_errors_raise(self):
        m, _ = meter()
        with self.assertRaises(MeterError) as cm:
            m.command("CURR:NPLC 0.001")
        self.assertIn("-222", str(cm.exception))
        with self.assertRaises(MeterError) as cm:
            m.command("CURR:DAMP ON")              # the 6485 has no damping filter
        self.assertIn("-113", str(cm.exception))
        with self.assertRaises(MeterError):
            m.query("TRAC:DATA?")                  # empty buffer: -230 and no reply
        with self.assertRaises(ValueError):
            m.configure_campbell(5e-9, 1.0, False)  # not a range
        with self.assertRaises(ValueError):
            m.configure_campbell(2e-9, 100.0, False)

    def test_errors_in_queue_at_connect_are_drained(self):
        sim = SimulatedK6485(SimPhysics(seed=1))
        sim.errors.append((-113, "Undefined header"))
        m = CampbellMeter(SimTransport(sim))
        m.connect()
        self.assertEqual(len(m.found_errors), 1)
        m.configure_campbell(20e-9, 1.0, False)     # does not trip over the old error


class Parsing(unittest.TestCase):
    def test_parse_strides_and_units(self):
        raw = "+1.000000E-09A,+0.000000E+00,+0.000000E+00,+1.100000E-09A,+1.750000E-02,+5.120000E+02"
        i, t, s = parse_trace(raw, 2)
        np.testing.assert_allclose(i, [1e-9, 1.1e-9])
        np.testing.assert_allclose(t, [0.0, 0.0175])
        self.assertEqual(list(s), [0, 512])
        with self.assertRaises(MeterError):
            parse_trace(raw, 3)
        with self.assertRaises(MeterError):
            parse_trace(raw + ",+1.0E-09", None)

    def test_default_elements_interleave(self):
        """With the default FORM:ELEM (all four) TRAC:DATA? has 3 fields per reading, so a
        one-value-per-reading parse (the DAQ driver's read_buffer) mixes in times and status."""
        m, sim = meter()
        for c in ("SYST:ZCH OFF", "TRIG:COUN 50", "TRAC:POIN 50", "TRAC:FEED SENS", "TRAC:FEED:CONT NEXT", "INIT"):
            m.command(c)
        m.query("*OPC?")
        raw = m.query("TRAC:DATA?")
        self.assertEqual(len(raw.split(",")), 3 * 50)
        i, t, s = parse_trace(raw, 50)
        self.assertTrue(np.all(np.diff(t) > 0))


class Bursts(unittest.TestCase):
    def test_chained_bursts(self):
        m, sim = meter(SimPhysics(current_a=1e-9, seed=3))
        m.configure_campbell(20e-9, 1.0, autozero=False)
        run = m.acquire(6000, 2500)
        self.assertEqual([b.n for b in run.bursts], [2500, 2500, 1000])
        self.assertEqual(list(np.bincount(run.burst)), [2500, 2500, 1000])
        for b in run.bursts:
            self.assertEqual(b.t_s[0], 0.0)
            self.assertTrue(np.all(np.diff(b.t_s) > 0))
            self.assertEqual(b.warnings, [])
            self.assertTrue(np.all(b.status == 0))
        tab = run.burst_table()
        # the simulated clock runs on during the 9600-baud transfer (~110 s for 2500 readings)
        self.assertTrue(np.all((tab["gap_instr_s"][1:] > 100) & (tab["gap_instr_s"][1:] < 130)),
                        tab["gap_instr_s"])
        self.assertTrue(np.all(tab["start_from_instrument"]))
        t = run.t_run_s()
        self.assertTrue(np.all(np.diff(t) > 0))

    def test_abort_keeps_partial_burst(self):
        m, sim = meter(SimPhysics(current_a=1e-9, seed=4), time_scale=0.01)
        m.configure_campbell(20e-9, 1.0, autozero=False)
        stop = threading.Event()

        def on_phase(k, nb, phase, frac):
            if k == 1 and phase == "acquire" and frac > 0.2:
                stop.set()

        run = m.acquire(5000, 2500, abort=stop.is_set, on_phase=on_phase)
        self.assertEqual(len(run.bursts), 2)
        self.assertEqual(run.bursts[0].n, 2500)
        self.assertTrue(run.bursts[-1].aborted)
        self.assertGreater(run.bursts[-1].n, 0)
        self.assertLess(run.n, 5000)
        self.assertEqual(m.drain_errors(), [])
        m.query("*IDN?")                            # still talking

    def test_overflow_flagged(self):
        st, run = run_stats(SimPhysics(current_a=21e-9, floor_a=0.3e-9, seed=12), 2500)
        self.assertGreater(st["n_overflow"], 100)
        self.assertEqual(st["n"] + st["n_invalid"], 2500)
        self.assertTrue(np.all(run.i_a[(run.status & 1) == 1] >= 9.9e37))

    def test_autozero_slows_readings(self):
        m, _ = meter()
        m.configure_campbell(20e-9, 1.0, autozero=True)
        on = m.acquire(200).bursts[0]
        m.configure_campbell(20e-9, 1.0, autozero=False)
        off = m.acquire(200).bursts[0]
        self.assertGreater(np.median(np.diff(on.t_s)), 1.5 * np.median(np.diff(off.t_s)))


if __name__ == "__main__":
    unittest.main()
