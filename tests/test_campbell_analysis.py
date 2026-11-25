"""Will's analyses (series fit, resistor control, Johnson T_eff), the quick check and run sequences, in simulation."""

import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from keithley6485.campbell import analysis                                         # noqa: E402
from keithley6485.campbell.session import (Connection, RunSettings, Session,        # noqa: E402
                                           build_sequence, range_advice)
from keithley6485.campbell.sim import SimPhysics                                    # noqa: E402
from keithley6485.campbell.stats import E_CHARGE                                    # noqa: E402

T1 = 1 / 60


def record(d, name, phys, kind="photocurrent", n=10000, nplc=1.0, range_a=20e-9, notes=None, zero_check=False):
    """One simulated run saved to d; returns its path."""
    s = Session()
    s.connect(Connection(port="simulation", sim=phys))
    s.notes.update(kind=kind, **(notes or {}))
    s.start(RunSettings(range_a=range_a, nplc=nplc, n_total=n, settle_s=0.0, out_dir=d, name=name,
                        zero_check=zero_check))
    s.wait(120)
    assert s.error is None, s.error
    p = s.last_path
    s.disconnect()
    return str(p)


class SeriesFit(unittest.TestCase):
    def test_floor_shot_and_laser_terms(self):
        floor, rel = 40e-15, 3e-5
        with tempfile.TemporaryDirectory() as d:
            paths = [record(d, f"nd{k}", SimPhysics(current_a=i, floor_a=floor, excess_rel_sd=rel, seed=100 + k))
                     for k, i in enumerate((0.3e-9, 0.6e-9, 1e-9, 1.6e-9, 2.5e-9, 4e-9, 6e-9))]
            runs = analysis.load_runs(paths)
            fits = analysis.fit_charge(runs, "naive")
        self.assertEqual(len(fits), 1)
        f = fits[0]
        self.assertTrue(f["quadratic"])
        self.assertLess(abs(f["e_over_e"] - 1), 4 * f["e_over_e_err"], f)
        self.assertLess(f["e_over_e_err"], 0.06)
        self.assertLess(abs(f["a_c_a"] - T1 * floor ** 2), 4 * f["a_err"], f)
        self.assertLess(abs(f["b_s"] - T1 * rel ** 2), 4 * f["b_err"], f)

    def test_groups_by_range_and_nplc_and_skips_floors(self):
        with tempfile.TemporaryDirectory() as d:
            paths = [record(d, "a", SimPhysics(current_a=1e-9, floor_a=0, seed=1), n=3000),
                     record(d, "b", SimPhysics(current_a=2e-9, floor_a=0, seed=2), n=3000),
                     record(d, "c", SimPhysics(current_a=1e-9, floor_a=0, seed=3), n=3000, nplc=2.0),
                     record(d, "z", SimPhysics(seed=4), n=3000, zero_check=True)]
            runs = analysis.load_runs(paths)
        self.assertEqual([r.kind for r in runs][-1], "zero-check floor")
        fits = analysis.fit_charge(runs)
        self.assertEqual([(f["nplc"], f["n_runs"]) for f in fits], [(1.0, 2), (2.0, 1)])
        self.assertIn("e_c", fits[0])
        self.assertIn("note", fits[1])
        table = analysis.nplc_table(runs)
        self.assertEqual(len(table), 3)


class Controls(unittest.TestCase):
    def test_resistor_control_removes_instrument_noise(self):
        floor, R = 200e-15, 1.5e9
        with tempfile.TemporaryDirectory() as d:
            pd = record(d, "pd", SimPhysics(current_a=1e-9, floor_a=floor, seed=11), n=20000)
            ctl = record(d, "ctl", SimPhysics(current_a=1e-9, floor_a=floor, shot_noise=False, johnson_r_ohm=R,
                                              seed=12), kind="resistor control", n=20000,
                         notes={"resistor_ohm": str(R), "battery_v": "1.5", "temperature_k": "295"})
            runs = analysis.load_runs([pd, ctl])
        naive = runs[0].stats["e_naive_over_e"]
        self.assertGreater(naive, 1.2)                  # the floor biases the plain estimate
        c = analysis.resistor_control(runs)
        self.assertEqual(len(c), 1)
        self.assertLess(abs(c[0]["e_over_e"] - 1), 4 * c[0]["e_over_e_err"], c[0])

    def test_johnson_teff(self):
        Rs = (1e7, 3e7, 1e8, 3e8, 1e9)
        with tempfile.TemporaryDirectory() as d:
            paths = [record(d, f"j{k}", SimPhysics(current_a=0.0, floor_a=2e-15, shot_noise=False, johnson_r_ohm=R,
                                                   temperature_k=296.0, seed=30 + k),
                            kind="johnson", n=10000, range_a=20e-9,
                            notes={"resistor_ohm": f"{R:g}", "temperature_k": "296"})
                     for k, R in enumerate(Rs)]
            runs = analysis.load_runs(paths)
        js = analysis.johnson_teff(runs)
        self.assertEqual(len(js), 1, js)
        j = js[0]
        self.assertTrue(j["quadratic"])
        self.assertLess(abs(j["t_eff_over_t_int"] - 1), 4 * j["t_eff_err_s"] / j["t_int_s"] + 0.02, j)


class CheckAndSequences(unittest.TestCase):
    def test_range_advice(self):
        self.assertIn("OVERFLOW", range_advice(2e-9, float("nan"), float("nan"), 100, 100))
        self.assertIn("use 2e-08 A", range_advice(2e-9, float("nan"), float("nan"), 3, 100))
        self.assertIn("50-67 %", range_advice(2e-9, 1.2e-9, 1e-13, 0, 100))
        self.assertIn("more sensitive", range_advice(200e-9, 1.2e-9, 1e-13, 0, 100))

    def test_check_saves_nothing_and_restores_on_disconnect(self):
        with tempfile.TemporaryDirectory() as d:
            s = Session()
            s.connect(Connection(port="simulation", sim=SimPhysics(current_a=3e-9, seed=5)))
            r = s.check(2e-9, 1.0)
            self.assertEqual(r["n_overflow"], 100)
            r = s.check(20e-9, 1.0)
            self.assertEqual(r["n_overflow"], 0)
            self.assertLess(abs(r["mean_a"] - 3e-9), 1e-11)
            self.assertEqual(os.listdir(d), [])
            self.assertEqual(s.history, [])
            s.disconnect()
            self.assertEqual(s.restore_diff, {})

    def test_nplc_sequence_with_floors(self):
        with tempfile.TemporaryDirectory() as d:
            s = Session()
            s.connect(Connection(port="simulation", sim=SimPhysics(current_a=1e-9, floor_a=0, seed=6)))
            seq = build_sequence(RunSettings(n_total=2000, settle_s=0.0, out_dir=d, name="scan"), [1, 2, 5],
                                 zc_floor=True)
            self.assertEqual([(r.nplc, r.zero_check) for r in seq],
                             [(1, False), (1, True), (2, False), (2, True), (5, False), (5, True)])
            s.start_sequence(seq)
            s.wait(120)
            self.assertEqual(len(s.history), 6)
            self.assertEqual([h["kind"] for h in s.history],
                             ["photocurrent", "zero-check floor"] * 3)
            self.assertEqual(len(list(Path(d).glob("scan_*plc*.h5"))), 6)
            self.assertEqual(s.restore_diff, {})       # restored once, after the last run
            out = s.fit_history()
            self.assertEqual(len(out["table"]), 3)
            for row in out["table"]:                     # no floor: T_int*Var/<I> = e at every NPLC
                self.assertLess(abs(row["t_v_over_i_e"] - 1), 4 * row["err_e"])
            s.disconnect()


class Attach(unittest.TestCase):
    def test_attach_external_transport(self):
        from keithley6485.campbell.sim import SimulatedK6485
        from keithley6485.campbell.transport import SimTransport

        class Borrowed(SimTransport):
            closed = False

            def close(self):
                Borrowed.closed = True

        sim = SimulatedK6485(SimPhysics(current_a=1e-9, floor_a=0, seed=9))
        with tempfile.TemporaryDirectory() as d:
            s = Session()
            s.attach(Borrowed(sim), label="daq hub")
            self.assertEqual(s.conn.port, "daq hub")
            s.start(RunSettings(n_total=1000, settle_s=0.0, out_dir=d, name="att"))
            s.wait(60)
            self.assertIsNone(s.error)
            self.assertEqual(len(s.history), 1)
            s.disconnect()
            self.assertTrue(Borrowed.closed)
            self.assertEqual(s.restore_diff, {})


if __name__ == "__main__":
    unittest.main()
