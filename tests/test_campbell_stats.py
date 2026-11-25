"""campbell_stats on synthetic arrays: estimators, segments, invalid readings, independence."""

import ast
import os
import subprocess
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from keithley6485.campbell.stats import (E_CHARGE, OVERFLOW_A, campbell_stats,  # noqa: E402
                                          floor_subtracted_charge, segment_bounds)

DT = 0.0175


def bursts_t(n, size=2500, dt=DT):
    return np.concatenate([np.arange(min(size, n - k)) * dt for k in range(0, n, size)])


class WhiteNoise(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(11)
        self.n = 40000
        self.sd = 3e-13
        self.x = 1e-9 + self.sd * rng.standard_normal(self.n)
        self.t = bursts_t(self.n)

    def test_estimators_agree_with_variance(self):
        st = campbell_stats(self.x, self.t, 1.0, 60.0)
        var = self.sd ** 2
        self.assertEqual(st["n"], self.n)
        self.assertEqual(st["n_segments"], 16)          # time stamps restart at 0 each burst
        self.assertLess(abs(st["var_a2"] / var - 1), 4 * st["var_rel_err"])
        self.assertLess(abs(st["sdiff_var_a2"] / var - 1), 4 * st["sdiff_rel_err"])
        self.assertLess(abs(st["psd_band_a2"] / var - 1), 4 * st["psd_band_rel_err"])
        for a in st["acf"]:
            self.assertLess(abs(a), 4 * st["acf_err"])
        for m, v, c in zip(st["allan_m"], st["allan_m_var_a2"], st["allan_count"]):
            if c > 200:
                self.assertLess(abs(v / var - 1), 4 * np.sqrt(2.0 / c) + 0.02, f"Allan m={m}")
        self.assertAlmostEqual(st["t_int_s"], 1 / 60)
        self.assertAlmostEqual(st["dt_median_s"], DT, places=9)
        self.assertEqual(st["psd_band_hz"], [1.0, 10.0])

    def test_charge_is_t_int_var_over_mean(self):
        st = campbell_stats(self.x, self.t, 2.0, 50.0)
        self.assertAlmostEqual(st["e_naive_c"], (2.0 / 50.0) * st["var_a2"] / st["mean_a"], delta=1e-30)
        self.assertAlmostEqual(st["e_naive_over_e"], st["e_naive_c"] / E_CHARGE)

    def test_burst_index_matches_time_resets(self):
        b = np.repeat(np.arange(16), 2500)
        a = campbell_stats(self.x, self.t, 1, 60)
        c = campbell_stats(self.x, self.t, 1, 60, burst=b)
        self.assertEqual(a["n_segments"], c["n_segments"])
        self.assertAlmostEqual(a["sdiff_var_a2"], c["sdiff_var_a2"], delta=1e-40)

    def test_band_clipped_at_nyquist(self):
        st = campbell_stats(self.x, self.t * 10, 6.0, 60.0, band_hz=(1.0, 10.0))   # dt 0.175 s: Nyquist 2.86 Hz
        self.assertAlmostEqual(st["psd_band_hz"][1], st["f_nyquist_hz"])


class InvalidReadings(unittest.TestCase):
    def test_overflow_and_overvoltage_left_out(self):
        rng = np.random.default_rng(2)
        x = 1e-9 + 1e-13 * rng.standard_normal(5000)
        t = np.arange(5000) * DT
        st_word = np.zeros(5000, int)
        x[100] = OVERFLOW_A
        st_word[100] = 1
        x[200] = 9.91e37                   # NAN as the meter sends it
        st_word[300] = 1 << 7              # overvoltage flagged on an ordinary value
        s = campbell_stats(x, t, 1, 60, status=st_word)
        self.assertEqual(s["n"], 4997)
        self.assertEqual(s["n_invalid"], 3)
        self.assertEqual(s["n_overflow"], 2)
        self.assertEqual(s["n_overvoltage"], 1)
        self.assertEqual(s["n_segments"], 4)          # split around each invalid reading
        self.assertLess(abs(s["mean_a"] - 1e-9), 1e-14)

    def test_segment_bounds(self):
        t = np.array([0, 1, 2, 0, 1, 2, 3, 10, 11, 12], float)
        self.assertEqual(segment_bounds(t), [(0, 3), (3, 7), (7, 10)])
        valid = np.ones(10, bool)
        valid[5] = False
        self.assertEqual(segment_bounds(t, valid=valid), [(0, 3), (3, 5), (6, 7), (7, 10)])

    def test_status_counts(self):
        x = np.full(10, 1e-12)
        s = campbell_stats(x + np.arange(10) * 1e-15, np.arange(10.0), 1, 60,
                           status=np.full(10, (1 << 9) | (1 << 1)))
        self.assertEqual(s["n_zero_check"], 10)
        self.assertEqual(s["n_filter_flag"], 10)


class FloorSubtraction(unittest.TestCase):
    def test_floor_subtracted_charge(self):
        light = {"t_int_s": 1 / 60, "var_a2": 2.0e-26, "var_rel_err": 0.01, "mean_a": 1e-9}
        dark = {"var_a2": 1.0e-26, "var_rel_err": 0.01, "mean_a": 0.0}
        q, err = floor_subtracted_charge(light, dark)
        self.assertAlmostEqual(q, (1 / 60) * 1.0e-26 / 1e-9)
        self.assertGreater(err, 0)


class Independence(unittest.TestCase):
    def test_stats_imports_only_stdlib_and_numpy(self):
        with open(os.path.join(ROOT, "keithley6485", "campbell", "stats.py")) as fh:
            tree = ast.parse(fh.read())
        mods = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                mods |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom):
                self.assertEqual(node.level, 0, "stats.py must not import from the package")
                mods.add(node.module.split(".")[0])
        self.assertLessEqual(mods, {"__future__", "math", "typing", "numpy"})

    def test_importing_stats_loads_no_driver(self):
        code = ("import sys; import keithley6485.campbell.stats; "
                "bad = [m for m in ('keithley6485.campbell.meter', 'keithley6485.campbell.transport', "
                "'keithley6485.campbell.sim', 'serial', 'h5py') if m in sys.modules]; print(bad); "
                "sys.exit(1 if bad else 0)")
        r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    def test_no_daq_imports_anywhere(self):
        pkg = os.path.join(ROOT, "keithley6485", "campbell")
        for fn in os.listdir(pkg):
            if not fn.endswith(".py"):
                continue
            with open(os.path.join(pkg, fn)) as fh:
                tree = ast.parse(fh.read())
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0:
                    names = [node.module]
                for nm in names:
                    self.assertFalse(nm == "daq" or nm.startswith("daq."), f"{fn} imports {nm}")


if __name__ == "__main__":
    unittest.main()
