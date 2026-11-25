"""HDF5 store, session (CLI/GUI backend), port guard, CLI."""

import http.server
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from keithley6485.campbell import guard                                   # noqa: E402
from keithley6485.campbell.session import Connection, RunSettings, Session  # noqa: E402
from keithley6485.campbell.sim import SimPhysics                           # noqa: E402
from keithley6485.campbell.stats import campbell_stats                     # noqa: E402
from keithley6485.campbell.store import read_run, write_run                # noqa: E402


class Store(unittest.TestCase):
    def test_round_trip_and_no_temp_left(self):
        with tempfile.TemporaryDirectory() as d:
            rng = np.random.default_rng(0)
            i = 1e-9 + 1e-13 * rng.standard_normal(3000)
            t = np.concatenate([np.arange(2500), np.arange(500)]) * 0.0175
            b = np.repeat([0, 1], [2500, 500])
            st = campbell_stats(i, t, 1.0, 60.0, burst=b)
            p = write_run(Path(d) / "x" / "pd_20261125_120000.h5", i_a=i, t_s=t, status=np.zeros(3000),
                          burst=b, bursts={"n": [2500, 500]},
                          settings={"nplc": 1.0, "line_freq_hz": 60.0, "range_a": 2e-8, "idn": "K",
                                    "_request": {"x": 1}},
                          stats=st, notes={"laser": "HeNe, ND 2", "box": "closed"})
            self.assertEqual(sorted(os.listdir(p.parent)), ["pd_20261125_120000.h5"])
            r = read_run(p)
            self.assertEqual(r["run_id"], "pd_20261125_120000")
            np.testing.assert_array_equal(r["i_a"], i)
            np.testing.assert_array_equal(r["t_s"], t)
            self.assertEqual(r["attrs"]["set_nplc"], 1.0)
            self.assertEqual(r["attrs"]["note_laser"], "HeNe, ND 2")
            self.assertAlmostEqual(r["attrs"]["stat_var_a2"], st["var_a2"])
            st2 = campbell_stats(r["i_a"], r["t_s"], r["settings"]["nplc"], r["settings"]["line_freq_hz"],
                                 burst=r["burst"], status=r["status"])
            self.assertEqual(st2["var_a2"], st["var_a2"])
            self.assertEqual(st2["acf"], st["acf"])

    def test_failed_write_leaves_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(Exception):
                write_run(Path(d) / "bad.h5", i_a=np.zeros(3), t_s=np.zeros(3), status=np.zeros(3),
                          burst=np.zeros(3), settings={"bad": object()})
            self.assertEqual(os.listdir(d), [])


class SessionRuns(unittest.TestCase):
    def test_sim_session_run_saves_and_restores(self):
        with tempfile.TemporaryDirectory() as d:
            s = Session()
            s.connect(Connection(port="simulation", sim=SimPhysics(current_a=2e-9, seed=2)))
            s.notes["laser"] = "sim"
            s.start(RunSettings(range_a=20e-9, nplc=1.0, n_total=5000, settle_s=0.0, out_dir=d, name="t"))
            s.wait(60)
            self.assertFalse(s.running)
            self.assertIsNone(s.error)
            self.assertEqual(s.restore_diff, {})
            self.assertIsNotNone(s.last_path)
            r = read_run(s.last_path)
            self.assertEqual(len(r["i_a"]), 5000)
            self.assertEqual(r["notes"]["laser"], "sim")
            self.assertEqual(r["sim"]["current_a"], 2e-9)
            self.assertTrue(r["attrs"]["restored_as_found"])
            self.assertEqual(r["found"]["nplc"], 6.0)
            self.assertEqual(r["settings"]["nplc"], 1.0)
            s.disconnect()
            self.assertFalse(s.connected)

    def test_stop_mid_run(self):
        with tempfile.TemporaryDirectory() as d:
            s = Session()
            s.connect(Connection(port="simulation", sim=SimPhysics(seed=3), time_scale=0.01))
            s.start(RunSettings(n_total=10000, settle_s=0.0, out_dir=d, autosave=False))
            time.sleep(1.0)
            s.stop()
            s.wait(60)
            self.assertLess(s.run.n, 10000)
            self.assertEqual(s.restore_diff, {})
            p = s.save(Path(d) / "partial.h5")
            self.assertTrue(read_run(p)["attrs"]["stopped_early"])
            s.disconnect()


class _StateHandler(http.server.BaseHTTPRequestHandler):
    state = {}

    def do_GET(self):
        body = json.dumps({"ok": True, "result": self.state}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


class Guard(unittest.TestCase):
    def test_missing_device(self):
        problems, _ = guard.port_problems("/dev/serial/by-id/no-such-adapter", check_daq=False)
        self.assertTrue(problems and "does not exist" in problems[0])

    @unittest.skipUnless(os.path.isdir("/proc"), "Linux only")
    def test_other_process_holding_the_device(self):
        with tempfile.NamedTemporaryFile() as f:
            child = subprocess.Popen([sys.executable, "-c", f"f = open({f.name!r}); import time; time.sleep(30)"])
            try:
                for _ in range(50):
                    users = guard.processes_using(f.name)
                    if users:
                        break
                    time.sleep(0.1)
                self.assertEqual([u[0] for u in users], [child.pid])
                problems, _ = guard.port_problems(f.name, check_daq=False)
                self.assertTrue(problems)
            finally:
                child.kill()
                child.wait()

    def test_daq_state(self):
        srv = http.server.HTTPServer(("127.0.0.1", 0), _StateHandler)
        th = threading.Thread(target=srv.serve_forever, daemon=True)
        th.start()
        old = guard.DAQ_URL, os.environ.get("DAQ_API_TOKEN")
        guard.DAQ_URL = f"http://127.0.0.1:{srv.server_address[1]}"
        os.environ["DAQ_API_TOKEN"] = "test"
        try:
            _StateHandler.state = {"status": {"k6485": "disconnected"}, "claims": [], "calls": [], "holders": {}}
            self.assertEqual(guard.daq_k6485_problems()[0], [])
            _StateHandler.state = {"status": {"k6485": "OK — K6485 @ /dev/ttyUSB0"}}
            self.assertEqual(len(guard.daq_k6485_problems()[0]), 1)
            _StateHandler.state = {"status": {"k6485": "disconnected"},
                                   "claims": [{"id": "c-1", "keys": ["k6485"], "state": "live", "actor": "x"},
                                              {"id": "c-2", "keys": ["k6485"], "state": "released"}]}
            self.assertEqual(len(guard.daq_k6485_problems()[0]), 1)
        finally:
            guard.DAQ_URL = old[0]
            if old[1] is None:
                os.environ.pop("DAQ_API_TOKEN", None)
            else:
                os.environ["DAQ_API_TOKEN"] = old[1]
            srv.shutdown()


class Cli(unittest.TestCase):
    def test_cli_sim_run_and_stats(self):
        with tempfile.TemporaryDirectory() as d:
            env = dict(os.environ, PYTHONPATH=ROOT)
            r = subprocess.run([sys.executable, "-m", "keithley6485.campbell", "run", "--sim", "--sim-seed", "1",
                                "--sim-floor", "0", "--n", "3000", "--settle", "0", "--name", "cli",
                                "--out-dir", d, "--note-nd", "ND 1"],
                               capture_output=True, text=True, env=env, timeout=120)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertIn("T_int*Var/<I>", r.stdout)
            files = [f for f in os.listdir(d) if f.endswith(".h5")]
            self.assertEqual(len(files), 1)
            r2 = subprocess.run([sys.executable, "-m", "keithley6485.campbell", "stats", os.path.join(d, files[0]),
                                 "--json"], capture_output=True, text=True, env=env, timeout=60)
            self.assertEqual(r2.returncode, 0, r2.stderr)
            self.assertEqual(json.loads(r2.stdout)["n"], 3000)

    def test_cli_refuses_missing_port(self):
        env = dict(os.environ, PYTHONPATH=ROOT)
        r = subprocess.run([sys.executable, "-m", "keithley6485.campbell", "snapshot", "--port",
                            "/dev/serial/by-id/no-such-adapter", "--no-daq-check"],
                           capture_output=True, text=True, env=env, timeout=60)
        self.assertEqual(r.returncode, 1)
        self.assertIn("does not exist", r.stderr)


class GuiHelpers(unittest.TestCase):
    def test_imports_and_helpers(self):
        try:
            import nicegui  # noqa: F401
        except ImportError:
            self.skipTest("nicegui not installed")
        from keithley6485.campbell import gui
        t = np.arange(10000.0)
        y = np.sin(t / 50.0)
        y[1234] = 5.0
        tt, yy = gui._decimate(t, y, 1000)
        self.assertLessEqual(len(yy), 1100)
        self.assertEqual(yy.max(), 5.0)
        st = campbell_stats(1e-9 + 1e-13 * np.random.default_rng(1).standard_normal(5000),
                            np.arange(5000) * 0.0175, 1, 60)
        rows = gui.stats_rows(st, {})
        self.assertTrue(any("T_int·Var/<I>" == r["q"] for r in rows))


if __name__ == "__main__":
    unittest.main()
