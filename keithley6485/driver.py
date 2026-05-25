"""
keithley6485/driver.py

Low-level SCPI driver for the Keithley 6485 picoammeter.

The 6485 is a current-measurement-only instrument (no voltage source).
It is used in this DAQ as a VUV flux monitor via an XUV photodiode.

Interface: Serial (RS-232) or GPIB via pyvisa.
Default baud: 57600, 8N1, CR+LF termination.

Two modes:
  "hardware"   — connects via pyvisa
  "simulation" — returns synthetic photocurrent for development
"""

import time
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_VISA      = "ASRL6::INSTR"     # adjust COM port as needed
TIMEOUT_MS        = 10_000
BAUD_RATE         = 57600
READ_TERMINATION  = "\r\n"
WRITE_TERMINATION = "\n"
IDN_EXPECTED      = "KEITHLEY INSTRUMENTS INC.,MODEL 6485"

# Current ranges (A) — 6485 has 8 ranges
CURRENT_RANGES = [2e-9, 20e-9, 200e-9, 2e-6, 20e-6, 200e-6, 2e-3, 20e-3]

# Simulation: XUV photodiode photocurrent model
# Roughly linear with VUV flux; typical value ~10 nA during illumination
SIM_PHOTOCURRENT_A = 10e-9     # A — illuminated baseline
SIM_DARK_A         = 50e-12    # A — dark current
SIM_NOISE_A        = 2e-12     # A — current noise RMS


class K6485Driver:
    """
    Low-level SCPI driver for the Keithley 6485 picoammeter.

    Parameters
    ----------
    visa : str
        VISA resource string (e.g. "ASRL6::INSTR" or "GPIB0::22::INSTR").
    mode : str
        "hardware" or "simulation".
    """

    def __init__(self, visa: str = DEFAULT_VISA, mode: str = "simulation",
                 baud_rate: int = BAUD_RATE,
                 read_termination:  str = READ_TERMINATION,
                 write_termination: str = WRITE_TERMINATION):
        if mode not in ("hardware", "simulation"):
            raise ValueError(f"mode must be 'hardware' or 'simulation', got {mode!r}")
        self._visa_str          = visa
        self._mode              = mode
        self._baud_rate         = baud_rate
        self._read_termination  = read_termination
        self._write_termination = write_termination
        self._inst              = None
        self._rm                = None
        self._connected         = False

        # Simulation state
        self._sim_illuminated = False
        self._sim_rng         = np.random.default_rng()

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self):
        if self._connected:
            return
        if self._mode == "hardware":
            self._connect_hardware()
        self._connected = True

    def disconnect(self):
        if not self._connected:
            return
        if self._mode == "hardware":
            self._disconnect_hardware()
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def mode(self) -> str:
        return self._mode

    # ------------------------------------------------------------------
    # SCPI primitives
    # ------------------------------------------------------------------

    def write(self, cmd: str):
        if self._mode == "hardware":
            self._inst.write(cmd)

    def query(self, cmd: str) -> str:
        if self._mode == "hardware":
            return self._inst.query(cmd).strip()
        return ""

    # ------------------------------------------------------------------
    # Instrument configuration
    # ------------------------------------------------------------------

    def reset(self):
        """Reset to factory defaults."""
        self.write("*RST")
        self.write("*CLS")
        if self._mode == "hardware":
            time.sleep(1.0)

    def zero_check_off(self):
        """Disable zero check (must be off to take measurements)."""
        self.write("SYST:ZCH OFF")

    def zero_check_on(self):
        """Enable zero check."""
        self.write("SYST:ZCH ON")

    def zero_correct(self):
        """
        Perform zero correction (zero check must be ON first, then correct,
        then zero check OFF before measuring).
        """
        self.write("SYST:ZCH ON")
        self.write("INIT")
        self.write("SYST:ZCOR:ACQ")
        self.write("SYST:ZCOR ON")
        self.write("SYST:ZCH OFF")

    def set_range(self, range_a: float | str = "AUTO"):
        """
        Set current measurement range.

        Parameters
        ----------
        range_a : float or "AUTO"
            Range in amperes, e.g. 20e-9 for 20 nA range.
            "AUTO" enables autorange.
        """
        if str(range_a).upper() == "AUTO":
            self.write("CURR:RANG:AUTO ON")
        else:
            self.write("CURR:RANG:AUTO OFF")
            self.write(f"CURR:RANG {float(range_a):.2e}")

    def set_nplc(self, nplc: float = 1.0):
        """
        Set integration rate in power-line cycles (NPLC).

        Parameters
        ----------
        nplc : float
            0.01 to 6.0 (at 60 Hz) or 0.01 to 5.0 (at 50 Hz).
            Higher = slower, less noise.
        """
        self.write(f"CURR:NPLC {nplc:.3f}")

    def set_average(self, n: int = 1, mode: str = "NONE"):
        """
        Configure digital averaging filter.

        Parameters
        ----------
        n : int
            Number of readings to average (1–100).
        mode : str
            "NONE", "SCAL" (scalar average), "ADV" (advanced).
        """
        if mode.upper() == "NONE" or n <= 1:
            self.write("AVER:STAT OFF")
        else:
            self.write(f"AVER:COUN {n}")
            self.write(f"AVER:TCON {mode.upper()}")
            self.write("AVER:STAT ON")

    def configure_buffer(self, n: int):
        """
        Configure the internal reading buffer.

        Parameters
        ----------
        n : int
            Number of readings to store (1–2500).
        """
        self.write("TRAC:CLE")
        self.write(f"TRAC:POIN {n}")
        self.write("TRAC:FEED SENS")
        self.write("TRAC:FEED:CONT NEXT")

    # ------------------------------------------------------------------
    # Measurement
    # ------------------------------------------------------------------

    def read_single(self) -> float:
        """
        Take one current reading.

        Returns
        -------
        float
            Current in amperes.
        """
        if self._mode == "hardware":
            raw = self.query("READ?")
            # Response format: "<value>A,<timestamp>,..."; take first field
            return float(raw.split(",")[0].replace("A", "").strip())
        else:
            return self._sim_single()

    def read_n(self, n: int, delay_s: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        """
        Take n current readings with optional inter-reading delay.

        Returns
        -------
        currents : np.ndarray
            Current readings in amperes.
        timestamps : np.ndarray
            UTC timestamps for each reading (s).
        """
        if self._mode == "hardware":
            return self._read_n_hardware(n, delay_s)
        else:
            return self._read_n_simulation(n, delay_s)

    def read_buffer(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Trigger n readings into the internal buffer, then retrieve all at once.
        More efficient than read_n for large n.

        Returns
        -------
        currents : np.ndarray
        timestamps : np.ndarray  (relative to first reading, seconds)
        """
        if self._mode == "hardware":
            return self._read_buffer_hardware(n)
        else:
            return self._read_n_simulation(n, delay_s=0.0)

    # ------------------------------------------------------------------
    # Hardware internals
    # ------------------------------------------------------------------

    def _connect_hardware(self):
        try:
            import pyvisa
        except ImportError as e:
            raise ImportError(
                "pyvisa not installed. Run: pip install pyvisa pyvisa-py"
            ) from e

        self._rm   = pyvisa.ResourceManager()
        self._inst = self._rm.open_resource(
            self._visa_str,
            read_termination  = self._read_termination,
            write_termination = self._write_termination,
        )
        self._inst.timeout    = TIMEOUT_MS
        self._inst.baud_rate  = self._baud_rate   # serial only; ignored for GPIB

        idn = self._inst.query("*IDN?").strip()
        if IDN_EXPECTED not in idn:
            self._inst.close()
            raise RuntimeError(
                f"IDN mismatch. Expected to contain {IDN_EXPECTED!r},\n"
                f"got: {idn!r}\nCheck VISA string: {self._visa_str!r}"
            )
        self.reset()

    def _disconnect_hardware(self):
        if self._inst is not None:
            try:
                self.zero_check_on()
                self._inst.close()
            except Exception:
                pass
        if self._rm is not None:
            try:
                self._rm.close()
            except Exception:
                pass

    def _read_n_hardware(self, n: int,
                          delay_s: float) -> tuple[np.ndarray, np.ndarray]:
        currents   = np.empty(n)
        timestamps = np.empty(n)
        for i in range(n):
            t0 = time.time()
            currents[i]   = self.read_single()
            timestamps[i] = t0
            if delay_s > 0 and i < n - 1:
                time.sleep(delay_s)
        return currents, timestamps

    def _read_buffer_hardware(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        self.configure_buffer(n)
        self.write(f"TRIG:COUN {n}")
        self.write("INIT")
        # Wait for buffer to fill
        deadline = time.monotonic() + n * 2.0 + 5.0
        while time.monotonic() < deadline:
            try:
                done = self.query("TRAC:POIN:ACT?")
                if int(done) >= n:
                    break
            except Exception:
                pass
            time.sleep(0.1)

        raw = self.query("TRAC:DATA?")
        t0  = time.time()
        vals = [float(v.replace("A", "").strip())
                for v in raw.split(",") if v.strip()]
        currents   = np.array(vals[:n], dtype=np.float64)
        timestamps = t0 + np.arange(n) * (1.0 / 60.0)  # approximate
        return currents, timestamps

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def _sim_single(self) -> float:
        base  = SIM_PHOTOCURRENT_A if self._sim_illuminated else SIM_DARK_A
        noise = self._sim_rng.normal(0.0, SIM_NOISE_A)
        return abs(base + noise)

    def _read_n_simulation(self, n: int,
                            delay_s: float) -> tuple[np.ndarray, np.ndarray]:
        t0         = time.time()
        dt         = max(delay_s, 1.0 / 60.0)
        currents   = np.array([self._sim_single() for _ in range(n)])
        timestamps = t0 + np.arange(n) * dt
        return currents, timestamps

    def sim_set_illuminated(self, state: bool):
        """Toggle simulated illumination (simulation mode only)."""
        self._sim_illuminated = state

    def sim_set_photocurrent(self, current_a: float):
        """Set simulated photocurrent level (simulation mode only)."""
        global SIM_PHOTOCURRENT_A
        SIM_PHOTOCURRENT_A = current_a

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *_):
        self.disconnect()
