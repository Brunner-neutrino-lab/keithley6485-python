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

BUFFER_MAX = 2500                  # readings (6485-901-01 Rev. A, Table 6-1)
# The default FORM:ELEM is READ,UNIT,TIME,STAT, with the unit glued to the
# reading: three comma-separated fields per reading either way (p. 13-6).
TRACE_ELEMENTS = "READ,TIME,STAT"
# A count readback at or above this is INF (the meter answers 9.9E37).
_INF_COUNT = 9.8e37
# TRAC:DATA? took 54-55 ms a reading at 9600 baud on the bench's meter.
TRANSFER_S_PER_READING_9600 = 0.056

# Simulation: XUV photodiode photocurrent model
# Roughly linear with VUV flux; typical value ~10 nA during illumination
SIM_PHOTOCURRENT_A = 10e-9     # A — illuminated baseline
SIM_DARK_A         = 50e-12    # A — dark current
SIM_NOISE_A        = 2e-12     # A — current noise RMS


def parse_trace(raw: str, n_expected: int | None = None
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Parse a TRAC:DATA? reply taken with FORM:ELEM READ,TIME,STAT.

    Returns (currents in A, buffer time stamps in s, status words). A unit
    letter glued to the reading is accepted. Raises RuntimeError when the
    reply is not three fields per reading or not ``n_expected`` readings.
    """
    fields = [f.strip() for f in raw.strip().split(",")]
    if fields == [""]:
        fields = []
    if len(fields) % 3:
        raise RuntimeError(
            f"TRAC:DATA? returned {len(fields)} fields, not 3 per reading")
    n = len(fields) // 3
    if n_expected is not None and n != n_expected:
        raise RuntimeError(
            f"TRAC:DATA? returned {n} readings, expected {n_expected}")
    currents = np.array([float(f.rstrip("Aa")) for f in fields[0::3]],
                        dtype=np.float64)
    stamps = np.array([float(f) for f in fields[1::3]], dtype=np.float64)
    status = np.array([int(round(float(f))) for f in fields[2::3]],
                      dtype=np.int64)
    return currents, stamps, status


def _count_arg(reply: str) -> str:
    """A TRIG:COUN? / ARM:COUN? readback as an argument for the same command."""
    try:
        v = float(reply)
    except ValueError:
        raise RuntimeError(f"K6485 count readback {reply!r} is not a number") from None
    return "INF" if v >= _INF_COUNT else str(int(round(v)))


def _elements_arg(reply: str) -> str:
    """A FORM:ELEM? readback as an argument for the same command."""
    elems = [e.strip() for e in reply.strip().strip('"').upper().split(",")]
    if not elems or not set(elems) <= {"READ", "UNIT", "TIME", "STAT"}:
        raise RuntimeError(f"K6485 FORM:ELEM readback {reply!r} is not an element list")
    return ",".join(elems)


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
        # Set by a program that drives the resource itself for a while (the
        # ETS DAQ's Campbell runs): every command through this driver is then
        # refused, so nothing else interleaves on the serial line.
        self._held_by: str | None = None

        # Simulation state
        self._sim_illuminated = False
        self._sim_rng         = np.random.default_rng()

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self):
        """Open the port and check *IDN?. Sends no *RST: call reset() for that."""
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

    def _check_free(self):
        if self._held_by is not None:
            raise RuntimeError(f"K6485 is in use: {self._held_by}")

    def write(self, cmd: str):
        self._check_free()
        if self._mode == "hardware":
            self._inst.write(cmd)

    def query(self, cmd: str) -> str:
        self._check_free()
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
        # An aborted burst leaves the feed armed; stop it before clearing.
        self.write("TRAC:FEED:CONT NEV")
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
        self._check_free()
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
        self._check_free()
        if self._mode == "hardware":
            return self._read_n_hardware(n, delay_s)
        else:
            return self._read_n_simulation(n, delay_s)

    def read_buffer(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Trigger n readings into the internal buffer, then retrieve all at once.
        More efficient than read_n for large n.

        The call, burst and transfer, ends within 2 s per reading + 5 s. The
        trigger and arm counts and FORM:ELEM are put back afterwards.

        Parameters
        ----------
        n : int
            1 to 2500 (the buffer size).

        Returns
        -------
        currents : np.ndarray
            Current readings in amperes.
        timestamps : np.ndarray
            UTC timestamps (s), like read_n's: the host clock when INIT was
            sent plus the meter's own buffer time stamps, so the spacing is the
            instrument's.
        """
        if not 1 <= n <= BUFFER_MAX:
            raise ValueError(f"n must be in [1, {BUFFER_MAX}], got {n!r}")
        self._check_free()
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

        # Accept a bare Linux serial path (e.g. "/dev/ttyUSB0") and translate
        # to the equivalent ASRL VISA resource string. Saves operators from
        # having to remember the "ASRL...::INSTR" wrapper for a port they
        # already know by path. Windows COM ports (COM5 etc.) are also
        # accepted bare by pyvisa-py.
        resource = self._visa_str
        if resource.startswith("/dev/"):
            resource = f"ASRL{resource}::INSTR"

        self._rm   = pyvisa.ResourceManager()
        self._inst = self._rm.open_resource(
            resource,
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

    def _disconnect_hardware(self):
        if self._inst is not None:
            try:
                # A holder may be mid-burst: leave the input as it is then.
                if self._held_by is None:
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

    def _until(self, deadline: float, fn, *args):
        """fn(*args) with the VISA timeout stretched to the monotonic deadline."""
        self._inst.timeout = max(1.0, deadline - time.monotonic()) * 1000.0
        try:
            return fn(*args)
        finally:
            self._inst.timeout = TIMEOUT_MS

    def _abort_burst(self):
        """ABORt a burst and read the held *OPC?'s late "1" off the line."""
        self.write("ABOR")
        try:
            self._until(time.monotonic() + 2.0, self._inst.read)
        except Exception:
            pass

    def _read_buffer_hardware(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        from pyvisa.constants import BufferOperation

        deadline = time.monotonic() + n * 2.0 + 5.0
        # A reply left by an earlier timeout would answer the next query.
        self._inst.flush(BufferOperation.discard_read_buffer)
        # READ? returns the whole trigger group in the FORM:ELEM layout, so
        # read_single and read_n need these back as they were. Parsed before
        # anything changes: an out-of-step line fails here, harmlessly.
        put_back = [f"TRIG:COUN {_count_arg(self.query('TRIG:COUN?'))}",
                    f"ARM:COUN {_count_arg(self.query('ARM:COUN?'))}",
                    f"FORM:ELEM {_elements_arg(self.query('FORM:ELEM?'))}"]
        try:
            result = self._burst(n, deadline)
        except BaseException:
            self._put_back(put_back)
            raise
        failed = self._put_back(put_back)
        if failed:
            raise RuntimeError("K6485 settings not put back after read_buffer: "
                               + "; ".join(failed))
        return result

    def _put_back(self, cmds: list[str]) -> list[str]:
        failed = []
        for cmd in cmds:
            try:
                self.write(cmd)
            except Exception as e:
                failed.append(f"{cmd}: {e}")
        return failed

    def _burst(self, n: int, deadline: float) -> tuple[np.ndarray, np.ndarray]:
        # pyvisa-py restarts its timeout at every read chunk, so TRAC:DATA?
        # is read in one chunk, and its time is kept out of the burst's.
        transfer_s = (1.5 * n * TRANSFER_S_PER_READING_9600
                      * max(1.0, 9600.0 / self._baud_rate) + 2.0)
        self.write(f"FORM:ELEM {TRACE_ELEMENTS}")
        self.write("TRAC:TST:FORM ABS")
        self.configure_buffer(n)
        self.write("ARM:COUN 1")
        self.write(f"TRIG:COUN {n}")
        t_init = time.time()
        self.write("INIT")
        # In the trigger model the meter holds every command but ABORt,
        # *RST and SYST:PRES until it is idle (p. 7-4), so *OPC? answers
        # when the burst is over; a TRAC:POIN:ACT? poll would just block.
        try:
            opc = self._until(deadline - transfer_s, self.query, "*OPC?")
        except Exception as e:
            self._abort_burst()
            raise RuntimeError(f"K6485 burst of {n} readings did not finish "
                               f"in time; aborted") from e
        if opc != "1":
            # Out of step: the burst may still be running.
            self._abort_burst()
            raise RuntimeError(f"K6485 *OPC? after INIT answered {opc!r}; aborted")
        chunk = self._inst.chunk_size
        self._inst.chunk_size = n * 64 + 1024
        try:
            raw = self._until(deadline, self.query, "TRAC:DATA?")
        finally:
            self._inst.chunk_size = chunk
        currents, stamps, _status = parse_trace(raw, n)
        # TRAC:TST:FORM ABS: the first reading is at 0, the others after it.
        return currents, t_init + stamps

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
