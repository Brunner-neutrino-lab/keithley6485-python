# keithley6485

Python code for the Keithley 6485 picoammeter:

- `keithley6485.campbell`: a standalone program (GUI + CLI) for Campbell
  (shot-noise) runs, e.g. measuring the electron charge from photodiode shot
  noise. Direct serial, no ETS DAQ needed. Described first below.
- `keithley6485.driver.K6485Driver`: the original pyvisa driver that the ETS DAQ
  web app loads (unchanged by the Campbell program).

## Campbell runs

The program takes raw, unfiltered, fixed-range readings into the meter's buffer,
in bursts of up to 2500. It records the instrument's own time stamps and status
words, and reads back every setting from the meter. Statistics are computed from
those arrays alone. Each run goes into one HDF5 file.

Two times are kept apart:

- `T_int = NPLC / f_line`, the integration aperture, enters Campbell:
  `Var(I) = e <I> / T_int`.
- `dt`, the reading interval from the time stamps, sets the spectra's frequency
  axis.

Nothing is averaged, filtered or autoranged, so correlated readings show up in
the autocorrelation, the successive-difference variance and the spectrum.

### Install

Python 3.9 or newer. The dependencies are numpy, pyserial, h5py and NiceGUI.

```bash
git clone git@github.com:Brunner-neutrino-lab/keithley6485-python.git   # or use this folder
cd keithley6485-python
python -m venv .venv && source .venv/bin/activate                         # optional
pip install .                    # or: pip install -e .   (editable)
```

This installs `k6485-campbell` (CLI) and `k6485-campbell-gui`. Without
installing, run from this folder with `python -m keithley6485.campbell ...`.

### Try it without a meter

```bash
k6485-campbell-gui                      # open http://127.0.0.1:8771/ ; port "simulation", connect, start
k6485-campbell run --sim --sim-current 1e-9 --sim-floor 0 --n 20000 --name sim_test
k6485-campbell stats data/campbell_k6485/sim_test_*.h5
```

The simulator speaks the meter's SCPI: settings, the error queue, the buffer
and `TRAC:DATA?`. Its readings are `e*Poisson(I*T/e)/T` plus a white floor.
The default floor is the specification's typical RMS noise, scaled as
`sqrt(6/NPLC)`. Two optional filters mimic a correlated meter: `ar1`
(phi < 0 is anti-correlated, like the B2987 at 1 PLC) and `boxcar`
(overlapping sub-conversions).

### Run a measurement

1. **The meter.** Warm it up for 1 h. Set COMM to RS-232 and note its baud rate,
   terminator and flow control. On the ETS bench these are 9600 baud, CR and no
   flow control; don't change the baud rate without Lucas. Use a
   straight-through DB-9 cable.
2. **Make sure no one else is using it.**
   - `k6485-campbell ports` lists the serial ports, and which process holds each.
   - On the ETS machine, the program also asks the DAQ web app. It refuses while
     the web app has the K6485 connected, claimed or in use.
   - Disconnect it there first ("release all instruments").
3. **GUI:** `k6485-campbell-gui`.
   - Pick the port under `/dev/serial/by-id/` and connect. The settings as
     found are read back and shown.
   - Choose the range, NPLC (an integer) and autozero, plus N readings.
   - Fill in the operator notes: laser, ND, photodiode, bias, box state.
   - Press start. When the run ends, the program restores the meter's settings
     and saves the file.
   - To serve the GUI on the lab network, add `--host 0.0.0.0`.

   **CLI:**
   ```bash
   k6485-campbell run --port /dev/serial/by-id/usb-Prolific_Technology_Inc._USB-Serial_Controller_D-if00-port0 \
       --range 20n --nplc 1 --no-autozero --n 20000 --name pd_nd2 \
       --note-laser "..." --note-nd "ND 2" --note-photodiode "..." --note-bias "0 V" --note-box "closed"
   ```
   Ctrl-C stops after the current burst's data are read back. The partial run
   is still restored and saved.
4. **The runs Will needs at each range and NPLC:**
   - a floor with zero check on (`--zero-check`, or input "floor" in the GUI);
   - a floor with the beam blocked;
   - photocurrent at six or more ND settings;
   - the battery and resistor control at the same currents.

   `floor_subtracted_charge(light, dark)` in `keithley6485.campbell.stats`
   combines a light run with a dark run.
5. **Check the statistics.** Lag-1 autocorrelation should be near 0, the
   spectrum flat, and Allan m·σ² flat. `T_int*Var/<I>`,
   `T_int*sdiff/<I>` and `T_int*P_band/<I>` should agree. If they don't, the
   readings are correlated, and `e / (T_int*Var/<I>)` measures T_eff/T_int.

**Timing at 9600 baud.** A 2500-reading burst is about 105 kB of ASCII
(reading, time and status, 42 characters each), so it takes about 110 s to
transfer. At 1 PLC the burst itself takes about 45 s. The time stamps make the
gaps between bursts visible. The variance does not care about them, and the
spectra are taken inside bursts. At 57.6 kbaud the transfer would take about
18 s.

### Output file

`data/campbell_k6485/<name>_<YYYYmmdd_HHMMSS>.h5`. It is written to a temporary
name and then renamed. It holds one group, named like the file:

| item | content |
|---|---|
| `i_a` | readings (A), raw; overflow readings are +9.9E37 |
| `t_s` | instrument time stamps (s); each burst starts at 0 |
| `status` | 16-bit status word per reading (bit 0 overflow, 7 overvoltage, 9 zero check, ...) |
| `burst` | burst index per reading |
| `bursts/*` | per burst: host times, start time, `SENS:DATA?` time of the last reading, gaps, acquire and transfer s |
| attrs `set_*` | every setting read back after configuring (range, NPLC, line frequency, filters, autozero, trigger, format, *IDN?) |
| attrs `found_*` | the settings as found, which are restored afterwards |
| attrs `stat_*`, `note_*`, `req_*`, `thr_*`, `sim_*` | statistics, operator notes, the request, throughput, simulation parameters |

`k6485-campbell stats FILE [--band LO HI]` recomputes everything from the
arrays. `keithley6485.campbell.stats` imports only numpy, so an analysis can
use it on its own.

### Instrument facts

The facts below are checked against the 6485 Instruction Manual, 6485-901-01
Rev. A:

- buffer of 1 to 2500 readings;
- `FORM:ELEM` defaults to all four elements. With it, `TRAC:DATA?` interleaves
  reading, time and status (three fields, the unit glued to the reading), so
  the driver sets `READ,TIME,STAT` and parses in strides of three;
- commands wait while the meter is in the trigger model, so the driver waits on
  `*OPC?`;
- there is no damping filter, only median and digital average;
- ranges run from 2 nA to 20 mA;
- RS-232 runs at 300 to 57600 baud, ASCII only.

`keithley6485/campbell/meter.py` lists them with page numbers.

**Not yet confirmed on the instrument:**
- the exact ASCII field format;
- whether `SENS:DATA?` carries the absolute timer, which would give the gaps
  between bursts on the instrument clock (the host clock is the fallback);
- whether NPLC above 6 is accepted over SCPI. The SCPI tables say 6, the
  front-panel menu says 60;
- the query reply formats.

The driver raises an error on any error-queue entry or readback mismatch, so a
wrong guess fails loudly instead of corrupting data.

### Tests

```bash
python -m unittest discover -s tests -t .
```

These are simulation only and never touch hardware. They check:
- that `campbell_stats` recovers e within its error;
- that the AR(1) and boxcar filters move the plain variance but not a
  low-band periodogram;
- the settings readback and restore, the error queue, chained and aborted
  bursts, the parser, the HDF5 round trip, the port guard and the CLI.

## Original driver (used by the ETS DAQ)

```python
from keithley6485 import K6485Driver

with K6485Driver(visa="ASRL/dev/ttyUSB0::INSTR", mode="hardware",
                 baud_rate=9600, read_termination="\r", write_termination="\r") as drv:
    drv.zero_check_off()
    drv.set_range(20e-9)
    drv.set_nplc(1.0)
    current = drv.read_single()                    # A
    currents, timestamps = drv.read_n(10, delay_s=0.05)
```

Its `connect()` sends `*RST`. Its `read_buffer()` assumes one value per reading
in `TRAC:DATA?` and makes up its time stamps, so use the Campbell program for
buffered data. Simulation: `K6485Driver(mode="simulation")`. NiceGUI panel:
`python -m keithley6485.gui`.
