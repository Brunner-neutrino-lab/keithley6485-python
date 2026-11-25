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
3. **Connect:** `k6485-campbell-gui` (on the ETS machine:
   `python scripts/k6485_campbell.py gui`).
   - Pick the port under `/dev/serial/by-id/` and connect. The settings as
     found are read back and shown.
   - To serve the GUI on the lab network, add `--host 0.0.0.0`.
4. **Set the light level: "check level".**
   - It takes 100 readings with zero check off and saves nothing.
   - It shows the current, its sd against the nominal shot sd, and the
     fraction of the range used.
   - It warns on overflow and says when a lower range would fit.
   - Keep the photocurrent in the top half to two-thirds of the range.
   - On the 2 nA range, a laser that's on without ND overflows.
5. **Take a run.**
   - Choose the range, NPLC (an integer) and autozero, plus N readings.
   - Fill in the notes: run kind, ND, laser, photodiode, bias, box state; the
     resistor value for control or Johnson runs.
   - Press start. When the run ends, the program restores the meter's settings
     and saves one file.
   - **NPLC sequence:** enter `1,2,5,10` to take one run per NPLC.
   - **Zero-check floor after each:** adds a zero-check floor run after every
     setting.
   - **Run kinds:**
     - photocurrent: laser on;
     - dark: beam blocked;
     - resistor control: battery + resistor at the same current;
     - johnson: resistor at 0 V.

     Zero-check runs are marked by themselves.
6. **Fit.** "fit runs" analyses the runs saved since connecting. From saved
   files: `k6485-campbell fit data/campbell_k6485/*.h5 [--estimator naive|sdiff|band]`.
   - **Series fit:** for each range and NPLC, fit
     `T_int*V = a + e<I> + b<I>^2` over the photocurrent runs. Here a is the
     floor, the slope is e, and b is the laser's classical noise. The b term
     is fitted with 4 or more runs; the brief asks for 6 or more currents.
   - **Resistor control:** each photocurrent run is paired with a control run
     at the same range, NPLC and current (within 10 %):
     `e = T_int (V_pd - V_ctl + 2kT/(R T_int)) / <I>`.
   - **Johnson runs:** fit `V = c + d/R (+ f/R^2)` and get
     `T_eff = 2 k T_R / d`, which needs no e. Put the resistance in the
     resistor note.
   - **NPLC table:** `T_int*V/<I>` per run. For shot noise it is the same at
     every NPLC.

**CLI equivalents:**
```bash
P=/dev/serial/by-id/usb-Prolific_Technology_Inc._USB-Serial_Controller_D-if00-port0
k6485-campbell check --port $P --range 2n --nplc 1
k6485-campbell run --port $P --range 2n --nplc 1 2 5 10 --zc-floor --n 20000 --name pd_nd1 \
    --kind photocurrent --note-nd "ND 1" --note-laser "..." --note-photodiode "..." --note-bias "0 V" --note-box "closed"
k6485-campbell run --port $P --range 2n --nplc 1 --n 20000 --name ctl_1G --kind "resistor control" \
    --note-resistor_ohm 1.5e9 --note-battery_v 1.5
k6485-campbell fit data/campbell_k6485/*.h5 --estimator sdiff
```
Ctrl-C stops after the current burst's data are read back. The partial run is
still restored and saved.

**Check the statistics.**
- Lag-1 autocorrelation should be near 0, and the spectrum and Allan m·σ²
  should be flat.
- `T_int*Var/<I>`, `T_int*sdiff/<I>` and `T_int*P_band/<I>` should agree. If
  they don't, the readings are correlated.
- With pure shot noise, `e / (T_int*Var/<I>)` is T_eff/T_int.

**What the meter showed (2026-11-25, S/N 4456711).**
- **Zero-check floor:** about 35 fA at 0.1 PLC, 4.5-5 fA at 1 PLC and
  3 fA at 5 PLC on the 2 nA and 20 nA ranges. It is 0.3-0.5 pA on 200 nA.
- **At 1 and 5 PLC** the floor is dominated by slow drift: lag-1 is
  +0.2 to +0.56, and the spectrum falls 6-20x by 10 Hz. Subtract floors with
  the sdiff or band estimate.
- **At 0.1 PLC** it is anti-correlated, with lag-1 -0.22, from line harmonics.
- **Readings are quantized** in steps of 3.55 fA on 2 nA and 20 nA, and
  0.23 pA on 200 nA.
- **The photodiode dark at 0.1 PLC** (laser off, zero check off) was
  19 pA ± 6.7 pA.

**Timing at 9600 baud** (measured).
- Transfer takes 54-55 ms per reading, so a 2500-reading burst is about 137 s.
- Acquisition at 1 PLC with autozero off runs at 58.8 readings/s, about 43 s
  per burst (T_int/dt = 0.98).
- At 0.1 PLC it is 498 readings/s; at 5 PLC, 11.95 readings/s.
- So 20 000 readings at 1 PLC take about 24 min.
- The gaps between bursts are timed on the instrument clock. The variance does
  not care about them, and the spectra are taken inside bursts.
- At 57.6 kbaud the transfer would take about 23 s per burst (Lucas's call).

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

**Confirmed on the instrument** (2026-11-25):
- `TRAC:DATA?` sends `+1.776357E-13,+0.000000E+00,+5.120000E+02,...`;
- `SENS:DATA?` carries the absolute timer, so the gaps between bursts are on
  the instrument clock;
- `*OPC?` after INIT works;
- every snapshot query is answered;
- the range reads back 5 % over nominal;
- restore leaves the meter as found.

**Still open:** whether NPLC above 6 is accepted over SCPI. The SCPI tables say
6, the front-panel menu says 60.

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
  bursts, the parser, the HDF5 round trip, the port guard and the CLI;
- the series fit (e, floor and laser terms), the resistor control, Johnson
  T_eff, the quick check and NPLC sequences.

## Original driver (used by the ETS DAQ)

```python
from keithley6485 import K6485Driver

with K6485Driver(visa="ASRL/dev/ttyUSB0::INSTR", mode="hardware",
                 baud_rate=9600, read_termination="\r", write_termination="\r") as drv:
    drv.reset()                                    # connect() sends no *RST
    drv.zero_check_off()
    drv.set_range(20e-9)
    drv.set_nplc(1.0)
    current = drv.read_single()                    # A
    currents, timestamps = drv.read_n(10, delay_s=0.05)
```

- 8 current ranges: 2 nA – 20 mA
- Interface: RS-232 (serial) or GPIB via pyvisa
- Zero check must be OFF before measuring (`drv.zero_check_off()`)
- `connect()` checks `*IDN?` and sends no `*RST`; call `reset()` for factory
  defaults.
- `read_buffer(n)` (n up to 2500) sets `FORM:ELEM READ,TIME,STAT`, arm count 1
  and trigger count n, waits on `*OPC?`, reads `TRAC:DATA?` (three fields per
  reading) and puts the trigger and arm counts and `FORM:ELEM` back as found:
  `READ?` returns the whole trigger group, so `read_single` depends on them.
  The time stamps are the host clock at INIT plus the meter's buffer stamps
  (1/1024 s ticks). The call ends within 2 s per reading + 5 s; at 9600 baud
  the transfer alone is about 56 ms per reading.

Simulation: `K6485Driver(mode="simulation")`. NiceGUI panel:
`python -m keithley6485.gui`.
