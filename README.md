# keithley6485

Python driver for the Keithley 6485 picoammeter.

Used in this DAQ as a VUV flux monitor: an XUV photodiode inside the cryostat
is read continuously to track and normalize the VUV photon flux delivered to
the SiPM under test.

## Quick Start

```bash
pip install -r requirements.txt

python examples/basic_usage.py   # headless demo (simulation mode)
```

## API

```python
from keithley6485 import K6485Driver

with K6485Driver(visa="ASRL6::INSTR", mode="hardware") as drv:
    drv.reset()
    drv.zero_check_off()
    drv.set_range("AUTO")
    drv.set_nplc(1.0)

    # Single reading
    current = drv.read_single()           # A

    # N readings with optional delay
    currents, timestamps = drv.read_n(10, delay_s=0.05)

    # Fast buffer burst (n readings, single INIT)
    currents, timestamps = drv.read_buffer(100)
```

## Simulation mode

```python
drv = K6485Driver(mode="simulation")
drv.sim_set_illuminated(True)    # switch between dark and illuminated baseline
drv.sim_set_photocurrent(15e-9)  # set illuminated current level (A)
```

## Instrument notes

- 8 current ranges: 2 nA – 20 mA
- Interface: RS-232 (serial) or GPIB via pyvisa
- Default baud: 57600, 8N1
- Zero check must be OFF before measuring (`drv.zero_check_off()`)
