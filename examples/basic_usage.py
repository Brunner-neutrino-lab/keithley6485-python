"""
Keithley 6485 basic usage example — simulation mode.

The 6485 is a picoammeter (no voltage source). It is used here as a VUV
flux monitor via an XUV photodiode inside the cryostat.

Demonstrates:
  1. Single current reading (dark)
  2. n-reading average
  3. Buffer mode (n readings in one trigger burst)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from keithley6485 import K6485Driver

drv = K6485Driver(mode="simulation")

with drv:
    drv.reset()
    drv.zero_check_off()
    drv.set_range("AUTO")
    drv.set_nplc(1.0)

    # --- 1. Single reading (dark) ---
    print("1. Single current reading (dark)")
    current = drv.read_single()
    print(f"   I={current:.4e} A")

    # --- 2. Illuminated readings ---
    print("\n2. Illuminated readings (10 pts, 0 delay)")
    drv.sim_set_illuminated(True)
    currents, timestamps = drv.read_n(10, delay_s=0.0)
    print(f"   mean = {currents.mean():.4e} A  (n=10)")
    print(f"   std  = {currents.std():.2e} A")

    # --- 3. Buffer mode ---
    print("\n3. Buffer mode (50 pts)")
    currents_buf, _ = drv.read_buffer(50)
    print(f"   {len(currents_buf)} readings acquired")
    print(f"   mean = {currents_buf.mean():.4e} A")

    drv.sim_set_illuminated(False)

print("\nDone.")
