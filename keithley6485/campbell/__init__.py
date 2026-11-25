"""Keithley 6485 Campbell (shot-noise) runs: driver, statistics, HDF5, simulation, CLI and GUI.

Independent of the ETS DAQ: it needs numpy, pyserial and h5py (and NiceGUI for
the GUI). Names are imported lazily, so ``keithley6485.campbell.stats`` loads
without the driver, pyserial or h5py.
"""

__version__ = "0.1.0"

_EXPORTS = {
    "E_CHARGE": "stats", "campbell_stats": "stats", "floor_subtracted_charge": "stats",
    "CampbellMeter": "meter", "MeterError": "meter", "RunData": "meter", "parse_trace": "meter",
    "SimPhysics": "sim", "SimulatedK6485": "sim",
    "read_run": "store", "write_run": "store",
    "SerialTransport": "transport", "SimTransport": "transport",
    "Session": "session", "RunSettings": "session", "Connection": "session",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        import importlib
        return getattr(importlib.import_module(f".{_EXPORTS[name]}", __name__), name)
    raise AttributeError(name)
