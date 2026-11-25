"""HDF5 files for Campbell runs.

One file per run, ``<out_dir>/<name>_<YYYYmmdd_HHMMSS>.h5``, holding one group
named like the file (so files can later be merged without collisions):

    /<run_id>/i_a       float64   readings (A), raw, overflow readings as +9.9E37
    /<run_id>/t_s       float64   instrument time stamps (s), each burst from 0
    /<run_id>/status    int32     status word per reading
    /<run_id>/burst     int32     burst index per reading
    /<run_id>/bursts/*            per-burst host times, gaps, instrument time of the last reading
    attrs on /<run_id>: set_* (meter settings read back), found_* (as found before the run),
                        req_* (what was asked for), stat_* (campbell_stats), note_* (operator),
                        sim_* (simulation parameters, simulated runs only), and JSON copies.

The file is written to a temporary name in the same directory and renamed, so
a crash never leaves a half-written file under the final name.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import re
from pathlib import Path
from typing import Optional

import numpy as np

FORMAT = "k6485_campbell"
FORMAT_VERSION = 1
# K6485_CAMPBELL_DATA moves it (the ETS DAQ launcher points it at the repo's data/campbell_k6485)
DEFAULT_DIR = Path(os.environ.get("K6485_CAMPBELL_DATA", os.path.join("data", "campbell_k6485")))


def run_id(name: str, when: Optional[_dt.datetime] = None) -> str:
    when = when or _dt.datetime.now()
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip()) or "run"
    return f"{safe}_{when:%Y%m%d_%H%M%S}"


def default_path(name: str, out_dir=DEFAULT_DIR, when: Optional[_dt.datetime] = None) -> Path:
    return Path(out_dir) / f"{run_id(name, when)}.h5"


def _jsonable(v):
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, np.ndarray):
        return [_jsonable(x) for x in v.tolist()]
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return f if np.isfinite(f) else None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def _set_attrs(g, prefix: str, d: Optional[dict]) -> None:
    for k, v in (d or {}).items():
        if k.startswith("_") or v is None:
            continue
        key = f"{prefix}{k}"
        if isinstance(v, (list, tuple, np.ndarray)):
            arr = np.asarray(v)
            if arr.dtype.kind in "fiub" or arr.size == 0:
                g.attrs[key] = arr.astype(float) if arr.size else np.array([], float)
            else:
                g.attrs[key] = json.dumps(_jsonable(v))
        elif isinstance(v, dict):
            g.attrs[key] = json.dumps(_jsonable(v))
        elif isinstance(v, (bool, np.bool_)):
            g.attrs[key] = bool(v)
        else:
            g.attrs[key] = v


def write_run(path, *, i_a, t_s, status, burst, bursts: Optional[dict] = None,
              settings: Optional[dict] = None, found: Optional[dict] = None,
              request: Optional[dict] = None, stats: Optional[dict] = None,
              notes: Optional[dict] = None, sim: Optional[dict] = None,
              extra: Optional[dict] = None) -> Path:
    """Write one run. Returns the final path."""
    import h5py

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    rid = path.stem
    try:
        with h5py.File(tmp, "w") as f:
            f.attrs["format"] = FORMAT
            f.attrs["format_version"] = FORMAT_VERSION
            f.attrs["created"] = _dt.datetime.now().astimezone().isoformat(timespec="seconds")
            from . import __version__
            f.attrs["program"] = f"keithley6485.campbell {__version__}"
            g = f.create_group(rid)
            g.create_dataset("i_a", data=np.asarray(i_a, np.float64))
            g.create_dataset("t_s", data=np.asarray(t_s, np.float64))
            g.create_dataset("status", data=np.asarray(status, np.int32))
            g.create_dataset("burst", data=np.asarray(burst, np.int32))
            g["i_a"].attrs["units"] = "A"
            g["t_s"].attrs["units"] = "s"
            g["t_s"].attrs["meaning"] = "instrument time stamp, TRAC:TST:FORM ABS: 0 at each burst's first reading"
            g["status"].attrs["bits"] = ("0 overflow, 1 averaging filter, 2 math, 3 null, 4 limits, "
                                         "5-6 limit results, 7 overvoltage, 9 zero check, 10 zero correct")
            if bursts:
                bg = g.create_group("bursts")
                for k, v in bursts.items():
                    bg.create_dataset(k, data=np.asarray(v))
            _set_attrs(g, "set_", settings)
            _set_attrs(g, "found_", found)
            _set_attrs(g, "req_", request)
            _set_attrs(g, "stat_", stats)
            _set_attrs(g, "note_", notes)
            _set_attrs(g, "sim_", sim)
            _set_attrs(g, "", extra)
            for key, d in (("settings_json", settings), ("found_json", found), ("stats_json", stats),
                           ("notes_json", notes), ("request_json", request), ("sim_json", sim)):
                if d is not None:
                    g.attrs[key] = json.dumps(_jsonable(d))
        os.replace(tmp, path)
    except BaseException:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        raise
    return path


def read_run(path) -> dict:
    """Read a file back: arrays, burst table and decoded attributes."""
    import h5py

    with h5py.File(path, "r") as f:
        groups = [k for k in f.keys() if isinstance(f[k], h5py.Group)]
        if len(groups) != 1:
            raise ValueError(f"{path}: expected one run group, found {groups}")
        g = f[groups[0]]
        out = {"run_id": groups[0],
               "i_a": g["i_a"][()], "t_s": g["t_s"][()], "status": g["status"][()],
               "burst": g["burst"][()],
               "bursts": {k: g["bursts"][k][()] for k in g["bursts"]} if "bursts" in g else {},
               "file_attrs": {k: _decode(v) for k, v in f.attrs.items()},
               "attrs": {k: _decode(v) for k, v in g.attrs.items()}}
    for key in ("settings", "found", "stats", "notes", "request", "sim"):
        js = out["attrs"].get(f"{key}_json")
        out[key] = json.loads(js) if js else {}
    return out


def _decode(v):
    if isinstance(v, bytes):
        return v.decode()
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    return v
