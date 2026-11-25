"""Checks that the meter's serial port is free before this program opens it.

Two independent checks, neither imports anything from the ETS DAQ:

1. Linux: any other process with the device open (a scan of /proc/*/fd).
2. The ETS DAQ web app, when this machine runs one (its API token file
   exists): a read-only GET of /api/v1/state. The K6485 must be disconnected
   there, with no live claim, call or job holding it.
"""

from __future__ import annotations

import glob
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

DAQ_URL = os.environ.get("DAQ_API_URL", "http://127.0.0.1:8765")
DAQ_TOKEN_FILE = Path(os.environ.get("DAQ_API_TOKEN_FILE", "~/.config/daq/api_token")).expanduser()
DAQ_KEY = "k6485"


def list_ports() -> list[str]:
    """Serial ports, stable /dev/serial/by-id names first."""
    ports = sorted(glob.glob("/dev/serial/by-id/*"))
    try:
        from serial.tools import list_ports as lp
        seen = {os.path.realpath(p) for p in ports}
        for info in sorted(lp.comports(), key=lambda i: i.device):
            if os.path.realpath(info.device) not in seen:
                ports.append(info.device)
    except Exception:
        pass
    return ports


def processes_using(device: str) -> list[tuple[int, str]]:
    """(pid, command line) of other processes holding ``device`` open (Linux only)."""
    if not os.path.isdir("/proc"):
        return []
    real = os.path.realpath(device)
    me = os.getpid()
    users = []
    for fd_dir in glob.glob("/proc/[0-9]*/fd"):
        pid = int(fd_dir.split("/")[2])
        if pid == me:
            continue
        try:
            for fd in os.listdir(fd_dir):
                try:
                    if os.readlink(os.path.join(fd_dir, fd)) == real:
                        with open(f"/proc/{pid}/cmdline", "rb") as fh:
                            cmd = fh.read().replace(b"\0", b" ").decode(errors="replace").strip()
                        users.append((pid, cmd))
                        break
                except OSError:
                    continue
        except OSError:
            continue
    return users


def daq_k6485_problems(timeout_s: float = 3.0) -> tuple[Optional[list[str]], str]:
    """(problems, note). problems is None when no DAQ web app is configured here."""
    token = os.environ.get("DAQ_API_TOKEN")
    if not token and DAQ_TOKEN_FILE.exists():
        token = DAQ_TOKEN_FILE.read_text().strip()
    if not token:
        return None, "no DAQ API token on this machine: DAQ check skipped"
    req = urllib.request.Request(f"{DAQ_URL}/api/v1/state",
                                 headers={"Authorization": f"Bearer {token}",
                                          "X-DAQ-Actor": "k6485-campbell"})
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as r:
            body = json.loads(r.read().decode())
    except (urllib.error.URLError, OSError, ValueError) as e:
        return [], f"DAQ web app not reachable ({e}); relying on the process check"
    st = body.get("result", body)
    problems = []
    status = str((st.get("status") or {}).get(DAQ_KEY, ""))
    if status.upper().startswith("OK"):
        problems.append(f"the DAQ web app has the K6485 connected ({status}): "
                        f"disconnect it there (or 'release all instruments') first")
    for c in st.get("claims") or []:
        if c.get("state") == "live" and DAQ_KEY in (c.get("keys") or []):
            problems.append(f"live DAQ claim {c.get('id')} on the K6485 by {c.get('actor')}: {c.get('purpose')}")
    for c in st.get("calls") or []:
        if DAQ_KEY in (c.get("keys") or []) or str(c.get("what", "")).startswith(DAQ_KEY + "."):
            problems.append(f"DAQ API call {c.get('what')} running on the K6485 ({c.get('actor')})")
    for k, h in (st.get("holders") or {}).items():
        if k == DAQ_KEY:
            problems.append(f"DAQ job holds the K6485: {h}")
    return problems, f"DAQ web app: K6485 {status or 'unknown'}"


def port_problems(device: str, check_daq: bool = True) -> tuple[list[str], list[str]]:
    """(problems, notes) for opening ``device``. An empty problems list means go."""
    problems, notes = [], []
    if device.startswith("/dev/") and not os.path.exists(device):
        problems.append(f"{device} does not exist (adapter unplugged?); present: {list_ports() or 'none'}")
        return problems, notes
    for pid, cmd in processes_using(device):
        problems.append(f"{device} is open in process {pid}: {cmd[:120]}")
    if check_daq:
        p, note = daq_k6485_problems()
        notes.append(note)
        if p:
            problems.extend(p)
    return problems, notes
