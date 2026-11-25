"""Line transports for the 6485: a serial port (pyserial) and the simulator.

Both expose the same three calls, so the meter code runs unchanged on the
bench and in simulation:

- ``write(cmd)`` sends one command line;
- ``readline(timeout, abort=None, on_bytes=None)`` returns one response line
  without its terminator, raising ``TransportTimeout`` or ``Aborted``;
- ``close()``.
"""

from __future__ import annotations

import time
from typing import Callable, Optional

TERMINATORS = {"CR": "\r", "LF": "\n", "CRLF": "\r\n", "LFCR": "\n\r"}


class TransportTimeout(TimeoutError):
    """No complete response line within the timeout."""


class Aborted(Exception):
    """The abort callback asked to stop while waiting for a response."""


def decode_terminator(t: str) -> str:
    """Accept "CR", "LF", "CRLF", "LFCR" or an escaped string like "\\r"."""
    if t.upper() in TERMINATORS:
        return TERMINATORS[t.upper()]
    return t.encode("ascii").decode("unicode_escape")


class SerialTransport:
    """RS-232 line transport.

    The 6485 processes a command after the <CR> it receives (6485-901-01,
    "Flow control", p. 9-17) and ends each response with the terminator set in
    its RS232 menu. It has no hardware handshaking; XON/XOFF is optional on the
    meter and must match ``xonxoff`` here.
    """

    def __init__(self, port: str, baud: int = 9600, read_term: str = "\r",
                 write_term: str = "\r", xonxoff: bool = False,
                 bytesize: int = 8, parity: str = "N", stopbits: int = 1):
        import serial  # pyserial

        self.port = port
        self.baud = int(baud)
        self.read_term = read_term.encode("ascii")
        self.write_term = write_term.encode("ascii")
        self._ser = serial.Serial(port=port, baudrate=self.baud, bytesize=bytesize,
                                  parity=parity, stopbits=stopbits, xonxoff=xonxoff,
                                  rtscts=False, dsrdtr=False, timeout=0.05,
                                  write_timeout=10.0, exclusive=True)
        self._buf = bytearray()
        self._ser.reset_input_buffer()

    def write(self, cmd: str) -> None:
        self._ser.write(cmd.encode("ascii") + self.write_term)
        self._ser.flush()

    def readline(self, timeout: float, abort: Optional[Callable[[], bool]] = None,
                 on_bytes: Optional[Callable[[int], None]] = None) -> str:
        deadline = time.monotonic() + timeout
        last_report = 0.0
        while True:
            i = self._buf.find(self.read_term)
            if i >= 0:
                line = bytes(self._buf[:i])
                del self._buf[:i + len(self.read_term)]
                return line.decode("ascii", errors="replace").strip()
            if abort is not None and abort():
                raise Aborted()
            if time.monotonic() > deadline:
                raise TransportTimeout(f"no response line within {timeout:.1f} s "
                                       f"({len(self._buf)} bytes pending)")
            chunk = self._ser.read(max(1, min(self._ser.in_waiting, 65536)))
            if chunk:
                self._buf.extend(chunk)
                now = time.monotonic()
                if on_bytes is not None and now - last_report > 0.25:
                    on_bytes(len(self._buf))
                    last_report = now

    def modem_lines(self) -> dict:
        """CTS, DSR, CD as the adapter reports them (empty if it cannot)."""
        try:
            return {"CTS": bool(self._ser.cts), "DSR": bool(self._ser.dsr), "CD": bool(self._ser.cd)}
        except Exception:
            return {}

    def flush_input(self) -> None:
        self._buf.clear()
        self._ser.reset_input_buffer()

    def close(self) -> None:
        try:
            self._ser.close()
        except Exception:
            pass

    def describe(self) -> str:
        return f"{self.port} @ {self.baud} baud"


class SimTransport:
    """Feeds commands to a ``SimulatedK6485`` and reads its replies.

    The simulator schedules each reply on its own clock; with ``time_scale``
    > 0 the transport waits that long in real time, so the GUI sees the same
    phases (acquiring, transferring) as on the bench.
    """

    def __init__(self, sim, baud: int = 9600):
        self.sim = sim
        self.baud = int(baud)
        sim.baud = self.baud

    @property
    def time_scale(self) -> float:
        return self.sim.time_scale

    def write(self, cmd: str) -> None:
        self.sim.receive(cmd)

    def readline(self, timeout: float, abort: Optional[Callable[[], bool]] = None,
                 on_bytes: Optional[Callable[[int], None]] = None) -> str:
        deadline = time.monotonic() + timeout
        last_report = 0.0
        while True:
            line, partial = self.sim.poll_output()
            if line is not None:
                return line
            if abort is not None and abort():
                raise Aborted()
            if time.monotonic() > deadline:
                raise TransportTimeout(f"no response line within {timeout:.1f} s")
            now = time.monotonic()
            if on_bytes is not None and partial and now - last_report > 0.25:
                on_bytes(partial)
                last_report = now
            time.sleep(min(0.02, max(0.0, self.sim.next_event_wait())))

    def flush_input(self) -> None:
        self.sim.clear_output()

    def close(self) -> None:
        pass

    def describe(self) -> str:
        return f"simulation @ {self.baud} baud"
