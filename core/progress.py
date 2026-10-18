"""Prints a simple progress line with the time remaining."""
import os
import sys
import time
from collections import deque

_BAR_WIDTH = 24


def _fmt_hms(seconds):
    """'3m12s', '1h04m', '45s' — compact and unambiguous."""
    if seconds is None or seconds != seconds or seconds < 0:
        return "--"
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"


def _is_tty():
    if os.environ.get("PROGRESS_PLAIN") == "1":
        return False
    try:
        return sys.stdout.isatty()
    except Exception:
        return False


class Progress:

    def __init__(self, total=None, label="run", unit="cell", interval=None,
                 window=32, stream=None):
        self.total = int(total) if total else 0
        self.label = label
        self.unit = unit
        self.interval = float(interval if interval is not None
                              else os.environ.get("PROGRESS_INTERVAL", 30.0))
        self.n = 0
        self.t0 = time.time()
        self._marks = deque(maxlen=max(2, window))   # (time, n) for the windowed rate
        self._marks.append((self.t0, 0))
        self._last_emit = 0.0
        self._tty = _is_tty()
        self._stream = stream or sys.stdout
        self._closed = False
        head = f"[{self.label}] starting"
        if self.total:
            head += f" — {self.total} {self.unit}s"
        print(head, flush=True, file=self._stream)

    # ------------------------------------------------------------------ rates
    def _rate(self):
        """Units per second over the trailing window."""
        if len(self._marks) < 2:
            return 0.0
        (t_old, n_old), (t_new, n_new) = self._marks[0], self._marks[-1]
        dt = t_new - t_old
        return (n_new - n_old) / dt if dt > 0 else 0.0

    def _eta(self):
        r = self._rate()
        if not self.total or r <= 0:
            return None
        return max(0.0, (self.total - self.n) / r)

    # ----------------------------------------------------------------- render
    def _line(self, extra=""):
        el = time.time() - self.t0
        r = self._rate()
        parts = []
        if self.total:
            frac = min(1.0, self.n / self.total)
            filled = int(round(frac * _BAR_WIDTH))
            parts.append(f"[{'#' * filled}{'.' * (_BAR_WIDTH - filled)}]")
            parts.append(f"{frac * 100:5.1f}%")
            parts.append(f"{self.n}/{self.total}")
        else:
            parts.append(f"{self.n} {self.unit}s")
        parts.append(f"{r:.2f}/s" if r >= 0.05 else f"{(1 / r) if r > 0 else float('inf'):.0f}s/{self.unit}")
        parts.append(f"elapsed {_fmt_hms(el)}")
        if self.total:
            parts.append(f"eta {_fmt_hms(self._eta())}")
        line = f"[{self.label}] " + "  ".join(parts)
        if extra:
            line += f"  | {extra}"
        return line

    def _emit(self, extra="", force=False):
        now = time.time()
        if self._tty:
            # redraw in place; pad so a shorter line does not leave debris
            sys.stdout.write("\r" + self._line(extra).ljust(140)[:140])
            sys.stdout.flush()
            self._last_emit = now
            return
        if force or (now - self._last_emit) >= self.interval:
            print(self._line(extra), flush=True, file=self._stream)
            self._last_emit = now

    # ------------------------------------------------------------------- api
    def update(self, n=1, extra=""):
        """Record `n` completed units."""
        self.n += n
        self._marks.append((time.time(), self.n))
        self._emit(extra)
        return self

    # alias, so either name reads naturally at a call site
    tick = update

    def close(self, extra=""):
        """Final summary line. Idempotent."""
        if self._closed:
            return
        self._closed = True
        if self._tty:
            sys.stdout.write("\r" + " " * 140 + "\r")
            sys.stdout.flush()
        el = time.time() - self.t0
        avg = self.n / el if el > 0 else 0.0
        msg = (f"[{self.label}] DONE {self.n}"
               + (f"/{self.total}" if self.total else "")
               + f" {self.unit}s in {_fmt_hms(el)}  ({avg:.2f}/s"
               + (f", {el / self.n:.1f}s per {self.unit}" if self.n else "") + ")")
        if extra:
            msg += f"  | {extra}"
        print(msg, flush=True, file=self._stream)

    # context-manager form
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def estimate(total, seconds_per_unit, label="estimate"):
    """Print a projected wall-clock cost before a run starts, so a six-hour job
    announces itself rather than being discovered."""
    secs = total * seconds_per_unit
    print(f"[{label}] {total} units x ~{seconds_per_unit:.1f}s = ~{_fmt_hms(secs)} "
          f"projected", flush=True)
    return secs
