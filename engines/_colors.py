"""ANSI color helpers and spinner for CLI output."""

import sys
import threading
import time

GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
RESET = "\033[0m"
DIM = "\033[2m"


def _supports_color():
    return hasattr(sys.stderr, "isatty") and sys.stderr.isatty()


if not _supports_color():
    GREEN = YELLOW = RED = RESET = DIM = ""


def green(text):
    return f"{GREEN}{text}{RESET}"


def yellow(text):
    return f"{YELLOW}{text}{RESET}"


def red(text):
    return f"{RED}{text}{RESET}"


class Spinner:
    """Animated spinner with elapsed time, shown on a single line."""

    _FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self, label=""):
        self.label = label
        self._stop = threading.Event()
        self._thread = None
        self._extra = ""
        self._start_time = 0
        self.elapsed = 0

    def _run(self):
        i = 0
        while not self._stop.is_set():
            elapsed = time.monotonic() - self._start_time
            frame = self._FRAMES[i % len(self._FRAMES)]
            extra = f"  {DIM}{self._extra}{RESET}" if self._extra else ""
            line = f"\r  {YELLOW}{frame} {self.label} [{elapsed:.0f}s]{RESET}{extra}"
            sys.stderr.write(line)
            sys.stderr.flush()
            i += 1
            self._stop.wait(0.1)
        # Clear the spinner line
        sys.stderr.write("\r\033[K")
        sys.stderr.flush()

    def update(self, text):
        """Update the extra text shown after the spinner."""
        self._extra = text

    def __enter__(self):
        self._start_time = time.monotonic()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *args):
        self._stop.set()
        self._thread.join()
        self.elapsed = time.monotonic() - self._start_time
