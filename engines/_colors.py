"""ANSI color helpers for CLI output."""

import sys

GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
RESET = "\033[0m"


def _supports_color():
    return hasattr(sys.stderr, "isatty") and sys.stderr.isatty()


if not _supports_color():
    GREEN = YELLOW = RED = RESET = ""


def green(text):
    return f"{GREEN}{text}{RESET}"


def yellow(text):
    return f"{YELLOW}{text}{RESET}"


def red(text):
    return f"{RED}{text}{RESET}"
