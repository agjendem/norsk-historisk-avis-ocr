#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ ! -f "$SCRIPT_DIR/.venv/bin/python" ]; then
    echo "Virtual environment not found. Running setup..."
    python3 "$SCRIPT_DIR/setup.py"
fi

exec "$SCRIPT_DIR/.venv/bin/python" "$SCRIPT_DIR/ocr.py" "$@"
