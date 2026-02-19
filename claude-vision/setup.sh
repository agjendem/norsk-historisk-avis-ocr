#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Install dependencies for Claude Vision OCR pipeline

# poppler: PDF rendering (provides pdftoppm)
brew install poppler

# Python venv + dependencies
python3 -m venv "$PROJECT_DIR/.venv"
source "$PROJECT_DIR/.venv/bin/activate"
pip install -r "$SCRIPT_DIR/requirements.txt"

echo ""
echo "Setup complete. Run: ./ocr claude-vision"
