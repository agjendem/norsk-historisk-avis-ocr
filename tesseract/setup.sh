#!/usr/bin/env bash
set -euo pipefail

# Install dependencies for Tesseract OCR pipeline

# poppler: PDF rendering (provides pdftoppm)
brew install poppler

# tesseract + language packs: OCR engine with Norwegian support
brew install tesseract tesseract-lang

echo ""
echo "Setup complete. Run: ./ocr tesseract"
