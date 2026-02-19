#!/usr/bin/env bash
set -euo pipefail

# OCR pipeline: PDF -> PNG -> Tesseract -> text file
#
# Usage:
#   ./tesseract/ocr.sh                      # process all PDFs in input/
#   ./tesseract/ocr.sh input/somefile.pdf   # process a single PDF

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

INPUT_DIR="$PROJECT_DIR/input"
OUTPUT_DIR="$PROJECT_DIR/output"
DPI=300
LANG="nor"

mkdir -p "$OUTPUT_DIR"

process_pdf() {
    local pdf="$1"
    local basename
    basename="$(basename "$pdf" .pdf)"

    local png="$OUTPUT_DIR/${basename}.png"
    local txt="$OUTPUT_DIR/${basename}.txt"

    echo "Processing: $pdf"

    # Step 1: Convert PDF to PNG at $DPI resolution
    pdftoppm -png -r "$DPI" -singlefile "$pdf" "$OUTPUT_DIR/${basename}"

    # Step 2: Run Tesseract OCR with Norwegian language
    tesseract "$png" "$OUTPUT_DIR/${basename}" -l "$LANG" 2>/dev/null

    # Step 3: Clean up intermediate PNG
    rm -f "$png"

    echo "  -> $txt"
}

if [[ $# -gt 0 ]]; then
    for pdf in "$@"; do
        # Resolve relative paths against project dir
        if [[ "$pdf" != /* ]]; then
            pdf="$PROJECT_DIR/$pdf"
        fi
        process_pdf "$pdf"
    done
else
    for pdf in "$INPUT_DIR"/*.pdf; do
        [[ -e "$pdf" ]] || { echo "No PDFs found in $INPUT_DIR/"; exit 1; }
        process_pdf "$pdf"
    done
fi

echo "Done."
