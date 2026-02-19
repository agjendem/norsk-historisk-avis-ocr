#!/usr/bin/env bash
set -euo pipefail

# OCR engine: processes a single PDF or image file via Tesseract
#
# Usage:
#   ./tesseract/ocr.sh <file>
#
# Supported formats: PDF, PNG, JPG, JPEG, TIFF, TIF

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

OUTPUT_DIR="$PROJECT_DIR/output"
DPI=300
LANG="nor"

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <file>" >&2
    echo "Supported formats: pdf, png, jpg, jpeg, tiff, tif" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

process_file() {
    local file="$1"
    local basename
    basename="$(basename "$file")"
    local stem="${basename%.*}"
    local ext="${basename##*.}"
    ext="${ext,,}" # lowercase

    local txt="$OUTPUT_DIR/${stem}.txt"

    echo "Processing: $file"

    case "$ext" in
        pdf)
            local png="$OUTPUT_DIR/${stem}.png"
            # Convert PDF to PNG at $DPI resolution
            pdftoppm -png -r "$DPI" -singlefile "$file" "$OUTPUT_DIR/${stem}"
            # Run Tesseract OCR with Norwegian language
            tesseract "$png" "$OUTPUT_DIR/${stem}" -l "$LANG" 2>/dev/null
            # Clean up intermediate PNG
            rm -f "$png"
            ;;
        png|jpg|jpeg|tiff|tif)
            # Tesseract natively supports these image formats
            tesseract "$file" "$OUTPUT_DIR/${stem}" -l "$LANG" 2>/dev/null
            ;;
        *)
            echo "Error: Unsupported file format '.$ext'" >&2
            exit 1
            ;;
    esac

    echo "  -> $txt"
}

FILE="$1"
# Resolve relative paths against project dir
if [[ "$FILE" != /* ]]; then
    FILE="$PROJECT_DIR/$FILE"
fi

process_file "$FILE"
