#!/usr/bin/env python3
"""Standalone column detection test — no API calls needed.

Loads a PDF or image, runs the column splitting algorithm, and saves
debug output (annotated page, column crops, detection info).

Usage:
    python3 test_columns.py input/RB_1957-06-03_s5u_....pdf
    python3 test_columns.py input/some_image.png
"""

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "output"


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <pdf_or_image>")
        sys.exit(1)

    file_path = Path(sys.argv[1])
    if not file_path.exists():
        print(f"Error: file not found: {file_path}")
        sys.exit(1)

    ext = file_path.suffix.lower()
    stem = file_path.stem

    # Load image
    if ext == ".pdf":
        from pdf2image import convert_from_path
        print(f"Converting PDF to image (300 DPI)...")
        images = convert_from_path(str(file_path), dpi=300)
        page_image = images[0]
    else:
        from PIL import Image
        page_image = Image.open(file_path)

    print(f"Image size: {page_image.size[0]} x {page_image.size[1]}")

    # Run column detection
    from engines._columns import _split_columns

    debug_dir = OUTPUT_DIR / f"{stem}_debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    header, columns = _split_columns(page_image, debug_dir=debug_dir)

    print(f"Columns detected: {len(columns)}")
    for i, col in enumerate(columns, 1):
        print(f"  Column {i}: {col.size[0]} x {col.size[1]}")

    print(f"\nDebug output saved to: {debug_dir}/")
    print(f"  page_annotated.png  — full page with boundary lines")
    print(f"  column_N_crop.png   — individual column crops")
    print(f"  detection_info.txt  — boundary positions and widths")


if __name__ == "__main__":
    main()
