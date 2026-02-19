#!/usr/bin/env python3
"""
OCR pipeline using Claude's vision API.

Converts PDFs to PNGs, sends them to Claude for transcription,
and writes the result to text files.

Usage (via root CLI):
    ./ocr claude-vision                         # process all PDFs in input/
    ./ocr claude-vision input/somefile.pdf       # process specific files
    ./ocr claude-vision --dpi 400 input/*.pdf    # custom DPI

Requires:
    - ANTHROPIC_API_KEY set in environment or .env file
    - poppler (pdftoppm) installed
"""

import argparse
import base64
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_DIR / ".env")

import anthropic

INPUT_DIR = PROJECT_DIR / "input"
OUTPUT_DIR = PROJECT_DIR / "output"

SYSTEM_PROMPT = """\
You are an expert OCR transcription assistant specializing in historical \
Norwegian newspaper scans. Your task is to produce an accurate, clean \
transcription of the text in the provided image.

Rules:
- Read columns left to right, top to bottom within each column.
- Join hyphenated words that are split across line breaks.
- Output flowing paragraph text, not line-by-line reproduction.
- Preserve paragraph breaks where they appear in the original.
- Reproduce poems/verses with their original line breaks.
- Use «» for quotes as in the original.
- Mark section headings on their own lines.
- If a word is truly illegible, write [?] after your best guess.
- Do NOT add commentary, headers, or metadata — output only the transcribed text.\
"""

USER_PROMPT = """\
Transcribe the full text of this newspaper page. \
Read the columns in order (left to right). \
Join hyphenated line-break words into whole words. \
Output clean flowing text with paragraph breaks preserved.\
"""


def pdf_to_png(pdf_path: Path, dpi: int) -> Path:
    """Convert a PDF to a PNG image using pdftoppm."""
    stem = pdf_path.stem
    out_prefix = OUTPUT_DIR / stem
    subprocess.run(
        ["pdftoppm", "-png", "-r", str(dpi), "-singlefile", str(pdf_path), str(out_prefix)],
        check=True,
        capture_output=True,
    )
    png_path = OUTPUT_DIR / f"{stem}.png"
    if not png_path.exists():
        raise FileNotFoundError(f"Expected PNG not found: {png_path}")
    return png_path


def transcribe_image(client: anthropic.Anthropic, png_path: Path, model: str) -> str:
    """Send a PNG to Claude's vision API and get the transcription."""
    image_data = base64.standard_b64encode(png_path.read_bytes()).decode("utf-8")

    message = client.messages.create(
        model=model,
        max_tokens=8192,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_data,
                        },
                    },
                    {
                        "type": "text",
                        "text": USER_PROMPT,
                    },
                ],
            }
        ],
    )

    return message.content[0].text


def process_pdf(client: anthropic.Anthropic, pdf_path: Path, dpi: int, model: str) -> None:
    """Full pipeline: PDF -> PNG -> Claude vision -> text file."""
    stem = pdf_path.stem
    txt_path = OUTPUT_DIR / f"{stem}.vision.txt"

    print(f"Processing: {pdf_path}")

    # Step 1: PDF to PNG
    print(f"  Converting to PNG (DPI={dpi})...")
    png_path = pdf_to_png(pdf_path, dpi)

    # Step 2: Send to Claude vision API
    print(f"  Sending to Claude ({model})...")
    text = transcribe_image(client, png_path, model)

    # Step 3: Write output
    txt_path.write_text(text + "\n", encoding="utf-8")
    print(f"  -> {txt_path}")

    # Step 4: Clean up intermediate PNG
    png_path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description="OCR newspaper PDFs using Claude vision API")
    parser.add_argument("files", nargs="*", help="PDF files to process (default: all in input/)")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for PDF to PNG conversion")
    parser.add_argument(
        "--model",
        default="claude-sonnet-4-20250514",
        help="Claude model to use (default: claude-sonnet-4-20250514)",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)

    client = anthropic.Anthropic()

    if args.files:
        pdf_files = [Path(f).resolve() for f in args.files]
    else:
        pdf_files = sorted(INPUT_DIR.glob("*.pdf"))
        if not pdf_files:
            print(f"No PDFs found in {INPUT_DIR}/")
            sys.exit(1)

    for pdf_path in pdf_files:
        process_pdf(client, pdf_path, args.dpi, args.model)

    print(f"\nDone. Processed {len(pdf_files)} file(s).")


if __name__ == "__main__":
    main()
