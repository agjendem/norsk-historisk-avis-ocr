#!/usr/bin/env python3
"""
OCR engine using Claude's vision API.

Processes a single PDF or image file: converts PDFs to PNGs first,
sends images directly to Claude for transcription, and writes the
result to a text file.

Usage (via root CLI):
    ./ocr claude-vision                         # interactive file picker
    ./ocr claude-vision --dpi 400               # custom DPI for PDF conversion

Requires:
    - ANTHROPIC_API_KEY set in environment or .env file
    - poppler (pdftoppm) installed (for PDF input only)
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

MEDIA_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
}

UNSUPPORTED_IMAGE_TYPES = {".tiff", ".tif"}


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


def transcribe_image(
    client: anthropic.Anthropic, image_path: Path, model: str, media_type: str
) -> str:
    """Send an image to Claude's vision API and get the transcription."""
    image_data = base64.standard_b64encode(image_path.read_bytes()).decode("utf-8")

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
                            "media_type": media_type,
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


def process_file(
    client: anthropic.Anthropic, file_path: Path, dpi: int, model: str
) -> None:
    """Process a single file: PDF or image -> Claude vision -> text file."""
    stem = file_path.stem
    ext = file_path.suffix.lower()
    txt_path = OUTPUT_DIR / f"{stem}.vision.txt"

    print(f"Processing: {file_path}")

    if ext == ".pdf":
        print(f"  Converting to PNG (DPI={dpi})...")
        png_path = pdf_to_png(file_path, dpi)
        print(f"  Sending to Claude ({model})...")
        text = transcribe_image(client, png_path, model, "image/png")
        png_path.unlink(missing_ok=True)
    elif ext in MEDIA_TYPES:
        media_type = MEDIA_TYPES[ext]
        print(f"  Sending to Claude ({model})...")
        text = transcribe_image(client, file_path, model, media_type)
    elif ext in UNSUPPORTED_IMAGE_TYPES:
        print(f"Error: TIFF format is not supported by Claude API. Use tesseract engine instead.", file=sys.stderr)
        sys.exit(1)
    else:
        print(f"Error: Unsupported file format '{ext}'", file=sys.stderr)
        sys.exit(1)

    txt_path.write_text(text + "\n", encoding="utf-8")
    print(f"  -> {txt_path}")


def main():
    parser = argparse.ArgumentParser(description="OCR a file using Claude vision API")
    parser.add_argument("file", help="File to process (PDF, PNG, JPG, JPEG)")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for PDF to PNG conversion")
    parser.add_argument(
        "--model",
        default="claude-sonnet-4-20250514",
        help="Claude model to use (default: claude-sonnet-4-20250514)",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)

    client = anthropic.Anthropic()
    file_path = Path(args.file).resolve()

    process_file(client, file_path, args.dpi, args.model)


if __name__ == "__main__":
    main()
