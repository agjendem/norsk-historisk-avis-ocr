"""Claude Vision OCR engine — cross-platform wrapper using anthropic SDK and pdf2image."""

import base64
import io
import os
import platform
import shutil
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
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
- Use \u00ab\u00bb for quotes as in the original.
- Mark section headings on their own lines.
- If a word is truly illegible, write [?] after your best guess.
- Do NOT add commentary, headers, or metadata \u2014 output only the transcribed text.\
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

# Auto-detect poppler on Windows
_poppler_path = None
if platform.system() == "Windows":
    _win_poppler = PROJECT_DIR / "vendor" / "poppler" / "Library" / "bin"
    if _win_poppler.exists():
        _poppler_path = str(_win_poppler)


def _load_dotenv():
    from dotenv import load_dotenv
    env_file = PROJECT_DIR / ".env"
    if env_file.exists():
        load_dotenv(env_file)


def ensure_api_key():
    """Check env, load .env, or prompt interactively for the Anthropic API key."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return

    _load_dotenv()
    if os.environ.get("ANTHROPIC_API_KEY"):
        return

    print("No ANTHROPIC_API_KEY found in environment or .env file.")
    key = input("Enter your Anthropic API key: ").strip()
    if not key:
        print("Error: API key is required for claude-vision.", file=sys.stderr)
        sys.exit(1)

    env_file = PROJECT_DIR / ".env"
    env_file.write_text(f"ANTHROPIC_API_KEY={key}\n", encoding="utf-8")
    os.environ["ANTHROPIC_API_KEY"] = key
    print("Saved to .env")


class ClaudeVisionEngine:
    name = "claude-vision"
    output_suffix = ".vision.txt"

    def __init__(self, dpi=300, model="claude-sonnet-4-20250514"):
        self.dpi = dpi
        self.model = model

    def check_dependencies(self):
        """Return list of missing dependencies (hard blockers only)."""
        missing = []
        if not shutil.which("pdftoppm") and _poppler_path is None:
            missing.append("poppler (provides pdftoppm for PDF conversion)")
        return missing

    def process_file(self, file_path):
        """Process a single file and write OCR output to output/."""
        import anthropic
        from pdf2image import convert_from_path

        ensure_api_key()

        file_path = Path(file_path)
        ext = file_path.suffix.lower()
        stem = file_path.stem
        txt_path = OUTPUT_DIR / f"{stem}{self.output_suffix}"

        OUTPUT_DIR.mkdir(exist_ok=True)
        print(f"Processing: {file_path}")

        if ext in UNSUPPORTED_IMAGE_TYPES:
            print(
                "Error: TIFF format is not supported by Claude API. "
                "Use tesseract engine instead.",
                file=sys.stderr,
            )
            return

        if ext == ".pdf":
            print(f"  Converting PDF to image (DPI={self.dpi})...")
            images = convert_from_path(
                str(file_path),
                dpi=self.dpi,
                first_page=1,
                last_page=1,
                poppler_path=_poppler_path,
            )
            buf = io.BytesIO()
            images[0].save(buf, format="PNG")
            image_data = base64.standard_b64encode(buf.getvalue()).decode("utf-8")
            media_type = "image/png"
        elif ext in MEDIA_TYPES:
            media_type = MEDIA_TYPES[ext]
            image_data = base64.standard_b64encode(
                file_path.read_bytes()
            ).decode("utf-8")
        else:
            print(f"Error: Unsupported file format '{ext}'", file=sys.stderr)
            return

        print(f"  Sending to Claude ({self.model})...")
        client = anthropic.Anthropic()
        message = client.messages.create(
            model=self.model,
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

        text = message.content[0].text
        txt_path.write_text(text + "\n", encoding="utf-8")
        print(f"  -> {txt_path}")
