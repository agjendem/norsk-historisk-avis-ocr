"""Claude Vision OCR engine — cross-platform wrapper using anthropic SDK and pdf2image."""

import base64
import getpass
import io
import os
import platform
import shutil
import sys
from pathlib import Path

from engines._colors import green, yellow, red

try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "output"

SYSTEM_PROMPT = """\
You are an expert OCR transcription assistant specializing in historical \
Norwegian newspaper scans. Your task is to produce an accurate, clean \
transcription of the text in the provided image.

Rules:
- Identify each column visually before reading. Process one column at a \
time, left to right. Read each column top to bottom before moving to the next.
- Do NOT mix text from adjacent columns. If a line seems to jump topics \
mid-sentence, you are likely reading across a column boundary.
- Join hyphenated words that are split across line breaks within a column.
- Output flowing paragraph text, not line-by-line reproduction.
- Preserve paragraph breaks where they appear in the original.
- Reproduce poems/verses with their original line breaks.
- Use \u00ab\u00bb for quotes as in the original.
- Mark section headings on their own lines.
- If a word is truly illegible, write [?] after your best guess.
- Do NOT add commentary, headers, or metadata \u2014 output only the transcribed text.
- NEVER summarize, skip, or abbreviate content. Transcribe every word on the \
page. If you run out of space, stop mid-sentence rather than adding a summary \
like \u00abcontent continues...\u00bb or similar.\
"""

USER_PROMPT = """\
Transcribe the COMPLETE text of this newspaper page. \
First identify the column layout, then read each column fully (left to right). \
Join hyphenated line-break words into whole words. \
Output clean flowing text with paragraph breaks preserved. \
Do not skip or summarize any content.\
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


def _has_anthropic_api_key():
    """Return True if ANTHROPIC_API_KEY is available in env or .env file."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return True
    _load_dotenv()
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def _has_aws_credentials():
    """Return True if AWS credentials are available (profile, env vars, or default session)."""
    if os.environ.get("AWS_PROFILE") or os.environ.get("AWS_ACCESS_KEY_ID"):
        return True
    try:
        import boto3
        session = boto3.Session()
        credentials = session.get_credentials()
        return credentials is not None
    except Exception:
        return False


def _prompt_api_key():
    """Prompt for Anthropic API key and save to .env."""
    print(yellow("No ANTHROPIC_API_KEY found in environment or .env file."))
    print(yellow("No AWS credentials detected for Bedrock."))
    key = getpass.getpass("Enter your Anthropic API key: ").strip()
    if not key:
        print(red("Error: API key is required for claude-vision."), file=sys.stderr)
        sys.exit(1)

    env_file = PROJECT_DIR / ".env"
    env_file.write_text(f"ANTHROPIC_API_KEY={key}\n", encoding="utf-8")
    os.environ["ANTHROPIC_API_KEY"] = key
    print("Saved to .env")


MAX_IMAGE_BYTES = 5 * 1024 * 1024  # 5 MB API limit


def _prepare_image(image):
    """Enhance image for OCR: sharpen and boost contrast."""
    from PIL import ImageEnhance, ImageFilter

    image = image.filter(ImageFilter.SHARPEN)
    image = ImageEnhance.Contrast(image).enhance(1.3)
    return image


def _encode_image_under_limit(image, max_bytes=MAX_IMAGE_BYTES):
    """Encode a PIL Image as JPEG, scaling down if needed to stay under max_bytes.

    Returns (base64_str, media_type).
    """
    image = _prepare_image(image)
    quality = 95
    while True:
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=quality)
        data = buf.getvalue()
        if len(data) <= max_bytes:
            return base64.standard_b64encode(data).decode("utf-8"), "image/jpeg"
        # Try lower quality first
        if quality > 50:
            quality -= 10
            continue
        # Quality alone isn't enough — scale down
        w, h = image.size
        image = image.resize((int(w * 0.8), int(h * 0.8)))
        quality = 95  # reset quality after resize


BEDROCK_MODEL_MAP = {
    "claude-sonnet-4-20250514": "us.anthropic.claude-sonnet-4-20250514-v1:0",
    "claude-opus-4-20250514": "us.anthropic.claude-opus-4-20250514-v1:0",
}


class ClaudeVisionEngine:
    name = "claude-vision"

    def __init__(self, dpi=300, model="claude-opus-4-20250514", region="eu-north-1", max_tokens=16384):
        self.dpi = dpi
        self.model = model
        self.region = region
        self.max_tokens = max_tokens
        self._model_short = model.split("-")[1] if "-" in model else model
        self.output_suffix = f".vision-{dpi}dpi-{self._model_short}.txt"

    def _get_client(self):
        """Create an Anthropic client using the best available auth method.

        Priority:
        1. ANTHROPIC_API_KEY → direct Anthropic API
        2. AWS credentials → Bedrock
        3. Prompt for API key → direct Anthropic API
        """
        import anthropic

        if _has_anthropic_api_key():
            print(yellow("  Auth: using Anthropic API key"))
            return anthropic.Anthropic()

        if _has_aws_credentials():
            print(yellow(f"  Auth: using AWS Bedrock (region={self.region})"))
            return anthropic.AnthropicBedrock(aws_region=self.region)

        _prompt_api_key()
        print(yellow("  Auth: using Anthropic API key"))
        return anthropic.Anthropic()

    def _resolve_model(self, client):
        """Return the model ID appropriate for the client type."""
        import anthropic

        if isinstance(client, anthropic.AnthropicBedrock):
            return BEDROCK_MODEL_MAP.get(self.model, self.model)
        return self.model

    def check_dependencies(self):
        """Return list of missing dependencies (hard blockers only)."""
        missing = []
        if not shutil.which("pdftoppm") and _poppler_path is None:
            missing.append("poppler (provides pdftoppm for PDF conversion)")
        return missing

    def process_file(self, file_path):
        """Process a single file and write OCR output to output/."""
        from pdf2image import convert_from_path

        client = self._get_client()
        model = self._resolve_model(client)

        file_path = Path(file_path)
        ext = file_path.suffix.lower()
        stem = file_path.stem
        txt_path = OUTPUT_DIR / f"{stem}{self.output_suffix}"

        OUTPUT_DIR.mkdir(exist_ok=True)
        print(f"Processing: {file_path}")

        if ext in UNSUPPORTED_IMAGE_TYPES:
            print(red(
                "Error: TIFF format is not supported by Claude API. "
                "Use tesseract engine instead."
            ), file=sys.stderr)
            return

        if ext == ".pdf":
            from PIL import Image

            print(f"  Converting PDF to image (DPI={self.dpi})...")
            images = convert_from_path(
                str(file_path),
                dpi=self.dpi,
                first_page=1,
                last_page=1,
                poppler_path=_poppler_path,
            )
            image_data, media_type = _encode_image_under_limit(images[0])
        elif ext in MEDIA_TYPES:
            from PIL import Image

            raw = file_path.read_bytes()
            if len(raw) <= MAX_IMAGE_BYTES:
                media_type = MEDIA_TYPES[ext]
                image_data = base64.standard_b64encode(raw).decode("utf-8")
            else:
                image_data, media_type = _encode_image_under_limit(
                    Image.open(file_path)
                )
        else:
            print(red(f"Error: Unsupported file format '{ext}'"), file=sys.stderr)
            return

        print(yellow(f"  Sending to Claude ({model})..."))
        try:
            with client.messages.stream(
                model=model,
                max_tokens=self.max_tokens,
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
            ) as stream:
                message = stream.get_final_message()
        except Exception as exc:
            import anthropic

            if isinstance(exc, anthropic.AuthenticationError):
                print(red(
                    "Error: Authentication failed — your API key is invalid or expired."
                ), file=sys.stderr)
                print(red(
                    "  Get a key at: https://platform.claude.com/settings/keys"
                ), file=sys.stderr)
                # Remove bad key from .env so next run re-prompts
                env_file = PROJECT_DIR / ".env"
                if env_file.exists():
                    env_file.unlink()
                    print(red("  (Removed .env — you will be prompted for a new key next run)"))
            elif isinstance(exc, anthropic.PermissionDeniedError):
                print(red(
                    "Error: Permission denied — your credentials lack access to this model."
                ), file=sys.stderr)
                if isinstance(client, anthropic.AnthropicBedrock):
                    print(red(
                        "  Ensure the model is enabled in your AWS Bedrock console for"
                        f" region {self.region}."
                    ), file=sys.stderr)
                else:
                    print(red(
                        "  Check your API key permissions at: https://platform.claude.com/settings/keys"
                    ), file=sys.stderr)
            elif isinstance(exc, anthropic.APIConnectionError):
                print(red(
                    "Error: Could not connect to the API."
                ), file=sys.stderr)
                print(red(
                    "  Check your internet connection and try again."
                ), file=sys.stderr)
                if isinstance(client, anthropic.AnthropicBedrock):
                    print(red(
                        f"  Also verify that Bedrock is available in region {self.region}."
                    ), file=sys.stderr)
            elif isinstance(exc, anthropic.APIStatusError):
                print(red(
                    f"Error: API returned {exc.status_code} — {exc.message}"
                ), file=sys.stderr)
            else:
                raise
            return

        usage = message.usage
        stop = message.stop_reason
        print(yellow(
            f"  Tokens: {usage.input_tokens} in / {usage.output_tokens} out"
            f"  (stop: {stop})"
        ))
        if stop == "max_tokens":
            print(yellow(
                f"  Warning: output was truncated at {self.max_tokens} tokens."
                " Re-run with a higher --max-tokens value to get the full text."
            ))

        text = message.content[0].text
        txt_path.write_text(text + "\n", encoding="utf-8")
        print(green(f"  -> {txt_path}"))
