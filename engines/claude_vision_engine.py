"""Claude Vision OCR engine — cross-platform wrapper using anthropic SDK and pdf2image."""

import base64
import getpass
import io
import os
import platform
import shutil
import sys
from pathlib import Path

from engines._colors import green, yellow, red, Spinner
from engines._columns import _split_columns

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

COLUMN_USER_PROMPT = """\
Transcribe the text in this single newspaper column. \
Join hyphenated line-break words into whole words. \
Output clean flowing text with paragraph breaks preserved. \
Do not skip or summarize any content.\
"""

HEADER_USER_PROMPT = """\
Transcribe the text in this newspaper header/title area. \
This is the top section of the page containing the article title, subtitle, \
and/or author byline. Output clean text preserving the heading structure. \
Do not skip or summarize any content.\
"""

CORRECTION_SYSTEM_PROMPT = """\
You are an expert proofreader specializing in historical Norwegian text. \
You are given raw OCR output from a 1950s Norwegian newspaper scan. The OCR \
contains errors from misread characters, especially in fraktur/antiqua typefaces.

Your task is to correct obvious OCR errors while preserving the original text \
as faithfully as possible.

Rules:
- Fix clear character-level OCR errors (e.g. rn\u2192m, li\u2192h, cl\u2192d, \
\u00f8\u2192o, \u00e6\u2192ae confusions, doubled/missing letters).
- Fix garbled words where the correct Norwegian word is obvious from context.
- Preserve the original paragraph structure, line breaks, and formatting exactly.
- Preserve \u00ab\u00bb quotes, headings, and verse formatting.
- Do NOT rewrite, modernize spelling, or rephrase. Keep the 1950s Norwegian \
orthography (e.g. \u00abbleven\u00bb not \u00abblitt\u00bb, \u00abhvad\u00bb not \u00abhva\u00bb).
- If a word is ambiguous and you cannot determine the correct reading, leave it \
as-is with [?] after it.
- Do NOT add commentary or notes. Output only the corrected text.
- NEVER remove or summarize content. The output must have the same amount of \
text as the input.\
"""

CORRECTION_USER_PROMPT = """\
Correct OCR errors in the following text from a 1950s Norwegian newspaper. \
Fix only clear misreadings. Preserve original spelling and structure.\n\n{text}\
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


# Claude API limit is 5 MB on the decoded image bytes (not the base64 string).
# Base64 inflates size by ~33%, but the API decodes before checking, so we
# compare against raw byte length throughout.
MAX_IMAGE_BYTES = 5 * 1024 * 1024


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
        self.output_dir_name = f"vision-{dpi}dpi-{self._model_short}"

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

    def _ocr_image(self, client, model, image_data, media_type, label, user_prompt=USER_PROMPT):
        """Send an image to Claude for OCR and return the result.

        Returns (text, input_tokens, output_tokens, elapsed) on success, or None on error.
        The first error in a session prints full diagnostics; subsequent errors are brief.
        """
        try:
            with Spinner(f"  {label}: sending to Claude") as spinner, \
                 client.messages.stream(
                    model=model,
                    max_tokens=self.max_tokens,
                    temperature=0,
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
                                    "text": user_prompt,
                                },
                            ],
                        }
                    ],
                 ) as stream:
                token_count = 0
                for text in stream.text_stream:
                    token_count += 1
                    if token_count % 20 == 0:
                        spinner.update(f"~{token_count} tokens")
                message = stream.get_final_message()
            elapsed = spinner.elapsed
        except Exception as exc:
            self._handle_api_error(exc, client)
            return None

        usage = message.usage
        stop = message.stop_reason
        status = "complete" if stop == "end_turn" else "truncated"
        print(yellow(
            f"  {label}: {status} in {elapsed:.1f}s"
            f"  ({usage.input_tokens} in / {usage.output_tokens} out)"
        ))
        if stop == "max_tokens":
            print(yellow(
                f"  Warning: output was truncated at {self.max_tokens} tokens."
                " Re-run with a higher --max-tokens value to get the full text."
            ))

        return (message.content[0].text, usage.input_tokens, usage.output_tokens, elapsed)

    def _handle_api_error(self, exc, client):
        """Print diagnostics for API errors."""
        import anthropic

        if isinstance(exc, anthropic.AuthenticationError):
            print(red(
                "Error: Authentication failed — your API key is invalid or expired."
            ), file=sys.stderr)
            print(red(
                "  Get a key at: https://platform.claude.com/settings/keys"
            ), file=sys.stderr)
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

    def process_file(self, file_path):
        """Process a single file: split into columns, OCR each, and write output."""
        from pdf2image import convert_from_path
        from PIL import Image

        client = self._get_client()
        model = self._resolve_model(client)

        file_path = Path(file_path)
        ext = file_path.suffix.lower()
        stem = file_path.stem

        OUTPUT_DIR.mkdir(exist_ok=True)
        print(f"Processing: {file_path}")

        if ext in UNSUPPORTED_IMAGE_TYPES:
            print(red(
                "Error: TIFF format is not supported by Claude API. "
                "Use tesseract engine instead."
            ), file=sys.stderr)
            return

        # Load the full page image
        if ext == ".pdf":
            print(f"  Converting PDF to image (DPI={self.dpi})...")
            images = convert_from_path(
                str(file_path),
                dpi=self.dpi,
                first_page=1,
                last_page=1,
                poppler_path=_poppler_path,
            )
            page_image = images[0]
        elif ext in MEDIA_TYPES:
            page_image = Image.open(file_path)
        else:
            print(red(f"Error: Unsupported file format '{ext}'"), file=sys.stderr)
            return

        # Create sub-folder for this file: output/{stem}/{output_dir_name}/
        sub_dir = OUTPUT_DIR / stem / self.output_dir_name
        sub_dir.mkdir(parents=True, exist_ok=True)

        # Split into columns (debug images saved alongside OCR output)
        header_image, column_images = _split_columns(page_image, debug_dir=sub_dir)
        n_cols = len(column_images)
        print(yellow(f"  Detected {n_cols} column{'s' if n_cols != 1 else ''}"))

        total_in = 0
        total_out = 0
        total_elapsed = 0.0
        sections = []  # (label, text) for concatenation

        # OCR header if present (currently unused — header detection disabled)
        if header_image:
            image_data, media_type = _encode_image_under_limit(header_image)
            result = self._ocr_image(client, model, image_data, media_type,
                                     "Header", user_prompt=HEADER_USER_PROMPT)
            if result is None:
                return
            text, in_tok, out_tok, elapsed = result
            total_in += in_tok
            total_out += out_tok
            total_elapsed += elapsed
            (sub_dir / "header.txt").write_text(text + "\n", encoding="utf-8")
            print(green(f"  -> {sub_dir / 'header.txt'}"))
            sections.append(("header", text))

        # OCR each column
        for i, col_image in enumerate(column_images, 1):
            image_data, media_type = _encode_image_under_limit(col_image)
            prompt = COLUMN_USER_PROMPT if n_cols > 1 else USER_PROMPT
            result = self._ocr_image(client, model, image_data, media_type,
                                     f"Column {i}/{n_cols}", user_prompt=prompt)
            if result is None:
                return
            text, in_tok, out_tok, elapsed = result
            total_in += in_tok
            total_out += out_tok
            total_elapsed += elapsed
            col_file = sub_dir / f"column-{i}.txt"
            col_file.write_text(text + "\n", encoding="utf-8")
            print(green(f"  -> {col_file}"))
            sections.append((f"column-{i}", text))

        # Concatenate all sections into combined file
        combined_text = "\n\n".join(text for _, text in sections)
        combined_path = sub_dir / "combined.txt"
        combined_path.write_text(combined_text + "\n", encoding="utf-8")
        print(green(f"  -> {combined_path}"))

        print(yellow(
            f"  Total: {total_elapsed:.1f}s"
            f"  ({total_in} in / {total_out} out)"
        ))

        # Post-processing: correct OCR errors via a second text-only pass
        corrected_path = sub_dir / "combined.corrected.txt"
        corrected = self._correct_ocr(client, model, combined_text)
        if corrected:
            corrected_path.write_text(corrected + "\n", encoding="utf-8")
            print(green(f"  -> {corrected_path}"))

        # Write best available output as transcribed.txt
        best_text = corrected if corrected else combined_text
        transcribed_path = sub_dir / "transcribed.txt"
        transcribed_path.write_text(best_text + "\n", encoding="utf-8")
        print(green(f"  -> {transcribed_path}"))

    def _correct_ocr(self, client, model, text):
        """Run a text-only correction pass on raw OCR output."""
        try:
            with Spinner("Correcting OCR errors") as spinner, \
                 client.messages.stream(
                    model=model,
                    max_tokens=self.max_tokens,
                    temperature=0,
                    system=CORRECTION_SYSTEM_PROMPT,
                    messages=[
                        {
                            "role": "user",
                            "content": CORRECTION_USER_PROMPT.format(text=text),
                        }
                    ],
                 ) as stream:
                token_count = 0
                for chunk in stream.text_stream:
                    token_count += 1
                    if token_count % 20 == 0:
                        spinner.update(f"~{token_count} tokens")
                message = stream.get_final_message()
            elapsed = spinner.elapsed

            usage = message.usage
            stop = message.stop_reason
            status = "complete" if stop == "end_turn" else "truncated"
            print(yellow(
                f"  {status.capitalize()} in {elapsed:.1f}s"
                f"  ({usage.input_tokens} in / {usage.output_tokens} out)"
            ))
            if stop == "max_tokens":
                print(yellow(
                    f"  Warning: correction was truncated at {self.max_tokens} tokens."
                    " Re-run with a higher --max-tokens value."
                ))
            return message.content[0].text
        except Exception as exc:
            print(red(f"  Correction pass failed: {exc}"), file=sys.stderr)
            return None
