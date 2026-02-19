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


def _detect_header_boundary(gray_pixels, width, height, threshold=200):
    """Find the y-coordinate where newspaper columns begin.

    Scans rows top-down looking for the first row where the vertical projection
    shows multiple distinct text regions separated by gaps — indicating the
    start of multi-column layout.

    Returns the y-coordinate of the column start, or 0 if no header detected.
    """
    # Minimum gap width (px) to count as a column separator
    min_gap = 15
    # Minimum number of column regions to consider it "multi-column"
    min_columns = 2
    # Scan in bands of rows to smooth out noise
    band_height = 20

    for y_start in range(0, height - band_height, band_height):
        y_end = min(y_start + band_height, height)
        # Count dark pixels per x-coordinate in this band
        x_dark = [0] * width
        for y in range(y_start, y_end):
            for x in range(width):
                if gray_pixels[x, y] < threshold:
                    x_dark[x] += 1

        # Classify each x as "has text" or "gap" in this band
        band_rows = y_end - y_start
        gap_threshold = band_rows * 0.01  # <1% dark pixels = gap
        in_text = False
        regions = 0
        gap_width = 0

        for x in range(width):
            if x_dark[x] > gap_threshold:
                if not in_text:
                    if gap_width >= min_gap or regions == 0:
                        regions += 1
                    in_text = True
                gap_width = 0
            else:
                in_text = False
                gap_width += 1

        if regions >= min_columns:
            # Found multi-column layout — header ends here
            # Go back a bit to avoid cutting into the first column row
            return max(0, y_start)

    return 0


def _find_gap_boundaries(gray_pixels, x_start, x_end, y_start, y_end,
                         expected_col_width, threshold=200, min_gap_px=8,
                         min_coverage=0.55):
    """Find column boundaries in a wide segment using gap coverage analysis.

    Computes a gap coverage profile: for each x-position, the fraction of rows
    where x is inside a run of consecutive light pixels (>=threshold) at least
    min_gap_px wide.  Then uses the expected column width to guide a search for
    boundaries at positions with maximum gap coverage.

    Args:
        gray_pixels: Pixel access object from a grayscale PIL Image.
        x_start, x_end: Horizontal extent of the segment (absolute coords).
        y_start, y_end: Vertical extent of the segment.
        expected_col_width: Median column width from Phase 1 (guides search).
        threshold: Grayscale value above which a pixel is considered light.
        min_gap_px: Minimum run length to count as a gap (filters inter-word spaces).
        min_coverage: Minimum gap coverage fraction to accept a boundary.

    Returns list of absolute x-coordinates for detected boundaries.
    """
    seg_width = x_end - x_start
    height = y_end - y_start
    if seg_width < 50 or height < 50:
        return []

    # Build gap coverage profile: for each relative x, count rows where x is
    # inside a gap run of >= min_gap_px consecutive light pixels.
    coverage = [0] * seg_width
    for y in range(y_start, y_end):
        run_start = None
        for rx in range(seg_width):
            ax = x_start + rx
            if gray_pixels[ax, y] >= threshold:
                if run_start is None:
                    run_start = rx
            else:
                if run_start is not None:
                    if rx - run_start >= min_gap_px:
                        for j in range(run_start, rx):
                            coverage[j] += 1
                    run_start = None
        if run_start is not None:
            if seg_width - run_start >= min_gap_px:
                for j in range(run_start, seg_width):
                    coverage[j] += 1

    # Smooth with 15px moving average
    half_w = 7
    smoothed = [0.0] * seg_width
    for i in range(seg_width):
        lo = max(0, i - half_w)
        hi = min(seg_width, i + half_w + 1)
        smoothed[i] = sum(coverage[lo:hi]) / (hi - lo) / height

    # Determine expected number of sub-columns and search for boundaries
    n_expected = round(seg_width / expected_col_width)
    if n_expected < 2:
        return []

    search_radius = int(expected_col_width * 0.3)
    boundaries = []
    for b in range(1, n_expected):
        expected_rx = int(b * seg_width / n_expected)
        lo = max(50, expected_rx - search_radius)
        hi = min(seg_width - 50, expected_rx + search_radius)
        if lo >= hi:
            continue

        best_rx = lo
        best_val = smoothed[lo]
        for rx in range(lo + 1, hi + 1):
            if smoothed[rx] > best_val:
                best_val = smoothed[rx]
                best_rx = rx

        if best_val >= min_coverage:
            boundaries.append(x_start + best_rx)

    return boundaries


def _save_debug_images(image, boundaries, debug_dir, body_top=0, overlap_px=0):
    """Save annotated page image, column crops, and detection info."""
    from PIL import Image, ImageDraw, ImageFont

    debug_dir.mkdir(parents=True, exist_ok=True)
    width, height = image.size

    # Annotated full page
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    for i, bx in enumerate(boundaries):
        if 0 < bx < width:
            draw.line([(bx, 0), (bx, height)], fill="blue", width=2)
    # Draw overlap regions as semi-transparent red shading
    if overlap_px > 0:
        overlay = Image.new("RGBA", annotated.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        for bx in boundaries:
            if 0 < bx < width:
                ol = max(0, bx - overlap_px)
                or_ = min(width, bx + overlap_px)
                overlay_draw.rectangle(
                    [(ol, 0), (or_, height)],
                    fill=(255, 0, 0, 40),
                )
        annotated = Image.alpha_composite(annotated.convert("RGBA"), overlay)
        annotated = annotated.convert("RGB")
    # Label columns
    draw = ImageDraw.Draw(annotated)
    for i in range(len(boundaries) - 1):
        cx = (boundaries[i] + boundaries[i + 1]) // 2
        draw.text((cx - 10, 10), str(i + 1), fill="blue")
    annotated.save(debug_dir / "page_annotated.png")

    # Column crops (with overlap padding matching OCR crops)
    for i in range(len(boundaries) - 1):
        left = max(0, boundaries[i] - overlap_px)
        right = min(width, boundaries[i + 1] + overlap_px)
        if right - left < 30:
            continue
        col_img = image.crop((left, body_top, right, height))
        col_img.save(debug_dir / f"column_{i + 1}_crop.png")

    # Detection info
    info_lines = [
        f"Image size: {width} x {height}",
        f"Body top: {body_top}",
        f"Overlap padding: {overlap_px}px",
        f"Boundaries: {boundaries}",
        f"Columns: {len(boundaries) - 1}",
        "",
    ]
    for i in range(len(boundaries) - 1):
        w = boundaries[i + 1] - boundaries[i]
        pad_left = min(overlap_px, boundaries[i])
        pad_right = min(overlap_px, width - boundaries[i + 1])
        crop_w = w + pad_left + pad_right
        info_lines.append(
            f"  Column {i + 1}: x={boundaries[i]}-{boundaries[i + 1]}, "
            f"width={w}px, crop={crop_w}px (pad L={pad_left} R={pad_right})"
        )
    (debug_dir / "detection_info.txt").write_text("\n".join(info_lines) + "\n", encoding="utf-8")


def _split_columns(image, debug_dir=None, overlap_px=20):
    """Split a newspaper page image into individual column images.

    Uses a three-phase algorithm:
    1. Detect ink divider lines via vertical projection profile
    2. Subdivide wide segments using row-by-row gap voting
    3. Merge boundaries, crop columns with overlap padding

    Args:
        image: PIL Image of the full page.
        debug_dir: Optional directory to save debug images.
        overlap_px: Pixels of padding to add on each side of every column
            crop. Compensates for non-linear scan distortion that shifts
            gutter positions across the page height. Default 20px.

    Returns (None, [column_images]).
    Header detection is not performed (headers that don't span full width
    are handled fine by per-column OCR).
    If only one column is detected, returns (None, [original_image]).
    """
    gray = image.convert("L")
    width, height = gray.size
    pixels = gray.load()

    body_top = 0

    # Phase 1: detect ink divider lines via vertical projection profile
    v_profile = [0] * width
    for x in range(width):
        for y in range(body_top, height):
            if pixels[x, y] < 200:
                v_profile[x] += 1

    body_height = height - body_top
    divider_threshold = body_height * 0.8
    divider_xs = []
    in_divider = False
    div_start = 0
    for x in range(width):
        if v_profile[x] >= divider_threshold:
            if not in_divider:
                div_start = x
                in_divider = True
        else:
            if in_divider:
                divider_xs.append((div_start + x) // 2)
                in_divider = False
    if in_divider:
        divider_xs.append((div_start + width - 1) // 2)

    # Phase 2: subdivide wide segments using gap voting
    phase1_boundaries = [0] + divider_xs + [width]

    # Compute median segment width from Phase 1
    seg_widths = [phase1_boundaries[i + 1] - phase1_boundaries[i]
                  for i in range(len(phase1_boundaries) - 1)]
    seg_widths_sorted = sorted(seg_widths)
    if seg_widths_sorted:
        mid = len(seg_widths_sorted) // 2
        median_width = seg_widths_sorted[mid]
    else:
        median_width = width

    all_boundaries = set(phase1_boundaries)

    for i in range(len(phase1_boundaries) - 1):
        seg_left = phase1_boundaries[i]
        seg_right = phase1_boundaries[i + 1]
        seg_w = seg_right - seg_left

        # Only subdivide segments wider than 1.5x the median
        if seg_w > median_width * 1.5:
            gap_bounds = _find_gap_boundaries(
                pixels, seg_left, seg_right, body_top, height,
                expected_col_width=median_width)
            all_boundaries.update(gap_bounds)

    # Phase 3: merge, sort, and crop
    boundaries = sorted(all_boundaries)

    # Crop column images, skipping narrow artifacts
    columns = []
    final_boundaries = [boundaries[0]]
    for i in range(len(boundaries) - 1):
        left = boundaries[i]
        right = boundaries[i + 1]
        if right - left < 30:
            continue
        # Apply overlap padding, clamped to image bounds
        crop_left = max(0, left - overlap_px)
        crop_right = min(width, right + overlap_px)
        col_img = image.crop((crop_left, body_top, crop_right, height))
        columns.append(col_img)
        final_boundaries.append(right)

    if debug_dir:
        _save_debug_images(image, final_boundaries, debug_dir, body_top,
                           overlap_px=overlap_px)

    if not columns:
        return (None, [image])

    return (None, columns)


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

        # Create sub-folder for this file
        sub_dir = OUTPUT_DIR / stem
        sub_dir.mkdir(exist_ok=True)

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
        combined_path = sub_dir / f"combined{self.output_suffix}"
        combined_path.write_text(combined_text + "\n", encoding="utf-8")
        print(green(f"  -> {combined_path}"))

        print(yellow(
            f"  Total: {total_elapsed:.1f}s"
            f"  ({total_in} in / {total_out} out)"
        ))

        # Post-processing: correct OCR errors via a second text-only pass
        corrected_path = sub_dir / f"combined{self.output_suffix.replace('.txt', '.corrected.txt')}"
        corrected = self._correct_ocr(client, model, combined_text)
        if corrected:
            corrected_path.write_text(corrected + "\n", encoding="utf-8")
            print(green(f"  -> {corrected_path}"))

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
