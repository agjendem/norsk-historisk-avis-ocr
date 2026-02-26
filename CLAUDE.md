# Project Guidelines

See [README.md](README.md) for full project documentation.

## Architecture

- `ocr.py` is the CLI entry point. Engine-specific args (`--model`, `--max-tokens`, `--region`) are only passed when `engine_name in ("claude-vision", "tesseract+claude")`.
- Engines live in `engines/` and follow a common interface: `__init__(**kwargs)`, `check_dependencies()`, `process_file(path)`, `output_suffix`, `output_dir_name`.
- Three engines: `tesseract` (local OCR), `claude-vision` (Claude API for OCR + correction), `tesseract+claude` (tesseract OCR + Claude correction). The `tesseract+claude` engine subclasses `TesseractEngine` and adds a Claude correction pass.
- Column splitting logic lives in `engines/_columns.py` and is shared by all engines. All engines split pages into columns before OCR.
- Shared Claude correction logic (prompts, auth, client creation, `correct_ocr()`) lives in `engines/_correction.py` and is used by both `claude-vision` and `tesseract+claude` engines.
- Engine imports are lazy (deferred in `engines/__init__.py`) so `--help` and setup work without dependencies installed.
- The `truststore` import at the top of `claude_vision_engine.py` must stay before any `anthropic` imports to inject OS trust store for SSL.

## Authentication flow

Claude auth priority: `ANTHROPIC_API_KEY` > AWS credentials (Bedrock) > interactive prompt. See `get_claude_client()` in `engines/_correction.py`. When using Bedrock, model IDs are mapped via `BEDROCK_MODEL_MAP`.

## Image processing pipeline

Both engines share a common pre-processing step: pdf2image conversion (PDF only) -> column splitting via `engines/_columns.py`. Column splitting uses a three-phase algorithm (ink divider detection, gap coverage analysis, merge and crop with overlap padding). The tesseract engine runs `_clean_divider_noise()` on output to remove column divider pipe artifacts — this handles trailing pipes with adjacent-column bleed (`text | s`, `text|g`), leading pipes (`-| text`), standalone pipes, stray 1-2 character garbage lines, and collapses excessive blank lines.

After column boundaries are determined, `_detect_title_region()` checks for title sections (large-font headings/bylines) that span multiple columns. It scans the top 40% of each column strip for large vertical gaps (>=40px of blank rows) between text lines — title text has these gaps while body text is dense and continuous. Adjacent columns with large gaps are grouped into a title region, cropped as a single image, and excluded from column crops (each title column's crop starts at its `body_start_y` instead of 0). `_split_columns()` returns `(title_image_or_none, [column_images])`. Both engines handle the title: claude-vision uses `HEADER_USER_PROMPT` to OCR it, tesseract runs `image_to_string` on it. Title text is written to `header.txt` and placed first in `combined.txt`. Debug output includes a green-shaded title overlay on `page_annotated.png` and a `title_crop.png` file.

For Claude Vision specifically, column images additionally go through: sharpen + contrast boost -> JPEG encoding with adaptive quality/scaling to stay under 5 MB API limit. This happens in `_prepare_image()` and `_encode_image_under_limit()`.

## Key constraints

- Claude API has a 5 MB image size limit on **decoded** bytes (not the base64 string). Base64 inflates by ~33%, but the API decodes before checking. All size comparisons in the code use raw byte length to match. Two paths: `_encode_image_under_limit()` loops until under the limit (PDFs, oversized images); small image files are passed through as-is after a raw size check.
- Opus with high `max_tokens` requires the streaming API (`client.messages.stream()`).
- TIFF files are not supported by the Claude API. Only tesseract handles TIFF.
- Output is written to `output/{stem}/{output_dir_name}/` (e.g. `output/s5u/vision-300dpi-opus/`). Each engine/config combination gets its own subfolder for side-by-side comparison. If a title is detected, `header.txt` contains the title text. Per-column files are `column-N.txt`, concatenated result is `combined.txt` (title first, then columns). All engines write `transcribed.txt` as the final best output: tesseract's version is reflowed (hyphenated words rejoined, line breaks collapsed to spaces, paragraph breaks preserved via `_reflow_text()`); claude-vision's and tesseract+claude's is the corrected LLM output (or raw combined/reflowed if correction failed). The tesseract+claude engine also writes `combined.corrected.txt`.

## OCR prompt tuning

The system/user prompts in `claude_vision_engine.py` are tuned for multi-column Norwegian newspaper scans. Key prompt rules:
- Explicit column-awareness instructions (identify columns first, process one at a time)
- Anti-truncation rule (never summarize, stop mid-sentence if needed)
- The prompts should be adjusted if the use case changes (e.g., single-column documents, non-Norwegian text).
