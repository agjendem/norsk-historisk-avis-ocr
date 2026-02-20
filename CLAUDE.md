# Project Guidelines

See [README.md](README.md) for full project documentation.

## Architecture

- `ocr.py` is the CLI entry point. Engine-specific args (`--model`, `--max-tokens`, `--region`) are only passed when `engine_name == "claude-vision"`.
- Engines live in `engines/` and follow a common interface: `__init__(**kwargs)`, `check_dependencies()`, `process_file(path)`, `output_suffix`, `output_dir_name`.
- Column splitting logic lives in `engines/_columns.py` and is shared by both engines. Both engines split pages into columns before OCR.
- Engine imports are lazy (deferred in `engines/__init__.py`) so `--help` and setup work without dependencies installed.
- The `truststore` import at the top of `claude_vision_engine.py` must stay before any `anthropic` imports to inject OS trust store for SSL.

## Authentication flow

Claude-vision auth priority: `ANTHROPIC_API_KEY` > AWS credentials (Bedrock) > interactive prompt. See `_get_client()` in `claude_vision_engine.py`. When using Bedrock, model IDs are mapped via `BEDROCK_MODEL_MAP`.

## Image processing pipeline

Both engines share a common pre-processing step: pdf2image conversion (PDF only) -> column splitting via `engines/_columns.py`. Column splitting uses a three-phase algorithm (ink divider detection, gap coverage analysis, merge and crop with overlap padding). The tesseract engine also strips leading/trailing `|` characters from output lines (column divider artifacts).

For Claude Vision specifically, column images additionally go through: sharpen + contrast boost -> JPEG encoding with adaptive quality/scaling to stay under 5 MB API limit. This happens in `_prepare_image()` and `_encode_image_under_limit()`.

## Key constraints

- Claude API has a 5 MB image size limit on **decoded** bytes (not the base64 string). Base64 inflates by ~33%, but the API decodes before checking. All size comparisons in the code use raw byte length to match. Two paths: `_encode_image_under_limit()` loops until under the limit (PDFs, oversized images); small image files are passed through as-is after a raw size check.
- Opus with high `max_tokens` requires the streaming API (`client.messages.stream()`).
- TIFF files are not supported by the Claude API. Only tesseract handles TIFF.
- Output is written to `output/{stem}/{output_dir_name}/` (e.g. `output/s5u/vision-300dpi-opus/`). Each engine/config combination gets its own subfolder for side-by-side comparison. Per-column files are `column-N.txt`, concatenated result is `combined.txt`.

## OCR prompt tuning

The system/user prompts in `claude_vision_engine.py` are tuned for multi-column Norwegian newspaper scans. Key prompt rules:
- Explicit column-awareness instructions (identify columns first, process one at a time)
- Anti-truncation rule (never summarize, stop mid-sentence if needed)
- The prompts should be adjusted if the use case changes (e.g., single-column documents, non-Norwegian text).
