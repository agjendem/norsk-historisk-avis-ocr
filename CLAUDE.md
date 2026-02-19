# Project Guidelines

See [README.md](README.md) for full project documentation.

## Architecture

- `ocr.py` is the CLI entry point. Engine-specific args (`--model`, `--max-tokens`, `--region`) are only passed when `engine_name == "claude-vision"`.
- Engines live in `engines/` and follow a common interface: `__init__(**kwargs)`, `check_dependencies()`, `process_file(path)`, `output_suffix`.
- Engine imports are lazy (deferred in `engines/__init__.py`) so `--help` and setup work without dependencies installed.
- The `truststore` import at the top of `claude_vision_engine.py` must stay before any `anthropic` imports to inject OS trust store for SSL.

## Authentication flow

Claude-vision auth priority: `ANTHROPIC_API_KEY` > AWS credentials (Bedrock) > interactive prompt. See `_get_client()` in `claude_vision_engine.py`. When using Bedrock, model IDs are mapped via `BEDROCK_MODEL_MAP`.

## Image processing pipeline

PDF/images go through: pdf2image conversion (PDF only) -> sharpen + contrast boost -> JPEG encoding with adaptive quality/scaling to stay under 5 MB API limit. This happens in `_prepare_image()` and `_encode_image_under_limit()`.

## Key constraints

- Claude API has a 5 MB image size limit. The `_encode_image_under_limit()` function handles this automatically.
- Opus with high `max_tokens` requires the streaming API (`client.messages.stream()`).
- TIFF files are not supported by the Claude API. Only tesseract handles TIFF.
- Output filenames include DPI and model short name to allow side-by-side comparison of different settings.

## OCR prompt tuning

The system/user prompts in `claude_vision_engine.py` are tuned for multi-column Norwegian newspaper scans. Key prompt rules:
- Explicit column-awareness instructions (identify columns first, process one at a time)
- Anti-truncation rule (never summarize, stop mid-sentence if needed)
- The prompts should be adjusted if the use case changes (e.g., single-column documents, non-Norwegian text).
