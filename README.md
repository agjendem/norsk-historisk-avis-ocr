# OCR CLI

Cross-platform CLI for OCR transcription of scanned documents (PDFs and images). Built for historical Norwegian newspaper scans, but adaptable to other use cases.

## Quick start

```bash
python setup.py              # creates .venv, installs dependencies
.venv/bin/python ocr.py      # interactive engine and file picker
```

Place files in `input/`. Results are written to `output/`.

## Engines

### tesseract

Local OCR using [Tesseract](https://github.com/tesseract-ocr/tesseract). Free, fast, runs entirely offline. Reasonable for clean printed text but struggles with historical/degraded scans and complex column layouts.

- Requires `tesseract` and `poppler` installed on the system
- Default language: Norwegian (`nor`)
- Supports: PDF, PNG, JPG, JPEG, TIFF
- Output: `{filename}.txt`

### claude-vision

Cloud OCR using the Anthropic Claude API. Sends the scanned page as an image to Claude with a specialized prompt for column-aware newspaper transcription. Significantly better than Tesseract for historical scans with complex layouts.

- Requires an Anthropic API key or AWS credentials (see [Authentication](#authentication))
- Requires `poppler` for PDF support
- Supports: PDF, PNG, JPG, JPEG (not TIFF)
- Output: `{filename}.vision-{dpi}dpi-{model}.txt`
- Images are sharpened, contrast-boosted, and compressed to JPEG to fit the 5 MB API limit
- Token usage (input/output) is printed after each call

## Authentication

The claude-vision engine auto-detects credentials in this order:

1. **`ANTHROPIC_API_KEY`** in environment or `.env` file -- uses the direct Anthropic API
2. **AWS credentials** (`AWS_PROFILE`, `AWS_ACCESS_KEY_ID`, or boto3 default session) -- uses AWS Bedrock
3. **Interactive prompt** -- asks for an API key and saves it to `.env`

Get an API key at: https://platform.claude.com/settings/keys

### Corporate proxies / SSL

If you're behind a corporate TLS-inspecting proxy, the `truststore` package (included in dependencies) uses your OS trust store instead of Python's bundled certificates. This handles self-signed corporate CAs automatically on macOS and Windows.

## CLI arguments

```
python ocr.py [engine] [options]
```

| Argument | Default | Description |
|---|---|---|
| `engine` | _(interactive)_ | `tesseract` or `claude-vision` |
| `--dpi` | `300` | DPI for PDF-to-image conversion |
| `--model` | `claude-opus-4-20250514` | Claude model ID (claude-vision only) |
| `--max-tokens` | `16384` | Max output tokens (claude-vision only) |
| `--region` | `eu-north-1` | AWS Bedrock region (claude-vision only) |

### Examples

```bash
# Interactive mode
.venv/bin/python ocr.py

# Tesseract on all files
.venv/bin/python ocr.py tesseract

# Claude Vision with defaults (Opus, 300 DPI)
.venv/bin/python ocr.py claude-vision

# Use Sonnet instead (cheaper, less accurate)
.venv/bin/python ocr.py claude-vision --model claude-sonnet-4-20250514

# Higher token limit for very dense pages
.venv/bin/python ocr.py claude-vision --max-tokens 32768

# Use Bedrock in us-east-1
.venv/bin/python ocr.py claude-vision --region us-east-1
```

## Model choices and cost

The claude-vision engine reports token usage after each call. Cost depends on the model:

| Model | Input | Output | Accuracy | Notes |
|---|---|---|---|---|
| `claude-opus-4-20250514` | $15/M tokens | $75/M tokens | Best | Default. ~$0.23/page. Best at column layouts and degraded text |
| `claude-sonnet-4-20250514` | $3/M tokens | $15/M tokens | Good | ~$0.04/page. Faster, but struggles with complex multi-column layouts |

Opus is recommended for historical newspaper scans. Sonnet is a reasonable choice for cleaner, single-column documents where cost matters.

The `--max-tokens` flag controls the output ceiling. It only affects cost if the model actually generates that many tokens. Set it higher for dense pages; the default of 16384 covers most single newspaper pages.

## Output files

Output filenames encode the engine settings for easy comparison:

- Tesseract: `Filename.txt`
- Claude Vision: `Filename.vision-300dpi-opus.txt`

Re-running with different settings (e.g., `--dpi 150` or `--model claude-sonnet-4-20250514`) produces separate output files so you can compare results side by side.

## Project structure

```
ocr.py                  # CLI entry point
setup.py                # Setup script (venv, deps, external tools)
requirements.txt        # Shared Python dependencies
engines/
  __init__.py           # Engine registry
  _colors.py            # ANSI color helpers
  tesseract_engine.py   # Tesseract engine
  claude_vision_engine.py # Claude Vision engine
  tesseract-requirements.txt
  claude-vision-requirements.txt
input/                  # Place scanned files here
output/                 # OCR results written here
.env                    # API key (auto-created, gitignored)
```
