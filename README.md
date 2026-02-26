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

Local OCR using [Tesseract](https://github.com/tesseract-ocr/tesseract). Free, fast, runs entirely offline. Pages are split into columns before OCR (same algorithm as Claude Vision) for better accuracy on multi-column layouts. Title sections that span multiple columns are detected automatically, extracted separately, and placed first in the output.

- Requires `tesseract` and `poppler` installed on the system
- Default language: Norwegian (`nor`)
- Supports: PDF, PNG, JPG, JPEG, TIFF
- Output: `output/{stem}/tesseract-{dpi}dpi/` with `header.txt` (if title detected), per-column files, `combined.txt`, and `transcribed.txt` (reflowed flowing text with hyphenated words rejoined)

### tesseract+claude

Hybrid engine: runs Tesseract OCR locally, then sends the reflowed text to Claude for intelligent error correction. Gets most of the accuracy benefit of Claude Vision at a fraction of the cost (text-only API call, no image tokens). Title sections are handled the same way as the other engines.

- Requires `tesseract`, `poppler`, and Claude credentials (see [Authentication](#authentication))
- Default language: Norwegian (`nor`)
- Supports: PDF, PNG, JPG, JPEG, TIFF
- Output: `output/{stem}/tesseract+claude-{dpi}dpi-{model}/` with `header.txt` (if title detected), per-column files, `combined.txt`, `transcribed-pre-claude.txt` (reflowed tesseract output before correction), `transcribed.txt` (corrected output), and `correction-changes.txt` (human-readable diff of Claude's edits)

### claude-vision

Cloud OCR using the Anthropic Claude API. Sends the scanned page as an image to Claude with a specialized prompt for column-aware newspaper transcription. Significantly better than Tesseract for historical scans with complex layouts. Title sections that span multiple columns are detected automatically, extracted separately, and placed first in the output.

- Requires an Anthropic API key or AWS credentials (see [Authentication](#authentication))
- Requires `poppler` for PDF support
- Supports: PDF, PNG, JPG, JPEG (not TIFF)
- Output: `output/{stem}/vision-{dpi}dpi-{model}/` with `header.txt` (if title detected), per-column files, `combined.txt`, `combined.corrected.txt`, and `transcribed.txt` (best available corrected output)
- Images are sharpened, contrast-boosted, and compressed to JPEG to fit the 5 MB API limit
- Token usage (input/output) is printed after each call
- Automatic post-processing pass corrects common OCR errors using a second text-only Claude call

## Authentication

The `claude-vision` and `tesseract+claude` engines auto-detect credentials in this order:

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
| `engine` | _(interactive)_ | `tesseract`, `tesseract+claude`, or `claude-vision` |
| `--dpi` | `300` | DPI for PDF-to-image conversion |
| `--model` | `claude-opus-4-20250514` | Claude model ID (`claude-vision` and `tesseract+claude`) |
| `--max-tokens` | `16384` | Max output tokens (`claude-vision` and `tesseract+claude`) |
| `--region` | `eu-north-1` | AWS Bedrock region (`claude-vision` and `tesseract+claude`) |

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

# Tesseract OCR + Claude correction (cheaper than full Claude Vision)
.venv/bin/python ocr.py tesseract+claude

# Tesseract+Claude with Sonnet for correction
.venv/bin/python ocr.py tesseract+claude --model claude-sonnet-4-20250514
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

Output is organized as `output/{stem}/{engine-config}/` so different engines and settings produce separate folders for side-by-side comparison:

```
output/
  RB_1957_s5u/
    tesseract-300dpi/
      header.txt                      # title section OCR (if detected)
      column-1.txt .. column-N.txt   # per-column OCR
      combined.txt                    # concatenated result (title first)
      transcribed.txt                 # reflowed flowing text (final output)
      page_annotated.png              # debug: column boundaries + title box
      title_crop.png                  # debug: title region crop (if detected)
      column_N_crop.png               # debug: column crops
      detection_info.txt              # debug: boundary positions + title info
    tesseract+claude-300dpi-opus/
      header.txt                      # title section OCR (if detected)
      column-1.txt .. column-N.txt
      combined.txt
      transcribed-pre-claude.txt      # reflowed tesseract output (before correction)
      transcribed.txt                 # Claude-corrected output (final output)
      correction-changes.txt          # human-readable diff of Claude's edits
      page_annotated.png
      title_crop.png                  # (if detected)
      column_N_crop.png
      detection_info.txt
    vision-300dpi-opus/
      header.txt                      # title section OCR (if detected)
      column-1.txt .. column-N.txt
      combined.txt
      combined.corrected.txt          # post-processed OCR correction
      transcribed.txt                 # best corrected output (final output)
      page_annotated.png
      title_crop.png                  # (if detected)
      column_N_crop.png
      detection_info.txt
```

Re-running with different settings (e.g., `--dpi 150` or `--model claude-sonnet-4-20250514`) produces separate subfolders.

## Project structure

```
ocr.py                  # CLI entry point
setup.py                # Setup script (venv, deps, external tools)
requirements.txt        # Shared Python dependencies
engines/
  __init__.py           # Engine registry
  _colors.py            # ANSI color helpers
  _columns.py           # Shared column splitting logic
  _correction.py        # Shared Claude correction logic (prompts, auth, client)
  tesseract_engine.py   # Tesseract engine
  tesseract_claude_engine.py # Tesseract+Claude hybrid engine
  claude_vision_engine.py # Claude Vision engine
  tesseract-requirements.txt
  claude-vision-requirements.txt
input/                  # Place scanned files here
output/                 # OCR results written here
.env                    # API key (auto-created, gitignored)
```
