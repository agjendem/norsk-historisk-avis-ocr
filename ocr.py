#!/usr/bin/env python3
"""
Cross-platform OCR CLI.

Interactive file picker to select files from input/, process them with
a chosen OCR engine, and write results to output/.

Usage:
    python ocr.py                          # pick engine, then pick files
    python ocr.py tesseract                # pick files for tesseract
    python ocr.py claude-vision --dpi 400  # pick files for claude-vision
"""

import argparse
import sys
from pathlib import Path

from engines import ENGINE_NAMES, _get_engines

PROJECT_DIR = Path(__file__).resolve().parent
INPUT_DIR = PROJECT_DIR / "input"
OUTPUT_DIR = PROJECT_DIR / "output"

SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".tif"}


def list_files(output_suffix):
    """List supported files in input/, marking processed ones with [done]."""
    files = sorted(
        f
        for f in INPUT_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not files:
        print(f"No supported files found in {INPUT_DIR}/")
        return None

    print()
    print("Files in input/:")
    for i, f in enumerate(files, 1):
        stem = f.stem
        marker = ""
        if (OUTPUT_DIR / f"{stem}{output_suffix}").exists():
            marker = " [done]"
        print(f"  {i}) {f.name}{marker}")
    print("  q) Quit")
    print()

    return files


def select_engine():
    """Interactive engine selection menu."""
    print("Select an OCR engine:")
    for i, name in enumerate(ENGINE_NAMES, 1):
        print(f"  {i}) {name}")
    choice = input(f"Choice [1-{len(ENGINE_NAMES)}]: ").strip()
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(ENGINE_NAMES):
            return ENGINE_NAMES[idx]
    except ValueError:
        pass
    print(f"Error: Invalid choice '{choice}'", file=sys.stderr)
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Cross-platform OCR CLI",
        epilog="Engines: tesseract, claude-vision",
    )
    parser.add_argument(
        "engine",
        nargs="?",
        choices=ENGINE_NAMES,
        help="OCR engine to use (interactive menu if omitted)",
    )
    parser.add_argument("--dpi", type=int, default=300, help="DPI for PDF conversion")
    parser.add_argument(
        "--model",
        default="claude-sonnet-4-20250514",
        help="Claude model (claude-vision only)",
    )
    parser.add_argument(
        "--region",
        default="eu-north-1",
        help="AWS Bedrock region (claude-vision only, default: eu-north-1)",
    )
    args = parser.parse_args()

    # Engine selection
    engine_name = args.engine or select_engine()

    # Lazy-load engine classes (deferred so --help works without deps installed)
    engines = _get_engines()
    engine_cls = engines[engine_name]

    # Build engine with applicable kwargs
    kwargs = {"dpi": args.dpi}
    if engine_name == "claude-vision":
        kwargs["model"] = args.model
        kwargs["region"] = args.region
    engine = engine_cls(**kwargs)

    # Dependency check
    missing = engine.check_dependencies()
    if missing:
        print(f"Missing dependencies for {engine_name}:")
        for item in missing:
            print(f"  - {item}")
        answer = input("Run setup.py now? [Y/n] ").strip().lower()
        if answer in ("", "y", "yes"):
            import subprocess
            subprocess.run([sys.executable, str(PROJECT_DIR / "setup.py")], check=True)
        else:
            sys.exit(1)

    # Ensure input/ exists
    INPUT_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Interactive file picker loop
    while True:
        files = list_files(engine.output_suffix)
        if files is None:
            break

        selection = input(f"Select file [1-{len(files)} or q]: ").strip()

        if selection.lower() == "q":
            print("Bye.")
            break

        try:
            idx = int(selection) - 1
            if 0 <= idx < len(files):
                print()
                try:
                    engine.process_file(files[idx])
                except Exception as e:
                    print(f"Error: {e}", file=sys.stderr)
                continue
        except ValueError:
            pass

        print(f"Invalid selection '{selection}'")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
