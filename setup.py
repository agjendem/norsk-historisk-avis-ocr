#!/usr/bin/env python3
"""
Cross-platform setup script for OCR CLI.

Creates a virtual environment, installs shared and engine-specific
dependencies, and checks for external tools (tesseract, poppler).
"""

import platform
import shutil
import subprocess
import sys
import venv
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
VENV_DIR = PROJECT_DIR / ".venv"

ENGINES = {
    "tesseract": PROJECT_DIR / "engines" / "tesseract-requirements.txt",
    "claude-vision": PROJECT_DIR / "engines" / "claude-vision-requirements.txt",
}


def get_venv_python():
    """Return path to the venv Python executable."""
    if platform.system() == "Windows":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def create_venv():
    """Create virtual environment if it doesn't exist."""
    venv_python = get_venv_python()
    if venv_python.exists():
        print(f"[OK] Virtual environment exists: {VENV_DIR}")
        return
    print(f"Creating virtual environment in {VENV_DIR}...")
    venv.create(str(VENV_DIR), with_pip=True)
    print(f"[OK] Virtual environment created")


def install_requirements(requirements_file, label):
    """Install requirements from a file into the venv."""
    venv_python = get_venv_python()
    print(f"Installing {label} dependencies...")
    subprocess.run(
        [str(venv_python), "-m", "pip", "install", "-r", str(requirements_file)],
        check=True,
    )
    print(f"[OK] {label} dependencies installed")


def check_tesseract():
    """Check if tesseract is available."""
    if shutil.which("tesseract"):
        print("[OK] tesseract found")
        return True

    if platform.system() == "Windows":
        win_path = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
        if win_path.exists():
            print(f"[OK] tesseract found at {win_path}")
            return True

    print("[MISSING] tesseract")
    system = platform.system()
    if system == "Darwin":
        answer = input("  Install via Homebrew (brew install tesseract tesseract-lang)? [Y/n] ").strip().lower()
        if answer in ("", "y", "yes"):
            subprocess.run(["brew", "install", "tesseract", "tesseract-lang"], check=True)
            return True
    elif system == "Windows":
        print("  Download from: https://github.com/UB-Mannheim/tesseract/wiki")
        print("  Install to the default path: C:\\Program Files\\Tesseract-OCR\\")
        print("  Make sure to select Norwegian language data during install.")
    else:
        print("  Install tesseract via your package manager (e.g. apt install tesseract-ocr tesseract-ocr-nor)")
    return False


def check_poppler():
    """Check if poppler (pdftoppm) is available."""
    if shutil.which("pdftoppm"):
        print("[OK] poppler (pdftoppm) found")
        return True

    if platform.system() == "Windows":
        vendor_path = PROJECT_DIR / "vendor" / "poppler" / "Library" / "bin"
        if vendor_path.exists() and (vendor_path / "pdftoppm.exe").exists():
            print(f"[OK] poppler found at {vendor_path}")
            return True

    print("[MISSING] poppler (provides pdftoppm for PDF conversion)")
    system = platform.system()
    if system == "Darwin":
        answer = input("  Install via Homebrew (brew install poppler)? [Y/n] ").strip().lower()
        if answer in ("", "y", "yes"):
            subprocess.run(["brew", "install", "poppler"], check=True)
            return True
    elif system == "Windows":
        print("  Download from: https://github.com/oschwartz10612/poppler-windows/releases")
        print("  Extract to: vendor/poppler/ inside the project directory")
        print("  Expected path: vendor/poppler/Library/bin/pdftoppm.exe")
    else:
        print("  Install poppler via your package manager (e.g. apt install poppler-utils)")
    return False


def main():
    print(f"OCR CLI Setup — {platform.system()}")
    print("=" * 40)
    print()

    # Step 1: Virtual environment
    create_venv()
    print()

    # Step 2: Shared dependencies
    install_requirements(PROJECT_DIR / "requirements.txt", "shared")
    print()

    # Step 3: Engine selection
    print("Which engine(s) do you want to set up?")
    print("  1) tesseract")
    print("  2) claude-vision")
    print("  3) both")
    choice = input("Choice [1-3]: ").strip()

    selected = []
    if choice == "1":
        selected = ["tesseract"]
    elif choice == "2":
        selected = ["claude-vision"]
    elif choice == "3":
        selected = ["tesseract", "claude-vision"]
    else:
        print(f"Invalid choice '{choice}', installing both.")
        selected = ["tesseract", "claude-vision"]

    print()
    for engine_name in selected:
        install_requirements(ENGINES[engine_name], engine_name)
    print()

    # Step 4: External tool checks
    print("Checking external dependencies...")
    if "tesseract" in selected:
        check_tesseract()
    check_poppler()
    print()

    print("Setup complete.")


if __name__ == "__main__":
    main()
