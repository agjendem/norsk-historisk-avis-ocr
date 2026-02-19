"""Tesseract OCR engine — cross-platform wrapper using pytesseract and pdf2image."""

import platform
import shutil
from pathlib import Path

import pytesseract
from pdf2image import convert_from_path
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "output"

# Auto-detect tesseract on Windows
if platform.system() == "Windows":
    _win_tesseract = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
    if _win_tesseract.exists():
        pytesseract.pytesseract.tesseract_cmd = str(_win_tesseract)

# Auto-detect poppler on Windows
_poppler_path = None
if platform.system() == "Windows":
    _win_poppler = PROJECT_DIR / "vendor" / "poppler" / "Library" / "bin"
    if _win_poppler.exists():
        _poppler_path = str(_win_poppler)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif"}


class TesseractEngine:
    name = "tesseract"
    output_suffix = ".txt"

    def __init__(self, dpi=300, lang="nor"):
        self.dpi = dpi
        self.lang = lang

    def check_dependencies(self):
        """Return list of missing dependencies."""
        missing = []
        if not shutil.which("tesseract") and not (
            platform.system() == "Windows"
            and Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe").exists()
        ):
            missing.append("tesseract")
        # Check poppler (needed for PDF support)
        if not shutil.which("pdftoppm") and _poppler_path is None:
            missing.append("poppler (provides pdftoppm for PDF conversion)")
        return missing

    def process_file(self, file_path):
        """Process a single file and write OCR output to output/."""
        file_path = Path(file_path)
        ext = file_path.suffix.lower()
        stem = file_path.stem
        txt_path = OUTPUT_DIR / f"{stem}{self.output_suffix}"

        OUTPUT_DIR.mkdir(exist_ok=True)
        print(f"Processing: {file_path}")

        if ext == ".pdf":
            print(f"  Converting PDF to image (DPI={self.dpi})...")
            images = convert_from_path(
                str(file_path),
                dpi=self.dpi,
                first_page=1,
                last_page=1,
                poppler_path=_poppler_path,
            )
            text = pytesseract.image_to_string(images[0], lang=self.lang)
        elif ext in IMAGE_EXTENSIONS:
            img = Image.open(file_path)
            text = pytesseract.image_to_string(img, lang=self.lang)
        else:
            print(f"Error: Unsupported file format '{ext}'")
            return

        txt_path.write_text(text, encoding="utf-8")
        print(f"  -> {txt_path}")
