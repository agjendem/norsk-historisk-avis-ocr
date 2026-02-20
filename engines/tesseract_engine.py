"""Tesseract OCR engine — cross-platform wrapper using pytesseract and pdf2image."""

import platform
import shutil
from pathlib import Path

from engines._colors import green, yellow, red
from engines._columns import _split_columns

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "output"

# Auto-detect poppler on Windows
_poppler_path = None
if platform.system() == "Windows":
    _win_poppler = PROJECT_DIR / "vendor" / "poppler" / "Library" / "bin"
    if _win_poppler.exists():
        _poppler_path = str(_win_poppler)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif"}


class TesseractEngine:
    name = "tesseract"

    def __init__(self, dpi=300, lang="nor"):
        self.dpi = dpi
        self.lang = lang
        self.output_suffix = f".tesseract-{dpi}dpi.txt"

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
        """Process a single file: split into columns, OCR each, and write output."""
        import pytesseract
        from pdf2image import convert_from_path
        from PIL import Image

        # Auto-detect tesseract on Windows
        if platform.system() == "Windows":
            _win_tesseract = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
            if _win_tesseract.exists():
                pytesseract.pytesseract.tesseract_cmd = str(_win_tesseract)

        file_path = Path(file_path)
        ext = file_path.suffix.lower()
        stem = file_path.stem

        OUTPUT_DIR.mkdir(exist_ok=True)
        print(f"Processing: {file_path}")

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
        elif ext in IMAGE_EXTENSIONS:
            page_image = Image.open(file_path)
        else:
            print(red(f"Error: Unsupported file format '{ext}'"))
            return

        # Create sub-folder for this file
        sub_dir = OUTPUT_DIR / stem
        sub_dir.mkdir(exist_ok=True)

        # Split into columns
        header_image, column_images = _split_columns(page_image, debug_dir=sub_dir)
        n_cols = len(column_images)
        print(yellow(f"  Detected {n_cols} column{'s' if n_cols != 1 else ''}"))

        sections = []

        # OCR each column
        for i, col_image in enumerate(column_images, 1):
            print(f"  Column {i}/{n_cols}: running tesseract...")
            text = pytesseract.image_to_string(col_image, lang=self.lang)
            col_file = sub_dir / f"column-{i}.txt"
            col_file.write_text(text, encoding="utf-8")
            print(green(f"  -> {col_file}"))
            sections.append(text)

        # Concatenate all sections into combined file
        combined_text = "\n\n".join(sections)
        combined_path = sub_dir / f"combined{self.output_suffix}"
        combined_path.write_text(combined_text + "\n", encoding="utf-8")
        print(green(f"  -> {combined_path}"))
