"""Tesseract OCR engine — cross-platform wrapper using pytesseract and pdf2image."""

import re
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


def _clean_divider_noise(text):
    """Remove column divider pipe artifacts from tesseract output.

    The overlap padding (20px per side) causes pipe characters to appear at
    line edges and mid-line, often followed/preceded by 1-3 garbage characters
    from the adjacent column bleeding through. This cleaner handles:
    - Trailing pipes: strip from | to end of line (adjacent column bleed)
    - Leading pipes: strip from start of line to | (other side bleed)
    - Standalone leading/trailing pipes (fallback)
    After cleaning, lines that are empty or contain only 1-2 characters are
    removed (stray garbage), and runs of 3+ blank lines are collapsed to 1.
    """
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        # Trailing pipe + optional garbage: "text | s", "text|g", "hadde | k"
        line = re.sub(r"\s*\|.{0,3}$", "", line)
        # Leading pipe + optional garbage: "-| text", ".| text", '"|sjonene'
        line = re.sub(r"^.{0,3}\|\s*", "", line)
        # Fallback: standalone leading/trailing pipes
        line = line.strip("|")
        line = line.rstrip()
        # Skip lines that became near-empty (1-2 chars = stray garbage).
        # Don't append them at all — replacing with "" would leave blank lines
        # that fragment the text and prevent hyphen-rejoining in reflow.
        stripped = line.strip()
        if len(stripped) > 0 and len(stripped) <= 2:
            continue
        cleaned.append(line)

    # Collapse runs of 2+ blank lines down to 1
    result = []
    blank_count = 0
    for line in cleaned:
        if line.strip() == "":
            blank_count += 1
            if blank_count <= 1:
                result.append(line)
        else:
            blank_count = 0
            result.append(line)
    return "\n".join(result)


def _reflow_text(section_text):
    """Convert a single OCR section (one column or header) into flowing text.

    Processes one section at a time. Tesseract output from overlapping column
    crops contains many spurious blank lines (from adjacent column bleed-through)
    that are indistinguishable from real paragraph breaks, so we collapse all
    whitespace into continuous flowing text:
    - Rejoins hyphenated line-break words: "word-\\ncontinuation" → "wordcontinuation"
    - Collapses all newlines (single and blank-line-separated) to spaces
    """
    section_text = section_text.strip()
    if not section_text:
        return ""

    # Join hyphenated line-break words: "word-\ncontinuation" → "wordcontinuation"
    section_text = re.sub(r"-\n\s*", "", section_text)
    # Collapse all newlines to spaces
    section_text = re.sub(r"\n+", " ", section_text)
    # Clean up multiple spaces
    section_text = re.sub(r"  +", " ", section_text)
    return section_text.strip()


class TesseractEngine:
    name = "tesseract"

    def __init__(self, dpi=300, lang="nor"):
        self.dpi = dpi
        self.lang = lang
        self.output_suffix = f".tesseract-{dpi}dpi.txt"
        self.output_dir_name = f"tesseract-{dpi}dpi"

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

        # Create sub-folder for this file: output/{stem}/{output_dir_name}/
        sub_dir = OUTPUT_DIR / stem / self.output_dir_name
        sub_dir.mkdir(parents=True, exist_ok=True)

        # Split into columns
        header_image, column_images = _split_columns(page_image, debug_dir=sub_dir)
        n_cols = len(column_images)
        print(yellow(f"  Detected {n_cols} column{'s' if n_cols != 1 else ''}"))

        sections = []

        # OCR header if present
        if header_image:
            print(f"  Header: running tesseract...")
            text = pytesseract.image_to_string(header_image, lang=self.lang)
            text = _clean_divider_noise(text)
            header_file = sub_dir / "header.txt"
            header_file.write_text(text, encoding="utf-8")
            print(green(f"  -> {header_file}"))
            sections.append(text)

        # OCR each column
        for i, col_image in enumerate(column_images, 1):
            print(f"  Column {i}/{n_cols}: running tesseract...")
            text = pytesseract.image_to_string(col_image, lang=self.lang)
            text = _clean_divider_noise(text)
            col_file = sub_dir / f"column-{i}.txt"
            col_file.write_text(text, encoding="utf-8")
            print(green(f"  -> {col_file}"))
            sections.append(text)

        # Concatenate all sections into combined file
        combined_text = "\n\n".join(sections)
        combined_path = sub_dir / "combined.txt"
        combined_path.write_text(combined_text + "\n", encoding="utf-8")
        print(green(f"  -> {combined_path}"))

        # Write flowing text output (reflow each section independently)
        reflowed_sections = [_reflow_text(s) for s in sections]
        transcribed_text = "\n\n".join(s for s in reflowed_sections if s)
        transcribed_path = sub_dir / "transcribed.txt"
        transcribed_path.write_text(transcribed_text + "\n", encoding="utf-8")
        print(green(f"  -> {transcribed_path}"))
