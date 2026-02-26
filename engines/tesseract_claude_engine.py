"""Tesseract+Claude hybrid engine — tesseract OCR with Claude text correction."""

from engines._colors import green, yellow, red
from engines._correction import (
    has_claude_auth,
    get_claude_client,
    resolve_model,
    correct_ocr,
)
from engines.tesseract_engine import TesseractEngine


class TesseractClaudeEngine(TesseractEngine):
    name = "tesseract+claude"

    def __init__(self, dpi=300, lang="nor", model="claude-opus-4-20250514",
                 region="eu-north-1", max_tokens=16384):
        super().__init__(dpi=dpi, lang=lang)
        self.model = model
        self.region = region
        self.max_tokens = max_tokens
        self._model_short = model.split("-")[1] if "-" in model else model
        self.output_suffix = f".tesseract+claude-{dpi}dpi-{self._model_short}.txt"
        self.output_dir_name = f"tesseract+claude-{dpi}dpi-{self._model_short}"

    def check_dependencies(self):
        """Return list of missing dependencies (tesseract + Claude auth)."""
        missing = super().check_dependencies()
        if not has_claude_auth():
            missing.append("Claude API credentials (set ANTHROPIC_API_KEY or configure AWS credentials)")
        return missing

    def process_file(self, file_path):
        """Run tesseract OCR, then correct the output with Claude."""
        # Run the full tesseract pipeline (columns, OCR, combined.txt, transcribed.txt)
        super().process_file(file_path)

        # Read the combined tesseract output
        from pathlib import Path
        stem = Path(file_path).stem
        sub_dir = Path(__file__).resolve().parent.parent / "output" / stem / self.output_dir_name
        combined_path = sub_dir / "combined.txt"
        if not combined_path.exists():
            print(red("  No combined.txt found — skipping correction."))
            return

        combined_text = combined_path.read_text(encoding="utf-8").strip()
        if not combined_text:
            print(yellow("  combined.txt is empty — skipping correction."))
            return

        # Set up Claude client and run correction
        client = get_claude_client(self.region)
        model = resolve_model(client, self.model)

        corrected = correct_ocr(client, model, self.max_tokens, combined_text)

        corrected_path = sub_dir / "combined.corrected.txt"
        if corrected:
            corrected_path.write_text(corrected + "\n", encoding="utf-8")
            print(green(f"  -> {corrected_path}"))

        # Overwrite transcribed.txt with corrected output (or keep reflowed version)
        best_text = corrected if corrected else combined_text
        transcribed_path = sub_dir / "transcribed.txt"
        transcribed_path.write_text(best_text + "\n", encoding="utf-8")
        print(green(f"  -> {transcribed_path} (corrected)"))
