"""Tesseract+Claude hybrid engine — tesseract OCR with Claude text correction."""

import difflib

from engines._colors import green, yellow, red
from engines._correction import (
    has_claude_auth,
    get_claude_client,
    resolve_model,
    correct_ocr,
)
from engines.tesseract_engine import TesseractEngine


def _readable_diff(before, after):
    """Generate a human-readable list of changes between two texts.

    Shows each change with a few words of surrounding context, formatted as:
        ...context «old» → «new» context...
    """
    before_words = before.split()
    after_words = after.split()

    sm = difflib.SequenceMatcher(None, before_words, after_words)
    changes = []
    CONTEXT = 3  # words of context on each side

    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            continue

        # Surrounding context from the "before" side
        ctx_before = before_words[max(0, i1 - CONTEXT):i1]
        ctx_after = before_words[i2:i2 + CONTEXT]

        old = " ".join(before_words[i1:i2]) if i1 < i2 else ""
        new = " ".join(after_words[j1:j2]) if j1 < j2 else ""

        parts = []
        if ctx_before:
            parts.append("..." + " ".join(ctx_before))
        if op == "replace":
            parts.append(f"\u00ab{old}\u00bb \u2192 \u00ab{new}\u00bb")
        elif op == "delete":
            parts.append(f"\u00ab{old}\u00bb \u2192 (deleted)")
        elif op == "insert":
            parts.append(f"(inserted) \u00ab{new}\u00bb")
        if ctx_after:
            parts.append(" ".join(ctx_after) + "...")

        changes.append(" ".join(parts))

    if not changes:
        return "No changes detected."

    header = f"Claude correction changes ({len(changes)} edits):\n"
    return header + "\n".join(f"  {i+1}. {c}" for i, c in enumerate(changes))


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
        """Run tesseract OCR, then correct the reflowed output with Claude."""
        from pathlib import Path
        from engines.tesseract_engine import _reflow_text

        # Run tesseract pipeline (columns, OCR, combined.txt) but skip transcribed.txt
        super().process_file(file_path, _skip_transcribed=True)

        # Reflow and write pre-correction version
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

        # Reflow each section (split on double newlines matching combined.txt format)
        sections = combined_text.split("\n\n")
        reflowed_sections = [_reflow_text(s) for s in sections]
        reflowed_text = "\n\n".join(s for s in reflowed_sections if s)

        pre_claude_path = sub_dir / "transcribed-pre-claude.txt"
        pre_claude_path.write_text(reflowed_text + "\n", encoding="utf-8")
        print(green(f"  -> {pre_claude_path}"))

        # Set up Claude client and run correction on the reflowed text
        client = get_claude_client(self.region)
        model = resolve_model(client, self.model)
        corrected = correct_ocr(client, model, self.max_tokens, reflowed_text)

        if corrected:
            transcribed_path.write_text(corrected + "\n", encoding="utf-8")
            print(green(f"  -> {transcribed_path}"))

            # Write human-readable diff of what Claude changed
            changes_path = sub_dir / "correction-changes.txt"
            changes_path.write_text(
                _readable_diff(reflowed_text, corrected) + "\n", encoding="utf-8"
            )
            print(green(f"  -> {changes_path}"))
        else:
            # Correction failed — restore reflowed version as transcribed.txt
            pre_claude_path.rename(transcribed_path)
            print(yellow("  Correction failed — keeping reflowed output."))
