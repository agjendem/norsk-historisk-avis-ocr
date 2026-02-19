"""Engine registry for OCR engines (lazy-loaded to avoid import errors before setup)."""


def _get_engines():
    from engines.tesseract_engine import TesseractEngine
    from engines.claude_vision_engine import ClaudeVisionEngine

    return {
        "tesseract": TesseractEngine,
        "claude-vision": ClaudeVisionEngine,
    }


ENGINE_NAMES = ["tesseract", "claude-vision"]
