"""Engine registry for OCR engines."""

from engines.tesseract_engine import TesseractEngine
from engines.claude_vision_engine import ClaudeVisionEngine

ENGINES = {
    "tesseract": TesseractEngine,
    "claude-vision": ClaudeVisionEngine,
}
