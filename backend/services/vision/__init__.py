"""Vision Services for EHR Screen Capture and OCR."""

from services.vision.screen_capture import ScreenCapture, get_screen_capture
from services.vision.deepseek_vl2 import (
    DeepSeekVL2,
    create_vision_service,
    shutdown_vision,
    get_vision_status,
)
from services.vision.ehr_extractor import EHRExtractor
from services.vision.ocr_fallback import (
    OCRFallback,
    get_ocr_fallback,
    initialize_ocr_fallback,
    get_ocr_status,
)

__all__ = [
    # Screen Capture
    "ScreenCapture",
    "get_screen_capture",
    # DeepSeek-VL2 Vision
    "DeepSeekVL2",
    "create_vision_service",
    "shutdown_vision",
    "get_vision_status",
    # EHR Extraction
    "EHRExtractor",
    # OCR Fallback
    "OCRFallback",
    "get_ocr_fallback",
    "initialize_ocr_fallback",
    "get_ocr_status",
]
