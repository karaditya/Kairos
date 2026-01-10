"""
OCR Fallback Service for CAE System.

Provides fallback OCR using EasyOCR or Tesseract when DeepSeek-VL2 is unavailable.
Lightweight alternative for text verification tasks.
"""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import asyncio

from PIL import Image

from utils.logging import get_logger
from utils.exceptions import VisionError

logger = get_logger(__name__)


@dataclass
class OCRResult:
    """OCR extraction result."""
    text: str
    confidence: float
    method: str
    bboxes: List[Dict[str, Any]]


class OCRFallback:
    """
    Fallback OCR service using EasyOCR or Tesseract.

    Priority:
    1. EasyOCR (better accuracy, especially for medical text)
    2. Tesseract (widely available, fast)
    3. Basic PIL text extraction (last resort)
    """

    def __init__(self):
        self._easyocr_reader = None
        self._tesseract_available = False
        self._initialized = False
        self._method = None

    async def initialize(self) -> None:
        """Initialize the best available OCR engine."""
        if self._initialized:
            return

        # Try EasyOCR first
        try:
            import easyocr
            # Initialize reader for French and English
            self._easyocr_reader = easyocr.Reader(
                ['fr', 'en'],
                gpu=self._check_gpu_available(),
                verbose=False,
            )
            self._method = "easyocr"
            self._initialized = True
            logger.info("OCR fallback initialized with EasyOCR")
            return
        except ImportError:
            logger.info("EasyOCR not available, trying Tesseract")
        except Exception as e:
            logger.warning("EasyOCR initialization failed", error=str(e))

        # Try Tesseract
        try:
            import pytesseract
            # Test if tesseract is installed
            pytesseract.get_tesseract_version()
            self._tesseract_available = True
            self._method = "tesseract"
            self._initialized = True
            logger.info("OCR fallback initialized with Tesseract")
            return
        except ImportError:
            logger.info("pytesseract not available")
        except Exception as e:
            logger.warning("Tesseract not available", error=str(e))

        # No OCR available
        self._method = "none"
        self._initialized = True
        logger.warning("No OCR engine available - verification will be limited")

    def _check_gpu_available(self) -> bool:
        """Check if GPU is available for EasyOCR."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    @property
    def is_ready(self) -> bool:
        """Check if OCR is available."""
        return self._initialized and self._method != "none"

    @property
    def method(self) -> str:
        """Get the active OCR method."""
        return self._method or "none"

    async def extract_text(
        self,
        image: Image.Image,
        language: str = "fr",
    ) -> Dict[str, Any]:
        """
        Extract text from image using available OCR engine.

        Args:
            image: PIL Image to process
            language: Primary language hint ("fr" or "en")

        Returns:
            Dict with:
                - text: Extracted text
                - confidence: Average confidence score
                - method: OCR method used
                - bboxes: List of detected text regions with coordinates
        """
        if not self._initialized:
            await self.initialize()

        if self._method == "easyocr":
            return await self._extract_with_easyocr(image, language)
        elif self._method == "tesseract":
            return await self._extract_with_tesseract(image, language)
        else:
            return {
                "text": "",
                "confidence": 0.0,
                "method": "none",
                "bboxes": [],
                "error": "No OCR engine available",
            }

    async def _extract_with_easyocr(
        self,
        image: Image.Image,
        language: str,
    ) -> Dict[str, Any]:
        """Extract text using EasyOCR."""
        try:
            import numpy as np

            # Convert PIL to numpy array
            img_array = np.array(image)

            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                lambda: self._easyocr_reader.readtext(img_array),
            )

            # Parse results
            texts = []
            confidences = []
            bboxes = []

            for (bbox, text, confidence) in results:
                texts.append(text)
                confidences.append(confidence)

                # Convert bbox to [x, y, width, height]
                x_coords = [p[0] for p in bbox]
                y_coords = [p[1] for p in bbox]
                bboxes.append({
                    "text": text,
                    "confidence": confidence,
                    "bbox": [
                        min(x_coords),
                        min(y_coords),
                        max(x_coords) - min(x_coords),
                        max(y_coords) - min(y_coords),
                    ],
                })

            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

            return {
                "text": " ".join(texts),
                "confidence": avg_confidence,
                "method": "easyocr",
                "bboxes": bboxes,
            }

        except Exception as e:
            logger.error("EasyOCR extraction failed", error=str(e))
            return {
                "text": "",
                "confidence": 0.0,
                "method": "easyocr",
                "bboxes": [],
                "error": str(e),
            }

    async def _extract_with_tesseract(
        self,
        image: Image.Image,
        language: str,
    ) -> Dict[str, Any]:
        """Extract text using Tesseract."""
        try:
            import pytesseract

            # Map language code to Tesseract language
            lang_map = {
                "fr": "fra",
                "en": "eng",
            }
            tess_lang = lang_map.get(language, "fra+eng")

            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()

            # Get text
            text = await loop.run_in_executor(
                None,
                lambda: pytesseract.image_to_string(image, lang=tess_lang),
            )

            # Get detailed data for confidence and bboxes
            data = await loop.run_in_executor(
                None,
                lambda: pytesseract.image_to_data(image, lang=tess_lang, output_type=pytesseract.Output.DICT),
            )

            # Calculate average confidence (filter out -1 values)
            confidences = [c for c in data.get("conf", []) if c != -1 and c > 0]
            avg_confidence = sum(confidences) / len(confidences) / 100 if confidences else 0.5

            # Build bboxes
            bboxes = []
            n_boxes = len(data.get("text", []))
            for i in range(n_boxes):
                if data["text"][i].strip():
                    bboxes.append({
                        "text": data["text"][i],
                        "confidence": data["conf"][i] / 100 if data["conf"][i] != -1 else 0.5,
                        "bbox": [
                            data["left"][i],
                            data["top"][i],
                            data["width"][i],
                            data["height"][i],
                        ],
                    })

            return {
                "text": text.strip(),
                "confidence": avg_confidence,
                "method": "tesseract",
                "bboxes": bboxes,
            }

        except Exception as e:
            logger.error("Tesseract extraction failed", error=str(e))
            return {
                "text": "",
                "confidence": 0.0,
                "method": "tesseract",
                "bboxes": [],
                "error": str(e),
            }

    async def verify_text_present(
        self,
        image: Image.Image,
        expected_text: str,
        case_sensitive: bool = False,
    ) -> Dict[str, Any]:
        """
        Verify that expected text is present in image.

        Args:
            image: PIL Image to check
            expected_text: Text to look for
            case_sensitive: Whether to match case

        Returns:
            Dict with:
                - found: Whether text was found
                - confidence: Confidence score
                - method: OCR method used
                - match_location: Bbox where match was found (if any)
        """
        result = await self.extract_text(image)

        extracted = result["text"]
        if not case_sensitive:
            extracted = extracted.lower()
            expected_text = expected_text.lower()

        found = expected_text in extracted

        # Find specific bbox containing the match
        match_location = None
        if found and result["bboxes"]:
            for bbox_info in result["bboxes"]:
                bbox_text = bbox_info["text"]
                if not case_sensitive:
                    bbox_text = bbox_text.lower()
                if expected_text in bbox_text:
                    match_location = bbox_info["bbox"]
                    break

        return {
            "found": found,
            "confidence": result["confidence"],
            "method": result["method"],
            "match_location": match_location,
            "extracted_text_length": len(result["text"]),
        }


# =============================================================================
# SINGLETON
# =============================================================================

_ocr_fallback: Optional[OCRFallback] = None


def get_ocr_fallback() -> OCRFallback:
    """Get singleton OCR fallback service."""
    global _ocr_fallback
    if _ocr_fallback is None:
        _ocr_fallback = OCRFallback()
    return _ocr_fallback


async def initialize_ocr_fallback() -> OCRFallback:
    """Initialize and return OCR fallback service."""
    service = get_ocr_fallback()
    await service.initialize()
    return service


async def get_ocr_status() -> str:
    """Get OCR fallback service status."""
    global _ocr_fallback
    if _ocr_fallback is None:
        return "not_initialized"
    if not _ocr_fallback._initialized:
        return "not_initialized"
    if _ocr_fallback.is_ready:
        return f"ready ({_ocr_fallback.method})"
    return "no_engine"
