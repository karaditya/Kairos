"""
EHR Extractor Service for CAE System.

Orchestrates the vision pipeline:
1. Screen capture
2. DeepSeek-VL2 extraction
3. Structured JSON output
"""

from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import uuid

from PIL import Image

from services.vision.screen_capture import ScreenCapture, CaptureResult, get_screen_capture
from services.vision.deepseek_vl2 import DeepSeekVL2, ExtractionResult, create_vision_service
from utils.logging import get_logger
from utils.exceptions import VisionError

logger = get_logger(__name__)


@dataclass
class EHRExtractionResult:
    """Complete EHR extraction result."""
    screenshot_id: str
    extracted_at: datetime
    extracted_data: Dict[str, Dict[str, Any]]  # field -> {value, confidence, source}
    ui_elements: List[Dict[str, Any]]
    raw_text: str
    window_title: Optional[str]
    image_dimensions: Tuple[int, int]


class EHRExtractor:
    """
    Orchestrates EHR screen capture and data extraction.

    Combines screen capture and vision model to extract
    structured data from legacy EHR systems.
    """

    def __init__(
        self,
        screen_capture: ScreenCapture,
        vision_model: DeepSeekVL2,
    ):
        self.screen_capture = screen_capture
        self.vision_model = vision_model

    async def extract_from_screen(
        self,
        window_title: Optional[str] = None,
        region: Optional[Tuple[int, int, int, int]] = None,
        extract_fields: Optional[List[str]] = None,
    ) -> EHRExtractionResult:
        """
        Capture screen and extract EHR data.

        Args:
            window_title: Optional window to capture
            region: Optional screen region (x, y, w, h)
            extract_fields: Specific fields to extract

        Returns:
            EHRExtractionResult with extracted data
        """
        # 1. Capture screen
        if window_title:
            capture = await self.screen_capture.capture_window(window_title)
            if capture is None:
                capture = await self.screen_capture.capture_screen(region=region)
        else:
            capture = await self.screen_capture.capture_screen(region=region)

        # 2. Extract with vision model
        extraction = await self.vision_model.extract_from_image(
            capture.image,
            extract_fields=extract_fields,
        )

        # 3. Build result
        return EHRExtractionResult(
            screenshot_id=capture.screenshot_id,
            extracted_at=capture.timestamp,
            extracted_data=extraction.extracted_fields,
            ui_elements=extraction.ui_elements,
            raw_text=extraction.raw_text,
            window_title=window_title,
            image_dimensions=(capture.width, capture.height),
        )

    async def extract_from_image(
        self,
        image: Image.Image,
        extract_fields: Optional[List[str]] = None,
    ) -> EHRExtractionResult:
        """
        Extract EHR data from provided image.

        Args:
            image: PIL Image
            extract_fields: Specific fields to extract

        Returns:
            EHRExtractionResult with extracted data
        """
        extraction = await self.vision_model.extract_from_image(
            image,
            extract_fields=extract_fields,
        )

        return EHRExtractionResult(
            screenshot_id=str(uuid.uuid4()),
            extracted_at=datetime.utcnow(),
            extracted_data=extraction.extracted_fields,
            ui_elements=extraction.ui_elements,
            raw_text=extraction.raw_text,
            window_title=None,
            image_dimensions=(image.width, image.height),
        )

    async def detect_ehr_layout(
        self,
        window_title: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Detect EHR layout and UI elements for calibration.

        Args:
            window_title: Optional window to capture

        Returns:
            Dict with detected UI elements and layout info
        """
        # Capture screen
        if window_title:
            capture = await self.screen_capture.capture_window(window_title)
            if capture is None:
                capture = await self.screen_capture.capture_screen()
        else:
            capture = await self.screen_capture.capture_screen()

        # Detect UI elements
        ui_elements = await self.vision_model.detect_ui_elements(capture.image)

        return {
            "screenshot_id": capture.screenshot_id,
            "window_title": window_title,
            "dimensions": {
                "width": capture.width,
                "height": capture.height,
            },
            "ui_elements": ui_elements,
            "detected_at": capture.timestamp.isoformat(),
        }

    async def get_available_windows(self) -> List[Dict[str, Any]]:
        """Get list of available windows for capture."""
        return await self.screen_capture.get_window_list()

    def result_to_dict(self, result: EHRExtractionResult) -> Dict[str, Any]:
        """Convert extraction result to JSON-serializable dict."""
        return {
            "screenshot_id": result.screenshot_id,
            "extracted_at": result.extracted_at.isoformat(),
            "extracted_data": result.extracted_data,
            "ui_elements": result.ui_elements,
            "raw_text": result.raw_text,
            "window_title": result.window_title,
            "image_dimensions": {
                "width": result.image_dimensions[0],
                "height": result.image_dimensions[1],
            },
        }


# =============================================================================
# FACTORY
# =============================================================================

_ehr_extractor: Optional[EHRExtractor] = None


async def get_ehr_extractor() -> EHRExtractor:
    """Get singleton EHR extractor."""
    global _ehr_extractor
    if _ehr_extractor is None:
        screen_capture = get_screen_capture()
        vision_model = await create_vision_service()
        _ehr_extractor = EHRExtractor(screen_capture, vision_model)
    return _ehr_extractor
