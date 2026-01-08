"""
DeepSeek-VL2 Vision Model for CAE System.

Provides semantic extraction from EHR screenshots using
DeepSeek-VL2 via HuggingFace transformers.
"""

from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import json
import re

from PIL import Image

from config import DEEPSEEK_VL2_MODEL, DEEPSEEK_DEVICE, VISION_ENABLED
from utils.logging import get_logger
from utils.exceptions import VisionModelError

logger = get_logger(__name__)


@dataclass
class ExtractionResult:
    """Result from vision extraction."""
    extracted_fields: Dict[str, Any]
    raw_text: str
    confidence: float
    ui_elements: List[Dict[str, Any]]


class DeepSeekVL2:
    """DeepSeek-VL2 vision-language model for EHR extraction."""

    def __init__(
        self,
        model_name: str = DEEPSEEK_VL2_MODEL,
        device: str = DEEPSEEK_DEVICE,
    ):
        self.model_name = model_name
        self.device = device
        self._model = None
        self._processor = None
        self._tokenizer = None
        self._initialized = False

    async def initialize(self) -> None:
        """Load DeepSeek-VL2 model."""
        if self._initialized:
            return

        if not VISION_ENABLED:
            logger.info("Vision service disabled by config")
            return

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoProcessor

            # Determine device
            device = self.device
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"

            logger.info(
                "Loading DeepSeek-VL2 model",
                model=self.model_name,
                device=device,
            )

            # Load processor
            self._processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=True,
            )

            # Load model
            dtype = torch.float16 if device == "cuda" else torch.float32
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=dtype,
                device_map=device if device == "cuda" else None,
            )

            if device == "cpu":
                self._model = self._model.to(device)

            self._model.eval()
            self._initialized = True

            logger.info("DeepSeek-VL2 model loaded")

        except ImportError as e:
            raise VisionModelError(f"Missing dependencies: {str(e)}")
        except Exception as e:
            raise VisionModelError(f"Failed to load model: {str(e)}")

    @property
    def is_ready(self) -> bool:
        return self._initialized and self._model is not None

    async def extract_from_image(
        self,
        image: Image.Image,
        prompt: Optional[str] = None,
        extract_fields: Optional[List[str]] = None,
    ) -> ExtractionResult:
        """
        Extract structured data from image.

        Args:
            image: PIL Image of EHR screen
            prompt: Custom extraction prompt
            extract_fields: Specific fields to extract

        Returns:
            ExtractionResult with extracted data
        """
        if not self.is_ready:
            await self.initialize()

        if not self.is_ready:
            # Return mock result if vision disabled
            return self._mock_extraction(extract_fields)

        # Build extraction prompt
        if prompt is None:
            prompt = self._build_extraction_prompt(extract_fields)

        try:
            import torch

            # Process image
            inputs = self._processor(
                text=prompt,
                images=image,
                return_tensors="pt",
            )

            # Move to device
            if self._model.device.type == "cuda":
                inputs = {k: v.cuda() for k, v in inputs.items()}

            # Generate
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    do_sample=False,
                    pad_token_id=self._processor.tokenizer.eos_token_id,
                )

            # Decode
            response = self._processor.decode(
                outputs[0],
                skip_special_tokens=True,
            )

            # Parse response
            return self._parse_extraction_response(response, extract_fields)

        except Exception as e:
            logger.error("Vision extraction failed", error=str(e))
            raise VisionModelError(f"Extraction failed: {str(e)}")

    def _build_extraction_prompt(
        self,
        extract_fields: Optional[List[str]] = None,
    ) -> str:
        """Build prompt for EHR data extraction."""
        base_prompt = """Analyze this medical software screenshot and extract the following information.

Return the data as a JSON object with these fields:
- patient_name: Patient's full name
- patient_dob: Date of birth
- patient_mrn: Medical record number
- chief_complaint: Main reason for visit
- vitals: Any vital signs visible (BP, HR, Temp, SpO2)
- allergies: Listed allergies
- medications: Current medications
- notes_field_location: Coordinates [x, y, width, height] of the main notes/text input field
- save_button_location: Coordinates of the save/submit button

"""
        if extract_fields:
            base_prompt += f"\nFocus specifically on these fields: {', '.join(extract_fields)}\n"

        base_prompt += """
Output ONLY valid JSON. Example:
{
    "patient_name": "Jean Dupont",
    "patient_dob": "1985-03-15",
    "chief_complaint": "Chest pain",
    "vitals": {"bp": "120/80", "hr": "72"},
    "notes_field_location": [100, 200, 400, 150],
    "confidence": 0.85
}
"""
        return base_prompt

    def _parse_extraction_response(
        self,
        response: str,
        extract_fields: Optional[List[str]] = None,
    ) -> ExtractionResult:
        """Parse model response into structured result."""
        # Try to extract JSON from response
        extracted_fields = {}
        ui_elements = []
        confidence = 0.5

        try:
            # Find JSON in response
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())

                # Extract confidence if present
                confidence = data.pop("confidence", 0.8)

                # Extract UI element locations
                for key in ["notes_field_location", "save_button_location"]:
                    if key in data and data[key]:
                        bbox = data.pop(key)
                        ui_elements.append({
                            "type": "input" if "notes" in key else "button",
                            "label": key.replace("_location", "").replace("_", " "),
                            "bbox": bbox,
                        })

                # Remaining fields are extracted data
                for key, value in data.items():
                    if value:
                        extracted_fields[key] = {
                            "value": value,
                            "confidence": confidence,
                            "source": "vision",
                        }

        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON from model response")

        return ExtractionResult(
            extracted_fields=extracted_fields,
            raw_text=response,
            confidence=confidence,
            ui_elements=ui_elements,
        )

    def _mock_extraction(
        self,
        extract_fields: Optional[List[str]] = None,
    ) -> ExtractionResult:
        """Return mock result when vision is disabled."""
        return ExtractionResult(
            extracted_fields={},
            raw_text="[Vision service disabled]",
            confidence=0.0,
            ui_elements=[],
        )

    async def detect_ui_elements(
        self,
        image: Image.Image,
    ) -> List[Dict[str, Any]]:
        """
        Detect UI elements (input fields, buttons) in image.

        Args:
            image: PIL Image of EHR screen

        Returns:
            List of detected UI elements with bounding boxes
        """
        prompt = """Identify all interactive UI elements in this screenshot.
For each element, provide:
- type: "input", "button", "dropdown", "checkbox"
- label: The text label or placeholder
- bbox: [x, y, width, height] coordinates

Return as JSON array:
[{"type": "input", "label": "Patient Name", "bbox": [10, 20, 200, 30]}, ...]
"""

        result = await self.extract_from_image(image, prompt=prompt)
        return result.ui_elements

    async def shutdown(self) -> None:
        """Unload model and free memory."""
        if self._model:
            del self._model
            self._model = None

        if self._processor:
            del self._processor
            self._processor = None

        self._initialized = False

        # Force garbage collection
        import gc
        gc.collect()

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        logger.info("DeepSeek-VL2 model unloaded")


# =============================================================================
# SINGLETON
# =============================================================================

_vision_service: Optional[DeepSeekVL2] = None


async def create_vision_service() -> DeepSeekVL2:
    """Create and initialize vision service."""
    global _vision_service
    if _vision_service is None:
        _vision_service = DeepSeekVL2()
        await _vision_service.initialize()
    return _vision_service


async def shutdown_vision() -> None:
    """Shutdown vision service."""
    global _vision_service
    if _vision_service:
        await _vision_service.shutdown()
        _vision_service = None


async def get_vision_status() -> str:
    """Get vision service status."""
    global _vision_service
    if not VISION_ENABLED:
        return "disabled"
    if _vision_service is None:
        return "not_initialized"
    if _vision_service.is_ready:
        return "ready"
    return "loading"
