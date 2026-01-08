"""Vision Pydantic schemas for CAE API."""

from typing import Optional, Dict, List, Any
from pydantic import BaseModel


class RegionRequest(BaseModel):
    """Screen region for capture."""
    x: int
    y: int
    width: int
    height: int


class VisionScrapeRequest(BaseModel):
    """Request to scrape EHR screen."""
    session_id: str
    window_title: Optional[str] = None
    region: Optional[RegionRequest] = None
    extract_fields: List[str] = []


class ExtractedField(BaseModel):
    """Extracted field from vision."""
    value: str
    confidence: float
    source: str = "vision"


class UIElement(BaseModel):
    """Detected UI element."""
    type: str  # "input", "button", "dropdown", "text"
    label: str
    bbox: List[int]  # [x, y, width, height]


class VisionScrapeResponse(BaseModel):
    """Response from vision scrape."""
    success: bool
    extracted_data: Dict[str, ExtractedField]
    ui_elements: List[UIElement]
    screenshot_id: str
    raw_text: Optional[str] = None


class FieldMapping(BaseModel):
    """Field coordinate mapping for calibration."""
    field_name: str
    bbox: List[int]  # [x, y, width, height]


class VisionCalibrateRequest(BaseModel):
    """Request to calibrate EHR coordinate mapping."""
    ehr_type: str = "custom"
    name: str
    field_mappings: List[FieldMapping]


class VisionCalibrateResponse(BaseModel):
    """Response after calibration."""
    success: bool
    coordinate_map_id: str
    message: str
