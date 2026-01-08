"""
Vision API Routes for CAE System.

Endpoints:
- POST /vision/scrape - Capture and extract EHR data
- POST /vision/calibrate - Save coordinate mapping
- GET /vision/windows - List available windows
"""

from typing import Optional, List

from fastapi import APIRouter, HTTPException, status

from api.schemas.vision import (
    VisionScrapeRequest,
    VisionScrapeResponse,
    VisionCalibrateRequest,
    VisionCalibrateResponse,
    ExtractedField,
    UIElement,
)
from services.vision.ehr_extractor import get_ehr_extractor
from services.rpa.coordinate_map import get_coordinate_manager, CoordinateMap, FieldCoordinate
from config import VISION_ENABLED
from utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.post("/scrape", response_model=VisionScrapeResponse)
async def scrape_ehr_screen(request: VisionScrapeRequest):
    """
    Capture EHR screen and extract data using DeepSeek-VL2.

    Returns extracted fields and detected UI elements.
    """
    if not VISION_ENABLED:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vision service disabled",
        )

    extractor = await get_ehr_extractor()

    # Convert region if provided
    region = None
    if request.region:
        region = (
            request.region.x,
            request.region.y,
            request.region.width,
            request.region.height,
        )

    # Extract
    result = await extractor.extract_from_screen(
        window_title=request.window_title,
        region=region,
        extract_fields=request.extract_fields if request.extract_fields else None,
    )

    # Convert to response
    extracted_data = {
        field_name: ExtractedField(
            value=str(data.get("value", "")),
            confidence=data.get("confidence", 0.0),
            source=data.get("source", "vision"),
        )
        for field_name, data in result.extracted_data.items()
    }

    ui_elements = [
        UIElement(
            type=elem.get("type", "input"),
            label=elem.get("label", ""),
            bbox=elem.get("bbox", [0, 0, 100, 30]),
        )
        for elem in result.ui_elements
    ]

    logger.info(
        "Vision scrape completed",
        session_id=request.session_id,
        fields_extracted=len(extracted_data),
        ui_elements=len(ui_elements),
    )

    return VisionScrapeResponse(
        success=True,
        extracted_data=extracted_data,
        ui_elements=ui_elements,
        screenshot_id=result.screenshot_id,
        raw_text=result.raw_text,
    )


@router.post("/calibrate", response_model=VisionCalibrateResponse)
async def calibrate_coordinates(request: VisionCalibrateRequest):
    """
    Save coordinate mapping for EHR fields.

    Used to configure where RPA should click/type.
    """
    import uuid

    coord_manager = get_coordinate_manager()

    # Build coordinate map
    fields = {}
    for mapping in request.field_mappings:
        fields[mapping.field_name] = FieldCoordinate(
            field_name=mapping.field_name,
            x=mapping.bbox[0],
            y=mapping.bbox[1],
            width=mapping.bbox[2],
            height=mapping.bbox[3],
        )

    coord_map = CoordinateMap(
        id=str(uuid.uuid4()),
        ehr_type=request.ehr_type,
        name=request.name,
        fields=fields,
    )

    # Save
    map_id = await coord_manager.save_map(coord_map)

    logger.info(
        "Coordinate map saved",
        map_id=map_id,
        ehr_type=request.ehr_type,
        fields=len(fields),
    )

    return VisionCalibrateResponse(
        success=True,
        coordinate_map_id=map_id,
        message=f"Saved {len(fields)} field coordinates",
    )


@router.get("/windows")
async def list_windows():
    """List available windows for capture."""
    extractor = await get_ehr_extractor()
    windows = await extractor.get_available_windows()

    return {"windows": windows}


@router.get("/coordinate-maps")
async def list_coordinate_maps(ehr_type: Optional[str] = None):
    """List saved coordinate mappings."""
    coord_manager = get_coordinate_manager()
    maps = await coord_manager.list_maps(ehr_type)

    return {"maps": maps}


@router.get("/coordinate-maps/{ehr_type}/{name}")
async def get_coordinate_map(ehr_type: str, name: str):
    """Get a specific coordinate mapping."""
    coord_manager = get_coordinate_manager()
    coord_map = await coord_manager.load_map(ehr_type, name)

    if not coord_map:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Coordinate map not found: {ehr_type}/{name}",
        )

    return coord_map.to_dict()


@router.delete("/coordinate-maps/{ehr_type}/{name}")
async def delete_coordinate_map(ehr_type: str, name: str):
    """Delete a coordinate mapping."""
    coord_manager = get_coordinate_manager()
    deleted = await coord_manager.delete_map(ehr_type, name)

    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Coordinate map not found: {ehr_type}/{name}",
        )

    return {"deleted": True}


@router.post("/detect-layout")
async def detect_ehr_layout(window_title: Optional[str] = None):
    """
    Auto-detect EHR layout and UI elements.

    Returns detected elements that can be used for calibration.
    """
    if not VISION_ENABLED:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vision service disabled",
        )

    extractor = await get_ehr_extractor()
    layout = await extractor.detect_ehr_layout(window_title)

    return layout
