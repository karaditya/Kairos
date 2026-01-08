"""Common Pydantic schemas for CAE API."""

from typing import Optional, Dict, Any
from pydantic import BaseModel


class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str
    message: str
    details: Optional[Dict[str, Any]] = None


class SuccessResponse(BaseModel):
    """Standard success response."""
    success: bool = True
    message: Optional[str] = None


class StatusResponse(BaseModel):
    """Service status response."""
    status: str
    service: str
    version: str


class ServiceStatusResponse(BaseModel):
    """Detailed service status response."""
    database: str
    parlant: str
    whisper: str
    vision: str
    qdrant: str
