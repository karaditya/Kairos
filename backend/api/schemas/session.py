"""Session Pydantic schemas for CAE API."""

from typing import Optional, Dict, Any, List
from datetime import datetime
from pydantic import BaseModel, Field


class SessionStartRequest(BaseModel):
    """Request to start a new clinical session."""
    patient_id: Optional[str] = None
    ehr_window_title: Optional[str] = None
    language: str = Field(default="fr", pattern="^(fr|en)$")


class SessionStartResponse(BaseModel):
    """Response after starting a session."""
    session_id: str
    websocket_url: str
    status: str


class TranscriptData(BaseModel):
    """Transcript data within session state."""
    text: str
    segments: List[Dict[str, Any]]
    keywords: List[str]


class EHRData(BaseModel):
    """EHR extracted data within session state."""
    fields: Dict[str, Any]
    extracted_at: Optional[str] = None


class SessionStateResponse(BaseModel):
    """Full session state response."""
    session_id: str
    status: str
    language: str
    patient_id: Optional[str] = None
    transcript: Optional[TranscriptData] = None
    ehr_data: Optional[EHRData] = None
    created_at: str
    updated_at: str


class SessionListItem(BaseModel):
    """Session item for list responses."""
    id: str
    patient_id: Optional[str]
    language: str
    status: str
    created_at: str


class SessionListResponse(BaseModel):
    """List of sessions response."""
    sessions: List[SessionListItem]
    total: int
