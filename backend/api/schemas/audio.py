"""Audio Pydantic schemas for CAE API."""

from typing import Optional, List
from pydantic import BaseModel


class TranscriptSegment(BaseModel):
    """Single transcript segment."""
    text: str
    start_time: float
    end_time: float
    confidence: float
    speaker: Optional[str] = None
    keywords: List[str] = []


class TranscriptResponse(BaseModel):
    """Full transcript response."""
    transcript_id: str
    session_id: str
    language: str
    full_text: str
    segments: List[TranscriptSegment]
    keywords: List[str]
    duration_seconds: float


class AudioUploadResponse(BaseModel):
    """Response after audio file upload."""
    transcript_id: str
    status: str  # "processing", "completed", "failed"
    message: Optional[str] = None


class AudioStreamMessage(BaseModel):
    """WebSocket message for audio streaming."""
    type: str  # "transcript_update", "keyword_detected", "error"
    segment: Optional[TranscriptSegment] = None
    keywords_detected: List[str] = []
    error: Optional[str] = None


class AudioStatusResponse(BaseModel):
    """Audio service status."""
    model_loaded: bool
    model_name: str
    device: str
    language: str
