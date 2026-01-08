"""CAE API Schemas."""

from api.schemas.session import (
    SessionStartRequest,
    SessionStartResponse,
    SessionStateResponse,
)
from api.schemas.vision import (
    VisionScrapeRequest,
    VisionScrapeResponse,
    VisionCalibrateRequest,
    ExtractedField,
    UIElement,
)
from api.schemas.agent import (
    AgentDraftResponse,
    SyncRequest,
    SyncPreviewResponse,
    SyncApproveRequest,
    SyncResultResponse,
)
from api.schemas.audio import (
    AudioUploadResponse,
    TranscriptSegment,
    TranscriptResponse,
)
from api.schemas.common import (
    ErrorResponse,
    SuccessResponse,
    StatusResponse,
)

__all__ = [
    # Session
    "SessionStartRequest",
    "SessionStartResponse",
    "SessionStateResponse",
    # Vision
    "VisionScrapeRequest",
    "VisionScrapeResponse",
    "VisionCalibrateRequest",
    "ExtractedField",
    "UIElement",
    # Agent
    "AgentDraftResponse",
    "SyncRequest",
    "SyncPreviewResponse",
    "SyncApproveRequest",
    "SyncResultResponse",
    # Audio
    "AudioUploadResponse",
    "TranscriptSegment",
    "TranscriptResponse",
    # Common
    "ErrorResponse",
    "SuccessResponse",
    "StatusResponse",
]
