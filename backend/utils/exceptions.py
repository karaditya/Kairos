"""
Custom exceptions for CAE System.

Structured exception hierarchy for clean error handling.
"""

from typing import Optional, Any, Dict


class CAEException(Exception):
    """Base exception for all CAE system errors."""

    def __init__(
        self,
        message: str,
        code: str = "CAE_ERROR",
        details: Optional[Dict[str, Any]] = None,
    ):
        self.message = message
        self.code = code
        self.details = details or {}
        super().__init__(self.message)

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for API responses."""
        return {
            "error": self.code,
            "message": self.message,
            "details": self.details,
        }


# =============================================================================
# SESSION ERRORS
# =============================================================================


class SessionNotFoundError(CAEException):
    """Session ID does not exist."""

    def __init__(self, session_id: str):
        super().__init__(
            message=f"Session not found: {session_id}",
            code="SESSION_NOT_FOUND",
            details={"session_id": session_id},
        )


class SessionExpiredError(CAEException):
    """Session has expired or been terminated."""

    def __init__(self, session_id: str):
        super().__init__(
            message=f"Session expired: {session_id}",
            code="SESSION_EXPIRED",
            details={"session_id": session_id},
        )


# =============================================================================
# VISION ERRORS
# =============================================================================


class VisionError(CAEException):
    """Error in vision/OCR processing."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            code="VISION_ERROR",
            details=details,
        )


class ScreenCaptureError(VisionError):
    """Failed to capture screen."""

    def __init__(self, reason: str):
        super().__init__(
            message=f"Screen capture failed: {reason}",
            details={"reason": reason},
        )


class VisionModelError(VisionError):
    """DeepSeek-VL2 model error."""

    def __init__(self, reason: str):
        super().__init__(
            message=f"Vision model error: {reason}",
            details={"reason": reason},
        )


# =============================================================================
# AUDIO ERRORS
# =============================================================================


class AudioError(CAEException):
    """Error in audio processing."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            code="AUDIO_ERROR",
            details=details,
        )


class WhisperError(AudioError):
    """Faster-Whisper transcription error."""

    def __init__(self, reason: str):
        super().__init__(
            message=f"Whisper transcription error: {reason}",
            details={"reason": reason},
        )


class AudioStreamError(AudioError):
    """WebSocket audio stream error."""

    def __init__(self, reason: str):
        super().__init__(
            message=f"Audio stream error: {reason}",
            details={"reason": reason},
        )


# =============================================================================
# RPA ERRORS
# =============================================================================


class RPAError(CAEException):
    """Error in RPA operations."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            code="RPA_ERROR",
            details=details,
        )


class RPANotApprovedError(RPAError):
    """RPA action not approved by human."""

    def __init__(self, verification_id: str):
        super().__init__(
            message="RPA action requires human approval before execution",
            details={"verification_id": verification_id},
        )


class RPAExecutionError(RPAError):
    """RPA action execution failed."""

    def __init__(self, action: str, reason: str):
        super().__init__(
            message=f"RPA execution failed: {reason}",
            details={"action": action, "reason": reason},
        )


class CoordinateMapError(RPAError):
    """EHR coordinate mapping error."""

    def __init__(self, field: str, reason: str):
        super().__init__(
            message=f"Coordinate map error for field '{field}': {reason}",
            details={"field": field, "reason": reason},
        )


# =============================================================================
# RAG ERRORS
# =============================================================================


class RAGError(CAEException):
    """Error in RAG operations."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            code="RAG_ERROR",
            details=details,
        )


class QdrantError(RAGError):
    """Qdrant vector DB error."""

    def __init__(self, reason: str):
        super().__init__(
            message=f"Qdrant error: {reason}",
            details={"reason": reason},
        )


class EmbeddingError(RAGError):
    """Embedding generation error."""

    def __init__(self, reason: str):
        super().__init__(
            message=f"Embedding error: {reason}",
            details={"reason": reason},
        )


# =============================================================================
# PARLANT ERRORS
# =============================================================================


class ParlantError(CAEException):
    """Error in Parlant agent operations."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            code="PARLANT_ERROR",
            details=details,
        )


class ParlantTimeoutError(ParlantError):
    """Parlant response timeout."""

    def __init__(self, timeout_seconds: float):
        super().__init__(
            message=f"Parlant response timeout after {timeout_seconds}s",
            details={"timeout_seconds": timeout_seconds},
        )


class ParlantNotReadyError(ParlantError):
    """Parlant agent not initialized."""

    def __init__(self):
        super().__init__(
            message="Parlant agent not ready",
            details={"hint": "Wait for initialization or check Ollama connection"},
        )


# =============================================================================
# VALIDATION ERRORS
# =============================================================================


class ValidationError(CAEException):
    """Input validation error."""

    def __init__(self, field: str, reason: str):
        super().__init__(
            message=f"Validation error for '{field}': {reason}",
            code="VALIDATION_ERROR",
            details={"field": field, "reason": reason},
        )


class UnsupportedLanguageError(ValidationError):
    """Language not supported."""

    def __init__(self, language: str, supported: list):
        super().__init__(
            field="language",
            reason=f"'{language}' not supported. Use one of: {supported}",
        )


# =============================================================================
# LLM ERRORS
# =============================================================================


class LLMError(CAEException):
    """Error in LLM operations."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            code="LLM_ERROR",
            details=details,
        )


class OllamaConnectionError(LLMError):
    """Cannot connect to Ollama."""

    def __init__(self, url: str):
        super().__init__(
            message=f"Cannot connect to Ollama at {url}",
            details={"url": url, "hint": "Ensure Ollama is running"},
        )


class ModelNotFoundError(LLMError):
    """Requested model not available."""

    def __init__(self, model: str):
        super().__init__(
            message=f"Model '{model}' not found",
            details={"model": model, "hint": "Pull the model with 'ollama pull'"},
        )
