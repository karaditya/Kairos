"""
Session API Routes for CAE System.

Endpoints:
- POST /session/start - Start new session with Whisper stream
- GET /session/{id} - Get session state
- POST /session/{id}/pause - Pause transcription
- POST /session/{id}/resume - Resume transcription
- POST /session/{id}/end - End session
"""

from typing import Optional

from fastapi import APIRouter, HTTPException, status

from api.schemas.session import (
    SessionStartRequest,
    SessionStartResponse,
    SessionStateResponse,
    SessionListResponse,
    SessionListItem,
)
from db.database import SessionRepository, TranscriptRepository
from utils.logging import get_logger
from utils.exceptions import SessionNotFoundError

logger = get_logger(__name__)

router = APIRouter()


@router.post("/start", response_model=SessionStartResponse)
async def start_session(request: SessionStartRequest):
    """
    Start a new clinical session.

    Initializes:
    - Session record in database
    - Transcript record for audio
    - WebSocket URL for audio streaming
    """
    # Create session
    session = await SessionRepository.create(
        language=request.language,
        patient_id=request.patient_id,
        ehr_window_title=request.ehr_window_title,
    )

    # Create transcript
    transcript = await TranscriptRepository.create(
        session_id=session["id"],
        language=request.language,
    )

    # Update session status
    await SessionRepository.update_status(session["id"], "active")

    logger.info("Session started", session_id=session["id"])

    return SessionStartResponse(
        session_id=session["id"],
        websocket_url=f"/audio/stream/{session['id']}",
        status="active",
    )


@router.get("/{session_id}", response_model=SessionStateResponse)
async def get_session(session_id: str):
    """Get session state including transcript and EHR data."""
    session = await SessionRepository.get(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session not found: {session_id}",
        )

    # Get transcript
    transcript = await TranscriptRepository.get_by_session(session_id)
    transcript_data = None
    if transcript:
        transcript_data = {
            "text": transcript["full_text"],
            "segments": transcript["segments"],
            "keywords": transcript["keywords"],
        }

    return SessionStateResponse(
        session_id=session["id"],
        status=session["status"],
        language=session["language"],
        patient_id=session.get("patient_id"),
        transcript=transcript_data,
        ehr_data=None,  # Will be populated by vision service
        created_at=session["created_at"],
        updated_at=session["updated_at"],
    )


@router.post("/{session_id}/pause")
async def pause_session(session_id: str):
    """Pause transcription for a session."""
    session = await SessionRepository.get(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session not found: {session_id}",
        )

    await SessionRepository.update_status(session_id, "paused")

    return {"status": "paused", "session_id": session_id}


@router.post("/{session_id}/resume")
async def resume_session(session_id: str):
    """Resume transcription for a session."""
    session = await SessionRepository.get(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session not found: {session_id}",
        )

    await SessionRepository.update_status(session_id, "active")

    return {"status": "active", "session_id": session_id}


@router.post("/{session_id}/end")
async def end_session(session_id: str):
    """End a session."""
    session = await SessionRepository.get(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session not found: {session_id}",
        )

    await SessionRepository.update_status(session_id, "completed")

    logger.info("Session ended", session_id=session_id)

    return {"status": "completed", "session_id": session_id}


@router.get("/", response_model=SessionListResponse)
async def list_sessions():
    """List active sessions."""
    sessions = await SessionRepository.list_active()

    return SessionListResponse(
        sessions=[
            SessionListItem(
                id=s["id"],
                patient_id=s.get("patient_id"),
                language=s["language"],
                status=s["status"],
                created_at=s["created_at"],
            )
            for s in sessions
        ],
        total=len(sessions),
    )


@router.delete("/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and all related data."""
    deleted = await SessionRepository.delete(session_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session not found: {session_id}",
        )

    return {"deleted": True, "session_id": session_id}
