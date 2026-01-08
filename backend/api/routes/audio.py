"""
Audio API Routes for CAE System.

Endpoints:
- WS /audio/stream/{session_id} - Real-time audio streaming
- POST /audio/upload - Upload audio file for transcription
- GET /audio/status - Get audio service status
"""

from typing import Optional
import tempfile
import os

from fastapi import APIRouter, WebSocket, UploadFile, File, HTTPException, status

from api.schemas.audio import (
    AudioUploadResponse,
    TranscriptResponse,
    TranscriptSegment,
    AudioStatusResponse,
)
from db.database import SessionRepository, TranscriptRepository
from services.audio.whisper_engine import create_whisper_engine, get_whisper_status
from services.audio.stream_handler import get_stream_handler
from services.audio.keyword_tagger import KeywordTagger
from config import AUDIO_ENABLED, WHISPER_MODEL, WHISPER_DEVICE, WHISPER_LANGUAGE
from utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.websocket("/stream/{session_id}")
async def audio_stream(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time audio streaming.

    Client sends binary audio data, server responds with transcript updates.
    """
    if not AUDIO_ENABLED:
        await websocket.close(code=1008, reason="Audio service disabled")
        return

    # Verify session exists
    session = await SessionRepository.get(session_id)
    if not session:
        await websocket.close(code=1008, reason="Session not found")
        return

    # Get or create transcript
    transcript = await TranscriptRepository.get_by_session(session_id)
    if not transcript:
        transcript = await TranscriptRepository.create(
            session_id=session_id,
            language=session.get("language", "fr"),
        )

    # Initialize services
    whisper = await create_whisper_engine()
    stream_handler = await get_stream_handler(whisper, session.get("language", "fr"))

    # Handle stream
    await stream_handler.handle_stream(
        websocket=websocket,
        session_id=session_id,
        transcript_id=transcript["id"],
        language=session.get("language", "fr"),
    )


@router.post("/upload", response_model=AudioUploadResponse)
async def upload_audio(
    session_id: str,
    file: UploadFile = File(...),
    language: str = WHISPER_LANGUAGE,
):
    """
    Upload audio file for batch transcription.

    Supports WAV, MP3, M4A formats.
    """
    if not AUDIO_ENABLED:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Audio service disabled",
        )

    # Verify session
    session = await SessionRepository.get(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session not found: {session_id}",
        )

    # Save uploaded file
    suffix = os.path.splitext(file.filename)[1] if file.filename else ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Transcribe
        whisper = await create_whisper_engine()
        full_text, segments = await whisper.transcribe_file(tmp_path, language)

        # Tag keywords
        tagger = KeywordTagger(language)
        keywords = tagger.get_keywords(full_text)

        # Get or create transcript
        transcript = await TranscriptRepository.get_by_session(session_id)
        if not transcript:
            transcript = await TranscriptRepository.create(
                session_id=session_id,
                language=language,
            )

        # Update transcript with segments
        for seg in segments:
            await TranscriptRepository.append_segment(
                transcript_id=transcript["id"],
                text=seg.text,
                start_time=seg.start_time,
                end_time=seg.end_time,
                confidence=seg.confidence,
                keywords=tagger.get_keywords(seg.text),
            )

        logger.info(
            "Audio uploaded and transcribed",
            session_id=session_id,
            duration=segments[-1].end_time if segments else 0,
            segments=len(segments),
        )

        return AudioUploadResponse(
            transcript_id=transcript["id"],
            status="completed",
            message=f"Transcribed {len(segments)} segments",
        )

    finally:
        # Cleanup temp file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@router.get("/transcript/{session_id}", response_model=TranscriptResponse)
async def get_transcript(session_id: str):
    """Get transcript for a session."""
    transcript = await TranscriptRepository.get_by_session(session_id)
    if not transcript:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Transcript not found for session: {session_id}",
        )

    return TranscriptResponse(
        transcript_id=transcript["id"],
        session_id=transcript["session_id"],
        language=transcript["language"],
        full_text=transcript["full_text"],
        segments=[
            TranscriptSegment(
                text=s.get("text", ""),
                start_time=s.get("start_time", 0.0),
                end_time=s.get("end_time", 0.0),
                confidence=s.get("confidence", 0.0),
                speaker=s.get("speaker"),
                keywords=s.get("keywords", []),
            )
            for s in transcript.get("segments", [])
        ],
        keywords=transcript.get("keywords", []),
        duration_seconds=transcript.get("duration_seconds", 0.0),
    )


@router.get("/status", response_model=AudioStatusResponse)
async def audio_status():
    """Get audio service status."""
    status_str = await get_whisper_status()

    return AudioStatusResponse(
        model_loaded=status_str == "ready",
        model_name=WHISPER_MODEL,
        device=WHISPER_DEVICE,
        language=WHISPER_LANGUAGE,
    )


@router.get("/keywords/{session_id}")
async def get_detected_keywords(session_id: str):
    """Get keywords detected in session transcript."""
    transcript = await TranscriptRepository.get_by_session(session_id)
    if not transcript:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Transcript not found for session: {session_id}",
        )

    # Get keywords by category
    tagger = KeywordTagger(transcript.get("language", "fr"))
    keywords_by_category = tagger.get_keywords_by_category(transcript["full_text"])

    return {
        "session_id": session_id,
        "keywords": transcript.get("keywords", []),
        "keywords_by_category": keywords_by_category,
    }
