"""
Audio Stream Handler for CAE System.

Manages WebSocket audio streaming with:
- Chunked processing
- Real-time transcription
- Keyword detection
"""

import asyncio
from typing import Optional, List, Dict, Any, Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime
import json

from fastapi import WebSocket, WebSocketDisconnect

from services.audio.whisper_engine import WhisperEngine, TranscriptSegment
from services.audio.keyword_tagger import KeywordTagger
from db.database import TranscriptRepository
from config import AUDIO_SAMPLE_RATE, AUDIO_CHUNK_DURATION_MS
from utils.logging import get_logger
from utils.exceptions import AudioStreamError

logger = get_logger(__name__)


@dataclass
class StreamSession:
    """Active streaming session."""
    session_id: str
    transcript_id: str
    language: str
    websocket: WebSocket
    started_at: datetime = field(default_factory=datetime.utcnow)
    total_duration: float = 0.0
    segments: List[TranscriptSegment] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    is_active: bool = True


class AudioStreamHandler:
    """Handles WebSocket audio streaming and transcription."""

    def __init__(
        self,
        whisper: WhisperEngine,
        keyword_tagger: KeywordTagger,
    ):
        self.whisper = whisper
        self.keyword_tagger = keyword_tagger
        self._active_sessions: Dict[str, StreamSession] = {}
        self._buffer_size = int(AUDIO_SAMPLE_RATE * AUDIO_CHUNK_DURATION_MS / 1000) * 2  # bytes

    async def handle_stream(
        self,
        websocket: WebSocket,
        session_id: str,
        transcript_id: str,
        language: str = "fr",
    ) -> None:
        """
        Handle WebSocket audio stream.

        Args:
            websocket: FastAPI WebSocket connection
            session_id: CAE session ID
            transcript_id: Transcript database ID
            language: Transcription language
        """
        await websocket.accept()

        # Create stream session
        stream_session = StreamSession(
            session_id=session_id,
            transcript_id=transcript_id,
            language=language,
            websocket=websocket,
        )
        self._active_sessions[session_id] = stream_session

        logger.info("Audio stream started", session_id=session_id)

        audio_buffer = bytearray()

        try:
            while stream_session.is_active:
                try:
                    # Receive audio data
                    data = await asyncio.wait_for(
                        websocket.receive_bytes(),
                        timeout=30.0,
                    )

                    # Add to buffer
                    audio_buffer.extend(data)

                    # Process when buffer is full
                    if len(audio_buffer) >= self._buffer_size:
                        await self._process_chunk(
                            stream_session,
                            bytes(audio_buffer[:self._buffer_size]),
                        )
                        audio_buffer = audio_buffer[self._buffer_size:]

                except asyncio.TimeoutError:
                    # Send keepalive
                    await self._send_message(websocket, {
                        "type": "keepalive",
                        "timestamp": datetime.utcnow().isoformat(),
                    })

        except WebSocketDisconnect:
            logger.info("WebSocket disconnected", session_id=session_id)

        except Exception as e:
            logger.error("Stream error", session_id=session_id, error=str(e))
            await self._send_message(websocket, {
                "type": "error",
                "error": str(e),
            })

        finally:
            # Process remaining buffer
            if len(audio_buffer) > 0:
                await self._process_chunk(stream_session, bytes(audio_buffer))

            # Cleanup
            stream_session.is_active = False
            self._active_sessions.pop(session_id, None)

            logger.info(
                "Audio stream ended",
                session_id=session_id,
                duration=stream_session.total_duration,
                segments=len(stream_session.segments),
            )

    async def _process_chunk(
        self,
        session: StreamSession,
        audio_chunk: bytes,
    ) -> None:
        """Process audio chunk and send results."""
        try:
            # Transcribe
            segment = await self.whisper.transcribe_stream(
                audio_chunk,
                language=session.language,
            )

            if segment and segment.text.strip():
                # Detect keywords
                keywords = self.keyword_tagger.get_keywords(segment.text)

                # Update session
                session.segments.append(segment)
                session.keywords.extend(keywords)
                session.total_duration = segment.end_time

                # Save to database
                await TranscriptRepository.append_segment(
                    session.transcript_id,
                    text=segment.text,
                    start_time=segment.start_time,
                    end_time=segment.end_time,
                    confidence=segment.confidence,
                    keywords=keywords,
                )

                # Send update to client
                await self._send_message(session.websocket, {
                    "type": "transcript_update",
                    "segment": {
                        "text": segment.text,
                        "start_time": segment.start_time,
                        "end_time": segment.end_time,
                        "confidence": segment.confidence,
                    },
                    "keywords_detected": keywords,
                })

        except Exception as e:
            logger.warning("Chunk processing error", error=str(e))

    async def _send_message(
        self,
        websocket: WebSocket,
        message: Dict[str, Any],
    ) -> None:
        """Send JSON message to WebSocket."""
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.debug("Send message error", error=str(e))

    async def stop_stream(self, session_id: str) -> bool:
        """Stop an active stream."""
        session = self._active_sessions.get(session_id)
        if session:
            session.is_active = False
            return True
        return False

    def get_active_sessions(self) -> List[str]:
        """Get list of active session IDs."""
        return list(self._active_sessions.keys())

    def get_session_stats(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get statistics for a session."""
        session = self._active_sessions.get(session_id)
        if not session:
            return None

        return {
            "session_id": session.session_id,
            "transcript_id": session.transcript_id,
            "language": session.language,
            "started_at": session.started_at.isoformat(),
            "duration_seconds": session.total_duration,
            "segment_count": len(session.segments),
            "keyword_count": len(set(session.keywords)),
            "is_active": session.is_active,
        }


# =============================================================================
# FACTORY
# =============================================================================

_stream_handler: Optional[AudioStreamHandler] = None


async def get_stream_handler(
    whisper: WhisperEngine,
    language: str = "fr",
) -> AudioStreamHandler:
    """Get or create stream handler."""
    global _stream_handler
    if _stream_handler is None:
        keyword_tagger = KeywordTagger(language)
        _stream_handler = AudioStreamHandler(whisper, keyword_tagger)
    return _stream_handler
