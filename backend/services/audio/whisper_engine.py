"""
Faster-Whisper Engine for CAE System.

Provides local audio transcription with:
- Real-time streaming support
- VAD (Voice Activity Detection)
- Multi-language support (French/English)
"""

from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
import io
import numpy as np

from config import (
    WHISPER_MODEL,
    WHISPER_DEVICE,
    WHISPER_COMPUTE_TYPE,
    WHISPER_LANGUAGE,
    AUDIO_SAMPLE_RATE,
)
from utils.logging import get_logger
from utils.exceptions import WhisperError

logger = get_logger(__name__)


@dataclass
class TranscriptSegment:
    """Single transcript segment."""
    text: str
    start_time: float
    end_time: float
    confidence: float
    words: Optional[List[Dict]] = None


class WhisperEngine:
    """Faster-Whisper transcription engine."""

    def __init__(
        self,
        model_name: str = WHISPER_MODEL,
        device: str = WHISPER_DEVICE,
        compute_type: str = WHISPER_COMPUTE_TYPE,
    ):
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self._model = None
        self._initialized = False

    async def initialize(self) -> None:
        """Load Whisper model."""
        if self._initialized:
            return

        try:
            from faster_whisper import WhisperModel

            # Determine device
            device = self.device
            if device == "auto":
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"

            logger.info(
                "Loading Whisper model",
                model=self.model_name,
                device=device,
                compute_type=self.compute_type,
            )

            self._model = WhisperModel(
                self.model_name,
                device=device,
                compute_type=self.compute_type if device == "cuda" else "int8",
            )

            self._initialized = True
            logger.info("Whisper model loaded")

        except ImportError:
            raise WhisperError("faster-whisper not installed. Run: pip install faster-whisper")
        except Exception as e:
            raise WhisperError(f"Failed to load Whisper model: {str(e)}")

    @property
    def is_ready(self) -> bool:
        return self._initialized and self._model is not None

    async def transcribe(
        self,
        audio_data: bytes,
        language: str = WHISPER_LANGUAGE,
        sample_rate: int = AUDIO_SAMPLE_RATE,
    ) -> Tuple[str, List[TranscriptSegment]]:
        """
        Transcribe audio data.

        Args:
            audio_data: Raw audio bytes (WAV/MP3/etc)
            language: Language code ("fr", "en", etc)
            sample_rate: Audio sample rate

        Returns:
            Tuple of (full_text, segments)
        """
        if not self.is_ready:
            await self.initialize()

        try:
            # Convert bytes to numpy array
            audio_array = self._bytes_to_array(audio_data, sample_rate)

            # Transcribe
            segments, info = self._model.transcribe(
                audio_array,
                language=language,
                beam_size=5,
                vad_filter=True,
                vad_parameters=dict(
                    min_silence_duration_ms=500,
                    speech_pad_ms=200,
                ),
            )

            # Collect segments
            transcript_segments = []
            full_text_parts = []

            for segment in segments:
                ts = TranscriptSegment(
                    text=segment.text.strip(),
                    start_time=segment.start,
                    end_time=segment.end,
                    confidence=segment.avg_logprob,
                )
                transcript_segments.append(ts)
                full_text_parts.append(segment.text.strip())

            full_text = " ".join(full_text_parts)

            logger.debug(
                "Transcription complete",
                duration=info.duration,
                segments=len(transcript_segments),
            )

            return full_text, transcript_segments

        except Exception as e:
            raise WhisperError(f"Transcription failed: {str(e)}")

    async def transcribe_file(
        self,
        file_path: str,
        language: str = WHISPER_LANGUAGE,
    ) -> Tuple[str, List[TranscriptSegment]]:
        """
        Transcribe audio file.

        Args:
            file_path: Path to audio file
            language: Language code

        Returns:
            Tuple of (full_text, segments)
        """
        if not self.is_ready:
            await self.initialize()

        try:
            segments, info = self._model.transcribe(
                file_path,
                language=language,
                beam_size=5,
                vad_filter=True,
            )

            transcript_segments = []
            full_text_parts = []

            for segment in segments:
                ts = TranscriptSegment(
                    text=segment.text.strip(),
                    start_time=segment.start,
                    end_time=segment.end,
                    confidence=segment.avg_logprob,
                )
                transcript_segments.append(ts)
                full_text_parts.append(segment.text.strip())

            return " ".join(full_text_parts), transcript_segments

        except Exception as e:
            raise WhisperError(f"File transcription failed: {str(e)}")

    async def transcribe_stream(
        self,
        audio_chunk: bytes,
        language: str = WHISPER_LANGUAGE,
        sample_rate: int = AUDIO_SAMPLE_RATE,
    ) -> Optional[TranscriptSegment]:
        """
        Transcribe audio chunk for streaming.

        Args:
            audio_chunk: Raw audio bytes
            language: Language code
            sample_rate: Audio sample rate

        Returns:
            TranscriptSegment if speech detected, None otherwise
        """
        if not self.is_ready:
            await self.initialize()

        try:
            audio_array = self._bytes_to_array(audio_chunk, sample_rate)

            # Quick transcription for streaming
            segments, _ = self._model.transcribe(
                audio_array,
                language=language,
                beam_size=1,
                vad_filter=True,
                vad_parameters=dict(
                    min_silence_duration_ms=300,
                ),
            )

            # Return first segment if any
            for segment in segments:
                if segment.text.strip():
                    return TranscriptSegment(
                        text=segment.text.strip(),
                        start_time=segment.start,
                        end_time=segment.end,
                        confidence=segment.avg_logprob,
                    )

            return None

        except Exception as e:
            logger.warning("Stream transcription error", error=str(e))
            return None

    def _bytes_to_array(
        self,
        audio_bytes: bytes,
        sample_rate: int,
    ) -> np.ndarray:
        """Convert audio bytes to numpy array."""
        try:
            import soundfile as sf

            # Try reading with soundfile
            audio_io = io.BytesIO(audio_bytes)
            audio_array, sr = sf.read(audio_io)

            # Resample if needed
            if sr != sample_rate:
                import librosa
                audio_array = librosa.resample(
                    audio_array,
                    orig_sr=sr,
                    target_sr=sample_rate,
                )

            # Convert to mono if stereo
            if len(audio_array.shape) > 1:
                audio_array = audio_array.mean(axis=1)

            return audio_array.astype(np.float32)

        except ImportError:
            # Fallback: assume raw PCM
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
            return audio_array.astype(np.float32) / 32768.0

    async def shutdown(self) -> None:
        """Unload model and free memory."""
        if self._model:
            del self._model
            self._model = None
            self._initialized = False

            # Force garbage collection
            import gc
            gc.collect()

            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

            logger.info("Whisper model unloaded")


# =============================================================================
# SINGLETON
# =============================================================================

_whisper_engine: Optional[WhisperEngine] = None


async def create_whisper_engine() -> WhisperEngine:
    """Create and initialize Whisper engine."""
    global _whisper_engine
    if _whisper_engine is None:
        _whisper_engine = WhisperEngine()
        await _whisper_engine.initialize()
    return _whisper_engine


async def shutdown_whisper() -> None:
    """Shutdown Whisper engine."""
    global _whisper_engine
    if _whisper_engine:
        await _whisper_engine.shutdown()
        _whisper_engine = None


async def get_whisper_status() -> str:
    """Get Whisper engine status."""
    global _whisper_engine
    if _whisper_engine is None:
        return "not_initialized"
    if _whisper_engine.is_ready:
        return "ready"
    return "loading"
