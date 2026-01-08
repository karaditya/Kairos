"""
CAE System Database Layer.

Async SQLite database with repositories for:
- Sessions (clinical encounters)
- Transcripts (audio transcriptions)
- Compte Rendus (clinical reports)
- RPA Verifications (human approval records)
"""

import json
import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

import aiosqlite

from config import DATABASE_PATH
from utils.logging import get_logger

logger = get_logger(__name__)

# Database connection pool (single connection for SQLite)
_db_connection: Optional[aiosqlite.Connection] = None


# =============================================================================
# DATABASE INITIALIZATION
# =============================================================================

async def init_db() -> None:
    """Initialize database and create tables."""
    global _db_connection

    logger.info("Initializing database", path=DATABASE_PATH)

    _db_connection = await aiosqlite.connect(DATABASE_PATH)
    _db_connection.row_factory = aiosqlite.Row

    # Enable WAL mode for better concurrency
    await _db_connection.execute("PRAGMA journal_mode=WAL")
    await _db_connection.execute("PRAGMA foreign_keys=ON")

    # Create tables
    await _db_connection.executescript(SCHEMA)
    await _db_connection.commit()

    logger.info("Database initialized successfully")


async def close_db() -> None:
    """Close database connection."""
    global _db_connection
    if _db_connection:
        await _db_connection.close()
        _db_connection = None
        logger.info("Database connection closed")


def get_db() -> aiosqlite.Connection:
    """Get database connection."""
    if _db_connection is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    return _db_connection


# =============================================================================
# SCHEMA
# =============================================================================

SCHEMA = """
-- Sessions: Clinical encounters
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    patient_id TEXT,
    language TEXT NOT NULL DEFAULT 'fr',
    status TEXT NOT NULL DEFAULT 'initializing',
    ehr_window_title TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    metadata TEXT DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions(status);
CREATE INDEX IF NOT EXISTS idx_sessions_created ON sessions(created_at DESC);

-- Transcripts: Audio transcriptions
CREATE TABLE IF NOT EXISTS transcripts (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    language TEXT NOT NULL DEFAULT 'fr',
    full_text TEXT NOT NULL DEFAULT '',
    segments TEXT DEFAULT '[]',
    keywords TEXT DEFAULT '[]',
    duration_seconds REAL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_transcripts_session ON transcripts(session_id);

-- Compte Rendus: Clinical reports
CREATE TABLE IF NOT EXISTS compte_rendus (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    language TEXT NOT NULL DEFAULT 'fr',

    -- Patient info
    patient_name TEXT,
    patient_dob TEXT,
    patient_mrn TEXT,

    -- Clinical sections (JSON for flexibility)
    sections TEXT DEFAULT '{}',

    -- Quality indicators
    missing_fields TEXT DEFAULT '[]',
    flagged_fields TEXT DEFAULT '[]',
    rag_citations TEXT DEFAULT '[]',

    -- Verification
    verified_by TEXT,
    verified_at TEXT,

    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_cr_session ON compte_rendus(session_id);
CREATE INDEX IF NOT EXISTS idx_cr_verified ON compte_rendus(verified_at);

-- RPA Verifications: Human approval records
CREATE TABLE IF NOT EXISTS rpa_verifications (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    compte_rendu_id TEXT NOT NULL,

    -- Actions to execute
    actions TEXT NOT NULL DEFAULT '[]',
    target_fields TEXT DEFAULT '[]',

    -- Status tracking
    status TEXT NOT NULL DEFAULT 'pending',

    -- Approval
    approved_by TEXT,
    approved_at TEXT,
    modifications TEXT DEFAULT '{}',

    -- Execution
    executed_at TEXT,
    execution_result TEXT,

    created_at TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE,
    FOREIGN KEY (compte_rendu_id) REFERENCES compte_rendus(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_rpa_session ON rpa_verifications(session_id);
CREATE INDEX IF NOT EXISTS idx_rpa_status ON rpa_verifications(status);

-- EHR Coordinate Maps: Saved field coordinates
CREATE TABLE IF NOT EXISTS coordinate_maps (
    id TEXT PRIMARY KEY,
    ehr_type TEXT NOT NULL,
    name TEXT NOT NULL,
    field_mappings TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_coord_ehr_name ON coordinate_maps(ehr_type, name);
"""


# =============================================================================
# HELPERS
# =============================================================================

def _now() -> str:
    """Get current timestamp as ISO string."""
    return datetime.utcnow().isoformat()


def _new_id() -> str:
    """Generate new UUID."""
    return str(uuid.uuid4())


def _json_dumps(obj: Any) -> str:
    """Serialize object to JSON string."""
    return json.dumps(obj, ensure_ascii=False)


def _json_loads(s: str) -> Any:
    """Parse JSON string."""
    if not s:
        return None
    return json.loads(s)


# =============================================================================
# SESSION REPOSITORY
# =============================================================================

class SessionRepository:
    """Repository for session operations."""

    @staticmethod
    async def create(
        language: str = "fr",
        patient_id: Optional[str] = None,
        ehr_window_title: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create a new session."""
        db = get_db()
        session_id = _new_id()
        now = _now()

        await db.execute(
            """
            INSERT INTO sessions (id, patient_id, language, status, ehr_window_title, created_at, updated_at, metadata)
            VALUES (?, ?, ?, 'initializing', ?, ?, ?, ?)
            """,
            (session_id, patient_id, language, ehr_window_title, now, now, _json_dumps(metadata or {})),
        )
        await db.commit()

        logger.info("Session created", session_id=session_id, language=language)
        return await SessionRepository.get(session_id)

    @staticmethod
    async def get(session_id: str) -> Optional[Dict[str, Any]]:
        """Get session by ID."""
        db = get_db()
        cursor = await db.execute(
            "SELECT * FROM sessions WHERE id = ?",
            (session_id,),
        )
        row = await cursor.fetchone()

        if not row:
            return None

        return {
            "id": row["id"],
            "patient_id": row["patient_id"],
            "language": row["language"],
            "status": row["status"],
            "ehr_window_title": row["ehr_window_title"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "metadata": _json_loads(row["metadata"]),
        }

    @staticmethod
    async def update_status(session_id: str, status: str) -> None:
        """Update session status."""
        db = get_db()
        await db.execute(
            "UPDATE sessions SET status = ?, updated_at = ? WHERE id = ?",
            (status, _now(), session_id),
        )
        await db.commit()
        logger.info("Session status updated", session_id=session_id, status=status)

    @staticmethod
    async def list_active() -> List[Dict[str, Any]]:
        """List all active sessions."""
        db = get_db()
        cursor = await db.execute(
            "SELECT * FROM sessions WHERE status IN ('initializing', 'active', 'paused') ORDER BY created_at DESC"
        )
        rows = await cursor.fetchall()

        return [
            {
                "id": row["id"],
                "patient_id": row["patient_id"],
                "language": row["language"],
                "status": row["status"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    @staticmethod
    async def delete(session_id: str) -> bool:
        """Delete session and related data."""
        db = get_db()
        cursor = await db.execute(
            "DELETE FROM sessions WHERE id = ?",
            (session_id,),
        )
        await db.commit()
        deleted = cursor.rowcount > 0
        if deleted:
            logger.info("Session deleted", session_id=session_id)
        return deleted


# =============================================================================
# TRANSCRIPT REPOSITORY
# =============================================================================

class TranscriptRepository:
    """Repository for transcript operations."""

    @staticmethod
    async def create(session_id: str, language: str = "fr") -> Dict[str, Any]:
        """Create a new transcript for a session."""
        db = get_db()
        transcript_id = _new_id()
        now = _now()

        await db.execute(
            """
            INSERT INTO transcripts (id, session_id, language, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (transcript_id, session_id, language, now, now),
        )
        await db.commit()

        logger.info("Transcript created", transcript_id=transcript_id, session_id=session_id)
        return await TranscriptRepository.get(transcript_id)

    @staticmethod
    async def get(transcript_id: str) -> Optional[Dict[str, Any]]:
        """Get transcript by ID."""
        db = get_db()
        cursor = await db.execute(
            "SELECT * FROM transcripts WHERE id = ?",
            (transcript_id,),
        )
        row = await cursor.fetchone()

        if not row:
            return None

        return {
            "id": row["id"],
            "session_id": row["session_id"],
            "language": row["language"],
            "full_text": row["full_text"],
            "segments": _json_loads(row["segments"]),
            "keywords": _json_loads(row["keywords"]),
            "duration_seconds": row["duration_seconds"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    @staticmethod
    async def get_by_session(session_id: str) -> Optional[Dict[str, Any]]:
        """Get transcript for a session."""
        db = get_db()
        cursor = await db.execute(
            "SELECT * FROM transcripts WHERE session_id = ? ORDER BY created_at DESC LIMIT 1",
            (session_id,),
        )
        row = await cursor.fetchone()

        if not row:
            return None

        return {
            "id": row["id"],
            "session_id": row["session_id"],
            "language": row["language"],
            "full_text": row["full_text"],
            "segments": _json_loads(row["segments"]),
            "keywords": _json_loads(row["keywords"]),
            "duration_seconds": row["duration_seconds"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    @staticmethod
    async def append_segment(
        transcript_id: str,
        text: str,
        start_time: float,
        end_time: float,
        confidence: float,
        keywords: Optional[List[str]] = None,
    ) -> None:
        """Append a new segment to transcript."""
        db = get_db()

        # Get current transcript
        transcript = await TranscriptRepository.get(transcript_id)
        if not transcript:
            return

        # Add new segment
        segments = transcript["segments"] or []
        segments.append({
            "text": text,
            "start_time": start_time,
            "end_time": end_time,
            "confidence": confidence,
            "keywords": keywords or [],
        })

        # Update full text
        full_text = transcript["full_text"] + " " + text if transcript["full_text"] else text

        # Merge keywords
        all_keywords = list(set(transcript["keywords"] + (keywords or [])))

        # Calculate duration
        duration = end_time

        await db.execute(
            """
            UPDATE transcripts
            SET full_text = ?, segments = ?, keywords = ?, duration_seconds = ?, updated_at = ?
            WHERE id = ?
            """,
            (full_text.strip(), _json_dumps(segments), _json_dumps(all_keywords), duration, _now(), transcript_id),
        )
        await db.commit()


# =============================================================================
# COMPTE RENDU REPOSITORY
# =============================================================================

class CompteRenduRepository:
    """Repository for compte rendu (clinical report) operations."""

    @staticmethod
    async def create(
        session_id: str,
        language: str = "fr",
        patient_name: Optional[str] = None,
        patient_dob: Optional[str] = None,
        patient_mrn: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a new compte rendu."""
        db = get_db()
        cr_id = _new_id()
        now = _now()

        await db.execute(
            """
            INSERT INTO compte_rendus (id, session_id, language, patient_name, patient_dob, patient_mrn, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (cr_id, session_id, language, patient_name, patient_dob, patient_mrn, now, now),
        )
        await db.commit()

        logger.info("Compte Rendu created", cr_id=cr_id, session_id=session_id)
        return await CompteRenduRepository.get(cr_id)

    @staticmethod
    async def get(cr_id: str) -> Optional[Dict[str, Any]]:
        """Get compte rendu by ID."""
        db = get_db()
        cursor = await db.execute(
            "SELECT * FROM compte_rendus WHERE id = ?",
            (cr_id,),
        )
        row = await cursor.fetchone()

        if not row:
            return None

        return {
            "id": row["id"],
            "session_id": row["session_id"],
            "language": row["language"],
            "patient_name": row["patient_name"],
            "patient_dob": row["patient_dob"],
            "patient_mrn": row["patient_mrn"],
            "sections": _json_loads(row["sections"]),
            "missing_fields": _json_loads(row["missing_fields"]),
            "flagged_fields": _json_loads(row["flagged_fields"]),
            "rag_citations": _json_loads(row["rag_citations"]),
            "verified_by": row["verified_by"],
            "verified_at": row["verified_at"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    @staticmethod
    async def get_by_session(session_id: str) -> Optional[Dict[str, Any]]:
        """Get latest compte rendu for a session."""
        db = get_db()
        cursor = await db.execute(
            "SELECT * FROM compte_rendus WHERE session_id = ? ORDER BY created_at DESC LIMIT 1",
            (session_id,),
        )
        row = await cursor.fetchone()

        if not row:
            return None

        return await CompteRenduRepository.get(row["id"])

    @staticmethod
    async def update_sections(
        cr_id: str,
        sections: Dict[str, str],
        missing_fields: Optional[List[Dict]] = None,
        flagged_fields: Optional[List[Dict]] = None,
        rag_citations: Optional[List[Dict]] = None,
    ) -> None:
        """Update compte rendu sections."""
        db = get_db()
        await db.execute(
            """
            UPDATE compte_rendus
            SET sections = ?, missing_fields = ?, flagged_fields = ?, rag_citations = ?, updated_at = ?
            WHERE id = ?
            """,
            (
                _json_dumps(sections),
                _json_dumps(missing_fields or []),
                _json_dumps(flagged_fields or []),
                _json_dumps(rag_citations or []),
                _now(),
                cr_id,
            ),
        )
        await db.commit()
        logger.info("Compte Rendu sections updated", cr_id=cr_id)

    @staticmethod
    async def verify(cr_id: str, verified_by: str) -> None:
        """Mark compte rendu as verified."""
        db = get_db()
        await db.execute(
            "UPDATE compte_rendus SET verified_by = ?, verified_at = ?, updated_at = ? WHERE id = ?",
            (verified_by, _now(), _now(), cr_id),
        )
        await db.commit()
        logger.info("Compte Rendu verified", cr_id=cr_id, verified_by=verified_by)


# =============================================================================
# RPA VERIFICATION REPOSITORY
# =============================================================================

class RPAVerificationRepository:
    """Repository for RPA verification/approval operations."""

    @staticmethod
    async def create(
        session_id: str,
        compte_rendu_id: str,
        actions: List[Dict[str, Any]],
        target_fields: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Create a new RPA verification request."""
        db = get_db()
        verification_id = _new_id()
        now = _now()

        await db.execute(
            """
            INSERT INTO rpa_verifications (id, session_id, compte_rendu_id, actions, target_fields, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (verification_id, session_id, compte_rendu_id, _json_dumps(actions), _json_dumps(target_fields or []), now),
        )
        await db.commit()

        logger.info("RPA verification created", verification_id=verification_id, session_id=session_id)
        return await RPAVerificationRepository.get(verification_id)

    @staticmethod
    async def get(verification_id: str) -> Optional[Dict[str, Any]]:
        """Get RPA verification by ID."""
        db = get_db()
        cursor = await db.execute(
            "SELECT * FROM rpa_verifications WHERE id = ?",
            (verification_id,),
        )
        row = await cursor.fetchone()

        if not row:
            return None

        return {
            "id": row["id"],
            "session_id": row["session_id"],
            "compte_rendu_id": row["compte_rendu_id"],
            "actions": _json_loads(row["actions"]),
            "target_fields": _json_loads(row["target_fields"]),
            "status": row["status"],
            "approved_by": row["approved_by"],
            "approved_at": row["approved_at"],
            "modifications": _json_loads(row["modifications"]),
            "executed_at": row["executed_at"],
            "execution_result": row["execution_result"],
            "created_at": row["created_at"],
        }

    @staticmethod
    async def approve(
        verification_id: str,
        approved_by: str,
        modifications: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Approve an RPA verification request."""
        db = get_db()
        await db.execute(
            """
            UPDATE rpa_verifications
            SET status = 'approved', approved_by = ?, approved_at = ?, modifications = ?
            WHERE id = ? AND status = 'pending'
            """,
            (approved_by, _now(), _json_dumps(modifications or {}), verification_id),
        )
        await db.commit()
        logger.info("RPA verification approved", verification_id=verification_id, approved_by=approved_by)

    @staticmethod
    async def reject(verification_id: str, rejected_by: str) -> None:
        """Reject an RPA verification request."""
        db = get_db()
        await db.execute(
            "UPDATE rpa_verifications SET status = 'rejected', approved_by = ?, approved_at = ? WHERE id = ?",
            (rejected_by, _now(), verification_id),
        )
        await db.commit()
        logger.info("RPA verification rejected", verification_id=verification_id, rejected_by=rejected_by)

    @staticmethod
    async def mark_executed(
        verification_id: str,
        success: bool,
        result: Optional[str] = None,
    ) -> None:
        """Mark RPA verification as executed."""
        db = get_db()
        status = "completed" if success else "failed"
        await db.execute(
            "UPDATE rpa_verifications SET status = ?, executed_at = ?, execution_result = ? WHERE id = ?",
            (status, _now(), result, verification_id),
        )
        await db.commit()
        logger.info("RPA verification executed", verification_id=verification_id, status=status)

    @staticmethod
    async def get_pending(session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get pending RPA verifications."""
        db = get_db()

        if session_id:
            cursor = await db.execute(
                "SELECT * FROM rpa_verifications WHERE status = 'pending' AND session_id = ?",
                (session_id,),
            )
        else:
            cursor = await db.execute(
                "SELECT * FROM rpa_verifications WHERE status = 'pending' ORDER BY created_at DESC"
            )

        rows = await cursor.fetchall()

        return [
            {
                "id": row["id"],
                "session_id": row["session_id"],
                "compte_rendu_id": row["compte_rendu_id"],
                "actions": _json_loads(row["actions"]),
                "status": row["status"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]
