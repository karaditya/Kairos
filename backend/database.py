"""
Database Module - SQLite Storage for Triage MVP

Handles all persistent storage:
- Patient sessions (in-progress triage)
- Completed cases (for staff review)
"""

import sqlite3
import json
import os
from datetime import datetime
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from contextlib import contextmanager

# =============================================================================
# Data Models
# =============================================================================

@dataclass
class Session:
    """In-progress patient session."""
    id: str
    language: str
    created_at: datetime
    status: str  # demographics, chief_complaint, triage, complete
    current_question_id: Optional[str]
    answers: Dict[str, Any]
    demographics: Dict[str, Any]

@dataclass
class Case:
    """Completed triage case for staff review."""
    id: str
    session_id: str
    ticket_id: str
    created_at: datetime
    risk_band: str  # red, amber, green
    demographics: Dict[str, Any]
    answers: Dict[str, Any]
    summary: str
    key_flags: List[str]
    triggered_rules: List[Dict[str, Any]]
    status: str = "pending"  # pending, reviewed, discharged

# =============================================================================
# Database Class
# =============================================================================

class Database:
    """SQLite database wrapper."""
    
    def __init__(self, db_path: str = "../data/triage.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    @contextmanager
    def get_connection(self):
        """Get database connection with automatic cleanup."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()
    
    def initialize(self):
        """Create database tables if they don't exist."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    language TEXT NOT NULL DEFAULT 'en',
                    created_at TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'demographics',
                    current_question_id TEXT,
                    answers TEXT NOT NULL DEFAULT '{}',
                    demographics TEXT NOT NULL DEFAULT '{}'
                )
            """)
            
            # Cases table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS cases (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    ticket_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    risk_band TEXT NOT NULL,
                    demographics TEXT NOT NULL,
                    answers TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    key_flags TEXT NOT NULL,
                    triggered_rules TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    FOREIGN KEY (session_id) REFERENCES sessions(id)
                )
            """)
            
            # Create indexes for common queries
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_cases_status ON cases(status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_cases_risk ON cases(risk_band)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_cases_created ON cases(created_at)")
    
    # =========================================================================
    # Session Operations
    # =========================================================================
    
    def create_session(self, session: Session) -> None:
        """Create a new session."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO sessions (id, language, created_at, status, current_question_id, answers, demographics)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                session.id,
                session.language,
                session.created_at.isoformat(),
                session.status,
                session.current_question_id,
                json.dumps(session.answers),
                json.dumps(session.demographics)
            ))
    
    def get_session(self, session_id: str) -> Optional[Session]:
        """Get session by ID."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
            row = cursor.fetchone()
            
            if row is None:
                return None
            
            return Session(
                id=row["id"],
                language=row["language"],
                created_at=datetime.fromisoformat(row["created_at"]),
                status=row["status"],
                current_question_id=row["current_question_id"],
                answers=json.loads(row["answers"]),
                demographics=json.loads(row["demographics"])
            )
    
    def update_session(self, session: Session) -> None:
        """Update an existing session."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE sessions
                SET language = ?, status = ?, current_question_id = ?, answers = ?, demographics = ?
                WHERE id = ?
            """, (
                session.language,
                session.status,
                session.current_question_id,
                json.dumps(session.answers),
                json.dumps(session.demographics),
                session.id
            ))
    
    def delete_session(self, session_id: str) -> None:
        """Delete a session."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
    
    # =========================================================================
    # Case Operations
    # =========================================================================
    
    def create_case(self, case: Case) -> None:
        """Create a new case."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO cases (id, session_id, ticket_id, created_at, risk_band, demographics, answers, summary, key_flags, triggered_rules, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                case.id,
                case.session_id,
                case.ticket_id,
                case.created_at.isoformat(),
                case.risk_band,
                json.dumps(case.demographics),
                json.dumps(case.answers),
                case.summary,
                json.dumps(case.key_flags),
                json.dumps(case.triggered_rules),
                case.status
            ))
    
    def get_case(self, case_id: str) -> Optional[Case]:
        """Get case by ID."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM cases WHERE id = ?", (case_id,))
            row = cursor.fetchone()
            
            if row is None:
                return None
            
            return self._row_to_case(row)
    
    def get_case_by_ticket(self, ticket_id: str) -> Optional[Case]:
        """Get case by ticket ID."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM cases WHERE ticket_id = ?", (ticket_id,))
            row = cursor.fetchone()
            
            if row is None:
                return None
            
            return self._row_to_case(row)
    
    def list_cases(self, status: Optional[str] = None, limit: int = 100) -> List[Case]:
        """List cases, optionally filtered by status."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            if status:
                cursor.execute(
                    "SELECT * FROM cases WHERE status = ? ORDER BY created_at DESC LIMIT ?",
                    (status, limit)
                )
            else:
                cursor.execute(
                    "SELECT * FROM cases ORDER BY created_at DESC LIMIT ?",
                    (limit,)
                )
            
            rows = cursor.fetchall()
            return [self._row_to_case(row) for row in rows]
    
    def update_case(self, case: Case) -> None:
        """Update an existing case."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE cases
                SET status = ?, summary = ?, key_flags = ?
                WHERE id = ?
            """, (
                case.status,
                case.summary,
                json.dumps(case.key_flags),
                case.id
            ))
    
    def _row_to_case(self, row: sqlite3.Row) -> Case:
        """Convert database row to Case object."""
        return Case(
            id=row["id"],
            session_id=row["session_id"],
            ticket_id=row["ticket_id"],
            created_at=datetime.fromisoformat(row["created_at"]),
            risk_band=row["risk_band"],
            demographics=json.loads(row["demographics"]),
            answers=json.loads(row["answers"]),
            summary=row["summary"],
            key_flags=json.loads(row["key_flags"]),
            triggered_rules=json.loads(row["triggered_rules"]),
            status=row["status"]
        )
    
    # =========================================================================
    # Statistics
    # =========================================================================
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Count cases by risk band
            cursor.execute("""
                SELECT risk_band, COUNT(*) as count
                FROM cases
                GROUP BY risk_band
            """)
            risk_counts = {row["risk_band"]: row["count"] for row in cursor.fetchall()}
            
            # Count cases by status
            cursor.execute("""
                SELECT status, COUNT(*) as count
                FROM cases
                GROUP BY status
            """)
            status_counts = {row["status"]: row["count"] for row in cursor.fetchall()}
            
            # Total counts
            cursor.execute("SELECT COUNT(*) FROM sessions")
            total_sessions = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM cases")
            total_cases = cursor.fetchone()[0]
            
            return {
                "total_sessions": total_sessions,
                "total_cases": total_cases,
                "by_risk": risk_counts,
                "by_status": status_counts
            }
