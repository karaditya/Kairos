"""Agent Pydantic schemas for CAE API."""

from typing import Optional, Dict, List, Any
from datetime import datetime
from pydantic import BaseModel


class MissingField(BaseModel):
    """Missing required field indicator."""
    field_name: str
    reason: str
    required_for: str


class FlaggedField(BaseModel):
    """Field flagged for clinical review."""
    field_name: str
    reason: str
    requires_clinical_input: bool = True


class RAGCitation(BaseModel):
    """RAG citation from protocol search."""
    protocol: str
    source: str
    relevance: float


class CompteRenduContent(BaseModel):
    """Compte Rendu content structure."""
    patient_name: Optional[str] = None
    patient_dob: Optional[str] = None
    patient_mrn: Optional[str] = None

    # Clinical sections
    motif_consultation: Optional[str] = None
    anamnese: Optional[str] = None
    antecedents: Optional[str] = None
    allergies: Optional[str] = None
    traitements_actuels: Optional[str] = None
    examen_clinique: Optional[str] = None
    examens_complementaires: Optional[str] = None
    hypotheses_diagnostiques: Optional[str] = None
    plan_therapeutique: Optional[str] = None

    # Quality indicators
    missing_fields: List[MissingField] = []
    flagged_fields: List[FlaggedField] = []


class AgentDraftResponse(BaseModel):
    """Response from agent draft generation."""
    session_id: str
    compte_rendu_id: str
    compte_rendu: CompteRenduContent
    rag_citations: List[RAGCitation]
    model_used: str
    disclaimer: str


class RPAAction(BaseModel):
    """Single RPA action."""
    action_type: str  # "click", "type", "clear", "scroll"
    target_field: str
    value: Optional[str] = None
    coordinates: Optional[List[int]] = None


class SyncRequest(BaseModel):
    """Request to sync compte rendu to EHR."""
    session_id: str
    compte_rendu_id: str
    target_fields: List[str] = []
    require_verification: bool = True


class SyncPreviewResponse(BaseModel):
    """Response with sync preview for approval."""
    status: str  # "pending_approval"
    verification_id: str
    preview: Dict[str, Any]


class SyncApproveRequest(BaseModel):
    """Request to approve RPA sync."""
    user_id: str
    modifications: Dict[str, str] = {}


class SyncResultResponse(BaseModel):
    """Response after sync execution."""
    status: str  # "completed", "failed"
    actions_executed: int
    timestamp: str
    errors: List[str] = []
