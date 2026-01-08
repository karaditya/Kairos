"""
Agent API Routes for CAE System.

Endpoints:
- GET /agent/draft - Generate Compte Rendu draft
- POST /agent/sync - Request RPA sync (creates verification)
- POST /agent/sync/{id}/approve - Approve RPA sync
- POST /agent/sync/{id}/reject - Reject RPA sync
- GET /agent/pending - List pending verifications
"""

from typing import Optional, List

from fastapi import APIRouter, HTTPException, status, Depends

from api.schemas.agent import (
    AgentDraftResponse,
    SyncRequest,
    SyncPreviewResponse,
    SyncApproveRequest,
    SyncResultResponse,
    CompteRenduContent,
    MissingField,
    FlaggedField,
    RAGCitation,
)
from api.deps import verify_staff_pin
from db.database import SessionRepository, TranscriptRepository, CompteRenduRepository
from services.parlant.session_handler import get_session_handler
from services.rpa.coordinator import get_rpa_coordinator
from config import ADMINISTRATIVE_DISCLAIMER
from utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.get("/draft", response_model=AgentDraftResponse)
async def generate_draft(
    session_id: str,
    include_transcript: bool = True,
    include_ehr: bool = True,
):
    """
    Generate Compte Rendu draft using Parlant + RAG.

    Combines transcript and EHR data to create clinical documentation.
    """
    # Get session
    session = await SessionRepository.get(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session not found: {session_id}",
        )

    # Get transcript
    transcript_text = ""
    if include_transcript:
        transcript = await TranscriptRepository.get_by_session(session_id)
        if transcript:
            transcript_text = transcript.get("full_text", "")

    # Get EHR data (placeholder - would come from vision service)
    ehr_data = {}

    # Generate CR via Parlant
    handler = await get_session_handler()
    cr_data = await handler.generate_compte_rendu(
        transcript=transcript_text,
        ehr_data=ehr_data if include_ehr else None,
        language=session.get("language", "fr"),
    )

    # Save CR to database
    cr_record = await CompteRenduRepository.create(
        session_id=session_id,
        language=session.get("language", "fr"),
        patient_name=cr_data.get("patient_name"),
        patient_dob=cr_data.get("patient_dob"),
        patient_mrn=cr_data.get("patient_mrn"),
    )

    # Update sections
    await CompteRenduRepository.update_sections(
        cr_id=cr_record["id"],
        sections=cr_data.get("sections", {}),
        missing_fields=cr_data.get("missing_fields", []),
        flagged_fields=cr_data.get("flagged_fields", []),
        rag_citations=cr_data.get("rag_citations", []),
    )

    logger.info(
        "Compte Rendu generated",
        session_id=session_id,
        cr_id=cr_record["id"],
    )

    # Build response
    sections = cr_data.get("sections", {})
    return AgentDraftResponse(
        session_id=session_id,
        compte_rendu_id=cr_record["id"],
        compte_rendu=CompteRenduContent(
            patient_name=cr_data.get("patient_name"),
            patient_dob=cr_data.get("patient_dob"),
            patient_mrn=cr_data.get("patient_mrn"),
            motif_consultation=sections.get("motif_consultation"),
            anamnese=sections.get("anamnese"),
            antecedents=sections.get("antecedents"),
            allergies=sections.get("allergies"),
            traitements_actuels=sections.get("traitements_actuels"),
            examen_clinique=sections.get("examen_clinique"),
            examens_complementaires=sections.get("examens_complementaires"),
            hypotheses_diagnostiques=sections.get("hypotheses_diagnostiques"),
            plan_therapeutique=sections.get("plan_therapeutique"),
            missing_fields=[
                MissingField(**m) for m in cr_data.get("missing_fields", [])
            ],
            flagged_fields=[
                FlaggedField(**f) for f in cr_data.get("flagged_fields", [])
            ],
        ),
        rag_citations=[
            RAGCitation(**c) for c in cr_data.get("rag_citations", [])
        ],
        model_used="Parlant (Scribe Agent)",
        disclaimer=cr_data.get("disclaimer", ADMINISTRATIVE_DISCLAIMER["fr"]),
    )


@router.post("/sync", response_model=SyncPreviewResponse)
async def request_sync(
    request: SyncRequest,
    _: bool = Depends(verify_staff_pin),
):
    """
    Request RPA sync to EHR.

    Creates a verification request that must be approved before execution.
    Returns preview of actions to be performed.
    """
    # Get compte rendu
    cr_data = await CompteRenduRepository.get(request.compte_rendu_id)
    if not cr_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Compte Rendu not found: {request.compte_rendu_id}",
        )

    # Convert to model
    from models.compte_rendu import CompteRendu
    compte_rendu = CompteRendu.from_dict(cr_data)

    # Request sync via coordinator
    coordinator = get_rpa_coordinator()

    try:
        result = await coordinator.request_sync(
            session_id=request.session_id,
            compte_rendu=compte_rendu,
            ehr_type="custom",  # Would come from session/config
            map_name="default",
            target_fields=request.target_fields if request.target_fields else None,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    logger.info(
        "Sync requested",
        session_id=request.session_id,
        verification_id=result["verification_id"],
    )

    return SyncPreviewResponse(
        status=result["status"],
        verification_id=result["verification_id"],
        preview=result["preview"],
    )


@router.post("/sync/{verification_id}/approve", response_model=SyncResultResponse)
async def approve_and_execute_sync(
    verification_id: str,
    request: SyncApproveRequest,
    _: bool = Depends(verify_staff_pin),
):
    """
    Approve and execute RPA sync.

    This executes the RPA actions after human approval.
    """
    from services.rpa.safety import get_safety_gate

    safety_gate = get_safety_gate()

    # Approve
    await safety_gate.approve(
        verification_id=verification_id,
        user_id=request.user_id,
        modifications=request.modifications if request.modifications else None,
    )

    # Execute
    coordinator = get_rpa_coordinator()
    result = await coordinator.execute_sync(verification_id)

    logger.info(
        "Sync executed",
        verification_id=verification_id,
        success=result["status"] == "completed",
    )

    return SyncResultResponse(
        status=result["status"],
        actions_executed=result["actions_executed"],
        timestamp=result.get("timestamp", ""),
        errors=result.get("errors", []),
    )


@router.post("/sync/{verification_id}/reject")
async def reject_sync(
    verification_id: str,
    user_id: str,
    _: bool = Depends(verify_staff_pin),
):
    """Reject a pending sync request."""
    from services.rpa.safety import get_safety_gate

    safety_gate = get_safety_gate()
    await safety_gate.reject(verification_id, user_id)

    logger.info("Sync rejected", verification_id=verification_id)

    return {"status": "rejected", "verification_id": verification_id}


@router.get("/pending")
async def list_pending_verifications(
    session_id: Optional[str] = None,
    _: bool = Depends(verify_staff_pin),
):
    """List pending RPA verification requests."""
    coordinator = get_rpa_coordinator()
    pending = await coordinator.get_pending(session_id)

    return {"pending": pending, "count": len(pending)}


@router.get("/compte-rendu/{cr_id}")
async def get_compte_rendu(cr_id: str):
    """Get a specific compte rendu."""
    cr_data = await CompteRenduRepository.get(cr_id)
    if not cr_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Compte Rendu not found: {cr_id}",
        )

    return cr_data


@router.post("/compte-rendu/{cr_id}/verify")
async def verify_compte_rendu(
    cr_id: str,
    user_id: str,
    _: bool = Depends(verify_staff_pin),
):
    """Mark compte rendu as verified by staff."""
    cr_data = await CompteRenduRepository.get(cr_id)
    if not cr_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Compte Rendu not found: {cr_id}",
        )

    await CompteRenduRepository.verify(cr_id, user_id)

    return {"verified": True, "verified_by": user_id}
