"""
Offline Triage Pre-Screen MVP - Main FastAPI Application

A fully functional offline triage system that:
- Guides patients through structured symptom questions
- Computes deterministic risk bands (green/amber/red)
- Uses local LLM for TRM-style reasoning and summaries
- Provides staff helper mode for case review

DISCLAIMER: This tool does not provide medical diagnosis or treatment.
A healthcare professional will review your case.
"""

import os
import json
import uuid
import hashlib
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, Header

logger = logging.getLogger(__name__)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel
import asyncio

from database import Database, Session, Case
from pdf_generator import generate_medical_report_pdf

# New Parlant-native imports
from triage_rules import get_triage_rules, TriageRules
from risk_calculator import get_risk_calculator, RiskCalculator
from parlant_engine import ParlantEngine, get_parlant_engine
from drbert_rag import get_drbert_rag, DrBERTRAG, CHROMADB_AVAILABLE, DRBERT_MODELS
from config import OLLAMA_MODEL, RAG_ENABLED, PARLANT_PORT, SUPPORTED_OLLAMA_MODELS
from ollama_manager import OllamaManager, get_ollama_manager

# Legacy imports for model downloading (still needed)
from model_downloader import (
    detect_gpu_capabilities,
    download_and_load_model,
    get_download_progress,
    get_all_downloads,
    check_model_downloaded
)

# Parlant is always available in the new architecture
PARLANT_AVAILABLE = True

# =============================================================================
# Configuration
# =============================================================================

MODEL_PATH = os.environ.get("MODEL_PATH", "../models/llama-3.2-1b-instruct-q4_k_m.gguf")
CONFIG_DIR = os.environ.get("CONFIG_DIR", "../config")
DB_PATH = os.environ.get("DB_PATH", "../data/triage.db")
STAFF_PIN = os.environ.get("STAFF_PIN", "1234")  # Default PIN for demo

# =============================================================================
# Pydantic Models
# =============================================================================

class StartSessionRequest(BaseModel):
    language: str = "en"

class StartSessionResponse(BaseModel):
    session_id: str
    first_question: Dict[str, Any]
    progress: float

class DemographicsRequest(BaseModel):
    age: int
    sex: str  # "male", "female", "other"
    pregnant: Optional[bool] = None

class AnswerRequest(BaseModel):
    question_id: str
    answer: Any  # Can be bool, str, int, or list

class AnswerResponse(BaseModel):
    next_question: Optional[Dict[str, Any]]
    progress: float
    is_complete: bool
    risk_band: Optional[str] = None

class SummaryResponse(BaseModel):
    session_id: str
    ticket_id: str
    risk_band: str
    risk_color: str
    triggered_rules: List[Dict[str, str]]
    summary: str
    key_flags: List[str]
    waiting_instruction: str
    demographics: Dict[str, Any]
    answers: Dict[str, Any]
    disclaimer: str

class StaffAuthRequest(BaseModel):
    pin: str

class StaffAskRequest(BaseModel):
    question: str

class StaffAskResponse(BaseModel):
    answer: str
    reasoning: Optional[str] = None
    has_reasoning: bool = False
    suggested_questions: List[str] = []
    cited_data: List[str]
    model_used: str
    disclaimer: str
    # RAG-specific fields
    rag_used: bool = False
    protocol_applied: Optional[str] = None  # Protocol title if RAG was used
    protocol_source: Optional[str] = None   # Protocol source (e.g., SFMU/HAS)

class CaseListResponse(BaseModel):
    cases: List[Dict[str, Any]]

class ModelSwitchRequest(BaseModel):
    model_id: str

class StaffAskRequestWithModel(BaseModel):
    question: str
    model_id: Optional[str] = None
    use_rag: bool = True


# Protocol ingestion models
class ProtocolInput(BaseModel):
    title: str
    text: str
    source: str = "SFMU/HAS"
    category: str = "general"
    priority_level: Optional[str] = None
    keywords: List[str] = []
    metadata: Dict[str, Any] = {}


class ProtocolIngestRequest(BaseModel):
    protocols: List[ProtocolInput]


class ProtocolSearchRequest(BaseModel):
    query: str
    n_results: int = 3
    category: Optional[str] = None

# =============================================================================
# Application Lifecycle
# =============================================================================

# Global instances
db: Database = None
triage_rules: TriageRules = None
risk_calculator: RiskCalculator = None
parlant_engine: ParlantEngine = None
drbert_rag: DrBERTRAG = None
ollama_manager: OllamaManager = None




@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize components on startup, cleanup on shutdown."""
    global db, triage_rules, risk_calculator, parlant_engine, drbert_rag, ollama_manager

    print("Starting Offline Triage MVP (Parlant-Native)...")

    # Initialize database
    db = Database(DB_PATH)
    db.initialize()
    print(f"  Database initialized: {DB_PATH}")

    # Initialize triage rules (deterministic question flow)
    triage_rules = get_triage_rules()
    print(f"  Triage rules loaded: {len(triage_rules.trees)} complaint trees")

    # Initialize risk calculator (deterministic risk bands)
    risk_calculator = get_risk_calculator()
    print(f"  Risk calculator loaded: {len(risk_calculator.rules_en)} EN rules, {len(risk_calculator.rules_fr)} FR rules")

    # Initialize DrBERT RAG (lazy load - doesn't load model by default)
    drbert_rag = get_drbert_rag()
    print(f"  DrBERT RAG initialized (load on demand)")

    # Initialize Ollama manager (model detection and switching)
    ollama_manager = await get_ollama_manager()
    if ollama_manager.is_available:
        ready_models = await ollama_manager.get_ready_models()
        print(f"  Ollama manager: {len(ready_models)} models ready")
    else:
        print("  Ollama manager: Ollama not available")

    # Initialize Parlant engine (main LLM engine with guidelines)
    parlant_engine = ParlantEngine()
    await parlant_engine.initialize(load_rag=RAG_ENABLED)
    print(f"  Parlant engine initialized: {parlant_engine.model_name}")

    # Report status
    print(f"  Parlant agent: ENABLED (guideline-based generation)")
    print(f"  Ollama model: {OLLAMA_MODEL}")
    print(f"  RAG available: {parlant_engine.rag_available}")

    if CHROMADB_AVAILABLE:
        stats = drbert_rag.get_stats()
        protocol_count = stats.get("total_protocols", 0)
        print(f"  VectorStore: {protocol_count} protocols indexed")
    else:
        print(f"  VectorStore: ChromaDB not available")

    print("Triage MVP ready!")
    print(f"   Patient interface: http://localhost:8000/")
    print(f"   Staff interface: http://localhost:8000/staff")
    print(f"   Parlant port: {PARLANT_PORT}")

    yield

    # Cleanup
    print("Shutting down...")
    if ollama_manager:
        await ollama_manager.shutdown()
    if parlant_engine:
        await parlant_engine.shutdown()
    if drbert_rag:
        drbert_rag.unload_model()

# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="Offline Triage Pre-Screen MVP",
    description="Local-first medical triage assistant",
    version="1.0.0",
    lifespan=lifespan
)

# CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# Patient Endpoints
# =============================================================================

@app.post("/session/start", response_model=StartSessionResponse)
async def start_session(request: StartSessionRequest):
    """Initialize a new patient session."""
    session_id = str(uuid.uuid4())

    # Create session in database
    session = Session(
        id=session_id,
        language=request.language,
        created_at=datetime.now(),
        status="demographics",
        current_question_id=None,
        answers={},
        demographics={}
    )
    db.create_session(session)

    # Return demographics question
    first_question = {
        "id": "demographics",
        "type": "demographics",
        "text": get_text("demographics_prompt", request.language),
        "fields": [
            {"id": "age", "label": get_text("age", request.language), "type": "number", "min": 0, "max": 120},
            {"id": "sex", "label": get_text("sex", request.language), "type": "choice", "options": ["male", "female", "other"]},
            {"id": "pregnant", "label": get_text("pregnant", request.language), "type": "yesno", "conditional": "sex=female"}
        ]
    }

    return StartSessionResponse(
        session_id=session_id,
        first_question=first_question,
        progress=0.0
    )

@app.post("/session/{session_id}/demographics")
async def submit_demographics(session_id: str, demographics: DemographicsRequest):
    """Submit patient demographics and get chief complaint selection."""
    session = db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Store demographics
    session.demographics = {
        "age": demographics.age,
        "sex": demographics.sex,
        "pregnant": demographics.pregnant
    }
    session.status = "chief_complaint"
    db.update_session(session)

    # Return chief complaint selection
    complaints = triage_rules.get_available_complaints(session.language)

    return {
        "next_question": {
            "id": "chief_complaint",
            "type": "choice",
            "text": get_text("chief_complaint_prompt", session.language),
            "options": complaints,
            "allow_free_text": True,
            "free_text_label": get_text("other_complaint", session.language)
        },
        "progress": 0.1
    }

@app.post("/session/{session_id}/complaint")
async def submit_chief_complaint(session_id: str, complaint: Dict[str, Any]):
    """Submit chief complaint and start triage questions."""
    session = db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    complaint_id = complaint.get("complaint_id", "general")
    free_text = complaint.get("free_text", "")

    # Store complaint
    session.answers["chief_complaint"] = complaint_id
    session.answers["chief_complaint_text"] = free_text

    # Load triage tree for this complaint
    tree = triage_rules.get_tree(complaint_id, session.language)
    if not tree:
        # Fallback: try first available tree for this language
        available = triage_rules.get_available_complaints(session.language)
        if available:
            fallback_id = available[0]["id"]
            tree = triage_rules.get_tree(fallback_id, session.language)
            logger.warning(f"Complaint '{complaint_id}' not found, using fallback: {fallback_id}")

    if not tree:
        raise HTTPException(status_code=400, detail=f"No triage tree available for complaint: {complaint_id}")

    # Get first question
    first_q = triage_rules.get_first_question(tree)
    session.current_question_id = first_q.id if hasattr(first_q, 'id') else first_q["id"]
    session.status = "triage"
    session.answers["_tree_id"] = tree.id if hasattr(tree, 'id') else tree["id"]
    db.update_session(session)

    return {
        "next_question": localize_question(first_q, session.language),
        "progress": 0.15,
        "is_complete": False
    }

@app.post("/session/{session_id}/answer", response_model=AnswerResponse)
async def submit_answer(session_id: str, request: AnswerRequest):
    """Submit answer to current question and get next question or summary."""
    session = db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Store answer
    session.answers[request.question_id] = request.answer

    # Get current tree
    tree_id = session.answers.get("_tree_id")
    if not tree_id:
        raise HTTPException(status_code=400, detail="No triage tree set for session")
    tree = triage_rules.get_tree(tree_id, session.language)
    if not tree:
        raise HTTPException(status_code=400, detail=f"Triage tree not found: {tree_id}")

    # Determine next question
    next_q = triage_rules.get_next_question(
        tree,
        request.question_id,
        request.answer,
        session.answers
    )

    # Calculate progress
    total_questions = triage_rules.count_questions(tree)
    answered = len([k for k in session.answers.keys() if not k.startswith("_")])
    progress = min(0.15 + (answered / total_questions) * 0.7, 0.85)

    if next_q is None:
        # Triage complete - compute risk
        session.status = "complete"
        session.current_question_id = None

        # Compute risk band (pass language to use appropriate system)
        risk_result = risk_calculator.compute_risk(
            session.demographics,
            session.answers,
            language=session.language
        ).to_dict()
        session.answers["_risk_band"] = risk_result["band"]
        session.answers["_triggered_rules"] = risk_result["triggered_rules"]

        # Store triage level data for both systems (Manchester for EN, FRENCH for FR)
        session.answers["_triage_level"] = risk_result.get("level", "4")
        session.answers["_level_info"] = risk_result.get("level_info", {})
        session.answers["_risk_color"] = risk_result.get("color", "#28a745")
        session.answers["_max_wait_minutes"] = risk_result.get("max_wait_minutes", 120)

        db.update_session(session)

        return AnswerResponse(
            next_question=None,
            progress=1.0,
            is_complete=True,
            risk_band=risk_result["band"]
        )
    else:
        # Handle both dict and Question dataclass
        session.current_question_id = next_q.id if hasattr(next_q, 'id') else next_q["id"]
        db.update_session(session)

        return AnswerResponse(
            next_question=localize_question(next_q, session.language),
            progress=progress,
            is_complete=False
        )

@app.get("/session/{session_id}/summary", response_model=SummaryResponse)
async def get_summary(session_id: str):
    """Get final triage summary with LLM-refined explanation."""
    session = db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if session.status != "complete":
        raise HTTPException(status_code=400, detail="Triage not complete")

    # Get risk data
    risk_band = session.answers.get("_risk_band", "amber")
    triggered_rules = session.answers.get("_triggered_rules", [])

    # Generate ticket ID
    ticket_id = generate_ticket_id(session_id)

    # Build clinical state for LLM
    clinical_state = {
        "demographics": session.demographics,
        "chief_complaint": session.answers.get("chief_complaint", "unknown"),
        "answers": {k: v for k, v in session.answers.items() if not k.startswith("_")},
        "risk_band": risk_band,
        "triggered_rules": triggered_rules
    }

    # Use the Parlant engine to generate a summary
    summary_prompt = "Provide a brief 2-3 sentence summary of this patient's condition and triage priority."
    ai_result = await parlant_engine.answer_staff_question(
        summary_prompt,
        clinical_state,
        user_language=session.language
    )
    summary_text = ai_result.get("answer", "Triage assessment complete. Please proceed as directed.")

    # Extract key flags from triggered rules
    key_flags = [rule.get("description", "") for rule in triggered_rules[:3]]

    # Determine waiting instruction based on language/system
    triage_level = session.answers.get("_triage_level", "4")
    level_info = session.answers.get("_level_info", {})
    risk_color = session.answers.get("_risk_color", "#28a745")

    if session.language == "fr":
        # Use FRENCH triage level for French system
        waiting_instruction = get_text(f"wait_level_{triage_level}", session.language)
        if not waiting_instruction or waiting_instruction == f"wait_level_{triage_level}":
            # Fallback to band-based instruction
            waiting_instruction = get_text(f"wait_{risk_band}", session.language)
    else:
        # Use Manchester triage levels for English system
        waiting_instruction = get_text(f"manchester_wait_{triage_level}", session.language)
        if not waiting_instruction or waiting_instruction == f"manchester_wait_{triage_level}":
            # Fallback to band-based instruction
            waiting_instructions = {
                "red": get_text("wait_red", session.language),
                "orange": get_text("wait_red", session.language),  # Orange maps to urgent
                "yellow": get_text("wait_amber", session.language),
                "green": get_text("wait_green", session.language),
                "blue": get_text("wait_green", session.language),
                "amber": get_text("wait_amber", session.language)
            }
            waiting_instruction = waiting_instructions.get(risk_band, waiting_instructions.get("green"))

    # Create case record for staff
    case = Case(
        id=str(uuid.uuid4()),
        session_id=session_id,
        ticket_id=ticket_id,
        created_at=datetime.now(),
        risk_band=risk_band,
        demographics=session.demographics,
        answers=session.answers,
        summary=summary_text,
        key_flags=key_flags,
        triggered_rules=triggered_rules,
        status="pending"
    )
    db.create_case(case)

    # Note: Case indexing is now handled automatically by the Parlant engine
    # when RAG is enabled, so we skip manual indexing here

    return SummaryResponse(
        session_id=session_id,
        ticket_id=ticket_id,
        risk_band=risk_band,
        risk_color=risk_color,
        triggered_rules=[{"rule": r["id"], "description": r["description"]} for r in triggered_rules],
        summary=summary_text,
        key_flags=key_flags,
        waiting_instruction=waiting_instruction,
        demographics=session.demographics,
        answers={k: v for k, v in session.answers.items() if not k.startswith("_")},
        disclaimer=get_text("disclaimer", session.language)
    )


@app.post("/session/{session_id}/summary/pdf")
async def generate_summary_pdf(session_id: str):
    """Generate a PDF medical report for the completed triage session."""
    session = db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if session.status != "complete":
        raise HTTPException(status_code=400, detail="Triage not complete")

    # Get risk data
    risk_band = session.answers.get("_risk_band", "amber")
    triggered_rules = session.answers.get("_triggered_rules", [])

    # Generate ticket ID
    ticket_id = generate_ticket_id(session_id)

    # Build clinical state for LLM
    clinical_state = {
        "demographics": session.demographics,
        "chief_complaint": session.answers.get("chief_complaint", "unknown"),
        "answers": {k: v for k, v in session.answers.items() if not k.startswith("_")},
        "risk_band": risk_band,
        "triggered_rules": triggered_rules
    }

    # Generate extended summary with diagnosis and conclusion using Parlant engine
    # Returns {"summary": ..., "diagnosis": ..., "conclusion": ...}
    pdf_sections = await parlant_engine.generate_pdf_summary(
        clinical_state,
        user_language=session.language
    )

    # Extract key flags from triggered rules
    key_flags = [rule.get("description", "") for rule in triggered_rules[:3]]

    # Waiting instruction based on risk
    waiting_instructions = {
        "red": get_text("wait_red", session.language),
        "amber": get_text("wait_amber", session.language),
        "green": get_text("wait_green", session.language)
    }

    # Generate PDF
    pdf_bytes = generate_medical_report_pdf(
        ticket_id=ticket_id,
        risk_band=risk_band,
        demographics=session.demographics,
        summary=pdf_sections.get("summary", "Summary unavailable."),
        diagnosis=pdf_sections.get("diagnosis", "Assessment unavailable."),
        conclusion=pdf_sections.get("conclusion", "Please consult clinical staff."),
        key_flags=key_flags,
        triggered_rules=[{"rule": r["id"], "description": r["description"]} for r in triggered_rules],
        answers={k: v for k, v in session.answers.items() if not k.startswith("_")},
        waiting_instruction=waiting_instructions.get(risk_band, waiting_instructions["amber"])
    )

    # Return PDF as downloadable file
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f"attachment; filename=triage_report_{ticket_id}.pdf"
        }
    )


# =============================================================================
# Staff Endpoints
# =============================================================================

def verify_staff_pin(x_staff_pin: str = Header(None)):
    """Verify staff PIN from header."""
    if x_staff_pin != STAFF_PIN:
        raise HTTPException(status_code=401, detail="Invalid staff PIN")
    return True

@app.post("/staff/auth")
async def staff_auth(request: StaffAuthRequest):
    """Authenticate staff with PIN."""
    if request.pin == STAFF_PIN:
        return {"authenticated": True, "token": hashlib.sha256(f"{request.pin}{datetime.now()}".encode()).hexdigest()[:32]}
    raise HTTPException(status_code=401, detail="Invalid PIN")

@app.get("/staff/cases", response_model=CaseListResponse)
async def list_cases(status: Optional[str] = None, _: bool = Depends(verify_staff_pin)):
    """List all cases for staff review."""
    cases = db.list_cases(status=status)
    return CaseListResponse(cases=[
        {
            "id": c.id,
            "session_id": c.session_id,
            "ticket_id": c.ticket_id,
            "created_at": c.created_at.isoformat(),
            "risk_band": c.risk_band,
            "demographics": c.demographics,
            "summary": c.summary,
            "key_flags": c.key_flags,
            "triggered_rules": c.triggered_rules,
            "status": c.status
        }
        for c in cases
    ])

@app.get("/staff/case/{case_id}")
async def get_case(case_id: str, _: bool = Depends(verify_staff_pin)):
    """Get full case details for staff."""
    case = db.get_case(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    return {
        "id": case.id,
        "session_id": case.session_id,
        "ticket_id": case.ticket_id,
        "created_at": case.created_at.isoformat(),
        "risk_band": case.risk_band,
        "demographics": case.demographics,
        "answers": {k: v for k, v in case.answers.items() if not k.startswith("_")},
        "summary": case.summary,
        "key_flags": case.key_flags,
        "triggered_rules": case.triggered_rules,
        "status": case.status,
        "disclaimer": "This tool does not provide medical diagnosis or treatment. Clinical judgment required."
    }

@app.post("/staff/case/{case_id}/ask", response_model=StaffAskResponse)
async def staff_ask(case_id: str, request: StaffAskRequestWithModel, _: bool = Depends(verify_staff_pin)):
    """
    Staff helper - ask questions about a case with optional model selection.

    Now supports French Pivot RAG:
    - If DrBERT is loaded and protocols are indexed, uses RAG for grounded answers
    - Set use_rag=false to force standard LLM response without protocol grounding
    """
    case = db.get_case(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    # Build clinical context
    clinical_state = {
        "demographics": case.demographics,
        "chief_complaint": case.answers.get("chief_complaint", "unknown"),
        "answers": {k: v for k, v in case.answers.items() if not k.startswith("_")},
        "risk_band": case.risk_band,
        "triggered_rules": case.triggered_rules,
        "summary": case.summary
    }

    # Determine user language from session if available
    session = db.get_session(case.session_id) if case.session_id else None
    user_language = session.language if session else "en"

    # Use Parlant engine to answer (with guideline-based generation)
    result = await parlant_engine.answer_staff_question(
        request.question,
        clinical_state,
        user_language=user_language
    )

    return StaffAskResponse(
        answer=result.get("answer", "No answer generated."),
        reasoning=result.get("reasoning", ""),
        has_reasoning=result.get("has_reasoning", bool(result.get("reasoning"))),
        suggested_questions=result.get("suggested_questions", []),
        cited_data=result.get("cited_data", []),
        model_used=result.get("model_used", "Unknown"),
        disclaimer="This is decision support only. Clinical judgment is required for all patient care decisions.",
        # RAG-specific fields
        rag_used=result.get("rag_used", False),
        protocol_applied=result.get("protocol_applied", None) if result.get("rag_used") else None,
        protocol_source=result.get("protocol_source", None) if result.get("rag_used") else None
    )

@app.post("/staff/case/{case_id}/status")
async def update_case_status(case_id: str, status: Dict[str, str], _: bool = Depends(verify_staff_pin)):
    """Update case status (pending/reviewed/discharged)."""
    case = db.get_case(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    case.status = status.get("status", case.status)
    db.update_case(case)

    return {"success": True, "new_status": case.status}

# =============================================================================
# Model Management Endpoints (Enhanced with OllamaManager)
# =============================================================================

@app.get("/models")
async def list_models():
    """
    List all models with their status.

    Returns models in two categories:
    - Ready: Models that are pulled and ready to use (local + pulled Ollama)
    - Available: Models that can be downloaded from Ollama

    Response includes current active model.
    """
    if not ollama_manager:
        # Fallback if ollama_manager not initialized
        return {
            "models": [],
            "current_model": OLLAMA_MODEL,
            "ollama_available": False,
            "message": "Ollama manager not initialized"
        }

    all_models = await ollama_manager.get_all_models()
    current_model_id = ollama_manager.get_current_model()

    # Format for frontend
    models_list = []
    for m in all_models:
        models_list.append({
            "id": m.id,
            "name": m.name,
            "size_gb": m.size_gb,
            "description": m.description,
            "status": m.status,  # 'ready', 'available', 'downloading'
            "source": m.source,  # 'local', 'ollama'
            "is_active": m.is_active,
            "is_ready": m.status == "ready",
            "is_available": m.status == "available",
        })

    return {
        "models": models_list,
        "current_model": current_model_id,
        "ollama_available": ollama_manager.is_available,
        "ready_count": len([m for m in all_models if m.status == "ready"]),
        "available_count": len([m for m in all_models if m.status == "available"]),
    }

@app.get("/models/ready")
async def list_ready_models():
    """List only models that are ready to use (pulled or local)."""
    if not ollama_manager:
        return {"models": [], "error": "Ollama manager not initialized"}

    ready_models = await ollama_manager.get_ready_models()
    return {
        "models": [m.to_dict() for m in ready_models],
        "current_model": ollama_manager.get_current_model()
    }

@app.get("/models/available")
async def list_available_models():
    """List models available for download (not yet pulled)."""
    if not ollama_manager:
        return {"models": [], "error": "Ollama manager not initialized"}

    available_models = await ollama_manager.get_available_models()
    return {
        "models": [m.to_dict() for m in available_models]
    }

@app.get("/models/current")
async def get_current_model():
    """Get currently active model information."""
    if not ollama_manager:
        return {
            "model_id": OLLAMA_MODEL,
            "name": f"Parlant ({OLLAMA_MODEL})",
            "ollama_available": False,
        }

    current_id = ollama_manager.get_current_model()
    is_available = await ollama_manager.check_model_available(current_id)

    return {
        "model_id": current_id,
        "name": f"Parlant ({current_id})",
        "is_ready": is_available,
        "ollama_available": ollama_manager.is_available,
        "parlant_initialized": parlant_engine.is_initialized if parlant_engine else False,
    }

@app.post("/models/select")
async def select_model(request: ModelSwitchRequest):
    """
    Select a model for use with Parlant.

    The model must be ready (pulled or local).
    """
    if not ollama_manager:
        raise HTTPException(status_code=503, detail="Ollama manager not initialized")

    model_id = request.model_id

    # Check if model is available
    is_available = await ollama_manager.check_model_available(model_id)
    if not is_available:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model_id}' is not ready. Pull it first using POST /models/pull"
        )

    # Set the model
    success = await ollama_manager.set_model(model_id)
    if not success:
        raise HTTPException(status_code=500, detail=f"Failed to select model: {model_id}")

    # Update Parlant engine to use new model
    if parlant_engine:
        await parlant_engine.set_model(model_id)

    return {
        "success": True,
        "model_id": model_id,
        "message": f"Model switched to {model_id}"
    }

@app.post("/models/pull/{model_id}")
async def pull_model(model_id: str):
    """
    Pull a model from Ollama with progress streaming.

    Returns Server-Sent Events with download progress.
    """
    if not ollama_manager:
        raise HTTPException(status_code=503, detail="Ollama manager not initialized")

    if not ollama_manager.is_available:
        raise HTTPException(status_code=503, detail="Ollama is not available")

    # Check if model is supported
    if model_id not in SUPPORTED_OLLAMA_MODELS and not model_id.startswith("local:"):
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model: {model_id}. Supported: {list(SUPPORTED_OLLAMA_MODELS.keys())}"
        )

    # Check if already pulled
    is_ready = await ollama_manager.check_model_available(model_id)
    if is_ready:
        return {
            "status": "already_ready",
            "model_id": model_id,
            "message": "Model is already pulled and ready to use"
        }

    import json

    async def generate_progress():
        async for progress in ollama_manager.pull_model(model_id):
            yield f"data: {json.dumps(progress.to_dict())}\n\n"

    return StreamingResponse(
        generate_progress(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

@app.get("/models/{model_id}/status")
async def get_model_status(model_id: str):
    """Get status of a specific model."""
    if not ollama_manager:
        raise HTTPException(status_code=503, detail="Ollama manager not initialized")

    is_ready = await ollama_manager.check_model_available(model_id)
    is_active = ollama_manager.get_current_model() == model_id

    # Get model info from supported list
    model_info = SUPPORTED_OLLAMA_MODELS.get(model_id, {})

    return {
        "model_id": model_id,
        "name": model_info.get("name", model_id),
        "description": model_info.get("description", ""),
        "size_gb": model_info.get("size_gb", 0),
        "status": "ready" if is_ready else "available",
        "is_active": is_active,
        "is_ready": is_ready,
    }

@app.get("/models/all")
async def list_all_models_with_download_status():
    """
    List all available models with download status.
    In Parlant-native mode, we use Ollama models + DrBERT for RAG.
    """
    gpu_info = detect_gpu_capabilities()

    all_models = []

    # Ollama model (used by Parlant)
    all_models.append({
        "id": OLLAMA_MODEL,
        "name": f"Parlant ({OLLAMA_MODEL})",
        "family": "ollama",
        "description": "Parlant agent with Ollama backend",
        "type": "ollama",
        "is_downloaded": True,  # Ollama manages its own downloads
        "is_loaded": parlant_engine.is_initialized if parlant_engine else False,
        "tags": ["parlant", "guideline-based"]
    })

    # DrBERT models for RAG
    for model_id, config in DRBERT_MODELS.items():
        drbert_dict = {
            "id": model_id,
            "name": config.name,
            "family": "drbert",
            "description": config.description,
            "type": "drbert",
            "size_gb": config.size_gb,
            "hf_repo": config.hf_repo,
            "is_downloaded": drbert_rag.is_loaded if drbert_rag and drbert_rag.model_id == model_id else False,
            "is_loaded": drbert_rag.is_loaded and drbert_rag.model_id == model_id if drbert_rag else False,
            "tags": ["french", "medical", "bert", "rag"]
        }
        all_models.append(drbert_dict)

    return {
        "models": all_models,
        "gpu_info": gpu_info,
        "current_parlant": {"id": OLLAMA_MODEL, "name": parlant_engine.model_name} if parlant_engine else None,
        "current_drbert": {"id": drbert_rag.model_id} if drbert_rag and drbert_rag.is_loaded else None
    }

@app.get("/models/stats", dependencies=[Depends(verify_staff_pin)])
async def get_model_stats():
    """Get engine statistics (staff only)."""
    return {
        "parlant_initialized": parlant_engine.is_initialized if parlant_engine else False,
        "ollama_model": OLLAMA_MODEL,
        "rag_available": parlant_engine.rag_available if parlant_engine else False,
        "drbert_loaded": drbert_rag.is_loaded if drbert_rag else False,
        "drbert_model": drbert_rag.model_id if drbert_rag and drbert_rag.is_loaded else None,
        "drbert_stats": drbert_rag.get_stats() if drbert_rag else {}
    }

@app.get("/models/{model_id}")
async def get_model_info(model_id: str):
    """Get detailed information about a specific model."""
    # In Parlant-native mode, we use Ollama for LLM and DrBERT for RAG
    if model_id == OLLAMA_MODEL:
        return {
            "id": OLLAMA_MODEL,
            "name": f"Parlant ({OLLAMA_MODEL})",
            "type": "ollama",
            "is_available": True,
            "is_loaded": parlant_engine.is_initialized if parlant_engine else False,
            "description": "Parlant agent with Ollama backend for guideline-based generation"
        }
    elif model_id in DRBERT_MODELS:
        config = DRBERT_MODELS[model_id]
        return {
            "id": model_id,
            "name": config.name,
            "type": "drbert",
            "is_available": True,
            "is_loaded": drbert_rag.is_loaded and drbert_rag.model_id == model_id if drbert_rag else False,
            "description": config.description,
            "size_gb": config.size_gb,
            "hf_repo": config.hf_repo
        }
    else:
        raise HTTPException(status_code=404, detail=f"Unknown model: {model_id}")

@app.post("/models/switch", dependencies=[Depends(verify_staff_pin)])
async def switch_model(request: ModelSwitchRequest):
    """Switch to a different model (staff only).

    In Parlant-native mode:
    - Ollama models are managed externally (use `ollama pull <model>`)
    - DrBERT models can be switched via /drbert/load/{model_id}
    """
    if request.model_id in DRBERT_MODELS:
        # Switch DrBERT model
        success = drbert_rag.load_model(request.model_id)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to switch DrBERT model")
        return {
            "success": True,
            "message": f"Switched DrBERT to {request.model_id}",
            "model": {"id": drbert_rag.model_id, "type": "drbert"}
        }
    else:
        return {
            "success": False,
            "message": f"To change Ollama model, update OLLAMA_MODEL in config and restart. Current: {OLLAMA_MODEL}",
            "current_model": {"id": OLLAMA_MODEL, "type": "ollama"}
        }

# =============================================================================
# DrBERT Endpoints (French Medical BERT)
# =============================================================================

@app.get("/drbert/models")
async def list_drbert_models():
    """List available DrBERT models."""
    models = []
    for model_id, config in DRBERT_MODELS.items():
        models.append({
            "id": model_id,
            "name": config.name,
            "description": config.description,
            "size_gb": config.size_gb,
            "hf_repo": config.hf_repo
        })

    return {
        "models": models,
        "current": {"id": drbert_rag.model_id} if drbert_rag and drbert_rag.is_loaded else None,
        "stats": drbert_rag.get_stats() if drbert_rag else {}
    }

@app.post("/drbert/load/{model_id}")
async def load_drbert_model(model_id: str):
    """
    Load a DrBERT model with automatic GPU/CPU distribution.
    Models: drbert-4gb, drbert-7gb, drbert-4gb-pubmed

    Note: DrBERT is for embeddings/RAG, not text generation.
    It runs alongside Parlant (Ollama) for the full RAG pipeline.
    """
    if model_id not in DRBERT_MODELS:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown model: {model_id}. Available: {list(DRBERT_MODELS.keys())}"
        )

    success = drbert_rag.load_model(model_id)
    if not success:
        raise HTTPException(
            status_code=500,
            detail="Failed to load model. Check if transformers is installed."
        )

    return {
        "success": True,
        "model": {"id": drbert_rag.model_id},
        "stats": drbert_rag.get_stats()
    }

@app.post("/drbert/unload")
async def unload_drbert_model():
    """Unload current DrBERT model to free memory."""
    drbert_rag.unload_model()
    return {"success": True, "message": "Model unloaded"}

@app.post("/drbert/embeddings")
async def get_drbert_embeddings(request: Dict[str, Any]):
    """
    Get embeddings for French medical texts.

    Request body:
    {
        "texts": ["Le patient présente une douleur thoracique", ...],
        "pooling": "mean"  // optional: mean, cls, max
    }
    """
    if not drbert_rag.is_loaded:
        raise HTTPException(status_code=400, detail="No DrBERT model loaded. Call /drbert/load/{model_id} first.")

    texts = request.get("texts", [])
    if not texts:
        raise HTTPException(status_code=400, detail="No texts provided")

    pooling = request.get("pooling", "mean")

    try:
        embeddings = drbert_rag.get_embeddings(texts, pooling=pooling)
        return {
            "embeddings": embeddings,
            "dimension": len(embeddings[0]) if embeddings else 0,
            "count": len(embeddings)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/drbert/similarity")
async def compute_drbert_similarity(request: Dict[str, Any]):
    """
    Compute semantic similarity between two French medical texts.

    Request body:
    {
        "text1": "Le patient souffre de dyspnée",
        "text2": "Difficulté respiratoire signalée"
    }
    """
    if not drbert_rag.is_loaded:
        raise HTTPException(status_code=400, detail="No DrBERT model loaded. Call /drbert/load/{model_id} first.")

    text1 = request.get("text1", "")
    text2 = request.get("text2", "")

    if not text1 or not text2:
        raise HTTPException(status_code=400, detail="Both text1 and text2 required")

    try:
        similarity = drbert_rag.compute_similarity(text1, text2)
        return {
            "similarity": similarity,
            "text1": text1[:100],
            "text2": text2[:100]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/drbert/fill-mask")
async def drbert_fill_mask(request: Dict[str, Any]):
    """
    Fill masked token in French medical text.

    Request body:
    {
        "text": "Le patient est atteint d'une <mask> cardiaque",
        "top_k": 5
    }
    """
    if not drbert_rag.is_loaded:
        raise HTTPException(status_code=400, detail="No DrBERT model loaded. Call /drbert/load/{model_id} first.")

    text = request.get("text", "")
    if not text:
        raise HTTPException(status_code=400, detail="No text provided")

    if "<mask>" not in text.lower():
        raise HTTPException(status_code=400, detail="Text must contain <mask> token")

    top_k = request.get("top_k", 5)

    try:
        predictions = drbert_rag.fill_mask(text, top_k=top_k)
        return {"predictions": predictions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Protocol RAG Endpoints (French Pivot Architecture)
# =============================================================================

@app.post("/admin/protocols/ingest", dependencies=[Depends(verify_staff_pin)])
async def ingest_protocols(request: ProtocolIngestRequest):
    """
    Ingest medical protocols into the vector store.

    Requires:
    - DrBERT model loaded (/drbert/load/{model_id})
    - ChromaDB available

    Request body:
    {
        "protocols": [
            {
                "title": "Protocole Douleur Thoracique",
                "text": "En cas de douleur thoracique...",
                "source": "SFMU",
                "category": "cardiac",
                "priority_level": "1",
                "keywords": ["douleur", "thorax", "coeur"]
            }
        ]
    }
    """
    if not drbert_rag.is_loaded:
        raise HTTPException(
            status_code=400,
            detail="DrBERT model not loaded. Call /drbert/load/{model_id} first."
        )

    if not drbert_rag.rag_available:
        raise HTTPException(
            status_code=400,
            detail="Vector store not available. Ensure ChromaDB is installed."
        )

    # Convert Pydantic models to dicts
    protocols_data = [p.model_dump() for p in request.protocols]

    result = drbert_rag.ingest_protocols(protocols_data)

    if not result.get("success", False):
        raise HTTPException(status_code=500, detail=result.get("error", "Ingestion failed"))

    return {
        "success": True,
        "added": result.get("added", 0),
        "skipped": result.get("skipped", 0),
        "total_in_store": result.get("total_in_store", 0),
        "errors": result.get("errors", [])
    }


@app.post("/admin/protocols/search")
async def search_protocols(request: ProtocolSearchRequest):
    """
    Search protocols using medical query text.

    Works with both English and French queries.
    DrBERT provides semantic understanding for accurate retrieval.

    Request body:
    {
        "query": "Douleur thoracique constrictive",
        "n_results": 3,
        "category": "cardiac"  // optional
    }
    """
    if not drbert_rag.is_loaded:
        raise HTTPException(
            status_code=400,
            detail="DrBERT model not loaded. Call /drbert/load/{model_id} first."
        )

    if not drbert_rag.rag_available:
        raise HTTPException(
            status_code=400,
            detail="Vector store not available. Ensure ChromaDB is installed."
        )

    # Use the search_protocols method from drbert_rag
    results = drbert_rag.search_protocols(
        query=request.query,
        n_results=request.n_results,
        category=request.category
    )

    return {
        "query": request.query,
        "results": [p.to_dict() for p in results.protocols] if hasattr(results, 'protocols') else results,
        "count": len(results.protocols) if hasattr(results, 'protocols') else len(results)
    }


@app.get("/admin/protocols/stats")
async def get_protocol_stats():
    """Get vector store statistics."""
    if not drbert_rag or not drbert_rag.rag_available:
        return {
            "available": False,
            "message": "Vector store not available. Ensure ChromaDB is installed."
        }

    return drbert_rag.get_stats()


@app.delete("/admin/protocols/clear", dependencies=[Depends(verify_staff_pin)])
async def clear_protocols():
    """Clear all protocols from the vector store."""
    if not drbert_rag or not drbert_rag.rag_available:
        raise HTTPException(
            status_code=400,
            detail="Vector store not available."
        )

    success = drbert_rag.clear_protocols()

    if not success:
        raise HTTPException(status_code=500, detail="Failed to clear protocols")

    return {"success": True, "message": "All protocols cleared"}


@app.get("/rag/status")
async def get_rag_status():
    """
    Get the status of the RAG pipeline.

    Returns information about:
    - Parlant/Ollama model status
    - DrBERT model status
    - Vector store status
    - Overall RAG readiness
    """
    drbert_stats = drbert_rag.get_stats() if drbert_rag else {}

    return {
        "rag_available": parlant_engine.rag_available if parlant_engine else False,
        "llm": {
            "loaded": parlant_engine.is_initialized if parlant_engine else False,
            "model": OLLAMA_MODEL,
            "type": "parlant+ollama"
        },
        "drbert": {
            "loaded": drbert_rag.is_loaded if drbert_rag else False,
            "model": drbert_rag.model_id if drbert_rag and drbert_rag.is_loaded else None,
            "rag_ready": drbert_rag.rag_available if drbert_rag else False
        },
        "vector_store": {
            "available": CHROMADB_AVAILABLE,
            "total_protocols": drbert_stats.get("total_protocols", 0)
        },
        "message": (
            "RAG pipeline ready" if (parlant_engine and parlant_engine.rag_available)
            else "Load DrBERT and ingest protocols to enable RAG"
        )
    }


@app.get("/parlant/status")
async def get_parlant_status():
    """
    Get the status of the Parlant agent system.

    Parlant provides guideline-based generation for improved
    reliability and strict protocol adherence.
    """
    from parlant_guidelines import CORE_GUIDELINES, PDF_GUIDELINES

    return {
        "parlant_available": PARLANT_AVAILABLE,
        "parlant_enabled": True,  # Always enabled in Parlant-native mode
        "agent_initialized": parlant_engine.is_initialized if parlant_engine else False,
        "configuration": {
            "ollama_model": OLLAMA_MODEL,
            "parlant_port": PARLANT_PORT,
            "core_guidelines_count": len(CORE_GUIDELINES),
            "pdf_guidelines_count": len(PDF_GUIDELINES),
            "rag_enabled": RAG_ENABLED,
        },
        "model_name": parlant_engine.model_name if parlant_engine else None,
        "rag_available": parlant_engine.rag_available if parlant_engine else False,
        "message": (
            "Parlant agent active with guideline-based generation"
            if (parlant_engine and parlant_engine.is_initialized)
            else "Parlant engine not initialized"
        )
    }


# =============================================================================
# Model Download Endpoints (Unified for GGUF and DrBERT)
# =============================================================================

@app.get("/gpu/info")
async def get_gpu_info():
    """Get GPU capabilities and recommended settings for model loading."""
    return detect_gpu_capabilities()

@app.get("/download/status")
async def get_download_status():
    """Get status of all active/recent downloads."""
    downloads = get_all_downloads()
    return {
        "downloads": {
            k: {
                "model_id": v.model_id,
                "model_type": v.model_type,
                "status": v.status,
                "progress": v.progress,
                "downloaded_mb": v.downloaded_mb,
                "total_mb": v.total_mb,
                "error": v.error
            }
            for k, v in downloads.items()
        }
    }

@app.get("/download/status/{model_id}")
async def get_model_download_status(model_id: str):
    """Get download status for a specific model."""
    progress = get_download_progress(model_id)
    if not progress:
        return {"status": "not_started", "model_id": model_id}

    return {
        "model_id": progress.model_id,
        "model_type": progress.model_type,
        "status": progress.status,
        "progress": progress.progress,
        "downloaded_mb": progress.downloaded_mb,
        "total_mb": progress.total_mb,
        "speed_mbps": progress.speed_mbps,
        "error": progress.error
    }

@app.post("/download/{model_id}")
async def download_model_endpoint(model_id: str, auto_load: bool = True):
    """
    Download a DrBERT model with progress streaming.

    In Parlant-native mode, Ollama models are managed externally.
    This endpoint handles DrBERT model downloads for RAG.

    Returns Server-Sent Events with download progress.
    After download, automatically loads with optimal GPU/CPU split.

    Args:
        model_id: Model ID (e.g., 'drbert-7gb', 'drbert-4gb-pubmed')
        auto_load: Whether to load after download (default: True)
    """
    # Check if it's a DrBERT model
    drbert_config = DRBERT_MODELS.get(model_id)

    if not drbert_config:
        # For Ollama models, return instructions
        if model_id == OLLAMA_MODEL or "ollama" in model_id.lower():
            return {
                "status": "external_management",
                "message": f"Ollama models are managed externally. Run: ollama pull {model_id}",
                "model_id": model_id
            }
        raise HTTPException(
            status_code=404,
            detail=f"Unknown model: {model_id}. Check /drbert/models for available models."
        )

    model_type = "drbert"
    config = {
        "hf_repo": drbert_config.hf_repo,
        "size_gb": drbert_config.size_gb
    }
    models_dir = os.path.dirname(MODEL_PATH)

    # Check if already downloaded (DrBERT uses HuggingFace cache)
    if drbert_rag and drbert_rag.is_loaded and drbert_rag.model_id == model_id:
        return {
            "status": "already_loaded",
            "model_id": model_id,
            "gpu_info": detect_gpu_capabilities()
        }

    # For DrBERT, loading handles download automatically via HuggingFace
    if auto_load:
        success = drbert_rag.load_model(model_id)
        return {
            "status": "downloaded_and_loaded" if success else "download_failed",
            "loaded": success,
            "model_id": model_id,
            "gpu_info": detect_gpu_capabilities()
        }

    # Stream download progress (for DrBERT via HuggingFace)
    async def generate_events():
        import json

        for update in download_and_load_model(
            model_id=model_id,
            model_type=model_type,
            config=config,
            models_dir=models_dir,
            auto_load=auto_load
        ):
            yield f"data: {json.dumps(update)}\n\n"
            await asyncio.sleep(0.01)  # Allow other tasks

    return StreamingResponse(
        generate_events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

# =============================================================================
# Helper Functions
# =============================================================================

TRANSLATIONS = {
    "en": {
        "demographics_prompt": "Please provide your information:",
        "age": "Age",
        "sex": "Sex",
        "pregnant": "Are you currently pregnant?",
        "chief_complaint_prompt": "What is your main concern today?",
        "other_complaint": "Other (please describe)",
        "wait_red": "Please proceed immediately to the emergency area. A staff member will assist you.",
        "wait_amber": "Please wait in the priority waiting area. You will be seen soon.",
        "wait_green": "Please take a seat in the general waiting area. You will be called when it's your turn.",
        # Manchester Triage System wait instructions
        "manchester_wait_1": "IMMEDIATE ATTENTION REQUIRED. Please proceed directly to the resuscitation area. You will be seen immediately.",
        "manchester_wait_2": "VERY URGENT. Please proceed to the emergency treatment area. Target wait time: 10 minutes.",
        "manchester_wait_3": "URGENT. Please wait in the priority area. Target wait time: 60 minutes.",
        "manchester_wait_4": "STANDARD. Please take a seat in the waiting area. Target wait time: 120 minutes.",
        "manchester_wait_5": "NON-URGENT. Please take a seat in the general waiting area. Target wait time: up to 4 hours.",
        "disclaimer": "This tool does not provide medical diagnosis or treatment. A healthcare professional will review your case."
    },
    "fr": {
        "demographics_prompt": "Veuillez fournir vos informations :",
        "age": "Âge",
        "sex": "Sexe",
        "pregnant": "Êtes-vous actuellement enceinte ?",
        "chief_complaint_prompt": "Quel est le motif principal de votre consultation ?",
        "other_complaint": "Autre (veuillez décrire)",
        "wait_red": "Veuillez vous rendre immédiatement à la salle de déchocage (SAUV). Un membre du personnel vous assistera.",
        "wait_amber": "Veuillez patienter dans la zone d'attente prioritaire. Vous serez pris en charge rapidement.",
        "wait_green": "Veuillez prendre place dans la salle d'attente générale. Vous serez appelé(e) à votre tour.",
        "wait_level_1": "Dirigez-vous immédiatement vers la SAUV (salle d'accueil des urgences vitales). Prise en charge médicale en moins d'1 minute.",
        "wait_level_2": "Rendez-vous au box d'examen ou en SAUV. Prise en charge médicale en moins de 20 minutes.",
        "wait_level_3A": "Attente prioritaire. Prise en charge médicale en moins de 60 minutes.",
        "wait_level_3B": "Veuillez patienter. Prise en charge médicale en moins de 90 minutes.",
        "wait_level_4": "Salle d'attente standard. Prise en charge médicale en moins de 2 heures.",
        "wait_level_5": "Salle d'attente générale ou circuit court. Prise en charge médicale en moins de 4 heures.",
        "disclaimer": "Cet outil ne fournit pas de diagnostic ou de traitement médical. Un professionnel de santé examinera votre cas."
    },
    "es": {
        "demographics_prompt": "Por favor proporcione su información:",
        "age": "Edad",
        "sex": "Sexo",
        "pregnant": "Esta actualmente embarazada?",
        "chief_complaint_prompt": "Cual es su principal preocupacion hoy?",
        "other_complaint": "Otro (por favor describa)",
        "wait_red": "Por favor dirijase inmediatamente al area de emergencias. Un miembro del personal le asistira.",
        "wait_amber": "Por favor espere en el area de espera prioritaria. Sera atendido pronto.",
        "wait_green": "Por favor tome asiento en el area de espera general. Sera llamado cuando sea su turno.",
        "disclaimer": "Esta herramienta no proporciona diagnóstico ni tratamiento médico. Un profesional de la salud revisará su caso."
    }
}

def get_text(key: str, language: str = "en") -> str:
    """Get localized text."""
    lang_dict = TRANSLATIONS.get(language, TRANSLATIONS["en"])
    return lang_dict.get(key, TRANSLATIONS["en"].get(key, key))

def localize_question(question, language: str) -> Dict:
    """Localize question text. Accepts both dict and Question dataclass."""
    # Convert Question dataclass to dict if needed
    if hasattr(question, 'to_dict'):
        q = question.to_dict()
    elif hasattr(question, '__dict__'):
        q = {k: v for k, v in question.__dict__.items() if not k.startswith('_')}
    else:
        q = question.copy()
    if language != "en" and "translations" in q:
        trans = q["translations"].get(language, {})
        q["text"] = trans.get("text", q["text"])
        if "options" in q and "options" in trans:
            q["options"] = trans["options"]
    return q

def generate_ticket_id(session_id: str) -> str:
    """Generate human-readable ticket ID."""
    hash_part = hashlib.md5(session_id.encode()).hexdigest()[:6].upper()
    return f"T-{datetime.now().strftime('%H%M')}-{hash_part}"

# =============================================================================
# Run Server
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
