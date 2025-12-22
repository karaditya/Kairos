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
from datetime import datetime
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel

from database import Database, Session, Case
from pdf_generator import generate_medical_report_pdf
from triage_engine import TriageEngine, RiskEngine
from multi_model_engine import MultiModelEngine, get_engine
from model_registry import get_all_models, get_model_config, model_to_dict
from drbert_engine import DrBERTEngine, get_drbert_engine, DRBERT_MODELS

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

class CaseListResponse(BaseModel):
    cases: List[Dict[str, Any]]

class ModelSwitchRequest(BaseModel):
    model_id: str

class StaffAskRequestWithModel(BaseModel):
    question: str
    model_id: Optional[str] = None

# =============================================================================
# Application Lifecycle
# =============================================================================

# Global instances
db: Database = None
triage_engine: TriageEngine = None
risk_engine: RiskEngine = None
reasoning_engine: MultiModelEngine = None
drbert_engine: DrBERTEngine = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize components on startup, cleanup on shutdown."""
    global db, triage_engine, risk_engine, reasoning_engine, drbert_engine

    print("Starting Offline Triage MVP...")

    # Initialize database
    db = Database(DB_PATH)
    db.initialize()
    print(f"  Database initialized: {DB_PATH}")

    # Initialize triage engine
    triage_engine = TriageEngine(CONFIG_DIR)
    print(f"  Triage engine loaded: {len(triage_engine.trees)} complaint trees")

    # Initialize risk engine
    risk_engine = RiskEngine(CONFIG_DIR)
    print(f"  Risk engine loaded: {len(risk_engine.rules)} red-flag rules")

    # Initialize multi-model reasoning engine
    models_dir = os.path.dirname(MODEL_PATH)
    reasoning_engine = get_engine(models_dir=models_dir, reinitialize=True)

    if reasoning_engine.is_loaded:
        current = reasoning_engine.get_current_model()
        if current:
            print(f"  LLM loaded: {current['name']} ({current['model_id']})")
        else:
            stats = reasoning_engine.get_engine_stats()
            print(f"  LLM loaded: {stats.get('model_loaded', 'Unknown')}")
    else:
        print(f"  No LLM loaded (will use fallback mode)")

    # List available models
    available = [m for m in reasoning_engine.get_available_models() if m.get("is_available")]
    print(f"  Available models: {len(available)}")

    # Initialize DrBERT engine (lazy load - doesn't load model by default)
    drbert_engine = get_drbert_engine(reinitialize=True)
    print(f"  DrBERT engine initialized (load on demand)")

    print("Triage MVP ready!")
    print(f"   Patient interface: http://localhost:8000/")
    print(f"   Staff interface: http://localhost:8000/staff")

    yield

    # Cleanup
    print("Shutting down...")
    if reasoning_engine:
        reasoning_engine.unload_model()
    if drbert_engine:
        drbert_engine.unload_model()

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
    complaints = triage_engine.get_available_complaints(session.language)

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
    tree = triage_engine.get_tree(complaint_id)
    if not tree:
        tree = triage_engine.get_tree("general")

    # Get first question
    first_q = triage_engine.get_first_question(tree)
    session.current_question_id = first_q["id"]
    session.status = "triage"
    session.answers["_tree_id"] = tree["id"]
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
    tree_id = session.answers.get("_tree_id", "general")
    tree = triage_engine.get_tree(tree_id)

    # Determine next question
    next_q = triage_engine.get_next_question(
        tree,
        request.question_id,
        request.answer,
        session.answers
    )

    # Calculate progress
    total_questions = triage_engine.count_questions(tree)
    answered = len([k for k in session.answers.keys() if not k.startswith("_")])
    progress = min(0.15 + (answered / total_questions) * 0.7, 0.85)

    if next_q is None:
        # Triage complete - compute risk
        session.status = "complete"
        session.current_question_id = None

        # Compute risk band (pass language to use appropriate system)
        risk_result = risk_engine.compute_risk(
            session.demographics,
            session.answers,
            language=session.language
        )
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
        session.current_question_id = next_q["id"]
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

    # Use the engine to generate a summary via answer_staff_question
    summary_prompt = "Provide a brief 2-3 sentence summary of this patient's condition and triage priority."
    ai_result = reasoning_engine.answer_staff_question(summary_prompt, clinical_state)
    summary_text = ai_result.get("answer", "Triage assessment complete. Please proceed as directed.")

    # Extract key flags from triggered rules
    key_flags = reasoning_engine._extract_key_flags(clinical_state)

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

    # Generate extended summary with diagnosis and conclusion using the engine
    # Returns {"summary": ..., "diagnosis": ..., "conclusion": ...}
    pdf_sections = reasoning_engine.generate_pdf_summary(clinical_state)

    # Extract key flags
    key_flags = reasoning_engine._extract_key_flags(clinical_state)

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
    """Staff helper - ask questions about a case with optional model selection."""
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

    # Use reasoning engine to answer (with optional model selection)
    result = reasoning_engine.answer_staff_question(
        request.question,
        clinical_state,
        model_id=request.model_id
    )

    return StaffAskResponse(
        answer=result.get("answer", "No answer generated."),
        reasoning=result.get("reasoning", ""),
        has_reasoning=result.get("has_reasoning", bool(result.get("reasoning"))),
        suggested_questions=result.get("suggested_questions", []),
        cited_data=result.get("cited_data", []),
        model_used=result.get("model_used", "Unknown"),
        disclaimer="This is decision support only. Clinical judgment is required for all patient care decisions."
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
# Model Management Endpoints
# =============================================================================

@app.get("/models")
async def list_models():
    """List all supported models with availability status."""
    return {
        "models": reasoning_engine.get_available_models(),
        "current_model": reasoning_engine.get_current_model(),
    }

@app.get("/models/current")
async def get_current_model():
    """Get currently loaded model information."""
    current = reasoning_engine.get_current_model()
    if not current:
        return {"loaded": False, "message": "No model currently loaded"}
    return {"loaded": True, "model": current}

@app.get("/models/{model_id}")
async def get_model_info(model_id: str):
    """Get detailed information about a specific model."""
    config = get_model_config(model_id)
    if not config:
        raise HTTPException(status_code=404, detail=f"Unknown model: {model_id}")

    model_info = model_to_dict(config)
    model_info["is_available"] = reasoning_engine._model_exists(model_id)

    current = reasoning_engine._current_model
    model_info["is_loaded"] = (current is not None and current.model_id == model_id)

    return model_info

@app.post("/models/switch", dependencies=[Depends(verify_staff_pin)])
async def switch_model(request: ModelSwitchRequest):
    """Switch to a different model (staff only)."""
    config = get_model_config(request.model_id)
    if not config:
        raise HTTPException(status_code=404, detail=f"Unknown model: {request.model_id}")

    if not reasoning_engine._model_exists(request.model_id):
        raise HTTPException(
            status_code=400,
            detail=f"Model not downloaded. Download {config.filename} first."
        )

    success = reasoning_engine.switch_model(request.model_id)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to switch model")

    return {
        "success": True,
        "message": f"Switched to {config.name}",
        "model": reasoning_engine.get_current_model()
    }

@app.get("/models/stats", dependencies=[Depends(verify_staff_pin)])
async def get_model_stats():
    """Get engine statistics (staff only)."""
    return reasoning_engine.get_engine_stats()

# =============================================================================
# DrBERT Endpoints (French Medical BERT)
# =============================================================================

@app.get("/drbert/models")
async def list_drbert_models():
    """List available DrBERT models."""
    return {
        "models": drbert_engine.get_available_models(),
        "current": drbert_engine.get_current_model(),
        "stats": drbert_engine.get_engine_stats()
    }

@app.post("/drbert/load/{model_id}")
async def load_drbert_model(model_id: str):
    """
    Load a DrBERT model with automatic GPU/CPU distribution.
    Models: drbert-4gb, drbert-7gb, drbert-4gb-pubmed
    """
    if model_id not in DRBERT_MODELS:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown model: {model_id}. Available: {list(DRBERT_MODELS.keys())}"
        )

    success = drbert_engine.load_model(model_id)
    if not success:
        raise HTTPException(
            status_code=500,
            detail="Failed to load model. Check if transformers is installed."
        )

    return {
        "success": True,
        "model": drbert_engine.get_current_model(),
        "stats": drbert_engine.get_engine_stats()
    }

@app.post("/drbert/unload")
async def unload_drbert_model():
    """Unload current DrBERT model to free memory."""
    drbert_engine.unload_model()
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
    if not drbert_engine.is_loaded:
        raise HTTPException(status_code=400, detail="No DrBERT model loaded. Call /drbert/load/{model_id} first.")

    texts = request.get("texts", [])
    if not texts:
        raise HTTPException(status_code=400, detail="No texts provided")

    pooling = request.get("pooling", "mean")

    try:
        embeddings = drbert_engine.get_embeddings(texts, pooling=pooling)
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
    if not drbert_engine.is_loaded:
        raise HTTPException(status_code=400, detail="No DrBERT model loaded. Call /drbert/load/{model_id} first.")

    text1 = request.get("text1", "")
    text2 = request.get("text2", "")

    if not text1 or not text2:
        raise HTTPException(status_code=400, detail="Both text1 and text2 required")

    try:
        similarity = drbert_engine.compute_similarity(text1, text2)
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
    if not drbert_engine.is_loaded:
        raise HTTPException(status_code=400, detail="No DrBERT model loaded. Call /drbert/load/{model_id} first.")

    text = request.get("text", "")
    if not text:
        raise HTTPException(status_code=400, detail="No text provided")

    if "<mask>" not in text.lower():
        raise HTTPException(status_code=400, detail="Text must contain <mask> token")

    top_k = request.get("top_k", 5)

    try:
        predictions = drbert_engine.fill_mask(text, top_k=top_k)
        return {"predictions": predictions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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

def localize_question(question: Dict, language: str) -> Dict:
    """Localize question text."""
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
