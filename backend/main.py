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
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from database import Database, Session, Case
from triage_engine import TriageEngine, RiskEngine
from reasoning_engine import ReasoningEngine

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
    cited_data: List[str]
    disclaimer: str

class CaseListResponse(BaseModel):
    cases: List[Dict[str, Any]]

# =============================================================================
# Application Lifecycle
# =============================================================================

# Global instances
db: Database = None
triage_engine: TriageEngine = None
risk_engine: RiskEngine = None
reasoning_engine: ReasoningEngine = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize components on startup, cleanup on shutdown."""
    global db, triage_engine, risk_engine, reasoning_engine
    
    print("🏥 Starting Offline Triage MVP...")
    
    # Initialize database
    db = Database(DB_PATH)
    db.initialize()
    print(f"  ✓ Database initialized: {DB_PATH}")
    
    # Initialize triage engine
    triage_engine = TriageEngine(CONFIG_DIR)
    print(f"  ✓ Triage engine loaded: {len(triage_engine.trees)} complaint trees")
    
    # Initialize risk engine
    risk_engine = RiskEngine(CONFIG_DIR)
    print(f"  ✓ Risk engine loaded: {len(risk_engine.rules)} red-flag rules")
    
    # Initialize reasoning engine (LLM)
    reasoning_engine = ReasoningEngine(MODEL_PATH)
    if reasoning_engine.is_loaded:
        print(f"  ✓ LLM loaded: {MODEL_PATH}")
    else:
        print(f"  ⚠ LLM not loaded (will use fallback mode)")
    
    print("🚀 Triage MVP ready!")
    print(f"   Patient interface: http://localhost:8000/")
    print(f"   Staff interface: http://localhost:8000/staff")
    
    yield
    
    # Cleanup
    print("Shutting down...")
    if reasoning_engine:
        reasoning_engine.unload()

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
        
        # Compute risk band
        risk_result = risk_engine.compute_risk(session.demographics, session.answers)
        session.answers["_risk_band"] = risk_result["band"]
        session.answers["_triggered_rules"] = risk_result["triggered_rules"]
        
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
    
    # Use TRM-style reasoning to generate summary
    clinical_state = {
        "demographics": session.demographics,
        "chief_complaint": session.answers.get("chief_complaint", "unknown"),
        "answers": {k: v for k, v in session.answers.items() if not k.startswith("_")},
        "risk_band": risk_band,
        "triggered_rules": triggered_rules
    }
    
    # Run reasoning loop
    summary, key_flags = reasoning_engine.generate_summary(clinical_state)
    
    # Waiting instruction based on risk
    waiting_instructions = {
        "red": get_text("wait_red", session.language),
        "amber": get_text("wait_amber", session.language),
        "green": get_text("wait_green", session.language)
    }
    
    # Create case record for staff
    case = Case(
        id=str(uuid.uuid4()),
        session_id=session_id,
        ticket_id=ticket_id,
        created_at=datetime.now(),
        risk_band=risk_band,
        demographics=session.demographics,
        answers=session.answers,
        summary=summary,
        key_flags=key_flags,
        triggered_rules=triggered_rules,
        status="pending"
    )
    db.create_case(case)
    
    return SummaryResponse(
        session_id=session_id,
        ticket_id=ticket_id,
        risk_band=risk_band,
        risk_color={"red": "#dc3545", "amber": "#ffc107", "green": "#28a745"}[risk_band],
        triggered_rules=[{"rule": r["id"], "description": r["description"]} for r in triggered_rules],
        summary=summary,
        key_flags=key_flags,
        waiting_instruction=waiting_instructions.get(risk_band, waiting_instructions["amber"]),
        demographics=session.demographics,
        answers={k: v for k, v in session.answers.items() if not k.startswith("_")},
        disclaimer="This tool does not provide medical diagnosis or treatment. A healthcare professional will review your case."
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
    print(f"[DEBUG] Auth attempt - Received PIN: '{request.pin}' (len={len(request.pin)}), Expected: '{STAFF_PIN}' (len={len(STAFF_PIN)})")
    print(f"[DEBUG] PIN match: {request.pin == STAFF_PIN}")
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
async def staff_ask(case_id: str, request: StaffAskRequest, _: bool = Depends(verify_staff_pin)):
    """Staff helper - ask questions about a case."""
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
    
    # Use reasoning engine to answer
    answer, cited_data = reasoning_engine.answer_staff_question(
        request.question, 
        clinical_state
    )
    
    return StaffAskResponse(
        answer=answer,
        cited_data=cited_data,
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
# Static Files & Frontend
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def patient_ui():
    """Serve patient interface."""
    return FileResponse("../frontend-old/patient.html")

@app.get("/staff", response_class=HTMLResponse)
async def staff_ui():
    """Serve staff interface."""
    return FileResponse("../frontend-old/staff.html")

# Mount static files
app.mount("/static", StaticFiles(directory="../frontend-old"), name="static")

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
        "wait_green": "Please take a seat in the general waiting area. You will be called when it's your turn."
    },
    "es": {
        "demographics_prompt": "Por favor proporcione su información:",
        "age": "Edad",
        "sex": "Sexo",
        "pregnant": "¿Está actualmente embarazada?",
        "chief_complaint_prompt": "¿Cuál es su principal preocupación hoy?",
        "other_complaint": "Otro (por favor describa)",
        "wait_red": "Por favor diríjase inmediatamente al área de emergencias. Un miembro del personal le asistirá.",
        "wait_amber": "Por favor espere en el área de espera prioritaria. Será atendido pronto.",
        "wait_green": "Por favor tome asiento en el área de espera general. Será llamado cuando sea su turno."
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
