"""
Parlant Tools for Medical Triage

Tools that Parlant agents use to access RAG, patient context,
risk calculation, and response formatting.

These tools provide grounded information - the agent uses these
to make factually accurate responses.
"""

import logging
from typing import Dict, List, Any, Optional

from parlant import sdk as p

from config import SUPPORTED_LANGUAGES, DEFAULT_LANGUAGE
from drbert_rag import get_drbert_rag, DrBERTRAG, detect_language
from risk_calculator import get_risk_calculator, RiskResult
from triage_rules import get_triage_rules

logger = logging.getLogger(__name__)

# Global reference to DrBERT engine (set via set_drbert_engine)
_drbert_engine: Optional[DrBERTRAG] = None


def set_drbert_engine(engine: DrBERTRAG):
    """Set the DrBERT engine for RAG tools."""
    global _drbert_engine
    _drbert_engine = engine
    logger.info("DrBERT engine connected to Parlant tools")


def get_drbert_engine() -> Optional[DrBERTRAG]:
    """Get the current DrBERT engine."""
    global _drbert_engine
    if _drbert_engine is None:
        _drbert_engine = get_drbert_rag()
    return _drbert_engine


# =============================================================================
# PROTOCOL SEARCH TOOL (RAG)
# =============================================================================

@p.tool
def search_medical_protocols(
    context: p.ToolContext,
    query: str,
    language: str = "auto",
    n_results: int = 3
) -> p.ToolResult:
    """
    Search medical protocols using DrBERT semantic search.

    Use this tool to find relevant medical triage protocols based on
    patient symptoms. The returned protocols should be cited in your response.

    Args:
        context: Parlant tool context
        query: Symptom description or medical query
        language: 'en' for English (MTS), 'fr' for French (SFMU), or 'auto'
        n_results: Number of protocols to return (default 3)

    Returns:
        ToolResult with matched protocols and their content
    """
    engine = get_drbert_engine()

    if engine is None or not engine.rag_available:
        return p.ToolResult(
            data={
                "success": False,
                "error": "RAG not available",
                "protocols": [],
            },
            metadata={"rag_available": False}
        )

    # Auto-detect language if needed
    if language == "auto":
        language = detect_language(query)

    try:
        results = engine.search_protocols(
            query=query,
            language=language,
            n_results=n_results
        )

        protocols = []
        for protocol in results.protocols:
            protocols.append({
                "id": protocol.id,
                "title": protocol.title,
                "text": protocol.text,
                "source": protocol.source,
                "category": protocol.category,
                "relevance_score": protocol.relevance_score,
                "keywords": protocol.keywords,
            })

        return p.ToolResult(
            data={
                "success": True,
                "protocols": protocols,
                "query_used": results.query_used,
                "language": results.language,
                "total_found": results.total_found,
            },
            metadata={
                "rag_available": True,
                "protocol_source": "MTS" if language == "en" else "SFMU",
            }
        )

    except Exception as e:
        logger.error(f"Protocol search error: {e}")
        return p.ToolResult(
            data={
                "success": False,
                "error": str(e),
                "protocols": [],
            },
            metadata={"rag_available": True, "error": True}
        )


# =============================================================================
# RISK CALCULATION TOOL
# =============================================================================

@p.tool
def calculate_risk_band(
    context: p.ToolContext,
    demographics_json: str,
    answers_json: str,
    language: str = "en"
) -> p.ToolResult:
    """
    Calculate deterministic risk band based on patient data.

    This uses rule-based computation - the result CANNOT be overridden.
    Use this to ground your triage priority in the validated rules.

    Args:
        context: Parlant tool context
        demographics_json: Patient demographics as JSON string (age, sex, pregnant)
        answers_json: Triage question answers as JSON string
        language: 'en' for Manchester, 'fr' for SFMU

    Returns:
        ToolResult with computed risk band and triggered rules
    """
    import json

    try:
        demographics = json.loads(demographics_json) if demographics_json else {}
        answers = json.loads(answers_json) if answers_json else {}

        calculator = get_risk_calculator()
        result = calculator.compute_risk(
            demographics=demographics,
            answers=answers,
            language=language
        )

        return p.ToolResult(
            data={
                "success": True,
                "level": result.level,
                "band": result.band,
                "color": result.color,
                "name": result.name,
                "description": result.description,
                "max_wait_minutes": result.max_wait_minutes,
                "triggered_rules": result.triggered_rules,
            },
            metadata={
                "language": language,
                "system": "SFMU" if language == "fr" else "Manchester",
            }
        )

    except Exception as e:
        logger.error(f"Risk calculation error: {e}")
        return p.ToolResult(
            data={
                "success": False,
                "error": str(e),
                "band": "green",
                "level": "4",
            },
            metadata={"error": True}
        )


# =============================================================================
# PATIENT CONTEXT TOOL
# =============================================================================

@p.tool
def get_patient_context(
    context: p.ToolContext,
    clinical_state_json: str = ""
) -> p.ToolResult:
    """
    Format patient clinical state into a structured context card.

    Use this to understand the complete patient picture before
    generating a response.

    Args:
        context: Parlant tool context
        clinical_state_json: Full clinical state as JSON string (optional, uses context if empty)

    Returns:
        ToolResult with formatted patient context
    """
    import json

    try:
        # Try to get from parameter or context
        if clinical_state_json:
            clinical_state = json.loads(clinical_state_json)
        else:
            clinical_state = context.customer_data.get("patient_data", {})

        # Extract key sections
        demographics = clinical_state.get("demographics", {})
        answers = clinical_state.get("answers", {})
        risk = clinical_state.get("risk_calculation", {})
        chief_complaint = clinical_state.get("chief_complaint", "Not specified")

        # Format patient card
        patient_card = format_patient_card(clinical_state)

        # Build structured context
        context_data = {
            "patient_card": patient_card,
            "demographics": {
                "age": demographics.get("age"),
                "sex": demographics.get("sex"),
                "pregnant": demographics.get("pregnant", False),
            },
            "chief_complaint": chief_complaint,
            "symptoms": _extract_symptoms(answers),
            "vital_signs": _extract_vitals(answers),
            "medical_history": _extract_history(answers),
            "current_risk": {
                "band": risk.get("band", "unknown"),
                "level": risk.get("level", "unknown"),
                "triggered_rules": risk.get("triggered_rules", []),
            },
        }

        return p.ToolResult(
            data=context_data,
            metadata={"has_vitals": bool(context_data["vital_signs"])}
        )

    except Exception as e:
        logger.error(f"Error building patient context: {e}")
        return p.ToolResult(
            data={"error": str(e), "patient_card": "Error loading patient data"},
            metadata={"error": True}
        )


# =============================================================================
# RESPONSE FORMATTING TOOL
# =============================================================================

@p.tool
def format_triage_response(
    context: p.ToolContext,
    answer: str,
    reasoning: str,
    follow_up_questions: List[str],
    protocol_applied: str = "",
    risk_band: str = "green",
    language: str = "en"
) -> p.ToolResult:
    """
    Format a triage response into the required structure.

    Use this to ensure your response has all required sections
    in the correct format.

    Args:
        context: Parlant tool context
        answer: Patient-friendly answer
        reasoning: Clinical reasoning for staff
        follow_up_questions: List of 3 follow-up questions
        protocol_applied: Name of protocol used (if any)
        risk_band: Computed risk band
        language: Response language

    Returns:
        ToolResult with formatted response structure
    """
    # Ensure exactly 3 follow-up questions
    if len(follow_up_questions) < 3:
        default_questions = _get_default_questions(language)
        follow_up_questions = (follow_up_questions + default_questions)[:3]
    elif len(follow_up_questions) > 3:
        follow_up_questions = follow_up_questions[:3]

    formatted_response = {
        "answer": answer,
        "reasoning": reasoning,
        "follow_up_questions": follow_up_questions,
        "suggested_questions": follow_up_questions,  # Alias for compatibility
        "protocol_applied": protocol_applied,
        "rag_used": bool(protocol_applied),
        "risk_band": risk_band,
        "has_reasoning": bool(reasoning),
        "language": language,
    }

    return p.ToolResult(
        data=formatted_response,
        metadata={"format_version": "2.0", "complete": True}
    )


# =============================================================================
# LANGUAGE DETECTION TOOL
# =============================================================================

@p.tool
def detect_query_language(
    context: p.ToolContext,
    text: str
) -> p.ToolResult:
    """
    Detect the language of medical text.

    Use this to determine whether to use English (MTS) or French (SFMU)
    protocols and response format.

    Args:
        context: Parlant tool context
        text: Text to analyze

    Returns:
        ToolResult with detected language code
    """
    detected = detect_language(text)

    return p.ToolResult(
        data={
            "language": detected,
            "language_name": "French" if detected == "fr" else "English",
            "triage_system": "SFMU" if detected == "fr" else "Manchester",
        },
        metadata={"confidence": "keyword-based"}
    )


# =============================================================================
# TRIAGE TREE TOOL
# =============================================================================

@p.tool
def get_available_complaints(
    context: p.ToolContext,
    language: str = "en"
) -> p.ToolResult:
    """
    Get list of available chief complaints for triage.

    Use this to understand what complaint categories are available
    in the triage system.

    Args:
        context: Parlant tool context
        language: 'en' or 'fr'

    Returns:
        ToolResult with list of complaint categories
    """
    try:
        triage = get_triage_rules()
        complaints = triage.get_available_complaints(language)

        return p.ToolResult(
            data={
                "success": True,
                "complaints": complaints,
                "count": len(complaints),
            },
            metadata={"language": language}
        )

    except Exception as e:
        logger.error(f"Error getting complaints: {e}")
        return p.ToolResult(
            data={"success": False, "error": str(e), "complaints": []},
            metadata={"error": True}
        )


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def format_patient_card(clinical_state: Dict[str, Any]) -> str:
    """
    Format patient clinical state as a human-readable card.

    Args:
        clinical_state: Full clinical state dictionary

    Returns:
        Formatted string for display
    """
    lines = []

    # Risk band header
    risk = clinical_state.get("risk_calculation", {})
    band = risk.get("band", "unknown").upper()
    lines.append(f"*** TRIAGE PRIORITY: {band} ***")
    lines.append("-" * 35)

    # Demographics
    demo = clinical_state.get("demographics", {})
    if demo:
        age = demo.get("age", "Unknown")
        sex = demo.get("sex", "Unknown")
        pregnant = demo.get("pregnant", False)

        demo_line = f"PATIENT: {age} year old {sex}"
        if pregnant:
            demo_line += " (pregnant)"
        lines.append(demo_line)

    # Chief complaint
    complaint = clinical_state.get("chief_complaint", "")
    if complaint:
        readable = complaint.replace("_", " ").title()
        lines.append(f"PRESENTING COMPLAINT: {readable}")

    # Answers/Symptoms
    answers = clinical_state.get("answers", {})
    if answers:
        lines.append("\nREPORTED INFORMATION:")
        for key, value in answers.items():
            if key.startswith("_"):
                continue
            # Format the key nicely
            display_key = key.replace("_", " ").title()
            if isinstance(value, bool):
                display_value = "Yes" if value else "No"
            elif isinstance(value, list):
                display_value = ", ".join(str(v) for v in value)
            else:
                display_value = str(value)
            lines.append(f"  - {display_key}: {display_value}")

    # Triggered rules
    triggered = risk.get("triggered_rules", [])
    if triggered:
        lines.append("\nCLINICAL ALERTS:")
        for rule in triggered[:5]:  # Show top 5
            rule_band = rule.get("band", "").upper()
            desc = rule.get("description", rule.get("id", ""))
            lines.append(f"  - [{rule_band}] {desc}")

    return "\n".join(lines)


def _extract_symptoms(answers: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract symptom information from answers."""
    symptoms = []

    symptom_keys = [
        "pain_type", "pain_severity", "pain_location", "pain_radiation",
        "pain_duration", "pain_onset", "shortness_breath", "breathing_severity",
        "sweating", "nausea", "dizziness", "palpitations", "palpitations_type",
        "fever", "temperature_value", "headache_severity", "neck_stiffness",
        "rash_description", "cough_type", "confusion",
    ]

    for key in symptom_keys:
        if key in answers and answers[key]:
            symptoms.append({
                "name": key.replace("_", " ").title(),
                "value": answers[key],
            })

    return symptoms


def _extract_vitals(answers: Dict[str, Any]) -> Dict[str, Any]:
    """Extract vital signs from answers."""
    vitals = {}

    vital_keys = {
        "temperature_value": "temperature",
        "heart_rate": "heart_rate",
        "blood_pressure_systolic": "bp_systolic",
        "blood_pressure_diastolic": "bp_diastolic",
        "spo2": "oxygen_saturation",
        "respiratory_rate": "respiratory_rate",
    }

    for answer_key, vital_name in vital_keys.items():
        if answer_key in answers and answers[answer_key]:
            vitals[vital_name] = answers[answer_key]

    return vitals


def _extract_history(answers: Dict[str, Any]) -> Dict[str, Any]:
    """Extract medical history from answers."""
    history = {}

    history_keys = [
        "heart_history", "chronic_conditions", "medications",
        "medication_list", "allergies", "previous_surgeries",
        "family_history", "smoking", "diagnosed_condition",
    ]

    for key in history_keys:
        if key in answers and answers[key]:
            history[key.replace("_", " ").title()] = answers[key]

    return history


def _get_default_questions(language: str) -> List[str]:
    """Get default follow-up questions."""
    if language == "fr":
        return [
            "Depuis combien de temps avez-vous ces symptomes?",
            "Avez-vous pris des medicaments pour cela?",
            "Y a-t-il d'autres symptomes que vous n'avez pas mentionnes?",
        ]
    return [
        "How long have you been experiencing these symptoms?",
        "Have you taken any medications for this?",
        "Are there any other symptoms you haven't mentioned?",
    ]


# =============================================================================
# TOOL COLLECTION
# =============================================================================

ALL_TOOLS = [
    search_medical_protocols,
    calculate_risk_band,
    get_patient_context,
    format_triage_response,
    detect_query_language,
    get_available_complaints,
]
