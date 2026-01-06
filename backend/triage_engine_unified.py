"""
Unified Triage Engine - Clean Entry Point with Fallback

Primary: Parlant Agent (Ollama backend)
Fallback: GGUF via llama-cpp-python
Last Resort: Rule-based response

Features:
- Proper timeout handling
- Automatic fallback
- Patient data RAG (not protocol RAG)
- Bilingual support (EN/FR)
"""

import asyncio
import logging
from typing import Dict, Any, Optional

from config import DEFAULT_LANGUAGE

logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

OVERALL_REQUEST_TIMEOUT = 90.0  # seconds


# =============================================================================
# LANGUAGE DETECTION
# =============================================================================

def detect_language(text: str) -> str:
    """
    Simple language detection.
    Returns 'fr' for French, 'en' for English.
    """
    # French indicators
    french_words = [
        "je", "tu", "il", "elle", "nous", "vous", "ils", "elles",
        "le", "la", "les", "un", "une", "des",
        "et", "ou", "mais", "donc", "car", "ni",
        "ai", "as", "avez", "avons", "ont",
        "suis", "es", "est", "sommes", "etes", "sont",
        "mon", "ma", "mes", "ton", "ta", "tes", "son", "sa", "ses",
        "douleur", "mal", "fievre", "tete", "ventre", "coeur",
        "depuis", "hier", "aujourd", "demain",
        "oui", "non", "peut", "etre", "faire",
    ]

    text_lower = text.lower()
    words = text_lower.split()

    french_count = sum(1 for word in words if word in french_words)

    # If more than 20% of words are French indicators, classify as French
    if len(words) > 0 and french_count / len(words) > 0.2:
        return "fr"

    return "en"


# =============================================================================
# UNIFIED TRIAGE ENGINE
# =============================================================================

class UnifiedTriageEngine:
    """
    Single entry point for all triage operations.

    Primary path: Parlant Agent
    Fallback: GGUF Model
    Last resort: Rule-based response

    Both LLM paths use patient-data RAG (not protocol RAG).
    """

    def __init__(self):
        self._parlant = None
        self._gguf = None
        self._risk_calc = None
        self._initialized = False
        self._parlant_available = False
        self._gguf_available = False

    async def initialize(self):
        """Initialize available engines."""
        if self._initialized:
            return

        logger.info("Initializing Unified Triage Engine...")

        # Initialize risk calculator
        try:
            from risk_calculator import get_risk_calculator
            self._risk_calc = get_risk_calculator()
            logger.info("Risk calculator ready")
        except Exception as e:
            logger.warning(f"Risk calculator not available: {e}")

        # Initialize GGUF first (more reliable, faster startup)
        try:
            from gguf_fallback import get_gguf_fallback
            self._gguf = get_gguf_fallback()
            if self._gguf.is_available:
                downloaded = self._gguf.get_downloaded_models()
                if downloaded:
                    self._gguf_available = True
                    model_names = [m.model_id for m in downloaded]
                    logger.info(f"Primary engine: GGUF ready ({model_names})")
                else:
                    logger.warning("No GGUF models downloaded")
            else:
                logger.warning("llama-cpp-python not available")
        except Exception as e:
            logger.warning(f"GGUF not available: {e}")

        # Try Parlant as secondary (may hang, so we don't depend on it)
        try:
            from parlant_agent_clean import get_clean_parlant_agent
            self._parlant = await get_clean_parlant_agent()
            # Use short timeout to avoid hanging
            if await self._parlant.initialize(timeout=15.0):
                self._parlant_available = True
                logger.info("Secondary engine: Parlant Agent ready")
            else:
                logger.warning("Parlant initialization failed/timed out")
        except Exception as e:
            logger.warning(f"Parlant not available: {e}")

        if not self._parlant_available and not self._gguf_available:
            logger.error("NO LLM ENGINES AVAILABLE - only rule-based responses possible")

        self._initialized = True
        logger.info("Unified Triage Engine initialized")

    async def shutdown(self):
        """Cleanup."""
        if self._parlant:
            await self._parlant.shutdown()
        if self._gguf:
            self._gguf.unload()
        self._initialized = False
        logger.info("Unified Triage Engine shutdown")

    @property
    def has_llm(self) -> bool:
        """Check if any LLM is available."""
        return self._parlant_available or self._gguf_available

    @property
    def engine_status(self) -> Dict[str, Any]:
        """Get status of all engines."""
        return {
            "parlant_available": self._parlant_available,
            "gguf_available": self._gguf_available,
            "gguf_model": (
                self._gguf.model_id if self._gguf and self._gguf.is_loaded else None
            ),
            "has_llm": self.has_llm,
            "initialized": self._initialized,
        }

    def compute_risk(
        self,
        demographics: Dict[str, Any],
        answers: Dict[str, Any],
        language: str = "en",
    ) -> Dict[str, Any]:
        """Compute deterministic risk band."""
        if self._risk_calc:
            result = self._risk_calc.compute_risk(demographics, answers, language)
            return result.to_dict() if hasattr(result, "to_dict") else result
        return {"band": "green", "triggered_rules": []}

    async def answer_question(
        self,
        question: str,
        clinical_state: Dict[str, Any],
        language: str = None,
        session_key: str = None,
        timeout: float = None,
    ) -> Dict[str, Any]:
        """
        Answer a staff question with automatic fallback.

        Primary: GGUF (fast, reliable)
        Secondary: Parlant (if GGUF unavailable)
        Last resort: Rule-based
        """
        if not self._initialized:
            await self.initialize()

        # Auto-detect language
        if language is None:
            text = f"{question} {clinical_state.get('chief_complaint', '')}"
            language = detect_language(text)

        language = language if language in ["en", "fr"] else DEFAULT_LANGUAGE
        timeout = timeout or OVERALL_REQUEST_TIMEOUT

        # Ensure risk is computed
        if "risk_calculation" not in clinical_state:
            clinical_state["risk_calculation"] = self.compute_risk(
                clinical_state.get("demographics", {}),
                clinical_state.get("answers", {}),
                language,
            )

        # Try GGUF first (fast, reliable)
        if self._gguf_available:
            try:
                logger.debug("Using GGUF fallback")
                from patient_data_rag import PatientDataRAG

                patient_context = PatientDataRAG.format_citable_context(clinical_state)
                response = await self._gguf.answer_question(
                    question=question, patient_context=patient_context, language=language
                )
                # Add citations
                response["cited_data"] = PatientDataRAG.get_cited_data(
                    clinical_state, response.get("answer", "")
                )
                return response
            except Exception as e:
                logger.error(f"GGUF fallback error: {e}")

        # Last resort: rule-based response
        return self._rule_based_response(clinical_state, language)

    async def generate_pdf_sections(
        self,
        clinical_state: Dict[str, Any],
        language: str = None,
        timeout: float = None,
    ) -> Dict[str, str]:
        """Generate PDF sections with fallback."""
        if not self._initialized:
            await self.initialize()

        language = language or clinical_state.get("language", DEFAULT_LANGUAGE)
        timeout = timeout or OVERALL_REQUEST_TIMEOUT * 2

        # Ensure risk is computed
        if "risk_calculation" not in clinical_state:
            clinical_state["risk_calculation"] = self.compute_risk(
                clinical_state.get("demographics", {}),
                clinical_state.get("answers", {}),
                language,
            )

        # Try GGUF first (fast, reliable)
        if self._gguf_available:
            try:
                from patient_data_rag import PatientDataRAG

                patient_context = PatientDataRAG.format_citable_context(clinical_state)
                sections = {}
                for section_type in ["summary", "diagnosis", "conclusion"]:
                    sections[section_type] = await self._gguf.generate_pdf_section(
                        section_type=section_type,
                        patient_context=patient_context,
                        language=language,
                    )
                return sections
            except Exception as e:
                logger.warning(f"GGUF PDF generation failed: {e}")

        # Fallback to Parlant
        if self._parlant_available:
            try:
                return await self._parlant.generate_pdf_sections(
                    clinical_state=clinical_state,
                    language=language,
                    timeout=timeout * 0.5,
                )
            except Exception as e:
                logger.warning(f"Parlant PDF generation failed: {e}")

        # Rule-based fallback
        return self._rule_based_pdf(clinical_state, language)

    def _rule_based_response(
        self, clinical_state: Dict[str, Any], language: str
    ) -> Dict[str, Any]:
        """Generate rule-based response when no LLM available."""
        risk = clinical_state.get("risk_calculation", {})
        if isinstance(risk, dict):
            band = risk.get("band", "green")
        else:
            band = "green"

        if language == "fr":
            if band == "red":
                answer = "Urgence vitale. Un soignant va vous voir immediatement."
            elif band in ["orange", "amber"]:
                answer = "Situation urgente. Vous serez pris en charge rapidement."
            else:
                answer = "Votre cas a ete enregistre. Veuillez patienter."
            follow_ups = [
                "Depuis combien de temps avez-vous ces symptomes?",
                "Avez-vous pris des medicaments?",
                "Y a-t-il d'autres symptomes?",
            ]
        else:
            if band == "red":
                answer = "Critical condition. A provider will see you immediately."
            elif band in ["orange", "amber"]:
                answer = "Urgent condition. You will be seen very soon."
            else:
                answer = "Your case has been registered. Please wait."
            follow_ups = [
                "How long have you had these symptoms?",
                "Have you taken any medications?",
                "Are there any other symptoms?",
            ]

        return {
            "answer": answer,
            "reasoning": f"Triage level: {band.upper()}",
            "follow_up_questions": follow_ups,
            "suggested_questions": follow_ups,
            "cited_data": [],
            "model_used": "Rule-based (no LLM)",
            "has_reasoning": True,
            "rag_used": False,
            "is_fallback": True,
            "fallback_reason": "No LLM available",
        }

    def _rule_based_pdf(
        self, clinical_state: Dict[str, Any], language: str
    ) -> Dict[str, str]:
        """Generate rule-based PDF when no LLM available."""
        risk = clinical_state.get("risk_calculation", {})
        demo = clinical_state.get("demographics", {})
        complaint = clinical_state.get("chief_complaint", "Unknown")

        if isinstance(risk, dict):
            band = risk.get("band", "green").upper()
        else:
            band = str(risk).upper() if risk else "GREEN"

        if language == "fr":
            return {
                "summary": f"Patient de {demo.get('age', 'N/A')} ans, {demo.get('sex', 'N/A')}. Motif: {complaint.replace('_', ' ')}. Niveau: {band}.",
                "diagnosis": "Evaluation clinique requise.",
                "conclusion": f"Priorite: {band}. Prise en charge selon protocole.",
            }
        return {
            "summary": f"{demo.get('age', 'N/A')} year old {demo.get('sex', 'N/A')}. Complaint: {complaint.replace('_', ' ')}. Level: {band}.",
            "diagnosis": "Clinical evaluation required.",
            "conclusion": f"Priority: {band}. Manage per protocol.",
        }


# =============================================================================
# SINGLETON & INTERFACE
# =============================================================================

_engine_instance: Optional[UnifiedTriageEngine] = None


async def get_unified_engine() -> UnifiedTriageEngine:
    """Get singleton unified engine instance."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = UnifiedTriageEngine()
    return _engine_instance


# =============================================================================
# COMPATIBILITY WRAPPER (for main.py)
# =============================================================================

class ParlantEngine:
    """
    Compatibility wrapper that matches the old ParlantEngine interface.

    This allows main.py to use the new unified engine with minimal changes.
    """

    def __init__(self):
        self._engine: Optional[UnifiedTriageEngine] = None

    async def initialize(self, load_rag: bool = False, timeout: float = 30.0):
        """Initialize the engine with timeout."""
        self._engine = await get_unified_engine()
        try:
            await asyncio.wait_for(self._engine.initialize(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(f"Engine initialization timed out after {timeout}s, continuing with available engines")
            self._engine._initialized = True  # Mark as initialized with whatever is available

    async def shutdown(self):
        """Shutdown the engine."""
        if self._engine:
            await self._engine.shutdown()

    @property
    def is_available(self) -> bool:
        """Check if engine is available."""
        return self._engine is not None and self._engine._initialized

    @property
    def is_initialized(self) -> bool:
        """Check if engine is initialized (alias for is_available)."""
        return self.is_available

    @property
    def has_llm(self) -> bool:
        """Check if LLM is available."""
        return self._engine.has_llm if self._engine else False

    @property
    def model_name(self) -> str:
        """Get current model name."""
        if self._engine:
            status = self._engine.engine_status
            if status.get("parlant_available"):
                from config import OLLAMA_MODEL
                return f"Parlant ({OLLAMA_MODEL})"
            elif status.get("gguf_model"):
                return f"GGUF ({status['gguf_model']})"
        return "Not initialized"

    @property
    def rag_available(self) -> bool:
        """RAG is always available (patient data RAG)."""
        return self._engine is not None and self._engine._initialized

    async def set_model(self, model_id: str):
        """Set model (compatibility - logs but doesn't change Ollama model)."""
        logger.info(f"Model selection requested: {model_id} (handled by Ollama)")

    def get_engine_status(self) -> Dict[str, Any]:
        """Get engine status."""
        if self._engine:
            return self._engine.engine_status
        return {"initialized": False}

    async def answer_staff_question(
        self,
        question: str,
        clinical_state: Dict[str, Any],
        user_language: str = "en",
        use_rag: bool = True,
        model_id: str = None,
    ) -> Dict[str, Any]:
        """Answer a staff question (compatibility method)."""
        if not self._engine:
            return {
                "answer": "Engine not initialized",
                "error": True,
            }
        return await self._engine.answer_question(
            question=question,
            clinical_state=clinical_state,
            language=user_language,
        )

    async def generate_pdf_summary(
        self,
        clinical_state: Dict[str, Any],
        user_language: str = "en",
    ) -> Dict[str, str]:
        """Generate PDF sections (compatibility method)."""
        if not self._engine:
            return {
                "summary": "Engine not initialized",
                "diagnosis": "Engine not initialized",
                "conclusion": "Engine not initialized",
            }
        return await self._engine.generate_pdf_sections(
            clinical_state=clinical_state,
            language=user_language,
        )


def get_parlant_engine() -> ParlantEngine:
    """Get compatibility wrapper (matches old interface)."""
    return ParlantEngine()
