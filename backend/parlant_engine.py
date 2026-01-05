"""
Parlant Engine - Main Entry Point for Parlant-Native Medical Triage

This is the main engine class that replaces multi_model_engine.py.
Uses Parlant agents with strict medical guidelines for all LLM operations.

Features:
- Single LLM call per request (vs 3-LLM review in old system)
- Architecturally enforced guidelines (no hallucination)
- Dual-language support (English MTS / French SFMU)
- DrBERT RAG integration for protocol grounding
- Clean, maintainable architecture
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from contextlib import asynccontextmanager

from config import (
    OLLAMA_MODEL,
    RAG_ENABLED,
    DEFAULT_LANGUAGE,
    SUPPORTED_LANGUAGES,
)
from parlant_agents import ParlantAgentManager, SyncParlantAgentManager
from drbert_rag import get_drbert_rag, DrBERTRAG
from risk_calculator import get_risk_calculator, RiskCalculator
from triage_rules import get_triage_rules, TriageRules

logger = logging.getLogger(__name__)

# Track active model (can be changed at runtime)
_active_model: str = OLLAMA_MODEL


# =============================================================================
# ASYNC PARLANT ENGINE
# =============================================================================

class ParlantEngine:
    """
    Parlant-native medical triage engine.

    Replaces the complex 3-LLM pipeline in MultiModelEngine with
    Parlant's guideline-based architecture.

    Key improvements:
    - Single LLM call per request
    - Guaranteed output structure via tools
    - No regex parsing needed
    - Explainable decisions (which guidelines triggered)
    - ~500 lines vs ~2100 lines
    """

    def __init__(self):
        """Initialize the Parlant engine."""
        self._agent_manager: Optional[ParlantAgentManager] = None
        self._drbert: Optional[DrBERTRAG] = None
        self._risk_calculator: Optional[RiskCalculator] = None
        self._triage_rules: Optional[TriageRules] = None
        self._initialized = False
        self._ollama_available = False
        self._current_model: str = OLLAMA_MODEL

    async def initialize(self, load_rag: bool = True):
        """
        Initialize all components.

        Args:
            load_rag: Whether to load DrBERT RAG (may take time)
        """
        if self._initialized:
            logger.info("ParlantEngine already initialized")
            return

        logger.info("Initializing ParlantEngine...")

        # Initialize rule engines (fast)
        self._risk_calculator = get_risk_calculator()
        self._triage_rules = get_triage_rules()
        logger.info("Risk calculator and triage rules loaded")

        # Initialize DrBERT RAG if enabled
        if load_rag and RAG_ENABLED:
            self._drbert = get_drbert_rag()
            if not self._drbert.is_loaded:
                logger.info("DrBERT RAG ready (model loads on demand)")
            logger.info("DrBERT RAG initialized")
        else:
            logger.info("RAG disabled or skipped")

        # Initialize Parlant agent manager (with graceful fallback)
        self._agent_manager = ParlantAgentManager(self._drbert)
        try:
            await self._agent_manager.initialize()
            # Check if agent manager actually initialized (Ollama available)
            self._ollama_available = self._agent_manager._initialized
            if self._ollama_available:
                logger.info("Parlant agent manager initialized with Ollama")
            else:
                logger.info("Running in fallback mode (rule-based only, no LLM)")
        except Exception as e:
            self._ollama_available = False
            logger.warning(f"Ollama not available: {e}")
            logger.info("Running in fallback mode (rule-based only, no LLM)")

        self._initialized = True
        logger.info("ParlantEngine initialization complete")

    async def shutdown(self):
        """Clean up resources."""
        logger.info("Shutting down ParlantEngine...")

        if self._agent_manager:
            await self._agent_manager.shutdown()
            self._agent_manager = None

        if self._drbert:
            self._drbert.unload_model()
            self._drbert = None

        self._initialized = False
        logger.info("ParlantEngine shutdown complete")

    async def set_model(self, model_id: str) -> bool:
        """
        Switch the active LLM model.

        Args:
            model_id: The Ollama model ID to use (e.g., 'mistral', 'llama3.2')

        Returns:
            True if model was switched successfully
        """
        global _active_model

        old_model = self._current_model
        self._current_model = model_id
        _active_model = model_id

        logger.info(f"Switched model from {old_model} to {model_id}")

        # Update agent manager if initialized
        if self._agent_manager:
            await self._agent_manager.set_model(model_id)

        return True

    def get_current_model(self) -> str:
        """Get the currently active model ID."""
        return self._current_model

    # =========================================================================
    # MAIN API: Answer Staff Questions
    # =========================================================================

    async def answer_staff_question(
        self,
        question: str,
        clinical_state: Dict[str, Any],
        user_language: str = None,
        session_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Answer a staff question about a patient case.

        This is the main method for chat interactions.

        Args:
            question: The question to answer
            clinical_state: Patient clinical data including:
                - demographics: {age, sex, pregnant}
                - answers: {symptom answers from triage}
                - chief_complaint: str
                - risk_calculation: {band, level, triggered_rules}
            user_language: 'en' or 'fr' (auto-detected if None)
            session_key: Optional session key for context continuity

        Returns:
            Response dictionary:
            {
                "answer": str,              # Patient-friendly response
                "reasoning": str,           # Clinical reasoning for staff
                "follow_up_questions": [str],  # 3 follow-up questions
                "suggested_questions": [str],  # Alias for compatibility
                "protocol_applied": str,    # Protocol name if RAG used
                "protocol_source": str,     # "MTS" or "SFMU"
                "rag_used": bool,
                "model_used": str,          # "Parlant (mistral)"
                "has_reasoning": bool,
                "cited_data": [str],        # Evidence from patient data
                "guidelines_triggered": [str],  # Which Parlant guidelines fired
            }
        """
        if not self._initialized:
            await self.initialize()

        # Auto-detect language if not provided
        if user_language is None:
            from drbert_rag import detect_language
            # Detect from question and chief complaint
            text = f"{question} {clinical_state.get('chief_complaint', '')}"
            user_language = detect_language(text)

        # Validate language
        if user_language not in SUPPORTED_LANGUAGES:
            user_language = DEFAULT_LANGUAGE

        # Compute risk if not already done
        if "risk_calculation" not in clinical_state:
            clinical_state["risk_calculation"] = self._compute_risk(
                clinical_state, user_language
            )

        # Check if Ollama/Parlant is available
        if not self._ollama_available or self._agent_manager is None:
            # Return fallback response based on rules
            return self._generate_fallback_response(clinical_state, user_language)

        # Get response from Parlant agent
        try:
            response = await self._agent_manager.answer_question(
                question=question,
                clinical_state=clinical_state,
                user_language=user_language,
                session_key=session_key
            )
        except Exception as e:
            logger.error(f"Parlant agent error: {e}")
            return self._generate_fallback_response(clinical_state, user_language)

        # Extract cited data from patient state
        response["cited_data"] = self._extract_cited_data(clinical_state)

        return response

    # =========================================================================
    # MAIN API: Generate PDF Summary
    # =========================================================================

    async def generate_pdf_summary(
        self,
        clinical_state: Dict[str, Any],
        user_language: str = None
    ) -> Dict[str, str]:
        """
        Generate PDF clinical referral letter sections.

        Args:
            clinical_state: Patient clinical data
            user_language: 'en' or 'fr' (auto-detected if None)

        Returns:
            Dictionary with three sections:
            {
                "summary": str,     # Clinical summary (triage nurse voice)
                "diagnosis": str,   # Diagnostic assessment (senior physician)
                "conclusion": str,  # Recommendations (attending physician)
            }
        """
        if not self._initialized:
            await self.initialize()

        # Auto-detect language if not provided
        if user_language is None:
            user_language = clinical_state.get("language", DEFAULT_LANGUAGE)

        # Compute risk if not already done
        if "risk_calculation" not in clinical_state:
            clinical_state["risk_calculation"] = self._compute_risk(
                clinical_state, user_language
            )

        # Check if Ollama/Parlant is available
        if not self._ollama_available or self._agent_manager is None:
            return self._generate_fallback_pdf(clinical_state, user_language)

        try:
            return await self._agent_manager.generate_pdf_sections(
                clinical_state=clinical_state,
                user_language=user_language
            )
        except Exception as e:
            logger.error(f"PDF generation error: {e}")
            return self._generate_fallback_pdf(clinical_state, user_language)

    # =========================================================================
    # RISK CALCULATION (Deterministic)
    # =========================================================================

    def _compute_risk(
        self,
        clinical_state: Dict[str, Any],
        language: str
    ) -> Dict[str, Any]:
        """Compute risk band using deterministic rules."""
        demographics = clinical_state.get("demographics", {})
        answers = clinical_state.get("answers", {})

        result = self._risk_calculator.compute_risk(
            demographics=demographics,
            answers=answers,
            language=language
        )

        return result.to_dict()

    def compute_risk(
        self,
        demographics: Dict[str, Any],
        answers: Dict[str, Any],
        language: str = "en"
    ) -> Dict[str, Any]:
        """
        Public method to compute risk band.

        Args:
            demographics: {age, sex, pregnant}
            answers: Triage question answers
            language: 'en' or 'fr'

        Returns:
            Risk result dictionary
        """
        if not self._risk_calculator:
            self._risk_calculator = get_risk_calculator()

        result = self._risk_calculator.compute_risk(
            demographics=demographics,
            answers=answers,
            language=language
        )
        return result.to_dict()

    # =========================================================================
    # TRIAGE TREE NAVIGATION
    # =========================================================================

    def get_available_complaints(self, language: str = "en") -> List[Dict[str, str]]:
        """Get list of available chief complaints."""
        if not self._triage_rules:
            self._triage_rules = get_triage_rules()
        return self._triage_rules.get_available_complaints(language)

    def get_triage_tree(self, tree_id: str, language: str = "en"):
        """Get a specific triage tree."""
        if not self._triage_rules:
            self._triage_rules = get_triage_rules()
        return self._triage_rules.get_tree(tree_id, language)

    def get_next_question(
        self,
        tree_id: str,
        current_question_id: str,
        answer: Any,
        all_answers: Dict[str, Any],
        language: str = "en"
    ):
        """Get next triage question based on current answer."""
        if not self._triage_rules:
            self._triage_rules = get_triage_rules()

        tree = self._triage_rules.get_tree(tree_id, language)
        if not tree:
            return None

        return self._triage_rules.get_next_question(
            tree=tree,
            current_question_id=current_question_id,
            answer=answer,
            all_answers=all_answers
        )

    # =========================================================================
    # RAG OPERATIONS
    # =========================================================================

    async def search_protocols(
        self,
        query: str,
        language: str = "en",
        n_results: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Search medical protocols via RAG.

        Args:
            query: Search query
            language: 'en' or 'fr'
            n_results: Number of results

        Returns:
            List of protocol dictionaries
        """
        if not self._drbert or not self._drbert.rag_available:
            return []

        results = self._drbert.search_protocols(
            query=query,
            language=language,
            n_results=n_results
        )

        return [p.to_dict() for p in results.protocols]

    def get_rag_stats(self) -> Dict[str, Any]:
        """Get RAG system statistics."""
        if not self._drbert:
            return {"available": False}
        return self._drbert.get_stats()

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _extract_cited_data(self, clinical_state: Dict[str, Any]) -> List[str]:
        """Extract key data points from clinical state for citation."""
        cited = []

        # Demographics
        demo = clinical_state.get("demographics", {})
        if demo.get("age"):
            cited.append(f"Age: {demo['age']}")
        if demo.get("sex"):
            cited.append(f"Sex: {demo['sex']}")
        if demo.get("pregnant"):
            cited.append("Pregnant: Yes")

        # Key symptoms
        answers = clinical_state.get("answers", {})
        key_fields = [
            "pain_severity", "pain_type", "pain_location",
            "breathing_severity", "temperature_value",
            "shortness_breath", "sweating", "nausea",
        ]
        for field in key_fields:
            if field in answers and answers[field]:
                display_name = field.replace("_", " ").title()
                cited.append(f"{display_name}: {answers[field]}")

        # Risk calculation
        risk = clinical_state.get("risk_calculation", {})
        if risk.get("band"):
            cited.append(f"Risk Band: {risk['band'].upper()}")
        if risk.get("triggered_rules"):
            for rule in risk["triggered_rules"][:2]:
                cited.append(f"Alert: {rule.get('description', '')}")

        return cited

    def _generate_fallback_response(
        self,
        clinical_state: Dict[str, Any],
        language: str
    ) -> Dict[str, Any]:
        """
        Generate a rule-based fallback response when LLM is unavailable.

        This provides basic triage information without LLM generation.
        """
        risk = clinical_state.get("risk_calculation", {})
        band = risk.get("band", "green")
        triggered = risk.get("triggered_rules", [])

        # Generate basic response based on risk band
        if language == "fr":
            if band == "red":
                answer = "Urgence vitale détectée. Un soignant va vous voir immédiatement."
            elif band in ["orange", "amber"]:
                answer = "Situation urgente. Vous serez pris en charge rapidement."
            else:
                answer = "Votre cas a été enregistré. Veuillez patienter en salle d'attente."

            follow_ups = [
                "Depuis combien de temps avez-vous ces symptômes?",
                "Avez-vous pris des médicaments?",
                "Y a-t-il d'autres symptômes?"
            ]
            reasoning = f"Niveau de tri: {band.upper()}. " + (
                f"Règles déclenchées: {len(triggered)}." if triggered else "Aucune alerte critique."
            )
        else:
            if band == "red":
                answer = "Critical condition detected. A healthcare provider will see you immediately."
            elif band in ["orange", "amber"]:
                answer = "Urgent condition. You will be seen very soon."
            else:
                answer = "Your case has been registered. Please wait in the waiting area."

            follow_ups = [
                "How long have you had these symptoms?",
                "Have you taken any medications?",
                "Are there any other symptoms?"
            ]
            reasoning = f"Triage level: {band.upper()}. " + (
                f"Triggered rules: {len(triggered)}." if triggered else "No critical alerts."
            )

        return {
            "answer": answer,
            "reasoning": reasoning,
            "follow_up_questions": follow_ups,
            "suggested_questions": follow_ups,
            "protocol_applied": "",
            "protocol_source": "SFMU" if language == "fr" else "MTS",
            "rag_used": False,
            "model_used": "Rule-based fallback (no LLM available)",
            "has_reasoning": True,
            "cited_data": self._extract_cited_data(clinical_state),
            "guidelines_triggered": [],
            "is_fallback": True,
            "fallback_reason": "Ollama LLM service not available - using deterministic rules only",
        }

    def _generate_fallback_pdf(
        self,
        clinical_state: Dict[str, Any],
        language: str
    ) -> Dict[str, str]:
        """Generate fallback PDF sections when LLM is unavailable."""
        risk = clinical_state.get("risk_calculation", {})
        band = risk.get("band", "green")
        demo = clinical_state.get("demographics", {})
        complaint = clinical_state.get("chief_complaint", "Unknown")

        if language == "fr":
            summary = f"Patient de {demo.get('age', 'N/A')} ans, {demo.get('sex', 'N/A')}. Motif: {complaint}. Niveau de tri: {band.upper()}."
            diagnosis = "Évaluation clinique requise par un médecin."
            conclusion = f"Priorité: {band.upper()}. Prise en charge selon protocole de tri."
        else:
            summary = f"{demo.get('age', 'N/A')} year old {demo.get('sex', 'N/A')}. Chief complaint: {complaint}. Triage level: {band.upper()}."
            diagnosis = "Clinical evaluation required by physician."
            conclusion = f"Priority: {band.upper()}. Manage according to triage protocol."

        return {
            "summary": summary,
            "diagnosis": diagnosis,
            "conclusion": conclusion,
        }

    # =========================================================================
    # PROPERTIES
    # =========================================================================

    @property
    def is_initialized(self) -> bool:
        """Check if engine is initialized."""
        return self._initialized

    @property
    def rag_available(self) -> bool:
        """Check if RAG is available."""
        return self._drbert is not None and self._drbert.rag_available

    @property
    def model_name(self) -> str:
        """Get the LLM model name."""
        return f"Parlant ({self._current_model})"


# =============================================================================
# SYNC PARLANT ENGINE
# =============================================================================

class SyncParlantEngine:
    """
    Synchronous wrapper around ParlantEngine.

    For compatibility with existing synchronous code.
    """

    def __init__(self):
        self._async_engine = ParlantEngine()
        self._loop = None

    def _get_loop(self):
        """Get or create an event loop."""
        try:
            self._loop = asyncio.get_event_loop()
            if self._loop.is_closed():
                raise RuntimeError("Loop closed")
        except RuntimeError:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
        return self._loop

    def initialize(self, load_rag: bool = True):
        """Initialize synchronously."""
        loop = self._get_loop()
        loop.run_until_complete(self._async_engine.initialize(load_rag))

    def shutdown(self):
        """Shutdown synchronously."""
        loop = self._get_loop()
        loop.run_until_complete(self._async_engine.shutdown())

    def answer_staff_question(
        self,
        question: str,
        clinical_state: Dict[str, Any],
        user_language: str = None,
        session_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """Answer a staff question synchronously."""
        loop = self._get_loop()
        return loop.run_until_complete(
            self._async_engine.answer_staff_question(
                question, clinical_state, user_language, session_key
            )
        )

    def generate_pdf_summary(
        self,
        clinical_state: Dict[str, Any],
        user_language: str = None
    ) -> Dict[str, str]:
        """Generate PDF sections synchronously."""
        loop = self._get_loop()
        return loop.run_until_complete(
            self._async_engine.generate_pdf_summary(clinical_state, user_language)
        )

    def compute_risk(
        self,
        demographics: Dict[str, Any],
        answers: Dict[str, Any],
        language: str = "en"
    ) -> Dict[str, Any]:
        """Compute risk band (already sync)."""
        return self._async_engine.compute_risk(demographics, answers, language)

    def get_available_complaints(self, language: str = "en") -> List[Dict[str, str]]:
        """Get available complaints (already sync)."""
        return self._async_engine.get_available_complaints(language)

    def search_protocols(
        self,
        query: str,
        language: str = "en",
        n_results: int = 3
    ) -> List[Dict[str, Any]]:
        """Search protocols synchronously."""
        loop = self._get_loop()
        return loop.run_until_complete(
            self._async_engine.search_protocols(query, language, n_results)
        )

    def get_rag_stats(self) -> Dict[str, Any]:
        """Get RAG stats (already sync)."""
        return self._async_engine.get_rag_stats()

    @property
    def is_initialized(self) -> bool:
        return self._async_engine.is_initialized

    @property
    def rag_available(self) -> bool:
        return self._async_engine.rag_available

    @property
    def model_name(self) -> str:
        return self._async_engine.model_name


# =============================================================================
# CONTEXT MANAGER
# =============================================================================

@asynccontextmanager
async def parlant_engine_context(load_rag: bool = True):
    """
    Async context manager for ParlantEngine.

    Usage:
        async with parlant_engine_context() as engine:
            response = await engine.answer_staff_question(...)
    """
    engine = ParlantEngine()
    try:
        await engine.initialize(load_rag)
        yield engine
    finally:
        await engine.shutdown()


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_engine_instance: Optional[ParlantEngine] = None


async def get_parlant_engine(reinitialize: bool = False) -> ParlantEngine:
    """
    Get singleton ParlantEngine instance.

    Args:
        reinitialize: Force create new instance

    Returns:
        ParlantEngine instance
    """
    global _engine_instance

    if _engine_instance is None or reinitialize:
        _engine_instance = ParlantEngine()
        await _engine_instance.initialize()

    return _engine_instance


def get_sync_parlant_engine() -> SyncParlantEngine:
    """
    Get a synchronous ParlantEngine wrapper.

    Returns:
        SyncParlantEngine instance
    """
    engine = SyncParlantEngine()
    engine.initialize()
    return engine
