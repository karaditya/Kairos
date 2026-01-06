"""
Parlant Agents for Medical Triage

Clean Parlant-native implementation of medical triage agents.
Uses strict guidelines to ensure protocol adherence and prevent hallucination.

Agents:
- TriageAgent: Answers staff questions about patient cases
- PDFAgent: Generates clinical referral letter sections
"""

import os
import asyncio
import logging
from typing import Dict, List, Any, Optional

# Set Ollama environment variables BEFORE importing Parlant
from config import OLLAMA_MODEL, OLLAMA_BASE_URL, OLLAMA_TIMEOUT

# Get model from environment or config (allows runtime changes)
_current_model = os.environ.get("OLLAMA_MODEL", OLLAMA_MODEL)

os.environ.setdefault("OLLAMA_BASE_URL", OLLAMA_BASE_URL)
os.environ.setdefault("OLLAMA_MODEL", _current_model)
os.environ.setdefault("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
os.environ.setdefault("OLLAMA_API_TIMEOUT", str(OLLAMA_TIMEOUT))

from parlant import sdk as p
from parlant.client import AsyncParlantClient

from config import PARLANT_PORT, PARLANT_TOOL_PORT, OLLAMA_MODEL
from parlant_guidelines import (
    ALL_TRIAGE_GUIDELINES,
    ALL_PDF_GUIDELINES,
    CANNED_RESPONSES,
)
from parlant_tools import (
    ALL_TOOLS,
    set_drbert_engine,
    format_patient_card,
)
from drbert_rag import DrBERTRAG, detect_language

logger = logging.getLogger(__name__)


# =============================================================================
# TRIAGE AGENT
# =============================================================================

class TriageAgent:
    """
    Parlant agent for answering staff questions about patient cases.

    Uses strict medical guidelines to:
    - Prevent hallucination (only use provided patient data)
    - Ensure protocol citation
    - Detect and escalate red flags
    - Provide structured responses with reasoning and follow-ups
    """

    def __init__(self):
        self._agent = None
        self._agent_id = None
        self._guidelines_registered = False

    async def initialize(self, server: p.Server):
        """
        Initialize the triage agent on the Parlant server.

        Args:
            server: Parlant server instance
        """
        logger.info("Initializing Triage Agent...")

        # Create the agent
        self._agent = await server.create_agent(
            name="MedicalTriageAssistant",
            description=(
                "A medical triage assistant that assesses patient symptoms "
                "and provides guidance based on Manchester Triage System (MTS) "
                "for English patients and SFMU/CIMU protocols for French patients. "
                "Prioritizes patient safety and follows evidence-based guidelines "
                "with strict protocol adherence. Never hallucinates - only uses "
                "information from patient data and retrieved protocols."
            )
        )
        self._agent_id = self._agent.id

        # Register tools with conditions
        for tool in ALL_TOOLS:
            # Get tool name from the wrapped function
            tool_name = getattr(tool, 'function', tool).__name__ if hasattr(tool, 'function') else str(tool)
            condition = f"when the agent needs to use {tool_name} for patient assessment"
            await self._agent.attach_tool(tool, condition)
        logger.info(f"Attached {len(ALL_TOOLS)} tools to Triage Agent")

        # Register guidelines
        for guideline in ALL_TRIAGE_GUIDELINES:
            await self._agent.create_guideline(
                condition=guideline["condition"],
                action=guideline["action"],
                metadata={
                    "id": guideline["id"],
                    "priority": guideline["priority"],
                }
            )
        logger.info(f"Registered {len(ALL_TRIAGE_GUIDELINES)} triage guidelines")

        # Register canned responses for critical scenarios
        # Note: Parlant SDK 3.x uses template-based canned responses
        for key, response_data in CANNED_RESPONSES.items():
            try:
                # Use the content as a template
                template = response_data["content"]
                await self._agent.create_canned_response(template=template)
            except Exception as e:
                logger.warning(f"Failed to register canned response '{key}': {e}")
        logger.info(f"Registered {len(CANNED_RESPONSES)} canned responses")

        self._guidelines_registered = True
        logger.info("Triage Agent initialized successfully")

    @property
    def agent_id(self) -> Optional[str]:
        return self._agent_id

    @property
    def is_ready(self) -> bool:
        return self._agent is not None and self._guidelines_registered


# =============================================================================
# PDF AGENT
# =============================================================================

class PDFAgent:
    """
    Parlant agent for generating PDF clinical referral letter sections.

    Generates three sections:
    - Clinical Summary: Factual documentation (triage nurse perspective)
    - Diagnosis/Assessment: Clinical reasoning (senior physician perspective)
    - Conclusion/Recommendations: Action plan (attending physician perspective)
    """

    def __init__(self):
        self._agent = None
        self._agent_id = None
        self._guidelines_registered = False

    async def initialize(self, server: p.Server):
        """
        Initialize the PDF agent on the Parlant server.

        Args:
            server: Parlant server instance
        """
        logger.info("Initializing PDF Agent...")

        # Create the agent
        self._agent = await server.create_agent(
            name="ClinicalDocumentWriter",
            description=(
                "A clinical documentation specialist that generates professional "
                "referral letters for emergency department patients. Writes in "
                "appropriate clinical voice for each section. Uses exact values "
                "from patient data - never invents or assumes information."
            )
        )
        self._agent_id = self._agent.id

        # Register only essential tools for PDF generation
        essential_tool_names = ['get_patient_context', 'search_medical_protocols', 'detect_query_language']
        essential_tools = [
            t for t in ALL_TOOLS
            if (getattr(t, 'function', t).__name__ if hasattr(t, 'function') else '') in essential_tool_names
        ]
        for tool in essential_tools:
            tool_name = getattr(tool, 'function', tool).__name__ if hasattr(tool, 'function') else str(tool)
            condition = f"when generating clinical documentation using {tool_name}"
            await self._agent.attach_tool(tool, condition)
        logger.info(f"Attached {len(essential_tools)} tools to PDF Agent")

        # Register PDF-specific guidelines
        for guideline in ALL_PDF_GUIDELINES:
            await self._agent.create_guideline(
                condition=guideline["condition"],
                action=guideline["action"],
                metadata={
                    "id": guideline["id"],
                    "priority": guideline["priority"],
                }
            )
        logger.info(f"Registered {len(ALL_PDF_GUIDELINES)} PDF guidelines")

        self._guidelines_registered = True
        logger.info("PDF Agent initialized successfully")

    @property
    def agent_id(self) -> Optional[str]:
        return self._agent_id

    @property
    def is_ready(self) -> bool:
        return self._agent is not None and self._guidelines_registered


# =============================================================================
# AGENT MANAGER
# =============================================================================

class ParlantAgentManager:
    """
    Manages Parlant server and all agents.

    Provides a clean interface for:
    - Starting/stopping the Parlant server
    - Managing agent sessions
    - Routing requests to appropriate agents
    """

    def __init__(self, drbert_engine: Optional[DrBERTRAG] = None):
        """
        Initialize the agent manager.

        Args:
            drbert_engine: DrBERT engine for RAG (optional)
        """
        self.drbert_engine = drbert_engine
        self._server = None
        self._client = None
        self._triage_agent = TriageAgent()
        self._pdf_agent = PDFAgent()
        self._sessions: Dict[str, str] = {}  # session_key -> parlant_session_id
        self._initialized = False

        # Connect DrBERT to tools if provided
        if drbert_engine:
            set_drbert_engine(drbert_engine)

    def set_drbert_engine(self, engine: DrBERTRAG):
        """Set the DrBERT engine for RAG."""
        self.drbert_engine = engine
        set_drbert_engine(engine)
        logger.info("DrBERT engine connected to agent manager")

    async def initialize(self):
        """Initialize Parlant server and all agents."""
        if self._initialized:
            logger.info("Agent manager already initialized")
            return

        # First check if Ollama is available
        ollama_available = await self._check_ollama_available()
        if not ollama_available:
            logger.warning("Ollama not available - running in fallback mode")
            self._initialized = False
            return

        try:
            logger.info("Starting Parlant server...")

            # Start Parlant server with Ollama
            self._server = p.Server(
                port=PARLANT_PORT,
                tool_service_port=PARLANT_TOOL_PORT,
                nlp_service=p.NLPServices.ollama,
                log_level=p.LogLevel.INFO
            )
            await self._server.__aenter__()

            # Initialize agents
            await self._triage_agent.initialize(self._server)
            await self._pdf_agent.initialize(self._server)

            # Initialize client for session management
            self._client = AsyncParlantClient(
                base_url=f"http://localhost:{PARLANT_PORT}"
            )

            self._initialized = True
            logger.info("Parlant agent manager initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize agent manager: {e}")
            # Don't raise - allow fallback mode
            self._initialized = False

    async def _check_ollama_available(self) -> bool:
        """Check if Ollama server is running and has required models."""
        try:
            import ollama
            from config import OLLAMA_BASE_URL

            # Try to list models
            client = ollama.Client(host=OLLAMA_BASE_URL)
            models_response = client.list()

            # Handle both old dict format and new object format
            # New ollama library returns ListResponse with .models attribute
            if hasattr(models_response, 'models'):
                # New format: ListResponse object
                models_list = models_response.models
                model_names = []
                for m in models_list:
                    # Each model is a Model object with .model attribute
                    full_name = getattr(m, 'model', '') or getattr(m, 'name', '')
                    if full_name:
                        # Extract base name (e.g., 'mistral' from 'mistral:latest')
                        model_names.append(full_name.split(':')[0])
            else:
                # Old dict format
                models_list = models_response.get('models', [])
                model_names = [m.get('name', '').split(':')[0] for m in models_list]

            # Check if we have any models at all
            if not model_names:
                logger.warning("No Ollama models found")
                return False

            # Check if required model is available
            required_model = OLLAMA_MODEL.split(':')[0]  # Handle 'mistral:latest' format

            if required_model not in model_names:
                logger.warning(f"Required Ollama model '{required_model}' not found. Available: {model_names}")
                return False

            logger.info(f"Ollama available with models: {model_names}")
            return True

        except Exception as e:
            logger.warning(f"Ollama check failed: {e}")
            return False

    async def shutdown(self):
        """Shutdown the Parlant server and clean up resources."""
        logger.info("Shutting down agent manager...")

        self._sessions.clear()

        if self._client:
            self._client = None

        if self._server:
            try:
                await self._server.__aexit__(None, None, None)
            except Exception as e:
                logger.warning(f"Error closing Parlant server: {e}")
            self._server = None

        self._initialized = False
        logger.info("Agent manager shutdown complete")

    async def set_model(self, model_id: str):
        """
        Set the Ollama model to use for generation.

        Note: Parlant uses environment variables for model configuration.
        Changing models may require server restart for full effect.

        Args:
            model_id: Ollama model ID (e.g., 'mistral', 'llama3.2')
        """
        # Update environment variable
        os.environ["OLLAMA_MODEL"] = model_id

        logger.info(f"Model set to: {model_id}")

        # Note: For full model change, Parlant server might need restart
        # The new model will be used for new sessions

    async def _get_or_create_session(
        self,
        agent_id: str,
        session_key: str,
        customer_id: Optional[str] = None
    ) -> str:
        """Get existing session or create new one."""
        full_key = f"{agent_id}:{session_key}"

        if full_key in self._sessions:
            return self._sessions[full_key]

        session = await self._client.sessions.create(
            agent_id=agent_id,
            customer_id=customer_id,
            title=f"Session {session_key}"
        )

        self._sessions[full_key] = session.id
        logger.debug(f"Created session: {session.id} for key: {full_key}")
        return session.id

    async def _send_message_and_wait(
        self,
        session_id: str,
        message: str,
        patient_data: Optional[Dict[str, Any]] = None,
        timeout: float = 60.0
    ) -> Dict[str, Any]:
        """
        Send a message and wait for the agent response.

        Args:
            session_id: Parlant session ID
            message: Message to send
            patient_data: Optional patient context
            timeout: Maximum wait time in seconds

        Returns:
            Agent response dictionary
        """
        # Build full message with patient context
        full_message = message
        if patient_data:
            patient_card = format_patient_card(patient_data)
            full_message = f"""PATIENT CONTEXT:
{patient_card}

TASK: {message}"""

        # Send customer message
        await self._client.sessions.create_event(
            session_id=session_id,
            kind="message",
            source="customer",
            message=full_message
        )

        # Wait for agent response
        start_time = asyncio.get_event_loop().time()
        while True:
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed > timeout:
                logger.warning(f"Timeout waiting for response after {timeout}s")
                return {
                    "error": "Response timeout",
                    "answer": "I apologize, but I couldn't generate a response in time. Please try again."
                }

            events = await self._client.sessions.list_events(session_id)

            for event in reversed(events):
                if event.source == "ai_agent" and event.kind == "message":
                    return self._parse_response(event)

            await asyncio.sleep(0.5)

    def _parse_response(self, event) -> Dict[str, Any]:
        """Parse Parlant event into response dictionary."""
        content = event.message or ""

        response = {
            "answer": content,
            "reasoning": "",
            "follow_up_questions": [],
            "suggested_questions": [],
            "protocol_applied": "",
            "rag_used": False,
            "has_reasoning": False,
            "parlant_event_id": getattr(event, 'id', None),
            "guidelines_triggered": [],
        }

        # Extract guidelines triggered
        if hasattr(event, 'guidelines') and event.guidelines:
            response["guidelines_triggered"] = [
                g.condition for g in event.guidelines
            ]

        # Extract structured data from tool calls
        if hasattr(event, 'data') and event.data:
            data = event.data
            if isinstance(data, dict):
                response.update({
                    "reasoning": data.get("reasoning", ""),
                    "answer": data.get("answer", content),
                    "follow_up_questions": data.get("follow_up_questions", []),
                    "protocol_applied": data.get("protocol_applied", ""),
                })
                response["suggested_questions"] = response["follow_up_questions"]
                response["has_reasoning"] = bool(response["reasoning"])
                if data.get("protocol_applied"):
                    response["rag_used"] = True

        return response

    # =========================================================================
    # PUBLIC API - TRIAGE
    # =========================================================================

    async def answer_question(
        self,
        question: str,
        clinical_state: Dict[str, Any],
        user_language: str = "en",
        session_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Answer a staff question about a patient case.

        Args:
            question: The question to answer
            clinical_state: Patient clinical data
            user_language: 'en' for English (MTS), 'fr' for French (SFMU)
            session_key: Optional session key for context continuity

        Returns:
            Response dictionary with answer, reasoning, follow_up_questions, etc.
        """
        if not self._initialized:
            await self.initialize()

        if not session_key:
            session_key = clinical_state.get("session_id", "default")

        try:
            session_id = await self._get_or_create_session(
                self._triage_agent.agent_id,
                session_key
            )

            # Pre-fetch protocol via RAG if available
            protocol_context = ""
            protocol_title = ""
            if self.drbert_engine and self.drbert_engine.rag_available:
                symptoms = self._build_symptoms_string(clinical_state)
                chief = clinical_state.get("chief_complaint", "")
                query = f"{chief} {symptoms}".strip()

                results = self.drbert_engine.search_protocols(
                    query=query,
                    language=user_language,
                    n_results=1
                )
                if results.protocols:
                    protocol = results.protocols[0]
                    protocol_title = protocol.title
                    protocol_context = f"""
RELEVANT PROTOCOL: {protocol.title}
Source: {protocol.source}
{protocol.text}

IMPORTANT: Cite specific criteria from this protocol in your response.
"""

            # Build the full question
            system_prefix = ""
            if user_language == "fr":
                system_prefix = "Respond in French. Use SFMU triage terminology.\n\n"
            else:
                system_prefix = "Respond in English. Use Manchester Triage System terminology.\n\n"

            full_question = f"""{system_prefix}{protocol_context}
QUESTION: {question}

Provide:
1. A patient-friendly ANSWER
2. Clinical REASONING for staff (cite protocol criteria)
3. Exactly 3 specific FOLLOW-UP QUESTIONS for this patient"""

            response = await self._send_message_and_wait(
                session_id=session_id,
                message=full_question,
                patient_data=clinical_state
            )

            # Ensure required fields
            response["model_used"] = f"Parlant ({OLLAMA_MODEL})"
            response["rag_used"] = bool(protocol_title)
            response["protocol_applied"] = protocol_title
            response["protocol_source"] = "SFMU" if user_language == "fr" else "MTS"

            # Ensure follow-up questions
            if not response.get("follow_up_questions"):
                response["follow_up_questions"] = self._get_default_questions(user_language)
            response["suggested_questions"] = response["follow_up_questions"]

            return response

        except Exception as e:
            logger.error(f"Error in answer_question: {e}")
            return self._error_response(str(e), user_language)

    # =========================================================================
    # PUBLIC API - PDF GENERATION
    # =========================================================================

    async def generate_pdf_sections(
        self,
        clinical_state: Dict[str, Any],
        user_language: str = "en"
    ) -> Dict[str, str]:
        """
        Generate PDF clinical referral letter sections.

        Args:
            clinical_state: Patient clinical data
            user_language: 'en' for English, 'fr' for French

        Returns:
            Dictionary with 'summary', 'diagnosis', 'conclusion' sections
        """
        if not self._initialized:
            await self.initialize()

        session_key = f"pdf_{clinical_state.get('session_id', 'default')}"

        try:
            session_id = await self._get_or_create_session(
                self._pdf_agent.agent_id,
                session_key
            )

            lang_instruction = "Respond in French." if user_language == "fr" else "Respond in English."

            # Generate Clinical Summary
            summary_prompt = f"""{lang_instruction}

Generate a CLINICAL SUMMARY for this patient as a triage nurse would document.

Include:
- Presenting complaint (in patient's words)
- Demographics (age, sex, pregnancy status if relevant)
- All reported symptoms with EXACT values (e.g., "pain severity 8/10", "temperature 38.5°C")
- Vital signs if available
- Time course of symptoms
- Relevant medical history

Be systematic, factual, and objective. Do NOT interpret or diagnose - just document."""

            summary_response = await self._send_message_and_wait(
                session_id=session_id,
                message=summary_prompt,
                patient_data=clinical_state
            )

            # Generate Diagnostic Assessment
            diagnosis_prompt = f"""{lang_instruction}

Generate a DIAGNOSTIC ASSESSMENT for this patient as a senior physician would.

Include:
- Differential considerations based on symptom pattern
- Reference to applicable triage protocol criteria (MTS/SFMU)
- Risk factors identified from patient data
- Red flags present or ruled out
- Clinical reasoning for triage categorization

Use appropriate medical terminology. Be thorough but concise."""

            diagnosis_response = await self._send_message_and_wait(
                session_id=session_id,
                message=diagnosis_prompt,
                patient_data=clinical_state
            )

            # Generate Conclusion/Recommendations
            conclusion_prompt = f"""{lang_instruction}

Generate RECOMMENDATIONS for this patient as an attending physician would.

Include:
1. Clear triage priority with specific justification
2. Immediate actions or interventions needed
3. Suggested investigations or monitoring
4. Disposition recommendation

Be decisive and action-oriented. State recommendations clearly without hedging."""

            conclusion_response = await self._send_message_and_wait(
                session_id=session_id,
                message=conclusion_prompt,
                patient_data=clinical_state
            )

            return {
                "summary": summary_response.get("answer", "Error generating summary"),
                "diagnosis": diagnosis_response.get("answer", "Error generating assessment"),
                "conclusion": conclusion_response.get("answer", "Error generating recommendations"),
            }

        except Exception as e:
            logger.error(f"Error generating PDF sections: {e}")
            return {
                "summary": f"Error: {str(e)}",
                "diagnosis": "Error generating diagnostic assessment",
                "conclusion": "Error generating recommendations",
            }

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _build_symptoms_string(self, clinical_state: Dict[str, Any]) -> str:
        """Extract key symptoms for protocol search."""
        symptoms = []
        answers = clinical_state.get("answers", {})

        for key, val in answers.items():
            if key.startswith("_"):
                continue
            if isinstance(val, bool) and val:
                symptoms.append(key.replace("_", " "))
            elif isinstance(val, str) and val:
                symptoms.append(val)
            elif isinstance(val, (int, float)) and "pain" in key.lower():
                symptoms.append(f"{key.replace('_', ' ')} {val}")

        return " ".join(symptoms[:5])

    def _get_default_questions(self, language: str) -> List[str]:
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

    def _error_response(self, error: str, language: str) -> Dict[str, Any]:
        """Generate error response dictionary."""
        if language == "fr":
            answer = "Je m'excuse, mais j'ai rencontre une erreur. Veuillez reessayer."
        else:
            answer = "I apologize, but I encountered an error. Please try again."

        return {
            "answer": answer,
            "reasoning": f"Error: {error}",
            "follow_up_questions": self._get_default_questions(language),
            "suggested_questions": self._get_default_questions(language),
            "protocol_applied": "",
            "rag_used": False,
            "model_used": "Parlant (error)",
            "has_reasoning": False,
            "error": error,
        }

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def rag_available(self) -> bool:
        return self.drbert_engine is not None and self.drbert_engine.rag_available


# =============================================================================
# SYNCHRONOUS WRAPPER
# =============================================================================

class SyncParlantAgentManager:
    """
    Synchronous wrapper around ParlantAgentManager.

    For compatibility with existing synchronous code.
    """

    def __init__(self, drbert_engine: Optional[DrBERTRAG] = None):
        self._async_manager = ParlantAgentManager(drbert_engine)
        self._loop = None

    def _get_loop(self):
        """Get or create an event loop."""
        try:
            self._loop = asyncio.get_event_loop()
            if self._loop.is_closed():
                raise RuntimeError("Loop is closed")
        except RuntimeError:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
        return self._loop

    def set_drbert_engine(self, engine: DrBERTRAG):
        """Set the DrBERT engine."""
        self._async_manager.set_drbert_engine(engine)

    def initialize(self):
        """Initialize synchronously."""
        loop = self._get_loop()
        loop.run_until_complete(self._async_manager.initialize())

    def shutdown(self):
        """Shutdown synchronously."""
        loop = self._get_loop()
        loop.run_until_complete(self._async_manager.shutdown())

    def answer_question(
        self,
        question: str,
        clinical_state: Dict[str, Any],
        user_language: str = "en",
        session_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """Answer a question synchronously."""
        loop = self._get_loop()
        return loop.run_until_complete(
            self._async_manager.answer_question(
                question, clinical_state, user_language, session_key
            )
        )

    def generate_pdf_sections(
        self,
        clinical_state: Dict[str, Any],
        user_language: str = "en"
    ) -> Dict[str, str]:
        """Generate PDF sections synchronously."""
        loop = self._get_loop()
        return loop.run_until_complete(
            self._async_manager.generate_pdf_sections(clinical_state, user_language)
        )

    @property
    def is_initialized(self) -> bool:
        return self._async_manager.is_initialized

    @property
    def rag_available(self) -> bool:
        return self._async_manager.rag_available
