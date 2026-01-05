"""
Parlant Medical Triage Agent

Main agent class that replaces the 3-LLM review pipeline in MultiModelEngine
with Parlant's guideline-based architecture for reliable medical triage.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from contextlib import asynccontextmanager

from parlant import sdk as p
from parlant.client import AsyncParlantClient

from parlant_config import (
    TRIAGE_GUIDELINES,
    PDF_GUIDELINES,
    CANNED_RESPONSES,
    OLLAMA_MODEL,
    PARLANT_PORT,
    PARLANT_TOOL_PORT
)
from parlant_tools import (
    search_medical_protocols,
    get_patient_context,
    format_triage_response,
    detect_query_language,
    set_drbert_engine,
    format_patient_card,
    ALL_TOOLS
)

logger = logging.getLogger(__name__)


class ParlantTriageAgent:
    """
    Parlant-based Medical Triage Agent.

    Replaces the complex 3-LLM review pipeline in MultiModelEngine with
    Parlant's guideline-based architecture for guaranteed behavioral compliance.

    Key features:
    - Guidelines architecturally enforce protocol adherence
    - DrBERT RAG integration via Parlant tools
    - Canned responses for critical scenarios
    - Explainability: traces which guidelines triggered
    """

    def __init__(self, drbert_engine=None):
        """
        Initialize the Parlant triage agent.

        Args:
            drbert_engine: DrBERT engine instance for RAG (optional, can set later)
        """
        self.drbert_engine = drbert_engine
        self._server = None
        self._server_task = None
        self._agent = None
        self._agent_id = None
        self._client = None
        self._initialized = False
        self._sessions: Dict[str, str] = {}  # session_key -> parlant_session_id

        # Connect DrBERT to tools if provided
        if drbert_engine:
            set_drbert_engine(drbert_engine)

    def set_drbert_engine(self, engine):
        """Set the DrBERT engine for RAG."""
        self.drbert_engine = engine
        set_drbert_engine(engine)
        logger.info("DrBERT engine connected to Parlant agent")

    async def initialize(self):
        """
        Initialize the Parlant server and agent.

        This starts the Parlant server, creates the medical triage agent,
        and registers all guidelines and tools.
        """
        if self._initialized:
            logger.info("Parlant agent already initialized")
            return

        try:
            logger.info("Initializing Parlant triage agent...")

            # Start Parlant server with Ollama
            self._server = p.Server(
                port=PARLANT_PORT,
                tool_service_port=PARLANT_TOOL_PORT,
                nlp_service=p.NLPServices.ollama,
                log_level=p.LogLevel.INFO
            )

            # Enter the server context
            await self._server.__aenter__()

            # Create the medical triage agent
            self._agent = await self._server.create_agent(
                name="MedicalTriageAssistant",
                description=(
                    "A medical triage assistant that assesses patient symptoms "
                    "and provides guidance based on Manchester Triage System (MTS) "
                    "and SFMU protocols. Prioritizes patient safety and follows "
                    "evidence-based guidelines with strict protocol adherence."
                )
            )
            self._agent_id = self._agent.id

            # Register tools
            for tool in ALL_TOOLS:
                await self._agent.attach_tool(tool)
            logger.info(f"Registered {len(ALL_TOOLS)} tools")

            # Register triage guidelines
            for guideline in TRIAGE_GUIDELINES:
                await self._agent.create_guideline(
                    condition=guideline["condition"],
                    action=guideline["action"],
                    metadata={"id": guideline["id"], "priority": guideline["priority"]}
                )
            logger.info(f"Registered {len(TRIAGE_GUIDELINES)} triage guidelines")

            # Register PDF guidelines
            for guideline in PDF_GUIDELINES:
                await self._agent.create_guideline(
                    condition=guideline["condition"],
                    action=guideline["action"],
                    metadata={"id": guideline["id"], "priority": guideline["priority"]}
                )
            logger.info(f"Registered {len(PDF_GUIDELINES)} PDF guidelines")

            # Register canned responses
            for key, response in CANNED_RESPONSES.items():
                await self._agent.create_canned_response(
                    name=key,
                    content=response["answer"],
                    metadata={"use_when": response["use_when"]}
                )
            logger.info(f"Registered {len(CANNED_RESPONSES)} canned responses")

            # Initialize client for session management
            self._client = AsyncParlantClient(
                base_url=f"http://localhost:{PARLANT_PORT}"
            )

            self._initialized = True
            logger.info("Parlant triage agent initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Parlant agent: {e}")
            raise

    async def _get_or_create_session(
        self,
        session_key: str,
        customer_id: Optional[str] = None
    ) -> str:
        """Get existing session or create new one."""
        if session_key in self._sessions:
            return self._sessions[session_key]

        # Create new session
        session = await self._client.sessions.create(
            agent_id=self._agent_id,
            customer_id=customer_id,
            title=f"Triage Session {session_key}"
        )

        self._sessions[session_key] = session.id
        logger.debug(f"Created new session: {session.id} for key: {session_key}")
        return session.id

    async def _send_message_and_wait(
        self,
        session_id: str,
        message: str,
        patient_data: Optional[Dict[str, Any]] = None,
        timeout: float = 30.0
    ) -> Dict[str, Any]:
        """
        Send a message and wait for the agent response.

        Args:
            session_id: Parlant session ID
            message: Message to send
            patient_data: Optional patient context to include
            timeout: Maximum time to wait for response

        Returns:
            Agent response dictionary
        """
        # Include patient context in the message if provided
        full_message = message
        if patient_data:
            patient_card = format_patient_card(patient_data)
            full_message = f"""PATIENT CONTEXT:
{patient_card}

QUESTION: {message}"""

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
                logger.warning(f"Timeout waiting for Parlant response after {timeout}s")
                return {"error": "Response timeout", "answer": "I apologize, but I couldn't generate a response in time. Please try again."}

            # Get events
            events = await self._client.sessions.list_events(session_id)

            # Look for agent response after our message
            for event in reversed(events):
                if event.source == "ai_agent" and event.kind == "message":
                    return self._parse_response(event)

            # Brief wait before checking again
            await asyncio.sleep(0.5)

    def _parse_response(self, event) -> Dict[str, Any]:
        """Parse Parlant event into our response format."""
        content = event.message or ""

        # Extract structured fields if present
        response = {
            "answer": content,
            "reasoning": "",
            "follow_up_questions": [],
            "protocol_applied": "",
            "rag_used": False,
            "parlant_event_id": event.id if hasattr(event, 'id') else None,
            "guidelines_triggered": []
        }

        # Check for guidelines in event metadata
        if hasattr(event, 'guidelines') and event.guidelines:
            response["guidelines_triggered"] = [
                g.condition for g in event.guidelines
            ]

        # Try to extract structured response if tool was used
        if hasattr(event, 'data') and event.data:
            data = event.data
            if isinstance(data, dict):
                response.update({
                    "reasoning": data.get("reasoning", response["reasoning"]),
                    "answer": data.get("answer", response["answer"]),
                    "follow_up_questions": data.get("follow_up_questions", []),
                    "protocol_applied": data.get("protocol_applied", ""),
                })
                if data.get("protocol_applied"):
                    response["rag_used"] = True

        # Ensure we have follow-up questions
        if not response["follow_up_questions"]:
            response["follow_up_questions"] = [
                "How long have you been experiencing these symptoms?",
                "Have you taken any medications for this?",
                "Do you have any other symptoms you haven't mentioned?"
            ]

        return response

    async def answer_question(
        self,
        question: str,
        clinical_state: Dict[str, Any],
        user_language: str = "en",
        session_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Answer a staff question about a case using Parlant agent.

        This replaces MultiModelEngine.answer_staff_question() with
        Parlant's guideline-enforced generation.

        Args:
            question: The question to answer
            clinical_state: Patient clinical data
            user_language: Language for response ('en' or 'fr')
            session_key: Optional session identifier for context continuity

        Returns:
            Response dictionary matching the existing format:
            {
                "answer": str,
                "reasoning": str,
                "follow_up_questions": List[str],
                "protocol_applied": str,
                "rag_used": bool,
                "model_used": str,
                ...
            }
        """
        if not self._initialized:
            await self.initialize()

        # Generate session key if not provided
        if not session_key:
            session_key = clinical_state.get("session_id", "default")

        try:
            # Get or create session
            session_id = await self._get_or_create_session(session_key)

            # Pre-fetch protocol via RAG if available
            protocol_data = {}
            if self.drbert_engine and self.drbert_engine.rag_available:
                symptoms = self._build_symptoms_string(clinical_state)
                chief = clinical_state.get("chief_complaint", "")
                query = f"{chief} {symptoms}".strip()

                results = self.drbert_engine.search_protocols(
                    query=query,
                    language=user_language,
                    n_results=1
                )
                if results:
                    protocol_data = results[0]

            # Build enhanced question with protocol context
            enhanced_question = question
            if protocol_data:
                enhanced_question = f"""Based on the following medical protocol:

PROTOCOL: {protocol_data.get('title', 'Unknown')}
{protocol_data.get('text', '')}

{question}

Remember to cite specific criteria from the protocol in your response."""

            # Send message and get response
            response = await self._send_message_and_wait(
                session_id=session_id,
                message=enhanced_question,
                patient_data=clinical_state
            )

            # Add metadata
            response["model_used"] = f"Parlant ({OLLAMA_MODEL})"
            response["rag_used"] = bool(protocol_data)
            if protocol_data:
                response["protocol_applied"] = protocol_data.get("title", "")
                response["protocol_source"] = protocol_data.get("source", "MTS")

            return response

        except Exception as e:
            logger.error(f"Error in Parlant answer_question: {e}")
            return {
                "answer": "I apologize, but I encountered an error processing your question. Please try again.",
                "reasoning": f"Error: {str(e)}",
                "follow_up_questions": [
                    "Could you rephrase your question?",
                    "Are there specific symptoms you'd like me to focus on?",
                    "Would you like to start over with the assessment?"
                ],
                "protocol_applied": "",
                "rag_used": False,
                "model_used": "Parlant (error)",
                "error": str(e)
            }

    async def generate_pdf_sections(
        self,
        clinical_state: Dict[str, Any],
        user_language: str = "en"
    ) -> Dict[str, str]:
        """
        Generate PDF summary sections using Parlant agent.

        Replaces MultiModelEngine.generate_pdf_summary() with
        guideline-enforced generation for clinical referral letters.

        Args:
            clinical_state: Patient clinical data
            user_language: Language for output

        Returns:
            Dictionary with 'summary', 'diagnosis', 'conclusion' sections
        """
        if not self._initialized:
            await self.initialize()

        sections = {}
        session_key = f"pdf_{clinical_state.get('session_id', 'default')}"

        try:
            session_id = await self._get_or_create_session(session_key)

            # Generate clinical summary
            summary_response = await self._send_message_and_wait(
                session_id=session_id,
                message="Generate a CLINICAL SUMMARY for this patient. Include: presenting complaint, demographics (age, sex), all reported symptoms with exact values, vital signs if available, and all clinical alerts. Write as a triage nurse - be systematic and factual.",
                patient_data=clinical_state
            )
            sections["summary"] = summary_response.get("answer", "")

            # Generate diagnosis/assessment
            diagnosis_response = await self._send_message_and_wait(
                session_id=session_id,
                message="Generate a DIAGNOSTIC ASSESSMENT for this patient. Provide differential considerations based on the symptoms. Reference the applicable MTS/SFMU protocol criteria. Discuss the clinical reasoning for the triage priority. Write as a senior physician.",
                patient_data=clinical_state
            )
            sections["diagnosis"] = diagnosis_response.get("answer", "")

            # Generate conclusion/recommendations
            conclusion_response = await self._send_message_and_wait(
                session_id=session_id,
                message="Generate RECOMMENDATIONS for this patient. State: 1) Recommended triage priority with justification, 2) Immediate actions needed, 3) Suggested investigations or monitoring, 4) Disposition recommendation. Write as an attending physician - be decisive.",
                patient_data=clinical_state
            )
            sections["conclusion"] = conclusion_response.get("answer", "")

            return sections

        except Exception as e:
            logger.error(f"Error generating PDF sections: {e}")
            return {
                "summary": "Error generating clinical summary",
                "diagnosis": "Error generating diagnostic assessment",
                "conclusion": "Error generating recommendations"
            }

    def _build_symptoms_string(self, clinical_state: Dict[str, Any]) -> str:
        """Extract key symptoms for protocol search."""
        symptoms = []
        answers = clinical_state.get("answers", {})

        for key, val in answers.items():
            if not key.startswith("_"):
                if isinstance(val, bool) and val:
                    symptoms.append(key.replace("_", " "))
                elif isinstance(val, str) and val:
                    symptoms.append(val)
                elif isinstance(val, (int, float)) and "pain" in key.lower():
                    symptoms.append(f"{key.replace('_', ' ')} {val}")

        # Limit to most relevant symptoms
        return " ".join(symptoms[:5])

    async def shutdown(self):
        """Clean up resources and stop the Parlant server."""
        logger.info("Shutting down Parlant agent...")

        # Clear sessions
        self._sessions.clear()

        # Close client
        if self._client:
            # AsyncParlantClient doesn't have explicit close, but we clear reference
            self._client = None

        # Exit server context
        if self._server:
            try:
                await self._server.__aexit__(None, None, None)
            except Exception as e:
                logger.warning(f"Error closing Parlant server: {e}")
            self._server = None

        self._initialized = False
        self._agent = None
        self._agent_id = None
        logger.info("Parlant agent shutdown complete")

    @property
    def is_initialized(self) -> bool:
        """Check if agent is initialized."""
        return self._initialized

    @property
    def rag_available(self) -> bool:
        """Check if RAG is available."""
        return self.drbert_engine is not None and self.drbert_engine.rag_available


# =============================================================================
# SYNCHRONOUS WRAPPER
# For compatibility with existing synchronous code
# =============================================================================

class SyncParlantTriageAgent:
    """
    Synchronous wrapper around ParlantTriageAgent.

    Provides the same interface as the async agent but runs
    operations synchronously for compatibility with existing code.
    """

    def __init__(self, drbert_engine=None):
        self._async_agent = ParlantTriageAgent(drbert_engine)
        self._loop = None

    def _get_loop(self):
        """Get or create an event loop."""
        try:
            self._loop = asyncio.get_event_loop()
        except RuntimeError:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
        return self._loop

    def set_drbert_engine(self, engine):
        """Set the DrBERT engine."""
        self._async_agent.set_drbert_engine(engine)

    def initialize(self):
        """Initialize the agent synchronously."""
        loop = self._get_loop()
        loop.run_until_complete(self._async_agent.initialize())

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
            self._async_agent.answer_question(
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
            self._async_agent.generate_pdf_sections(clinical_state, user_language)
        )

    def shutdown(self):
        """Shutdown synchronously."""
        loop = self._get_loop()
        loop.run_until_complete(self._async_agent.shutdown())

    @property
    def is_initialized(self) -> bool:
        return self._async_agent.is_initialized

    @property
    def rag_available(self) -> bool:
        return self._async_agent.rag_available
