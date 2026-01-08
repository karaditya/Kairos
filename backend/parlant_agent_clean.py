"""
Clean Parlant Agent - Subprocess-based Parlant Integration

Uses ParlantSubprocess to start Parlant HTTP server in a separate process,
then connects via AsyncParlantClient for all operations.

This approach ensures the HTTP server actually runs (unlike the SDK Server class
which never starts the HTTP server while the context is active).
"""

import os
import asyncio
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

from config import (
    OLLAMA_MODEL,
    OLLAMA_BASE_URL,
    PARLANT_PORT,
)

# Local imports
from parlant_subprocess import get_parlant_subprocess, ParlantSubprocess
from parlant_guidelines_minimal import get_all_guidelines
from patient_data_rag import PatientDataRAG

# Parlant HTTP client
from parlant.client import AsyncParlantClient

# Get model from environment or config
_current_model = os.environ.get("OLLAMA_MODEL", OLLAMA_MODEL)

# Configuration
PARLANT_RESPONSE_TIMEOUT = 90.0  # seconds (Ollama via Parlant can be slow)


class CleanParlantAgent:
    """
    Subprocess-based Parlant agent for medical triage.

    Uses HTTP API exclusively (no SDK context manager issues).

    Features:
    - Subprocess-based Parlant server
    - HTTP client for all operations
    - Agent creation with guidelines via HTTP
    - Session management via HTTP
    """

    def __init__(self):
        self._subprocess: Optional[ParlantSubprocess] = None
        self._client: Optional[AsyncParlantClient] = None
        self._agent_id: Optional[str] = None
        self._initialized = False
        self._sessions: Dict[str, str] = {}  # session_key -> session_id
        self._guidelines_created = False

    @property
    def is_available(self) -> bool:
        """Check if Parlant can be started."""
        return True  # Will be verified during initialize

    @property
    def is_initialized(self) -> bool:
        """Check if agent is initialized."""
        return self._initialized

    async def initialize(self, timeout: float = 120.0) -> bool:
        """Initialize Parlant subprocess and agent."""
        if self._initialized:
            return True

        # Check Ollama availability first
        if not await self._check_ollama():
            logger.warning("Ollama not available - Parlant will not initialize")
            return False

        try:
            logger.info("Initializing Parlant subprocess (timeout: %.1fs)...", timeout)

            # Start subprocess
            self._subprocess = get_parlant_subprocess()
            if not await self._subprocess.start(timeout=timeout):
                logger.error("Failed to start Parlant subprocess")
                return False

            # Initialize HTTP client
            self._client = AsyncParlantClient(
                base_url=self._subprocess.base_url,
                timeout=PARLANT_RESPONSE_TIMEOUT * 2,
            )

            # Get agent ID from subprocess (it creates the agent)
            self._agent_id = self._subprocess.agent_id
            if not self._agent_id:
                # Try to find via HTTP client
                self._agent_id = await self._ensure_agent()

            if not self._agent_id:
                logger.error("Failed to get/create agent")
                await self.shutdown()
                return False

            logger.info(f"Parlant agent ready: {self._agent_id}")

            # Create guidelines via HTTP
            await self._create_guidelines()

            self._initialized = True
            logger.info("Parlant agent initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Parlant initialization failed: {e}")
            await self.shutdown()
            return False

    async def _check_ollama(self) -> bool:
        """Check if Ollama is available with required model."""
        import httpx

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
                if response.status_code == 200:
                    data = response.json()
                    models = [
                        m.get("name", "").split(":")[0] for m in data.get("models", [])
                    ]
                    required = _current_model.split(":")[0]
                    if required in models:
                        logger.info(f"Ollama available with model: {required}")
                        return True
                    logger.warning(f"Model {required} not found in Ollama")
            return False
        except Exception as e:
            logger.warning(f"Ollama check failed: {e}")
            return False

    async def _ensure_agent(self) -> Optional[str]:
        """Get existing agent or create new one via HTTP."""
        try:
            # List existing agents
            agents = await self._client.agents.list()

            # Look for our agent
            for agent in agents:
                if agent.name == "MedicalTriageAgent":
                    return agent.id

            # Create new agent if not found
            agent = await self._client.agents.create(
                name="MedicalTriageAgent",
                description=(
                    "Medical triage assistant for emergency department. "
                    "Assesses patient symptoms, provides triage guidance, "
                    "and generates clinical summaries. "
                    "Supports English (Manchester Triage) and French (SFMU)."
                ),
            )
            logger.info(f"Created new agent via HTTP: {agent.id}")
            return agent.id

        except Exception as e:
            logger.error(f"Failed to ensure agent: {e}")
            return None

    async def _create_guidelines(self):
        """Create guidelines for the agent via HTTP API."""
        if self._guidelines_created:
            return

        try:
            guidelines = get_all_guidelines()
            created_count = 0

            for guideline in guidelines:
                try:
                    await self._client.agents.create_guideline(
                        agent_id=self._agent_id,
                        condition=guideline["condition"],
                        action=guideline["action"],
                    )
                    created_count += 1
                except Exception as e:
                    # May already exist or other error
                    logger.debug(f"Guideline creation note: {e}")

            self._guidelines_created = True
            logger.info(f"Created {created_count}/{len(guidelines)} guidelines")

        except Exception as e:
            logger.warning(f"Guideline creation error: {e}")

    async def shutdown(self):
        """Clean shutdown."""
        self._sessions.clear()
        self._client = None
        self._agent_id = None
        self._guidelines_created = False

        if self._subprocess:
            await self._subprocess.stop()
            self._subprocess = None

        self._initialized = False
        logger.info("Parlant agent shutdown")

    async def _get_session(self, session_key: str) -> str:
        """Get or create session via HTTP."""
        if session_key in self._sessions:
            return self._sessions[session_key]

        session = await self._client.sessions.create(
            agent_id=self._agent_id,
            title=f"Triage-{session_key[:8]}"
        )
        self._sessions[session_key] = session.id
        logger.debug(f"Created session: {session.id} for key: {session_key}")
        return session.id

    async def answer_question(
        self,
        question: str,
        clinical_state: Dict[str, Any],
        language: str = "en",
        session_key: str = None,
        timeout: float = None,
    ) -> Dict[str, Any]:
        """
        Answer a staff question about a patient.

        Args:
            question: The question to answer
            clinical_state: Patient data
            language: 'en' or 'fr'
            session_key: Session identifier
            timeout: Response timeout

        Returns:
            Response dictionary
        """
        if not self._initialized:
            raise RuntimeError("Agent not initialized")

        timeout = timeout or PARLANT_RESPONSE_TIMEOUT
        session_key = session_key or clinical_state.get("session_id", "default")

        try:
            session_id = await self._get_session(session_key)

            # Format patient context with citations
            patient_context = PatientDataRAG.format_citable_context(clinical_state)

            # Build message
            if language == "fr":
                lang_prefix = "Reponds en francais. Utilise la terminologie SFMU.\n\n"
            else:
                lang_prefix = ""

            full_message = f"""{lang_prefix}PATIENT DATA:
{patient_context}

QUESTION: {question}

Provide a clear answer, brief reasoning citing patient data, and 3 follow-up questions."""

            # Send message with timeout
            response = await asyncio.wait_for(
                self._send_and_wait(session_id, full_message),
                timeout=timeout
            )

            # Extract citations
            response["cited_data"] = PatientDataRAG.get_cited_data(
                clinical_state, response.get("answer", "")
            )
            response["model_used"] = f"Parlant ({_current_model})"
            response["is_fallback"] = False

            return response

        except asyncio.TimeoutError:
            logger.warning(f"Parlant response timeout after {timeout}s")
            raise
        except Exception as e:
            logger.error(f"Parlant error: {e}")
            raise

    async def _send_and_wait(self, session_id: str, message: str) -> Dict[str, Any]:
        """Send message and wait for response via HTTP."""
        # Send customer message
        await self._client.sessions.create_event(
            session_id=session_id,
            kind="message",
            source="customer",
            message=message
        )

        # Poll for response (max 180 polls = 90 seconds at 0.5s intervals)
        max_polls = 180
        for _ in range(max_polls):
            events = await self._client.sessions.list_events(session_id)

            for event in reversed(events):
                if event.source == "ai_agent" and event.kind == "message":
                    return self._parse_event(event)

            await asyncio.sleep(0.5)

        raise asyncio.TimeoutError("No response from agent")

    def _parse_event(self, event) -> Dict[str, Any]:
        """Parse Parlant event."""
        import re

        content = event.message or ""

        # Extract follow-up questions
        follow_ups = []
        fq_match = re.search(
            r"(?:follow-up|questions?).*?:\s*(.*?)$",
            content, re.DOTALL | re.IGNORECASE
        )
        if fq_match:
            questions = re.findall(r"\d+\.\s*(.+?)(?=\d+\.|$)", fq_match.group(1))
            follow_ups = [q.strip() for q in questions if q.strip()][:3]

        # Try to extract reasoning
        reasoning = ""
        reason_match = re.search(
            r"(?:reasoning|because|based on).*?:\s*(.*?)(?=follow-up|questions?:|$)",
            content, re.DOTALL | re.IGNORECASE
        )
        if reason_match:
            reasoning = reason_match.group(1).strip()

        return {
            "answer": content,
            "reasoning": reasoning,
            "follow_up_questions": follow_ups,
            "suggested_questions": follow_ups,
            "has_reasoning": bool(reasoning),
            "rag_used": False,
            "protocol_applied": "",
            "guidelines_triggered": getattr(event, "guidelines", []),
        }

    async def generate_pdf_sections(
        self,
        clinical_state: Dict[str, Any],
        language: str = "en",
        timeout: float = None,
    ) -> Dict[str, str]:
        """Generate PDF clinical summary sections."""
        if not self._initialized:
            raise RuntimeError("Agent not initialized")

        timeout = timeout or PARLANT_RESPONSE_TIMEOUT * 3  # Longer for 3 sections
        session_key = f"pdf_{clinical_state.get('session_id', 'default')}"

        try:
            session_id = await self._get_session(session_key)
            patient_context = PatientDataRAG.format_citable_context(clinical_state)

            lang = "French" if language == "fr" else "English"

            # Generate all sections
            sections = {}

            prompts = {
                "summary": f"""PATIENT DATA:
{patient_context}

Write a CLINICAL SUMMARY in {lang} as a triage nurse. Document objectively:
- Presenting complaint
- Demographics
- All symptoms with exact values
- Time course
Be factual. Do not interpret or diagnose.""",

                "diagnosis": f"""PATIENT DATA:
{patient_context}

Write a DIAGNOSTIC ASSESSMENT in {lang} as a senior physician:
- Differential considerations
- Triage protocol criteria met
- Risk factors identified
- Red flags present/ruled out
Use medical terminology appropriately.""",

                "conclusion": f"""PATIENT DATA:
{patient_context}

Write RECOMMENDATIONS in {lang} as an attending physician:
- Triage priority with justification
- Immediate actions needed
- Suggested investigations
- Disposition recommendation
Be decisive and clear.""",
            }

            section_timeout = timeout / 3
            for section_name, prompt in prompts.items():
                try:
                    response = await asyncio.wait_for(
                        self._send_and_wait(session_id, prompt),
                        timeout=section_timeout
                    )
                    sections[section_name] = response.get("answer", f"Error generating {section_name}")
                except asyncio.TimeoutError:
                    sections[section_name] = f"{section_name.title()} generation timed out"
                except Exception as e:
                    sections[section_name] = f"Error: {e}"

            return sections

        except Exception as e:
            logger.error(f"PDF generation error: {e}")
            return {
                "summary": f"Error: {e}",
                "diagnosis": f"Error: {e}",
                "conclusion": f"Error: {e}",
            }


# Singleton
_agent_instance: Optional[CleanParlantAgent] = None


async def get_clean_parlant_agent() -> CleanParlantAgent:
    """Get singleton Parlant agent instance."""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = CleanParlantAgent()
    return _agent_instance
