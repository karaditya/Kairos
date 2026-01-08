"""
Parlant Session Handler for CAE System.

Manages Parlant sessions for Compte Rendu generation,
integrating transcript, EHR data, and RAG context.
"""

from typing import Optional, Dict, Any, List
import json

from services.parlant.agent_factory import ScribeAgent, create_scribe_agent
from services.parlant.guidelines import get_guideline_prompt, get_cr_generation_prompt
from services.rag.protocol_store import ProtocolStore, get_protocol_store
from config import REQUIRED_CR_FIELDS, COMPTE_RENDU_SECTIONS, ADMINISTRATIVE_DISCLAIMER
from utils.logging import get_logger
from utils.exceptions import ParlantError

logger = get_logger(__name__)


class ParlantSessionHandler:
    """
    Handles Parlant sessions for document generation.

    Orchestrates:
    - RAG context retrieval
    - Prompt construction
    - Response parsing
    """

    def __init__(
        self,
        agent: ScribeAgent,
        protocol_store: ProtocolStore,
    ):
        self.agent = agent
        self.protocol_store = protocol_store

    async def generate_compte_rendu(
        self,
        transcript: str,
        ehr_data: Optional[Dict[str, Any]] = None,
        language: str = "fr",
        patient_info: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Generate a Compte Rendu from transcript and EHR data.

        Args:
            transcript: Audio transcript text
            ehr_data: Extracted EHR field data
            language: Output language
            patient_info: Patient name, DOB, MRN

        Returns:
            Compte Rendu with sections, missing fields, citations
        """
        # 1. Retrieve relevant protocols via RAG
        rag_context = await self._get_rag_context(transcript, language)

        # 2. Build the generation prompt
        user_prompt = get_cr_generation_prompt(
            transcript=transcript,
            ehr_data=ehr_data or {},
            language=language,
            rag_context=rag_context,
        )

        # 3. Get system prompt with guidelines
        system_prompt = get_guideline_prompt(language)

        # 4. Generate via Parlant
        full_prompt = f"{system_prompt}\n\n---\n\n{user_prompt}"
        raw_response = await self.agent.generate_response(full_prompt)

        # 5. Parse the response into structured CR
        compte_rendu = self._parse_cr_response(
            raw_response,
            language,
            patient_info,
        )

        # 6. Add RAG citations
        compte_rendu["rag_citations"] = await self._extract_citations(transcript, language)

        # 7. Add disclaimer
        compte_rendu["disclaimer"] = ADMINISTRATIVE_DISCLAIMER.get(language, ADMINISTRATIVE_DISCLAIMER["fr"])

        return compte_rendu

    async def _get_rag_context(
        self,
        transcript: str,
        language: str,
    ) -> str:
        """Retrieve relevant protocols from RAG."""
        if not transcript:
            return ""

        try:
            # Search for relevant protocols
            results = await self.protocol_store.search(
                query=transcript[:1000],  # First 1000 chars for search
                limit=3,
                language=language,
                min_score=0.5,
            )

            if not results:
                return ""

            # Format context
            context_parts = []
            for r in results:
                context_parts.append(
                    f"### {r['title']} (Source: {r['source']}, Pertinence: {r['score']:.2f})\n{r['content'][:500]}..."
                )

            return "\n\n".join(context_parts)

        except Exception as e:
            logger.warning("RAG context retrieval failed", error=str(e))
            return ""

    async def _extract_citations(
        self,
        transcript: str,
        language: str,
    ) -> List[Dict[str, Any]]:
        """Extract protocol citations for the CR."""
        if not transcript:
            return []

        try:
            results = await self.protocol_store.search(
                query=transcript[:1000],
                limit=3,
                language=language,
                min_score=0.5,
            )

            return [
                {
                    "protocol": r["title"],
                    "source": r["source"],
                    "relevance": round(r["score"], 2),
                }
                for r in results
            ]

        except Exception:
            return []

    def _parse_cr_response(
        self,
        response: str,
        language: str,
        patient_info: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Parse raw LLM response into structured CR.

        Extracts sections, identifies missing fields, flags for validation.
        """
        patient_info = patient_info or {}

        # Initialize structure
        cr = {
            "patient_name": patient_info.get("name"),
            "patient_dob": patient_info.get("dob"),
            "patient_mrn": patient_info.get("mrn"),
            "sections": {},
            "missing_fields": [],
            "flagged_fields": [],
        }

        # Parse sections from response
        for section in COMPTE_RENDU_SECTIONS:
            cr["sections"][section] = self._extract_section(response, section, language)

        # Detect missing fields
        missing_marker = "[MANQUANT:" if language == "fr" else "[MISSING:"
        for line in response.split("\n"):
            if missing_marker in line:
                # Parse [MANQUANT: field - reason]
                try:
                    content = line.split(missing_marker)[1].split("]")[0]
                    parts = content.split(" - ")
                    field = parts[0].strip()
                    reason = parts[1].strip() if len(parts) > 1 else "Required"
                    cr["missing_fields"].append({
                        "field_name": field,
                        "reason": reason,
                        "required_for": reason,
                    })
                except Exception:
                    pass

        # Check for required fields not mentioned
        for req_field in REQUIRED_CR_FIELDS:
            section_content = cr["sections"].get(req_field, "")
            if not section_content or section_content.strip() == "":
                if not any(m["field_name"] == req_field for m in cr["missing_fields"]):
                    cr["missing_fields"].append({
                        "field_name": req_field,
                        "reason": "Non mentionne" if language == "fr" else "Not mentioned",
                        "required_for": "Documentation complete",
                    })

        # Detect flagged fields
        validation_marker = "[A VALIDER]" if language == "fr" else "[TO VALIDATE]"
        for section, content in cr["sections"].items():
            if validation_marker in str(content):
                cr["flagged_fields"].append({
                    "field_name": section,
                    "reason": "Requires clinical validation",
                    "requires_clinical_input": True,
                })

        return cr

    def _extract_section(
        self,
        response: str,
        section: str,
        language: str,
    ) -> str:
        """Extract a specific section from the response."""
        # Section header patterns
        section_labels = {
            "motif_consultation": ["Motif de consultation", "Reason for Visit", "Motif:"],
            "anamnese": ["Anamnese", "History of Present Illness", "Anamnèse:"],
            "antecedents": ["Antecedents", "Past Medical History", "Antécédents:"],
            "allergies": ["Allergies", "Allergies:"],
            "traitements_actuels": ["Traitements actuels", "Current Medications", "Traitements:"],
            "examen_clinique": ["Examen clinique", "Physical Examination", "Examen:"],
            "examens_complementaires": ["Examens complementaires", "Additional Tests", "Examens:"],
            "hypotheses_diagnostiques": ["Hypotheses diagnostiques", "Diagnostic Hypotheses", "Hypothèses:"],
            "plan_therapeutique": ["Plan therapeutique", "Treatment Plan", "Plan:"],
        }

        labels = section_labels.get(section, [section])

        # Try to find section in response
        response_lower = response.lower()
        for label in labels:
            label_lower = label.lower()
            if label_lower in response_lower:
                # Find start position
                start_idx = response_lower.find(label_lower)
                if start_idx == -1:
                    continue

                # Find end (next section or end of text)
                end_idx = len(response)
                for other_section, other_labels in section_labels.items():
                    if other_section == section:
                        continue
                    for other_label in other_labels:
                        other_idx = response_lower.find(other_label.lower(), start_idx + len(label))
                        if other_idx != -1 and other_idx < end_idx:
                            end_idx = other_idx

                # Extract content
                content = response[start_idx + len(label):end_idx].strip()
                # Clean up common prefixes
                for prefix in [":", "-", "."]:
                    if content.startswith(prefix):
                        content = content[1:].strip()
                return content

        return ""


# =============================================================================
# FACTORY
# =============================================================================

_session_handler: Optional[ParlantSessionHandler] = None


async def get_session_handler() -> ParlantSessionHandler:
    """Get singleton session handler."""
    global _session_handler
    if _session_handler is None:
        agent = await create_scribe_agent()
        protocol_store = await get_protocol_store()
        _session_handler = ParlantSessionHandler(agent, protocol_store)
    return _session_handler
