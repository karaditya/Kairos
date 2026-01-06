"""
Patient Data RAG - Cite Patient Data in Responses

Simple in-memory approach: format patient data as citations
that the LLM can reference directly in its response.

This is MUCH simpler than protocol RAG:
- No vector store needed
- No embedding models
- Works in-memory
- Cites the specific patient's data
"""

import re
from typing import Dict, Any, List


class PatientDataRAG:
    """
    Simple patient data citation system.

    Unlike protocol RAG, this:
    - Works in-memory (no vector store needed)
    - Cites the specific patient's data, not generic protocols
    - Provides structured context for the LLM
    """

    @staticmethod
    def format_citable_context(clinical_state: Dict[str, Any]) -> str:
        """
        Format patient data as citable context for LLM.

        Returns structured patient data with citation markers.
        """
        sections = []

        # Demographics Section
        demo = clinical_state.get("demographics", {})
        if demo:
            demo_items = []
            if demo.get("age"):
                demo_items.append(f"Age: {demo['age']} years [CITE:demo_age]")
            if demo.get("sex"):
                demo_items.append(f"Sex: {demo['sex']} [CITE:demo_sex]")
            if demo.get("pregnant"):
                demo_items.append("Pregnant: Yes [CITE:demo_pregnant]")
            if demo_items:
                sections.append(
                    "DEMOGRAPHICS:\n" + "\n".join(f"  - {item}" for item in demo_items)
                )

        # Chief Complaint
        complaint = clinical_state.get("chief_complaint", "")
        if complaint:
            readable = complaint.replace("_", " ").title()
            sections.append(f"CHIEF COMPLAINT: {readable} [CITE:chief_complaint]")

        # Risk Band
        risk = clinical_state.get(
            "risk_calculation", clinical_state.get("risk_band", {})
        )
        if isinstance(risk, dict):
            band = risk.get("band", "").upper()
            if band:
                sections.append(f"TRIAGE PRIORITY: {band} [CITE:risk_band]")
        elif risk:
            sections.append(f"TRIAGE PRIORITY: {str(risk).upper()} [CITE:risk_band]")

        # Symptoms Section
        answers = clinical_state.get("answers", {})
        if answers:
            symptom_items = []
            for key, value in answers.items():
                if key.startswith("_"):
                    continue
                display_key = key.replace("_", " ").title()
                cite_key = f"CITE:{key}"

                if isinstance(value, bool):
                    symptom_items.append(
                        f"{display_key}: {'Yes' if value else 'No'} [{cite_key}]"
                    )
                elif value is not None and value != "":
                    symptom_items.append(f"{display_key}: {value} [{cite_key}]")

            if symptom_items:
                sections.append(
                    "REPORTED SYMPTOMS:\n"
                    + "\n".join(f"  - {item}" for item in symptom_items)
                )

        # Triggered Rules (if any)
        triggered = clinical_state.get("triggered_rules", [])
        if not triggered and isinstance(risk, dict):
            triggered = risk.get("triggered_rules", [])

        if triggered:
            alert_items = []
            for i, rule in enumerate(triggered[:5]):  # Max 5 alerts
                if isinstance(rule, dict):
                    desc = rule.get("description", rule.get("id", "Unknown"))
                    band = rule.get("band", "").upper()
                    alert_items.append(f"[{band}] {desc} [CITE:alert_{i}]")
                else:
                    alert_items.append(f"{rule} [CITE:alert_{i}]")
            sections.append(
                "CLINICAL ALERTS:\n"
                + "\n".join(f"  - {item}" for item in alert_items)
            )

        return "\n\n".join(sections) if sections else "No patient data available."

    @staticmethod
    def extract_citations(response: str) -> List[str]:
        """
        Extract citation keys from response text.

        Returns list of cited data points.
        """
        citations = re.findall(r"\[CITE:(\w+)\]", response)
        return list(set(citations))

    @staticmethod
    def get_cited_data(clinical_state: Dict[str, Any], response: str) -> List[str]:
        """
        Get human-readable list of cited patient data.
        """
        citations = PatientDataRAG.extract_citations(response)
        cited = []

        demo = clinical_state.get("demographics", {})
        answers = clinical_state.get("answers", {})
        risk = clinical_state.get("risk_calculation", {})

        for cite in citations:
            if cite == "demo_age" and demo.get("age"):
                cited.append(f"Age: {demo['age']}")
            elif cite == "demo_sex" and demo.get("sex"):
                cited.append(f"Sex: {demo['sex']}")
            elif cite == "demo_pregnant" and demo.get("pregnant"):
                cited.append("Pregnant: Yes")
            elif cite == "chief_complaint":
                complaint = clinical_state.get("chief_complaint", "N/A")
                cited.append(f"Chief complaint: {complaint.replace('_', ' ').title()}")
            elif cite == "risk_band":
                if isinstance(risk, dict):
                    cited.append(f"Risk: {risk.get('band', 'N/A').upper()}")
                else:
                    cited.append(f"Risk: {str(risk).upper()}")
            elif cite.startswith("alert_"):
                pass  # Alerts handled separately
            elif cite in answers:
                value = answers[cite]
                display_key = cite.replace("_", " ").title()
                if isinstance(value, bool):
                    cited.append(f"{display_key}: {'Yes' if value else 'No'}")
                elif value:
                    cited.append(f"{display_key}: {value}")

        return cited

    @staticmethod
    def format_for_pdf(clinical_state: Dict[str, Any]) -> str:
        """
        Format patient data for PDF generation (no citation markers).
        """
        sections = []

        # Demographics
        demo = clinical_state.get("demographics", {})
        if demo:
            parts = []
            if demo.get("age"):
                parts.append(f"{demo['age']} years old")
            if demo.get("sex"):
                parts.append(demo["sex"])
            if demo.get("pregnant"):
                parts.append("pregnant")
            if parts:
                sections.append(f"Patient: {', '.join(parts)}")

        # Chief Complaint
        complaint = clinical_state.get("chief_complaint", "")
        if complaint:
            readable = complaint.replace("_", " ").title()
            sections.append(f"Presenting complaint: {readable}")

        # Risk
        risk = clinical_state.get("risk_calculation", {})
        if isinstance(risk, dict) and risk.get("band"):
            sections.append(f"Triage level: {risk['band'].upper()}")

        # Symptoms
        answers = clinical_state.get("answers", {})
        if answers:
            symptom_lines = []
            for key, value in answers.items():
                if key.startswith("_"):
                    continue
                display_key = key.replace("_", " ").title()
                if isinstance(value, bool):
                    symptom_lines.append(f"- {display_key}: {'Yes' if value else 'No'}")
                elif value is not None and value != "":
                    symptom_lines.append(f"- {display_key}: {value}")
            if symptom_lines:
                sections.append("Symptoms:\n" + "\n".join(symptom_lines))

        return "\n\n".join(sections) if sections else "No patient data."
