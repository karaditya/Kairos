"""
Reasoning Engine - TRM-Style Iterative Reasoning with Local LLM

Implements the core TRM concept:
- Maintain latent reasoning state (z) and output state (y)
- Iterate multiple times to refine reasoning
- Generate final summary after T iterations

Uses llama-cpp-python for local CPU inference.

CRITICAL: The model CANNOT override risk bands. It can only:
- Summarize patient data
- Suggest follow-up questions
- Explain triage logic
"""

import os
import json
from typing import Dict, List, Any, Tuple, Optional

# =============================================================================
# LLM Backend
# =============================================================================

# Try to import llama-cpp-python
try:
    from llama_cpp import Llama
    LLAMA_AVAILABLE = True
except ImportError:
    LLAMA_AVAILABLE = False
    print("Warning: llama-cpp-python not installed. Using fallback mode.")
    print("Install with: pip install llama-cpp-python")

# =============================================================================
# System Prompts
# =============================================================================

SYSTEM_PROMPT = """You are a medical triage assistant. You do NOT diagnose or prescribe.

Your role is ONLY to:
1. Summarize patient-reported symptoms and data
2. Explain why certain risk flags were triggered
3. Suggest follow-up questions based on clinical protocols
4. Help staff understand the triage logic

CRITICAL RULES:
- NEVER provide medical diagnosis
- NEVER recommend treatments or medications
- NEVER override the risk band assigned by the system
- ALWAYS remind users that a healthcare professional must review all cases
- Cite specific data points when explaining decisions

The risk band (red/amber/green) is determined by deterministic rules, not by you."""

SUMMARY_PROMPT_TEMPLATE = """Based on the following patient data, provide a brief clinical summary.

PATIENT DATA:
- Age: {age}
- Sex: {sex}
- Pregnant: {pregnant}
- Chief Complaint: {chief_complaint}

REPORTED SYMPTOMS:
{symptoms}

RISK ASSESSMENT:
- Risk Band: {risk_band}
- Triggered Rules: {triggered_rules}

TASK: Write a 2-4 sentence summary for the receiving healthcare provider. Include:
1. Brief patient description
2. Main presenting concerns
3. Key risk factors identified

Do NOT diagnose. Do NOT recommend treatment. Just summarize the data.

SUMMARY:"""

REASONING_PROMPT_TEMPLATE = """You are refining your understanding of this case.

CURRENT REASONING STATE:
{z}

PATIENT DATA:
{patient_data}

CURRENT SUMMARY DRAFT:
{y}

Update your reasoning. What patterns do you notice? What's most important for the clinician to know?
Keep your reasoning notes brief (2-3 sentences).

UPDATED REASONING:"""

STAFF_QA_PROMPT_TEMPLATE = """You are helping a healthcare staff member understand a patient case.

PATIENT CASE:
{case_data}

STAFF QUESTION: {question}

INSTRUCTIONS:
- Answer based ONLY on the data provided
- Cite specific data points (e.g., "Patient reported fever of 39°C")
- If asked about diagnosis, remind them you cannot diagnose
- If asked about treatment, remind them a physician must decide
- Focus on explaining the triage logic and suggesting follow-up questions

RESPONSE:"""

# =============================================================================
# Reasoning Engine
# =============================================================================

class ReasoningEngine:
    """
    TRM-style reasoning engine using local LLM.
    
    Implements iterative refinement:
    1. Initialize z (reasoning state) and y (summary draft)
    2. Loop T times: z, y = refine(z, y, clinical_data)
    3. Output final summary
    """
    
    def __init__(self, model_path: str, n_iterations: int = 3):
        self.model_path = model_path
        self.n_iterations = n_iterations
        self.model = None
        self.is_loaded = False
        
        self._load_model()
    
    def _load_model(self):
        """Load the local LLM."""
        if not LLAMA_AVAILABLE:
            print("LLM library not available - using fallback mode")
            return

        if not os.path.exists(self.model_path):
            print(f"Model not found at {self.model_path}")
            print("Download a GGUF model and place it there.")
            print("See README for instructions.")
            return

        try:
            # GPU layers: 0 = CPU only, -1 = all GPU, or specify number of layers
            n_gpu_layers = int(os.environ.get("N_GPU_LAYERS", "0"))

            print(f"Loading LLM from {self.model_path}...")
            print(f"GPU layers: {n_gpu_layers} ({'CPU only' if n_gpu_layers == 0 else 'GPU accelerated'})")

            self.model = Llama(
                model_path=self.model_path,
                n_ctx=2048,
                n_threads=4,
                n_gpu_layers=n_gpu_layers,
                verbose=False
            )
            self.is_loaded = True
            print("LLM loaded successfully!")
        except Exception as e:
            print(f"Failed to load LLM: {e}")
            self.is_loaded = False
    
    def unload(self):
        """Unload model to free memory."""
        if self.model:
            del self.model
            self.model = None
            self.is_loaded = False
    
    def _generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.3) -> str:
        """Generate text from prompt."""
        if not self.is_loaded:
            return self._fallback_generate(prompt)
        
        try:
            response = self.model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=["</s>", "\n\n\n", "PATIENT DATA:", "TASK:"],
                echo=False
            )
            return response["choices"][0]["text"].strip()
        except Exception as e:
            print(f"Generation error: {e}")
            return self._fallback_generate(prompt)
    
    def _fallback_generate(self, prompt: str) -> str:
        """Fallback generation when LLM is not available."""
        # Simple template-based fallback
        if "SUMMARY:" in prompt:
            return "Patient presents with reported symptoms requiring clinical evaluation. Risk assessment completed per protocol. Healthcare provider review recommended."
        elif "REASONING:" in prompt:
            return "Reviewing symptom patterns and risk factors. Key data points identified for clinical handoff."
        elif "RESPONSE:" in prompt:
            return "Based on the patient data provided, I can help explain the triage logic. Please note that clinical decisions must be made by a qualified healthcare provider."
        else:
            return "Information processed. Healthcare provider review required for clinical decisions."
    
    # =========================================================================
    # TRM-Style Reasoning Loop
    # =========================================================================
    
    def generate_summary(self, clinical_state: Dict[str, Any]) -> Tuple[str, List[str]]:
        """
        Generate patient summary using TRM-style iterative reasoning.
        
        Args:
            clinical_state: {
                "demographics": {"age": ..., "sex": ..., "pregnant": ...},
                "chief_complaint": str,
                "answers": {...},
                "risk_band": str,
                "triggered_rules": [...]
            }
        
        Returns:
            (summary_text, list_of_key_flags)
        """
        # Initialize states
        z = ""  # Latent reasoning state (scratchpad)
        y = ""  # Output draft
        
        # Format patient data for prompts
        patient_data = self._format_patient_data(clinical_state)
        
        # === TRM Reasoning Loop ===
        for iteration in range(self.n_iterations):
            # Update reasoning state
            reasoning_prompt = REASONING_PROMPT_TEMPLATE.format(
                z=z if z else "Initial analysis.",
                patient_data=patient_data,
                y=y if y else "No draft yet."
            )
            z = self._generate(reasoning_prompt, max_tokens=128, temperature=0.3)
        
        # === Final Summary Generation ===
        demographics = clinical_state.get("demographics", {})
        answers = clinical_state.get("answers", {})
        
        # Format symptoms from answers
        symptoms_list = []
        for key, value in answers.items():
            if not key.startswith("_") and key not in ["chief_complaint", "chief_complaint_text"]:
                symptoms_list.append(f"- {key}: {value}")
        symptoms_text = "\n".join(symptoms_list) if symptoms_list else "No specific symptoms recorded"
        
        # Format triggered rules
        triggered = clinical_state.get("triggered_rules", [])
        rules_text = ", ".join([r.get("description", r.get("id", "Unknown")) for r in triggered]) if triggered else "None"
        
        summary_prompt = SUMMARY_PROMPT_TEMPLATE.format(
            age=demographics.get("age", "Unknown"),
            sex=demographics.get("sex", "Unknown"),
            pregnant=demographics.get("pregnant", "N/A"),
            chief_complaint=clinical_state.get("chief_complaint", "Unknown"),
            symptoms=symptoms_text,
            risk_band=clinical_state.get("risk_band", "Unknown").upper(),
            triggered_rules=rules_text
        )
        
        summary = self._generate(summary_prompt, max_tokens=256, temperature=0.3)
        
        # Extract key flags
        key_flags = self._extract_key_flags(clinical_state)
        
        return summary, key_flags
    
    def _format_patient_data(self, clinical_state: Dict[str, Any]) -> str:
        """Format clinical state as readable text."""
        lines = []
        
        demo = clinical_state.get("demographics", {})
        lines.append(f"Patient: {demo.get('age', '?')} y/o {demo.get('sex', '?')}")
        if demo.get("pregnant"):
            lines.append("Currently pregnant")
        
        lines.append(f"Chief complaint: {clinical_state.get('chief_complaint', 'Unknown')}")
        lines.append(f"Risk band: {clinical_state.get('risk_band', 'Unknown').upper()}")
        
        answers = clinical_state.get("answers", {})
        if answers:
            lines.append("Responses:")
            for k, v in answers.items():
                if not k.startswith("_"):
                    lines.append(f"  - {k}: {v}")
        
        return "\n".join(lines)
    
    def _extract_key_flags(self, clinical_state: Dict[str, Any]) -> List[str]:
        """Extract key clinical flags from the data."""
        flags = []
        
        demo = clinical_state.get("demographics", {})
        answers = clinical_state.get("answers", {})
        
        # Age flags
        age = demo.get("age", 0)
        if age < 2:
            flags.append("⚠️ Infant patient (under 2 years)")
        elif age > 65:
            flags.append("⚠️ Elderly patient (over 65 years)")
        
        # Pregnancy flag
        if demo.get("pregnant"):
            flags.append("⚠️ Patient is pregnant")
        
        # Check answers for concerning patterns
        for key, value in answers.items():
            key_lower = key.lower()
            
            # Fever flags
            if "fever" in key_lower or "temperature" in key_lower:
                if isinstance(value, (int, float)) and value > 39.5:
                    flags.append(f"🔴 High fever ({value}°C)")
                elif isinstance(value, (int, float)) and value > 38:
                    flags.append(f"🟡 Fever present ({value}°C)")
            
            # Pain severity
            if "pain" in key_lower and "severity" in key_lower:
                if isinstance(value, (int, float)) and value >= 8:
                    flags.append("🔴 Severe pain reported")
            
            # Breathing difficulty
            if "breathing" in key_lower or "breath" in key_lower:
                if value in [True, "yes", "Yes"]:
                    flags.append("🔴 Breathing difficulty reported")
            
            # Chest pain
            if "chest" in key_lower and "pain" in key_lower:
                if value in [True, "yes", "Yes"]:
                    flags.append("🔴 Chest pain reported")
            
            # Duration flags
            if "duration" in key_lower or "days" in key_lower:
                if isinstance(value, (int, float)) and value > 7:
                    flags.append(f"🟡 Prolonged symptoms ({value} days)")
        
        # Add triggered rule flags
        for rule in clinical_state.get("triggered_rules", []):
            if rule.get("band") == "red":
                flags.append(f"🔴 {rule.get('description', 'Red flag rule triggered')}")
        
        return flags
    
    # =========================================================================
    # Staff Q&A
    # =========================================================================
    
    def answer_staff_question(
        self, 
        question: str, 
        clinical_state: Dict[str, Any]
    ) -> Tuple[str, List[str]]:
        """
        Answer a staff question about a case.
        
        Args:
            question: The staff member's question
            clinical_state: Full case data
        
        Returns:
            (answer_text, list_of_cited_data_points)
        """
        # Format case data
        case_text = self._format_case_for_staff(clinical_state)
        
        prompt = STAFF_QA_PROMPT_TEMPLATE.format(
            case_data=case_text,
            question=question
        )
        
        answer = self._generate(prompt, max_tokens=512, temperature=0.3)
        
        # Extract cited data points (simple heuristic)
        cited_data = self._extract_citations(answer, clinical_state)
        
        return answer, cited_data
    
    def _format_case_for_staff(self, clinical_state: Dict[str, Any]) -> str:
        """Format case data for staff view."""
        lines = []
        
        # Demographics
        demo = clinical_state.get("demographics", {})
        lines.append("=== PATIENT DEMOGRAPHICS ===")
        lines.append(f"Age: {demo.get('age', 'Unknown')}")
        lines.append(f"Sex: {demo.get('sex', 'Unknown')}")
        lines.append(f"Pregnant: {demo.get('pregnant', 'N/A')}")
        
        # Chief complaint
        lines.append("\n=== CHIEF COMPLAINT ===")
        lines.append(clinical_state.get("chief_complaint", "Unknown"))
        
        # All answers
        lines.append("\n=== PATIENT RESPONSES ===")
        for key, value in clinical_state.get("answers", {}).items():
            if not key.startswith("_"):
                lines.append(f"{key}: {value}")
        
        # Risk assessment
        lines.append("\n=== RISK ASSESSMENT ===")
        lines.append(f"Risk Band: {clinical_state.get('risk_band', 'Unknown').upper()}")
        
        lines.append("\nTriggered Rules:")
        for rule in clinical_state.get("triggered_rules", []):
            lines.append(f"- [{rule.get('band', '?').upper()}] {rule.get('description', 'Unknown rule')}")
        
        # Summary
        if clinical_state.get("summary"):
            lines.append("\n=== AUTO-GENERATED SUMMARY ===")
            lines.append(clinical_state["summary"])
        
        return "\n".join(lines)
    
    def _extract_citations(self, answer: str, clinical_state: Dict[str, Any]) -> List[str]:
        """Extract data points that appear to be cited in the answer."""
        citations = []
        answer_lower = answer.lower()
        
        # Check demographics
        demo = clinical_state.get("demographics", {})
        if str(demo.get("age", "")) in answer:
            citations.append(f"Age: {demo.get('age')}")
        
        # Check answers
        for key, value in clinical_state.get("answers", {}).items():
            if str(value).lower() in answer_lower or key.lower() in answer_lower:
                citations.append(f"{key}: {value}")
        
        # Check triggered rules
        for rule in clinical_state.get("triggered_rules", []):
            desc = rule.get("description", "")
            if desc.lower() in answer_lower:
                citations.append(f"Rule: {desc}")
        
        return citations[:5]  # Limit to 5 citations


# =============================================================================
# Standalone Testing
# =============================================================================

if __name__ == "__main__":
    # Test the reasoning engine
    engine = ReasoningEngine("../models/llama-3.2-1b-instruct-q4_k_m.gguf")
    
    test_case = {
        "demographics": {"age": 45, "sex": "male", "pregnant": None},
        "chief_complaint": "chest_discomfort",
        "answers": {
            "chest_pain": True,
            "pain_severity": 7,
            "shortness_of_breath": True,
            "pain_radiating": "left arm",
            "duration_hours": 2
        },
        "risk_band": "red",
        "triggered_rules": [
            {"id": "chest_pain_sob", "description": "Chest pain with shortness of breath", "band": "red"}
        ]
    }
    
    print("Testing summary generation...")
    summary, flags = engine.generate_summary(test_case)
    print(f"\nSummary:\n{summary}")
    print(f"\nKey Flags:\n{flags}")
    
    print("\n" + "="*50)
    print("Testing staff Q&A...")
    answer, citations = engine.answer_staff_question(
        "Why was this patient flagged as red?",
        test_case
    )
    print(f"\nAnswer:\n{answer}")
    print(f"\nCitations:\n{citations}")
