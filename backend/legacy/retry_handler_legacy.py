"""
Retry Handler for Medical Triage LLM Generation

Provides retry logic with:
1. Progressive simplification on each retry
2. Validation feedback loop
3. Context-aware graceful degradation
"""

import logging
from typing import Dict, List, Any, Optional, Callable, TYPE_CHECKING

from response_validator import TriageResponseValidator, ValidationResult

if TYPE_CHECKING:
    from multi_model_engine import MultiModelEngine

logger = logging.getLogger(__name__)


class GenerationRetryHandler:
    """Handles generation retries with validation and progressive fallback."""

    def __init__(self,
                 max_retries: int = 3,
                 validator: Optional[TriageResponseValidator] = None):
        """
        Initialize retry handler.

        Args:
            max_retries: Maximum number of generation attempts
            validator: Response validator instance (creates new if None)
        """
        self.max_retries = max_retries
        self.validator = validator or TriageResponseValidator()
        self._last_errors: List[str] = []

    def generate_with_retry(self,
                           generate_fn: Callable[..., Dict[str, Any]],
                           schema: Dict,
                           context: Optional[Dict] = None,
                           **kwargs) -> Dict[str, Any]:
        """
        Generate response with validation and retry logic.

        Args:
            generate_fn: Generation function to call
            schema: JSON schema for validation
            context: Patient context for context-aware repairs
            **kwargs: Additional arguments passed to generate_fn

        Returns:
            Validated (possibly repaired) response dictionary
        """
        self._last_errors = []

        for attempt in range(self.max_retries):
            try:
                # Generate response
                response = generate_fn(**kwargs)

                # Validate response
                validation = self.validator.validate(response, schema, context)

                if validation.is_valid:
                    logger.debug(f"Generation succeeded on attempt {attempt + 1}")
                    return response

                # Try repaired response if available
                if validation.repaired_response:
                    revalidation = self.validator.validate(
                        validation.repaired_response, schema, context
                    )
                    if revalidation.is_valid:
                        logger.info(f"Using repaired response from attempt {attempt + 1}")
                        return validation.repaired_response

                # Log issues for debugging
                self._last_errors.extend(validation.format_errors)
                self._last_errors.extend(validation.quality_issues)
                logger.warning(
                    f"Attempt {attempt + 1} validation failed: "
                    f"format_errors={validation.format_errors}, "
                    f"quality_issues={validation.quality_issues}"
                )

            except Exception as e:
                self._last_errors.append(str(e))
                logger.error(f"Generation error on attempt {attempt + 1}: {e}")

        # All retries exhausted - use graceful degradation
        logger.warning(f"All {self.max_retries} attempts failed, using fallback")
        return self._graceful_fallback(context, schema)

    def _graceful_fallback(self, context: Optional[Dict], schema: Dict) -> Dict[str, Any]:
        """Provide graceful degradation when all retries fail."""
        if not context:
            context = {}

        risk_band = context.get("risk_band", "amber")
        chief_complaint = str(context.get("chief_complaint", "")).lower()

        # Risk-appropriate fallback responses
        fallback_responses = {
            "red": {
                "patient_response": (
                    "Your symptoms indicate an urgent situation. "
                    "A healthcare provider will see you right away. "
                    "Please remain calm and stay where you are."
                ),
                "clinical_assessment": (
                    "High-priority case requiring immediate clinical evaluation. "
                    "Automated assessment incomplete due to technical issues. "
                    "Manual clinical review mandatory."
                ),
                "answer": (
                    "Your symptoms indicate an urgent situation. "
                    "A healthcare provider will see you right away."
                ),
                "reasoning": (
                    "High-priority case requiring immediate clinical evaluation. "
                    "Manual clinical review mandatory."
                ),
            },
            "amber": {
                "patient_response": (
                    "Thank you for providing your information. "
                    "Your symptoms have been recorded and a healthcare provider "
                    "will review your case shortly."
                ),
                "clinical_assessment": (
                    "Moderate priority based on initial screening. "
                    "Standard triage evaluation pathway recommended. "
                    "Clinical assessment pending provider review."
                ),
                "answer": (
                    "Thank you for providing your information. "
                    "A healthcare provider will review your case shortly."
                ),
                "reasoning": (
                    "Moderate priority based on initial screening. "
                    "Standard triage evaluation pathway recommended."
                ),
            },
            "green": {
                "patient_response": (
                    "Your symptoms have been recorded. "
                    "You will be seen in order of arrival. "
                    "Please let staff know if your symptoms change."
                ),
                "clinical_assessment": (
                    "Low-risk presentation based on initial assessment. "
                    "Standard evaluation pathway appropriate. "
                    "No immediate interventions required."
                ),
                "answer": (
                    "Your symptoms have been recorded. "
                    "You will be seen in order of arrival."
                ),
                "reasoning": (
                    "Low-risk presentation based on initial assessment. "
                    "Standard evaluation pathway appropriate."
                ),
            }
        }

        base_response = fallback_responses.get(risk_band, fallback_responses["amber"])

        # Generate context-appropriate questions
        questions = self._generate_fallback_questions(chief_complaint)

        # Build response matching expected schema
        result = {
            "follow_up_questions": questions,
            "confidence_score": 0.3,
            "fallback_used": True,
            "fallback_reason": f"Generation failed after {self.max_retries} attempts"
        }

        # Add fields based on what schema expects
        required = schema.get("required", [])
        if "answer" in required or "answer" in schema.get("properties", {}):
            result["answer"] = base_response.get("answer", base_response["patient_response"])
        if "patient_response" in required or "patient_response" in schema.get("properties", {}):
            result["patient_response"] = base_response["patient_response"]
        if "reasoning" in required or "reasoning" in schema.get("properties", {}):
            result["reasoning"] = base_response.get("reasoning", base_response["clinical_assessment"])
        if "clinical_assessment" in required or "clinical_assessment" in schema.get("properties", {}):
            result["clinical_assessment"] = base_response["clinical_assessment"]
        if "protocol_applied" in required or "protocol_applied" in schema.get("properties", {}):
            result["protocol_applied"] = ""

        return result

    def _generate_fallback_questions(self, chief_complaint: str) -> List[str]:
        """Generate contextually appropriate fallback questions."""
        complaint_questions = {
            "chest": [
                "Is the pain getting worse or staying the same?",
                "Have you taken any medication for this?",
                "Do you have any history of heart problems?"
            ],
            "head": [
                "Is this the worst headache you've ever had?",
                "Have you had any recent head injuries?",
                "Are you sensitive to light or sound?"
            ],
            "abdomen": [
                "Where exactly is the pain located?",
                "Have you been able to eat or drink?",
                "When was your last bowel movement?"
            ],
            "breath": [
                "Do you have a history of asthma or COPD?",
                "Are you wheezing or making sounds when breathing?",
                "Have you been exposed to any allergens?"
            ],
            "fever": [
                "What is your current temperature?",
                "How long have you had the fever?",
                "Have you taken any fever-reducing medication?"
            ]
        }

        # Match complaint to questions
        for keyword, qs in complaint_questions.items():
            if keyword in chief_complaint:
                return qs

        # Default questions if no match
        return [
            "Have you experienced these symptoms before?",
            "Are you currently taking any medications?",
            "Do you have any known allergies?"
        ]

    @property
    def last_errors(self) -> List[str]:
        """Get errors from the last generation attempt."""
        return self._last_errors.copy()


class FallbackManager:
    """Manages multi-level fallback strategy for LLM generation."""

    FALLBACK_LEVELS = [
        "full_generation",      # Normal LLM with JSON grammar
        "simplified_prompt",    # Simpler prompt, same grammar
        "no_grammar",           # No grammar, regex parsing
        "template_based",       # Pre-built templates with variable substitution
        "static_response",      # Completely static, rule-based response
    ]

    def __init__(self):
        self.failure_history: List[tuple] = []
        self.validator = TriageResponseValidator()

    def get_response(self,
                    engine: 'MultiModelEngine',
                    patient_context: Dict,
                    user_query: str,
                    schema: Dict) -> Dict[str, Any]:
        """
        Get response using appropriate fallback level.

        Tries progressively simpler generation strategies until one succeeds.
        """
        self.failure_history = []

        for level in self.FALLBACK_LEVELS:
            try:
                result = self._try_level(level, engine, patient_context, user_query, schema)

                # Validate result
                validation = self.validator.validate(result, schema, patient_context)
                if validation.is_valid:
                    result["_fallback_level"] = level
                    return result

                if validation.repaired_response:
                    revalidation = self.validator.validate(
                        validation.repaired_response, schema, patient_context
                    )
                    if revalidation.is_valid:
                        validation.repaired_response["_fallback_level"] = f"{level}_repaired"
                        return validation.repaired_response

                self.failure_history.append((level, "validation_failed"))

            except Exception as e:
                self.failure_history.append((level, str(e)))
                logger.warning(f"Fallback level '{level}' failed: {e}")
                continue

        # Absolute last resort
        return self._static_response(patient_context, schema)

    def _try_level(self,
                   level: str,
                   engine: 'MultiModelEngine',
                   context: Dict,
                   query: str,
                   schema: Dict) -> Dict:
        """Attempt generation at specified fallback level."""

        if level == "full_generation":
            return engine._generate_structured(context, query)

        elif level == "simplified_prompt":
            return engine._generate_with_simple_prompt(context, query)

        elif level == "no_grammar":
            return engine._generate_without_grammar(context, query)

        elif level == "template_based":
            return self._template_based_response(engine, context, query)

        else:  # static_response
            return self._static_response(context, schema)

    def _template_based_response(self,
                                 engine: 'MultiModelEngine',
                                 context: Dict,
                                 query: str) -> Dict:
        """Generate response using templates with LLM for specific parts."""
        risk_band = context.get("risk_band", "amber")
        chief_complaint = str(context.get("chief_complaint", "general"))

        template = {
            "patient_response": self._get_patient_template(risk_band),
            "answer": self._get_patient_template(risk_band),
            "clinical_assessment": self._get_assessment_template(context),
            "reasoning": self._get_assessment_template(context),
            "follow_up_questions": self._get_questions_for_complaint(chief_complaint),
            "confidence_score": 0.5,
            "protocol_applied": ""
        }

        # Try to enhance with LLM if available
        if engine.is_loaded:
            try:
                enhanced = engine._generate_single_field(
                    f"In one sentence, assess this case: {chief_complaint}",
                    max_tokens=100
                )
                if enhanced and len(enhanced) > 20:
                    template["clinical_assessment"] = enhanced
                    template["reasoning"] = enhanced
            except Exception:
                pass

        return template

    def _static_response(self, context: Dict, schema: Dict) -> Dict:
        """Completely static, rule-based response."""
        risk_band = context.get("risk_band", "amber") if context else "amber"
        chief_complaint = str(context.get("chief_complaint", "")) if context else ""

        responses = {
            "red": {
                "text": "Your symptoms indicate an urgent situation. A healthcare provider will see you right away.",
                "assessment": "High-priority case requiring immediate clinical evaluation."
            },
            "amber": {
                "text": "Your symptoms have been recorded. You will be seen as soon as possible.",
                "assessment": "Moderate priority. Standard triage pathway recommended."
            },
            "green": {
                "text": "Thank you for your patience. You will be called when it's your turn.",
                "assessment": "Low priority based on initial assessment."
            }
        }

        base = responses.get(risk_band, responses["amber"])
        questions = self._get_questions_for_complaint(chief_complaint)

        result = {
            "patient_response": base["text"],
            "answer": base["text"],
            "clinical_assessment": base["assessment"],
            "reasoning": base["assessment"],
            "follow_up_questions": questions,
            "confidence_score": 0.2,
            "protocol_applied": "",
            "fallback_used": True,
            "fallback_reason": "All generation methods failed"
        }

        return result

    def _get_patient_template(self, risk_band: str) -> str:
        """Get patient-facing template based on risk band."""
        templates = {
            "red": "Your symptoms require immediate attention. A healthcare provider will see you right away.",
            "amber": "Thank you for providing your information. A healthcare provider will review your case shortly.",
            "green": "Your symptoms have been recorded. You will be seen in order of arrival."
        }
        return templates.get(risk_band, templates["amber"])

    def _get_assessment_template(self, context: Dict) -> str:
        """Get clinical assessment template."""
        risk_band = context.get("risk_band", "amber") if context else "amber"
        complaint = context.get("chief_complaint", "presenting symptoms") if context else "presenting symptoms"

        return f"Patient presents with {complaint}. Risk assessment: {risk_band.upper()} priority. Clinical evaluation recommended."

    def _get_questions_for_complaint(self, complaint: str) -> List[str]:
        """Get appropriate questions for complaint type."""
        complaint_lower = complaint.lower()

        question_map = {
            "chest": ["Is the pain getting worse?", "Any history of heart problems?", "Taking any heart medications?"],
            "head": ["Worst headache ever?", "Any vision changes?", "Recent head injury?"],
            "abdomen": ["Where is the pain located?", "Able to eat/drink?", "Any vomiting?"],
            "breath": ["History of lung conditions?", "Wheezing present?", "Recent illness?"],
            "fever": ["Current temperature?", "How long with fever?", "Any other symptoms?"]
        }

        for keyword, questions in question_map.items():
            if keyword in complaint_lower:
                return questions

        return ["How long have symptoms lasted?", "Any medications?", "Any allergies?"]
