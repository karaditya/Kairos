"""
Response Validator for Medical Triage LLM Outputs

Validates and repairs LLM responses to ensure:
1. Schema compliance (required fields, correct types)
2. Quality standards (min length, clinical terms, no instruction echoes)
3. Medical appropriateness (proper terminology, safety checks)
"""

import re
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of response validation."""
    is_valid: bool
    format_errors: List[str] = field(default_factory=list)
    quality_issues: List[str] = field(default_factory=list)
    severity: str = "pass"  # "critical", "warning", "info", "pass"
    repaired_response: Optional[Dict] = None


class TriageResponseValidator:
    """Validates and repairs LLM responses for medical triage."""

    # Quality thresholds
    MIN_ASSESSMENT_LENGTH = 50
    MIN_PATIENT_RESPONSE_LENGTH = 30
    MIN_QUESTIONS = 3
    MAX_QUESTIONS = 5

    # Forbidden patterns - instruction echoes and placeholders
    FORBIDDEN_PATTERNS = [
        r'\[Your [^\]]+\]',
        r'YOUR (CLINICAL|FIRST|SECOND|THIRD)',
        r'^Based on the (data|information|card|patient)',
        r'^As (a|an|the) (nurse|doctor|physician|clinician)',
        r'(TONE FOR|OUTPUT FORMAT|STRICT RULES)',
        r'Patient-friendly response here',
        r'\[.*?analysis.*?\]',
        r'\[.*?response.*?\]',
        r'\[Insert.*?\]',
        r'<your.*?>',
        r'YOUR PATIENT RESPONSE',
        r'YOUR CLINICAL ANALYSIS',
    ]

    # Clinical terminology that should appear in medical assessments
    CLINICAL_TERMS = [
        'pain', 'symptom', 'patient', 'assess', 'risk',
        'urgent', 'evaluation', 'condition', 'history',
        'present', 'report', 'indicate', 'concern', 'severity',
        'acute', 'chronic', 'onset', 'duration', 'location'
    ]

    # Context-aware fallback questions by complaint category
    FALLBACK_QUESTIONS = {
        "chest": [
            "Is the pain getting worse or staying the same?",
            "Do you have any history of heart problems?",
            "Are you taking any blood thinners or heart medications?"
        ],
        "head": [
            "Is this the worst headache you've ever had?",
            "Have you noticed any vision changes or sensitivity to light?",
            "Have you had any recent head injuries?"
        ],
        "abdomen": [
            "Where exactly is the pain located?",
            "Have you been able to eat or drink normally?",
            "When was your last bowel movement?"
        ],
        "breath": [
            "Do you have a history of asthma or lung conditions?",
            "Are you making any unusual sounds when breathing?",
            "Have you been exposed to any allergens recently?"
        ],
        "fever": [
            "What is your current temperature?",
            "How long have you had the fever?",
            "Have you taken any medication for the fever?"
        ],
        "default": [
            "How long have you been experiencing these symptoms?",
            "Are you currently taking any medications?",
            "Do you have any known allergies?"
        ]
    }

    def validate(self, response: Dict, schema: Dict, context: Optional[Dict] = None) -> ValidationResult:
        """
        Full validation of response against schema and quality criteria.

        Args:
            response: The LLM response dictionary
            schema: JSON schema to validate against
            context: Optional patient context for context-aware repairs

        Returns:
            ValidationResult with validity status and any repairs
        """
        format_errors = []
        quality_issues = []

        # 1. Schema validation
        format_errors.extend(self._validate_schema(response, schema))

        # 2. Content quality validation
        quality_issues.extend(self._validate_content_quality(response))

        # 3. Check for forbidden patterns (instruction echoes)
        quality_issues.extend(self._check_forbidden_patterns(response))

        # 4. Medical appropriateness (only for clinical_assessment/reasoning)
        quality_issues.extend(self._validate_medical_content(response))

        # Determine severity
        if format_errors:
            severity = "critical"
        elif any(
            "missing" in q.lower() or
            "too short" in q.lower() or
            "not enough" in q.lower()
            for q in quality_issues
        ):
            severity = "warning"
        elif quality_issues:
            severity = "info"
        else:
            severity = "pass"

        is_valid = len(format_errors) == 0 and severity not in ["critical", "warning"]

        # Attempt repair if there are issues
        repaired = None
        if not is_valid:
            repaired = self._attempt_repair(response, context)
            # Re-validate repaired response for format only
            revalidation_errors = self._validate_schema(repaired, schema)
            if not revalidation_errors:
                is_valid = True

        return ValidationResult(
            is_valid=is_valid,
            format_errors=format_errors,
            quality_issues=quality_issues,
            severity=severity,
            repaired_response=repaired
        )

    def _validate_schema(self, response: Dict, schema: Dict) -> List[str]:
        """Validate response against JSON schema."""
        errors = []
        required = schema.get("required", [])
        properties = schema.get("properties", {})

        for field_name in required:
            if field_name not in response:
                errors.append(f"Missing required field: {field_name}")
            elif response[field_name] is None:
                errors.append(f"Field '{field_name}' is null")
            else:
                field_type = properties.get(field_name, {}).get("type")
                if field_type == "array" and not isinstance(response[field_name], list):
                    errors.append(f"Field '{field_name}' should be array, got {type(response[field_name]).__name__}")
                elif field_type == "string" and not isinstance(response[field_name], str):
                    errors.append(f"Field '{field_name}' should be string, got {type(response[field_name]).__name__}")
                elif field_type == "number" and not isinstance(response[field_name], (int, float)):
                    errors.append(f"Field '{field_name}' should be number, got {type(response[field_name]).__name__}")

        return errors

    def _validate_content_quality(self, response: Dict) -> List[str]:
        """Check content quality metrics."""
        issues = []

        # Check clinical assessment / reasoning length
        assessment = response.get("clinical_assessment", response.get("reasoning", ""))
        if isinstance(assessment, str) and len(assessment) < self.MIN_ASSESSMENT_LENGTH:
            issues.append(f"Clinical assessment too short ({len(assessment)} chars, min {self.MIN_ASSESSMENT_LENGTH})")

        # Check patient response / answer length
        patient_response = response.get("patient_response", response.get("answer", ""))
        if isinstance(patient_response, str) and len(patient_response) < self.MIN_PATIENT_RESPONSE_LENGTH:
            issues.append(f"Patient response too short ({len(patient_response)} chars, min {self.MIN_PATIENT_RESPONSE_LENGTH})")

        # Check questions
        questions = response.get("follow_up_questions", [])
        if isinstance(questions, list):
            if len(questions) < self.MIN_QUESTIONS:
                issues.append(f"Not enough questions ({len(questions)}, need {self.MIN_QUESTIONS})")

            # Check individual question quality
            for i, q in enumerate(questions):
                if isinstance(q, str):
                    if not q.strip().endswith("?"):
                        issues.append(f"Question {i+1} missing question mark")
                    if len(q.strip()) < 10:
                        issues.append(f"Question {i+1} too short: '{q[:30]}'")

        return issues

    def _check_forbidden_patterns(self, response: Dict) -> List[str]:
        """Detect instruction echoing and other forbidden patterns."""
        issues = []

        for field_name, value in response.items():
            if not isinstance(value, str):
                continue

            for pattern in self.FORBIDDEN_PATTERNS:
                if re.search(pattern, value, re.IGNORECASE | re.MULTILINE):
                    issues.append(f"Forbidden pattern in '{field_name}': matches '{pattern[:30]}...'")
                    break  # One issue per field is enough

        return issues

    def _validate_medical_content(self, response: Dict) -> List[str]:
        """Ensure response contains appropriate medical content."""
        issues = []

        # Check for clinical terminology in assessment
        assessment = response.get("clinical_assessment", response.get("reasoning", ""))
        if isinstance(assessment, str) and len(assessment) > 20:
            assessment_lower = assessment.lower()
            term_count = sum(1 for term in self.CLINICAL_TERMS if term in assessment_lower)
            if term_count < 2:
                issues.append("Assessment lacks clinical terminology")

        return issues

    def _attempt_repair(self, response: Dict, context: Optional[Dict] = None) -> Dict:
        """Attempt to repair issues in the response."""
        repaired = response.copy()

        # 1. Clean forbidden patterns from text fields
        text_fields = ["clinical_assessment", "reasoning", "patient_response", "answer"]
        for field_name in text_fields:
            if field_name in repaired and isinstance(repaired[field_name], str):
                repaired[field_name] = self._clean_text(repaired[field_name])

        # 2. Fix questions
        questions = repaired.get("follow_up_questions", [])
        fixed_questions = []

        if isinstance(questions, list):
            for q in questions:
                if isinstance(q, str):
                    cleaned = self._clean_question(q)
                    if cleaned and len(cleaned) >= 8:  # Allow shorter valid questions
                        fixed_questions.append(cleaned)

        # 3. Add context-appropriate fallback questions if needed
        while len(fixed_questions) < self.MIN_QUESTIONS:
            fallback = self._get_context_appropriate_question(context, fixed_questions)
            if fallback and fallback not in fixed_questions:
                fixed_questions.append(fallback)
            else:
                break  # Avoid infinite loop

        repaired["follow_up_questions"] = fixed_questions[:self.MAX_QUESTIONS]

        # 4. Ensure answer/patient_response exists
        answer_field = "answer" if "answer" in response else "patient_response"
        if answer_field not in repaired or not repaired.get(answer_field):
            repaired[answer_field] = self._generate_minimal_answer(context)

        # 5. Ensure reasoning/clinical_assessment exists
        reasoning_field = "reasoning" if "reasoning" in response else "clinical_assessment"
        if reasoning_field not in repaired:
            repaired[reasoning_field] = ""

        return repaired

    def _clean_text(self, text: str) -> str:
        """Remove instruction echoes and clean text."""
        cleaned = text

        for pattern in self.FORBIDDEN_PATTERNS:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE | re.MULTILINE)

        # Clean up multiple spaces and newlines
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        cleaned = re.sub(r' {2,}', ' ', cleaned)

        return cleaned.strip()

    def _clean_question(self, question: str) -> str:
        """Clean and normalize a question."""
        cleaned = question.strip()

        # Remove common placeholders
        placeholders = [
            r'^\[.*?\]\s*',
            r'^YOUR\s+\w+\s+QUESTION\??',
            r'^Question\s*\d*[:.]?\s*',
            r'^-\s*',
            r'^\d+[.)]\s*',
        ]
        for pattern in placeholders:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)

        cleaned = cleaned.strip()

        # Add question mark if missing and looks like a question
        if cleaned and not cleaned.endswith("?"):
            question_words = ['what', 'when', 'where', 'why', 'how', 'is', 'are',
                           'do', 'does', 'did', 'have', 'has', 'can', 'could',
                           'would', 'should', 'will', 'any']
            if any(cleaned.lower().startswith(w) for w in question_words):
                cleaned += "?"

        return cleaned

    def _get_context_appropriate_question(self, context: Optional[Dict],
                                          existing: List[str]) -> Optional[str]:
        """Generate a context-appropriate fallback question."""
        if not context:
            category = "default"
        else:
            complaint = str(context.get("chief_complaint", "")).lower()

            # Determine category from complaint
            if any(word in complaint for word in ["chest", "heart", "cardiac"]):
                category = "chest"
            elif any(word in complaint for word in ["head", "headache", "migraine"]):
                category = "head"
            elif any(word in complaint for word in ["abdomen", "stomach", "belly", "abdominal"]):
                category = "abdomen"
            elif any(word in complaint for word in ["breath", "breathing", "respiratory", "lung"]):
                category = "breath"
            elif any(word in complaint for word in ["fever", "temperature", "hot"]):
                category = "fever"
            else:
                category = "default"

        # Get questions for this category
        questions = self.FALLBACK_QUESTIONS.get(category, self.FALLBACK_QUESTIONS["default"])

        # Return first question not already in existing
        for q in questions:
            if q not in existing:
                return q

        # If all category questions used, try default
        if category != "default":
            for q in self.FALLBACK_QUESTIONS["default"]:
                if q not in existing:
                    return q

        return None

    def _generate_minimal_answer(self, context: Optional[Dict]) -> str:
        """Generate a minimal but appropriate answer when none provided."""
        if not context:
            return "Thank you for providing your information. A healthcare provider will review your case."

        risk_band = context.get("risk_band", "amber")

        if risk_band == "red":
            return "Your symptoms require immediate attention. A healthcare provider will see you right away."
        elif risk_band == "green":
            return "Your symptoms have been recorded. You will be seen in order of arrival."
        else:
            return "Thank you for providing your information. A healthcare provider will review your case shortly."


def validate_triage_response(response: Dict,
                             schema: Dict,
                             context: Optional[Dict] = None) -> ValidationResult:
    """
    Convenience function to validate a triage response.

    Args:
        response: The LLM response dictionary
        schema: JSON schema to validate against
        context: Optional patient context for context-aware repairs

    Returns:
        ValidationResult with validity status and any repairs
    """
    validator = TriageResponseValidator()
    return validator.validate(response, schema, context)
