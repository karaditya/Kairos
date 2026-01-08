"""
Medical Keyword Tagger for CAE System.

Detects medical keywords in transcripts for:
- Smart tagging in UI
- Protocol triggering
- Missing data detection
"""

import re
from typing import List, Set, Dict, Any
from dataclasses import dataclass

from config import MEDICAL_KEYWORDS
from utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class KeywordMatch:
    """Keyword match result."""
    keyword: str
    category: str
    position: int
    context: str


class KeywordTagger:
    """Tags medical keywords in text."""

    # Extended keyword categories
    KEYWORD_CATEGORIES: Dict[str, Dict[str, List[str]]] = {
        "fr": {
            "vitaux": [
                "tension", "pouls", "frequence cardiaque", "saturation",
                "temperature", "fievre", "pression arterielle", "fc",
                "spo2", "ta", "bpm",
            ],
            "symptomes": [
                "douleur", "mal", "gene", "difficulte", "nausee", "vomissement",
                "vertige", "fatigue", "faiblesse", "essoufflement", "dyspnee",
                "toux", "cephalee", "migraine",
            ],
            "allergies": [
                "allergie", "allergique", "reaction", "intolerance",
                "anaphylaxie", "urticaire",
            ],
            "antecedents": [
                "antecedent", "chirurgie", "operation", "hospitalisation",
                "diagnostic", "maladie chronique", "diabete", "hypertension",
                "asthme", "cancer",
            ],
            "traitements": [
                "medicament", "traitement", "prescription", "posologie",
                "comprime", "gelule", "injection", "perfusion", "dose",
            ],
            "examens": [
                "examen", "analyse", "radiographie", "scanner", "irm",
                "echographie", "prise de sang", "electrocardiogramme", "ecg",
            ],
            "localisation": [
                "thorax", "abdomen", "tete", "bras", "jambe", "dos",
                "poitrine", "ventre", "cou", "gorge", "oreille", "oeil",
            ],
            "temporel": [
                "depuis", "il y a", "debut", "apparition", "progression",
                "aggravation", "amelioration", "soudain", "brutal",
            ],
        },
        "en": {
            "vitals": [
                "blood pressure", "pulse", "heart rate", "saturation",
                "temperature", "fever", "bp", "hr", "spo2", "bpm",
            ],
            "symptoms": [
                "pain", "ache", "discomfort", "difficulty", "nausea", "vomiting",
                "dizziness", "fatigue", "weakness", "shortness of breath", "dyspnea",
                "cough", "headache", "migraine",
            ],
            "allergies": [
                "allergy", "allergic", "reaction", "intolerance",
                "anaphylaxis", "hives",
            ],
            "history": [
                "history", "surgery", "operation", "hospitalization",
                "diagnosis", "chronic disease", "diabetes", "hypertension",
                "asthma", "cancer",
            ],
            "treatments": [
                "medication", "treatment", "prescription", "dosage",
                "tablet", "capsule", "injection", "infusion", "dose",
            ],
            "exams": [
                "exam", "test", "x-ray", "ct scan", "mri",
                "ultrasound", "blood test", "electrocardiogram", "ecg",
            ],
            "location": [
                "chest", "abdomen", "head", "arm", "leg", "back",
                "stomach", "neck", "throat", "ear", "eye",
            ],
            "temporal": [
                "since", "ago", "onset", "started", "progression",
                "worsening", "improving", "sudden", "acute",
            ],
        },
    }

    def __init__(self, language: str = "fr"):
        self.language = language
        self._compiled_patterns: Dict[str, re.Pattern] = {}
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Pre-compile regex patterns for efficiency."""
        categories = self.KEYWORD_CATEGORIES.get(self.language, {})
        for category, keywords in categories.items():
            # Create pattern that matches any keyword (word boundary)
            pattern = r'\b(' + '|'.join(re.escape(kw) for kw in keywords) + r')\b'
            self._compiled_patterns[category] = re.compile(pattern, re.IGNORECASE)

    def tag_text(self, text: str) -> List[KeywordMatch]:
        """
        Find all keyword matches in text.

        Args:
            text: Input text to analyze

        Returns:
            List of KeywordMatch objects
        """
        if not text:
            return []

        matches = []
        text_lower = text.lower()

        for category, pattern in self._compiled_patterns.items():
            for match in pattern.finditer(text_lower):
                # Get context (surrounding words)
                start = max(0, match.start() - 30)
                end = min(len(text), match.end() + 30)
                context = text[start:end].strip()
                if start > 0:
                    context = "..." + context
                if end < len(text):
                    context = context + "..."

                matches.append(KeywordMatch(
                    keyword=match.group(0),
                    category=category,
                    position=match.start(),
                    context=context,
                ))

        # Sort by position
        matches.sort(key=lambda m: m.position)

        return matches

    def get_keywords(self, text: str) -> List[str]:
        """
        Get unique keywords found in text.

        Args:
            text: Input text

        Returns:
            List of unique keywords
        """
        matches = self.tag_text(text)
        return list(set(m.keyword for m in matches))

    def get_keywords_by_category(self, text: str) -> Dict[str, List[str]]:
        """
        Get keywords grouped by category.

        Args:
            text: Input text

        Returns:
            Dict mapping category to list of keywords
        """
        matches = self.tag_text(text)
        result: Dict[str, Set[str]] = {}

        for match in matches:
            if match.category not in result:
                result[match.category] = set()
            result[match.category].add(match.keyword)

        return {k: list(v) for k, v in result.items()}

    def detect_missing_required(
        self,
        text: str,
        required_categories: List[str] = None,
    ) -> List[str]:
        """
        Detect missing required keyword categories.

        Args:
            text: Input text
            required_categories: Categories that should be present

        Returns:
            List of missing category names
        """
        if required_categories is None:
            # Default required categories
            required_categories = ["allergies", "symptomes" if self.language == "fr" else "symptoms"]

        found_categories = set(self.get_keywords_by_category(text).keys())
        missing = [cat for cat in required_categories if cat not in found_categories]

        return missing

    def highlight_keywords(self, text: str, marker: str = "**") -> str:
        """
        Return text with keywords highlighted.

        Args:
            text: Input text
            marker: Markdown marker for highlighting

        Returns:
            Text with keywords wrapped in markers
        """
        matches = self.tag_text(text)

        # Process in reverse order to preserve positions
        result = text
        for match in reversed(matches):
            start = match.position
            end = start + len(match.keyword)
            result = result[:start] + marker + result[start:end] + marker + result[end:]

        return result


# =============================================================================
# FACTORY
# =============================================================================

def create_keyword_tagger(language: str = "fr") -> KeywordTagger:
    """Create keyword tagger for language."""
    return KeywordTagger(language)
