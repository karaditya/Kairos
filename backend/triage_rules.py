"""
Triage Rules - Clean Deterministic Triage Tree Navigation

Handles question flow and navigation through triage trees.
This is the source of truth for triage logic - LLM cannot override.

Supports:
- English: Manchester Triage System (MTS) - 5 levels
- French: SFMU/CIMU - 6 levels (1, 2, 3A, 3B, 4, 5)
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

from config import CONFIG_DIR

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Question:
    """A triage question."""
    id: str
    text: str
    type: str  # yesno, choice, numeric, text
    options: List[Dict[str, str]] = field(default_factory=list)
    next: Optional[str] = None
    next_if_yes: Optional[str] = None
    next_if_no: Optional[str] = None
    next_conditions: Dict[str, str] = field(default_factory=dict)
    thresholds: List[Dict[str, Any]] = field(default_factory=list)
    multiple: bool = False
    min_value: Optional[float] = None
    max_value: Optional[float] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Question":
        """Create Question from dictionary."""
        return cls(
            id=data.get("id", ""),
            text=data.get("text", ""),
            type=data.get("type", "text"),
            options=data.get("options", []),
            next=data.get("next"),
            next_if_yes=data.get("next_if_yes"),
            next_if_no=data.get("next_if_no"),
            next_conditions=data.get("next_conditions", {}),
            thresholds=data.get("thresholds", []),
            multiple=data.get("multiple", False),
            min_value=data.get("min"),
            max_value=data.get("max"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to API-friendly dictionary."""
        result = {
            "id": self.id,
            "text": self.text,
            "type": self.type,
        }
        if self.options:
            result["options"] = self.options
        if self.multiple:
            result["multiple"] = True
        if self.min_value is not None:
            result["min"] = self.min_value
        if self.max_value is not None:
            result["max"] = self.max_value
        return result


@dataclass
class TriageTree:
    """A complete triage decision tree."""
    id: str
    name: str
    name_fr: str
    description: str
    description_fr: str
    system: str  # "english" or "french"
    questions: List[Question]
    version: str = "1.0"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TriageTree":
        """Create TriageTree from dictionary."""
        questions = [
            Question.from_dict(q) for q in data.get("questions", [])
        ]
        return cls(
            id=data.get("id", ""),
            name=data.get("name", data.get("name_en", "")),
            name_fr=data.get("name_fr", data.get("name", "")),
            description=data.get("description", data.get("description_en", "")),
            description_fr=data.get("description_fr", data.get("description", "")),
            system=data.get("system", "english"),
            questions=questions,
            version=data.get("version", "1.0"),
        )


# =============================================================================
# TRIAGE RULES ENGINE
# =============================================================================

class TriageRules:
    """
    Deterministic triage tree navigation.

    Manages question trees and navigates based on patient answers.
    This is the source of truth - LLM uses these rules for grounding.
    """

    def __init__(self, config_dir: str = None):
        """
        Initialize triage rules engine.

        Args:
            config_dir: Path to config directory containing triage_trees/
        """
        self.config_dir = config_dir or CONFIG_DIR
        self.trees: Dict[str, TriageTree] = {}
        self.trees_en: Dict[str, TriageTree] = {}
        self.trees_fr: Dict[str, TriageTree] = {}
        self._load_trees()

    def _load_trees(self):
        """Load all triage trees from config/triage_trees/."""
        trees_dir = os.path.join(self.config_dir, "triage_trees")

        if not os.path.exists(trees_dir):
            logger.warning(f"Triage trees directory not found: {trees_dir}")
            return

        for filename in os.listdir(trees_dir):
            if not filename.endswith(".json"):
                continue

            filepath = os.path.join(trees_dir, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    tree = TriageTree.from_dict(data)

                    self.trees[tree.id] = tree

                    # Categorize by language/system
                    if tree.system == "french" or tree.id.endswith("_fr"):
                        self.trees_fr[tree.id] = tree
                    else:
                        self.trees_en[tree.id] = tree

                    logger.debug(f"Loaded triage tree: {tree.id}")

            except Exception as e:
                logger.error(f"Error loading triage tree {filename}: {e}")

        logger.info(
            f"Loaded {len(self.trees)} triage trees "
            f"(EN: {len(self.trees_en)}, FR: {len(self.trees_fr)})"
        )

    def get_available_complaints(self, language: str = "en") -> List[Dict[str, str]]:
        """
        Get list of available chief complaints.

        Args:
            language: 'en' or 'fr'

        Returns:
            List of {id, name, description} dictionaries
        """
        complaints = []

        # Select appropriate trees
        if language == "fr":
            trees_to_use = self.trees_fr if self.trees_fr else self.trees_en
        else:
            trees_to_use = self.trees_en if self.trees_en else self.trees

        for tree_id, tree in trees_to_use.items():
            name = tree.name_fr if language == "fr" else tree.name
            description = tree.description_fr if language == "fr" else tree.description

            complaints.append({
                "id": tree_id,
                "name": name,
                "description": description,
            })

        return complaints

    def get_tree(self, tree_id: str, language: str = "en") -> Optional[TriageTree]:
        """
        Get triage tree by ID for specified language.

        Args:
            tree_id: Tree identifier
            language: 'en' or 'fr'

        Returns:
            TriageTree or None if not found
        """
        # Try language-specific tree first
        if language == "fr":
            fr_id = tree_id if tree_id.endswith("_fr") else f"{tree_id}_fr"
            if fr_id in self.trees:
                return self.trees[fr_id]

        # Fall back to base tree
        base_id = tree_id.replace("_fr", "")
        return self.trees.get(base_id) or self.trees.get(tree_id)

    def get_first_question(self, tree: TriageTree) -> Optional[Question]:
        """Get the first question in a tree."""
        if tree.questions:
            return tree.questions[0]
        return None

    def get_question_by_id(self, tree: TriageTree, question_id: str) -> Optional[Question]:
        """Get a specific question by ID."""
        for q in tree.questions:
            if q.id == question_id:
                return q
        return None

    def get_next_question(
        self,
        tree: TriageTree,
        current_question_id: str,
        answer: Any,
        all_answers: Dict[str, Any] = None
    ) -> Optional[Question]:
        """
        Determine next question based on current answer.

        Args:
            tree: The triage tree
            current_question_id: ID of current question
            answer: The answer provided
            all_answers: All answers so far (for complex conditions)

        Returns:
            Next Question or None if triage complete
        """
        current_q = self.get_question_by_id(tree, current_question_id)
        if not current_q:
            return None

        next_id = self._determine_next_id(current_q, answer)

        if next_id is None or next_id == "end":
            return None

        return self.get_question_by_id(tree, next_id)

    def _determine_next_id(self, question: Question, answer: Any) -> Optional[str]:
        """Determine next question ID based on question type and answer."""
        q_type = question.type

        if q_type == "yesno":
            if self._is_yes(answer):
                return question.next_if_yes
            else:
                return question.next_if_no

        elif q_type == "choice":
            # Check conditional next based on choice
            if question.next_conditions:
                answer_str = str(answer)
                if answer_str in question.next_conditions:
                    return question.next_conditions[answer_str]
            return question.next

        elif q_type == "numeric":
            # Check threshold conditions
            for threshold in question.thresholds:
                op = threshold.get("op", ">=")
                value = threshold.get("value", 0)
                if self._compare(answer, op, value):
                    return threshold.get("next")
            return question.next

        else:
            # Default: just go to next
            return question.next

    def _is_yes(self, value: Any) -> bool:
        """Check if value represents 'yes'."""
        return value in [True, "yes", "Yes", "YES", 1, "true", "True", "oui", "Oui", "OUI"]

    def _compare(self, value: Any, op: str, threshold: Any) -> bool:
        """Compare value against threshold."""
        try:
            value = float(value)
            threshold = float(threshold)

            if op == ">=":
                return value >= threshold
            elif op == ">":
                return value > threshold
            elif op == "<=":
                return value <= threshold
            elif op == "<":
                return value < threshold
            elif op == "==":
                return value == threshold
            elif op == "!=":
                return value != threshold
            else:
                return False
        except (ValueError, TypeError):
            return False

    def count_questions(self, tree: TriageTree) -> int:
        """Count total questions in tree."""
        return len(tree.questions)

    def calculate_progress(
        self,
        tree: TriageTree,
        answered_questions: List[str]
    ) -> float:
        """
        Calculate progress through triage tree.

        Args:
            tree: The triage tree
            answered_questions: List of answered question IDs

        Returns:
            Progress percentage (0-100)
        """
        total = self.count_questions(tree)
        if total == 0:
            return 100.0
        return min(100.0, (len(answered_questions) / total) * 100)


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_triage_rules_instance: Optional[TriageRules] = None


def get_triage_rules(reinitialize: bool = False) -> TriageRules:
    """
    Get singleton TriageRules instance.

    Args:
        reinitialize: Force create new instance

    Returns:
        TriageRules instance
    """
    global _triage_rules_instance

    if _triage_rules_instance is None or reinitialize:
        _triage_rules_instance = TriageRules()

    return _triage_rules_instance
