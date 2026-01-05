"""
Risk Calculator - Deterministic Risk Band Computation

Computes risk bands based on patient data and rule conditions.
This is the source of truth - LLM cannot override risk bands.

Supports:
- English: Manchester Triage System (MTS) - 5 levels (1-5)
- French: SFMU/CIMU - 6 levels (1, 2, 3A, 3B, 4, 5)
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

from config import (
    CONFIG_DIR,
    ENGLISH_RISK_BANDS,
    FRENCH_RISK_BANDS,
)

logger = logging.getLogger(__name__)


# =============================================================================
# TRIAGE LEVEL DEFINITIONS
# =============================================================================

# Manchester Triage System (MTS) - English
MANCHESTER_LEVELS = {
    "1": {
        "color": "#dc3545",
        "name": "Immediate",
        "band": "red",
        "max_wait_minutes": 0,
        "description": "Life-threatening condition requiring immediate intervention",
    },
    "2": {
        "color": "#ff6b35",
        "name": "Very Urgent",
        "band": "orange",
        "max_wait_minutes": 10,
        "description": "Severe condition requiring very rapid assessment",
    },
    "3": {
        "color": "#ffc107",
        "name": "Urgent",
        "band": "yellow",
        "max_wait_minutes": 60,
        "description": "Serious condition but patient is stable",
    },
    "4": {
        "color": "#28a745",
        "name": "Standard",
        "band": "green",
        "max_wait_minutes": 120,
        "description": "Standard condition requiring routine care",
    },
    "5": {
        "color": "#17a2b8",
        "name": "Non-urgent",
        "band": "blue",
        "max_wait_minutes": 240,
        "description": "Minor condition that can safely wait",
    },
}

# SFMU Triage System - French
SFMU_LEVELS = {
    "1": {
        "color": "#dc3545",
        "name": "Tri 1",
        "name_en": "Level 1",
        "band": "red",
        "max_wait_minutes": 1,
        "description": "Detresse vitale majeure",
        "description_en": "Life-threatening emergency",
    },
    "2": {
        "color": "#ff6b35",
        "name": "Tri 2",
        "name_en": "Level 2",
        "band": "red",
        "max_wait_minutes": 20,
        "description": "Atteinte patente d'un organe",
        "description_en": "Severe organ involvement",
    },
    "3A": {
        "color": "#ffc107",
        "name": "Tri 3A",
        "name_en": "Level 3A",
        "band": "amber",
        "max_wait_minutes": 60,
        "description": "Urgence avec comorbidite",
        "description_en": "Urgent with comorbidity",
    },
    "3B": {
        "color": "#ffda6b",
        "name": "Tri 3B",
        "name_en": "Level 3B",
        "band": "amber",
        "max_wait_minutes": 90,
        "description": "Urgence sans comorbidite",
        "description_en": "Urgent without comorbidity",
    },
    "4": {
        "color": "#28a745",
        "name": "Tri 4",
        "name_en": "Level 4",
        "band": "green",
        "max_wait_minutes": 120,
        "description": "Atteinte fonctionnelle stable",
        "description_en": "Stable functional impairment",
    },
    "5": {
        "color": "#20c997",
        "name": "Tri 5",
        "name_en": "Level 5",
        "band": "green",
        "max_wait_minutes": 240,
        "description": "Pas d'atteinte evidente",
        "description_en": "No obvious impairment",
    },
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class RiskRule:
    """A single risk assessment rule."""
    id: str
    description: str
    description_en: str
    band: str  # red, orange, yellow, green, blue, amber
    level: str  # 1, 2, 3, 3A, 3B, 4, 5
    conditions: List[Dict[str, Any]]
    priority: int = 0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RiskRule":
        """Create RiskRule from dictionary."""
        return cls(
            id=data.get("id", ""),
            description=data.get("description", ""),
            description_en=data.get("description_en", data.get("description", "")),
            band=data.get("band", "green"),
            level=data.get("level", "4"),
            conditions=data.get("conditions", []),
            priority=data.get("priority", 0),
        )


@dataclass
class RiskResult:
    """Result of risk computation."""
    level: str
    band: str
    color: str
    name: str
    description: str
    max_wait_minutes: int
    triggered_rules: List[Dict[str, Any]] = field(default_factory=list)
    language: str = "en"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to API-friendly dictionary."""
        return {
            "level": self.level,
            "band": self.band,
            "color": self.color,
            "name": self.name,
            "description": self.description,
            "max_wait_minutes": self.max_wait_minutes,
            "triggered_rules": self.triggered_rules,
            "level_info": {
                "color": self.color,
                "name": self.name,
                "band": self.band,
                "max_wait_minutes": self.max_wait_minutes,
            },
        }


# =============================================================================
# RISK CALCULATOR
# =============================================================================

class RiskCalculator:
    """
    Deterministic risk band calculator.

    The LLM CANNOT override these rules - they are the source of truth.
    Risk is computed purely from patient data and rule conditions.
    """

    def __init__(self, config_dir: str = None):
        """
        Initialize risk calculator.

        Args:
            config_dir: Path to config directory containing risk_rules.json
        """
        self.config_dir = config_dir or CONFIG_DIR
        self.rules_en: List[RiskRule] = []
        self.rules_fr: List[RiskRule] = []
        self.manchester_levels = MANCHESTER_LEVELS
        self.sfmu_levels = SFMU_LEVELS
        self._load_rules()

    def _load_rules(self):
        """Load risk rules from JSON files."""
        # Load English (Manchester) rules
        en_path = os.path.join(self.config_dir, "risk_rules.json")
        if os.path.exists(en_path):
            try:
                with open(en_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for rule_data in data.get("rules", []):
                        self.rules_en.append(RiskRule.from_dict(rule_data))
                logger.info(f"Loaded {len(self.rules_en)} English risk rules")
            except Exception as e:
                logger.error(f"Error loading English risk rules: {e}")

        # Load French (SFMU) rules
        fr_path = os.path.join(self.config_dir, "risk_rules_french.json")
        if os.path.exists(fr_path):
            try:
                with open(fr_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for rule_data in data.get("rules", []):
                        self.rules_fr.append(RiskRule.from_dict(rule_data))
                logger.info(f"Loaded {len(self.rules_fr)} French risk rules")
            except Exception as e:
                logger.error(f"Error loading French risk rules: {e}")

        # Fallback rules if none loaded
        if not self.rules_en:
            self._load_fallback_rules()

        if not self.rules_fr:
            # Use English rules as fallback for French
            self.rules_fr = self.rules_en.copy()

    def _load_fallback_rules(self):
        """Load minimal fallback rules."""
        self.rules_en = [
            RiskRule(
                id="infant",
                description="Patient is an infant (under 2 years)",
                description_en="Patient is an infant (under 2 years)",
                band="red",
                level="1",
                conditions=[{"field": "demographics.age", "op": "<", "value": 2}],
                priority=100,
            ),
            RiskRule(
                id="elderly",
                description="Patient is elderly (over 65 years)",
                description_en="Patient is elderly (over 65 years)",
                band="orange",
                level="2",
                conditions=[{"field": "demographics.age", "op": ">", "value": 65}],
                priority=60,
            ),
            RiskRule(
                id="default",
                description="No high-risk factors identified",
                description_en="No high-risk factors identified",
                band="green",
                level="4",
                conditions=[],
                priority=-100,
            ),
        ]
        logger.warning("Using fallback risk rules")

    def compute_risk(
        self,
        demographics: Dict[str, Any],
        answers: Dict[str, Any],
        language: str = "en"
    ) -> RiskResult:
        """
        Compute risk band from patient data.

        Args:
            demographics: Patient demographics (age, sex, pregnant)
            answers: Triage question answers
            language: 'en' for Manchester, 'fr' for SFMU

        Returns:
            RiskResult with computed risk band and triggered rules
        """
        # Combine data for rule evaluation
        data = {
            "demographics": demographics,
            "answers": answers,
        }

        # Select rules and levels based on language
        if language == "fr":
            rules = self.rules_fr if self.rules_fr else self.rules_en
            levels = self.sfmu_levels
            level_priority = {"5": 0, "4": 1, "3B": 2, "3A": 3, "2": 4, "1": 5}
            default_level = "5"
        else:
            rules = self.rules_en
            levels = self.manchester_levels
            level_priority = {"5": 0, "4": 1, "3": 2, "2": 3, "1": 4}
            default_level = "5"

        # Evaluate rules (sorted by priority, highest first)
        sorted_rules = sorted(rules, key=lambda r: r.priority, reverse=True)
        triggered_rules = []
        highest_level = default_level

        for rule in sorted_rules:
            if self._evaluate_rule(rule, data):
                # Add to triggered rules
                rule_info = {
                    "id": rule.id,
                    "description": rule.description if language == "fr" else rule.description_en,
                    "band": rule.band,
                    "level": rule.level,
                }
                triggered_rules.append(rule_info)

                # Update highest level
                if level_priority.get(rule.level, 0) > level_priority.get(highest_level, 0):
                    highest_level = rule.level

        # Get level info
        level_info = levels.get(highest_level, levels.get(default_level, {}))

        return RiskResult(
            level=highest_level,
            band=level_info.get("band", "green"),
            color=level_info.get("color", "#28a745"),
            name=level_info.get("name", "Standard"),
            description=level_info.get("description", ""),
            max_wait_minutes=level_info.get("max_wait_minutes", 120),
            triggered_rules=triggered_rules,
            language=language,
        )

    def _evaluate_rule(self, rule: RiskRule, data: Dict[str, Any]) -> bool:
        """
        Evaluate if a rule's conditions are met.

        All conditions must be true for the rule to trigger.
        """
        if not rule.conditions:
            return True  # Rules with no conditions always match

        for condition in rule.conditions:
            if not self._evaluate_condition(condition, data):
                return False

        return True

    def _evaluate_condition(self, condition: Dict[str, Any], data: Dict[str, Any]) -> bool:
        """Evaluate a single condition."""
        field_path = condition.get("field", "")
        op = condition.get("op", "==")
        expected = condition.get("value")

        # Get actual value using dot notation
        actual = self._get_nested_value(data, field_path)

        if actual is None and op not in ["exists", "not_exists"]:
            return False

        try:
            if op == "==":
                return actual == expected
            elif op == "!=":
                return actual != expected
            elif op == ">":
                return float(actual) > float(expected)
            elif op == ">=":
                return float(actual) >= float(expected)
            elif op == "<":
                return float(actual) < float(expected)
            elif op == "<=":
                return float(actual) <= float(expected)
            elif op == "in":
                if isinstance(actual, list):
                    return any(item in expected for item in actual)
                return actual in expected
            elif op == "contains":
                if isinstance(actual, list):
                    return expected in actual
                return expected in str(actual).lower()
            elif op == "exists":
                return actual is not None
            elif op == "not_exists":
                return actual is None
            elif op == "is_true":
                return actual in [True, "yes", "Yes", "YES", 1, "true", "True", "oui", "Oui"]
            elif op == "is_false":
                return actual in [False, "no", "No", "NO", 0, "false", "False", "non", "Non"]
            else:
                return False
        except (ValueError, TypeError):
            return False

    def _get_nested_value(self, data: Dict, path: str) -> Any:
        """Get value from nested dict using dot notation."""
        keys = path.split(".")
        value = data

        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            else:
                return None

            if value is None:
                return None

        return value

    def get_level_info(self, level: str, language: str = "en") -> Dict[str, Any]:
        """Get information about a specific triage level."""
        if language == "fr":
            return self.sfmu_levels.get(level, {})
        return self.manchester_levels.get(level, {})

    def get_all_levels(self, language: str = "en") -> Dict[str, Dict[str, Any]]:
        """Get all triage levels for a language."""
        if language == "fr":
            return self.sfmu_levels
        return self.manchester_levels


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_risk_calculator_instance: Optional[RiskCalculator] = None


def get_risk_calculator(reinitialize: bool = False) -> RiskCalculator:
    """
    Get singleton RiskCalculator instance.

    Args:
        reinitialize: Force create new instance

    Returns:
        RiskCalculator instance
    """
    global _risk_calculator_instance

    if _risk_calculator_instance is None or reinitialize:
        _risk_calculator_instance = RiskCalculator()

    return _risk_calculator_instance
