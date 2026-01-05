"""
Triage Engine - Decision Trees and Risk Assessment

Handles:
- Loading triage question trees from JSON
- Navigating through questions based on answers
- Deterministic risk band calculation (green/amber/red)
- Support for multiple triage systems (English default, FRENCH)

The model CANNOT override risk bands - they are computed purely by rules.
"""

import os
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# =============================================================================
# Triage Engine - Question Flow
# =============================================================================

class TriageEngine:
    """
    Manages triage question trees.

    Each tree is a JSON file defining:
    - Questions with IDs, text, type (yesno, choice, numeric)
    - Branching logic (next_if_yes, next_if_no, next_if conditions)

    Supports multiple triage systems:
    - English (default): Standard 3-level (red/amber/green)
    - French (FRENCH): 6-level SFMU scale (1, 2, 3A, 3B, 4, 5)
    """

    def __init__(self, config_dir: str = "../config"):
        self.config_dir = config_dir
        self.trees: Dict[str, Dict] = {}  # All trees (both systems)
        self.trees_en: Dict[str, Dict] = {}  # English trees
        self.trees_fr: Dict[str, Dict] = {}  # French trees
        self._load_trees()

    def _load_trees(self):
        """Load all triage trees from config/triage_trees/."""
        trees_dir = os.path.join(self.config_dir, "triage_trees")

        if not os.path.exists(trees_dir):
            print(f"Warning: Triage trees directory not found: {trees_dir}")
            self._create_default_trees(trees_dir)
            return

        for filename in os.listdir(trees_dir):
            if filename.endswith(".json"):
                filepath = os.path.join(trees_dir, filename)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        tree = json.load(f)
                        tree_id = tree.get("id", filename.replace(".json", ""))
                        tree_system = tree.get("system", "english")

                        self.trees[tree_id] = tree

                        # Categorize by system
                        if tree_system == "french" or tree_id.endswith("_fr"):
                            self.trees_fr[tree_id] = tree
                        else:
                            self.trees_en[tree_id] = tree
                except Exception as e:
                    print(f"Error loading triage tree {filename}: {e}")
    
    def _create_default_trees(self, trees_dir: str):
        """Create default triage trees if none exist."""
        os.makedirs(trees_dir, exist_ok=True)
        
        # These will be created by the config files we write later
        print(f"Creating default triage trees in {trees_dir}")
    
    def get_available_complaints(self, language: str = "en") -> List[Dict[str, str]]:
        """Get list of available chief complaints based on language/system."""
        complaints = []

        # Select trees based on language
        if language == "fr":
            # Use French trees if available, otherwise use English trees
            trees_to_use = self.trees_fr if self.trees_fr else self.trees_en
        else:
            trees_to_use = self.trees_en if self.trees_en else self.trees

        for tree_id, tree in trees_to_use.items():
            # Get localized name and description
            if language == "fr":
                name = tree.get("name", tree.get("name_en", tree_id))
                description = tree.get("description", tree.get("description_en", ""))
            else:
                name = tree.get("name_en", tree.get("name", tree_id))
                description = tree.get("description_en", tree.get("description", ""))

            complaints.append({
                "id": tree_id,
                "name": name,
                "description": description
            })
        return complaints

    def get_tree_for_language(self, tree_id: str, language: str = "en") -> Optional[Dict]:
        """Get the appropriate tree for a given complaint and language."""
        # First try language-specific tree
        if language == "fr":
            # Try French version first
            fr_tree_id = tree_id if tree_id.endswith("_fr") else f"{tree_id}_fr"
            if fr_tree_id in self.trees:
                return self.trees[fr_tree_id]
            # Fall back to base tree
            base_id = tree_id.replace("_fr", "")
            if base_id in self.trees:
                return self.trees[base_id]

        # For English or fallback
        if tree_id in self.trees:
            return self.trees[tree_id]
        # Try without _fr suffix
        base_id = tree_id.replace("_fr", "")
        return self.trees.get(base_id)
    
    def get_tree(self, tree_id: str) -> Optional[Dict]:
        """Get triage tree by ID."""
        return self.trees.get(tree_id)
    
    def get_first_question(self, tree: Dict) -> Dict:
        """Get the first question in a tree."""
        questions = tree.get("questions", [])
        if not questions:
            return self._default_question()
        return questions[0]
    
    def get_question_by_id(self, tree: Dict, question_id: str) -> Optional[Dict]:
        """Get a specific question by ID."""
        for q in tree.get("questions", []):
            if q.get("id") == question_id:
                return q
        return None
    
    def get_next_question(
        self, 
        tree: Dict, 
        current_id: str, 
        answer: Any,
        all_answers: Dict[str, Any]
    ) -> Optional[Dict]:
        """
        Determine the next question based on current answer.
        
        Returns None if triage is complete.
        """
        current_q = self.get_question_by_id(tree, current_id)
        if not current_q:
            return None
        
        # Determine next question ID based on answer type
        next_id = None
        
        q_type = current_q.get("type", "yesno")
        
        if q_type == "yesno":
            if answer in [True, "yes", "Yes", "YES", 1]:
                next_id = current_q.get("next_if_yes")
            else:
                next_id = current_q.get("next_if_no")
        
        elif q_type == "choice":
            # Check conditional next based on choice
            conditionals = current_q.get("next_conditions", {})
            if str(answer) in conditionals:
                next_id = conditionals[str(answer)]
            else:
                next_id = current_q.get("next")
        
        elif q_type == "numeric":
            # Check threshold conditions
            thresholds = current_q.get("thresholds", [])
            for threshold in thresholds:
                op = threshold.get("op", ">=")
                value = threshold.get("value", 0)
                
                if self._compare(answer, op, value):
                    next_id = threshold.get("next")
                    break
            
            if not next_id:
                next_id = current_q.get("next")
        
        else:
            # Default: just go to next
            next_id = current_q.get("next")
        
        # Check if we have a valid next question
        if next_id is None or next_id == "end":
            return None
        
        return self.get_question_by_id(tree, next_id)
    
    def count_questions(self, tree: Dict) -> int:
        """Count total questions in tree (for progress calculation)."""
        return len(tree.get("questions", []))
    
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
            else:
                return False
        except (ValueError, TypeError):
            return False
    
    def _default_question(self) -> Dict:
        """Return a default fallback question."""
        return {
            "id": "default_symptoms",
            "type": "text",
            "text": "Please describe your symptoms:",
            "next": "end"
        }


# =============================================================================
# Risk Engine - Deterministic Risk Banding
# =============================================================================

@dataclass
class RiskRule:
    """A single risk rule that can trigger a risk band."""
    id: str
    description: str
    band: str  # red, amber, green
    conditions: List[Dict[str, Any]]
    priority: int = 0
    level: str = ""  # FRENCH triage level (1, 2, 3A, 3B, 4, 5)
    description_en: str = ""  # English description for French rules


# Manchester Triage System (MTS) level configuration - English system
MANCHESTER_LEVELS = {
    "1": {"color": "#dc3545", "name": "Immediate", "description": "Life-threatening condition requiring immediate intervention", "max_wait_minutes": 0, "band": "red"},
    "2": {"color": "#ff6b35", "name": "Very Urgent", "description": "Severe condition requiring very rapid assessment", "max_wait_minutes": 10, "band": "orange"},
    "3": {"color": "#ffc107", "name": "Urgent", "description": "Serious condition but patient is stable", "max_wait_minutes": 60, "band": "yellow"},
    "4": {"color": "#28a745", "name": "Standard", "description": "Standard condition requiring routine care", "max_wait_minutes": 120, "band": "green"},
    "5": {"color": "#17a2b8", "name": "Non-urgent", "description": "Minor condition that can safely wait", "max_wait_minutes": 240, "band": "blue"}
}

# FRENCH triage level configuration (SFMU - Société Française de Médecine d'Urgence)
FRENCH_LEVELS = {
    "1": {"color": "#dc3545", "name": "Tri 1", "name_en": "Level 1", "description": "Détresse vitale majeure", "description_en": "Life-threatening emergency", "max_wait_minutes": 1, "band": "red"},
    "2": {"color": "#ff6b35", "name": "Tri 2", "name_en": "Level 2", "description": "Atteinte patente d'un organe", "description_en": "Severe organ involvement", "max_wait_minutes": 20, "band": "red"},
    "3A": {"color": "#ffc107", "name": "Tri 3A", "name_en": "Level 3A", "description": "Urgence avec comorbidité", "description_en": "Urgent with comorbidity", "max_wait_minutes": 60, "band": "amber"},
    "3B": {"color": "#ffda6b", "name": "Tri 3B", "name_en": "Level 3B", "description": "Urgence sans comorbidité", "description_en": "Urgent without comorbidity", "max_wait_minutes": 90, "band": "amber"},
    "4": {"color": "#28a745", "name": "Tri 4", "name_en": "Level 4", "description": "Atteinte fonctionnelle stable", "description_en": "Stable functional impairment", "max_wait_minutes": 120, "band": "green"},
    "5": {"color": "#20c997", "name": "Tri 5", "name_en": "Level 5", "description": "Pas d'atteinte évidente", "description_en": "No obvious impairment", "max_wait_minutes": 240, "band": "green"}
}


class RiskEngine:
    """
    Deterministic risk band calculator.

    The model CANNOT override these rules - they are the source of truth.
    Risk is computed purely from patient data and rule conditions.

    Supports two systems:
    - English: Manchester Triage System - 5-level (Red/Orange/Yellow/Green/Blue)
    - French (FRENCH): 6-level SFMU scale (1, 2, 3A, 3B, 4, 5)
    """

    def __init__(self, config_dir: str = "../config"):
        self.config_dir = config_dir
        self.rules_en: List[RiskRule] = []  # English rules (Manchester)
        self.rules_fr: List[RiskRule] = []  # French (FRENCH) rules
        self.rules: List[RiskRule] = []  # Default (English)
        self.manchester_levels = MANCHESTER_LEVELS
        self.french_levels = FRENCH_LEVELS
        self._load_rules()
    
    def _load_rules(self):
        """Load risk rules from config/risk_rules.json and risk_rules_french.json."""
        # Load English rules
        rules_path = os.path.join(self.config_dir, "risk_rules.json")
        if os.path.exists(rules_path):
            try:
                with open(rules_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for rule_data in data.get("rules", []):
                        rule = RiskRule(
                            id=rule_data["id"],
                            description=rule_data["description"],
                            band=rule_data["band"],
                            conditions=rule_data["conditions"],
                            priority=rule_data.get("priority", 0)
                        )
                        self.rules_en.append(rule)
                        self.rules.append(rule)  # Default to English
            except Exception as e:
                print(f"Error loading English risk rules: {e}")

        # Load French (FRENCH) rules
        french_rules_path = os.path.join(self.config_dir, "risk_rules_french.json")
        if os.path.exists(french_rules_path):
            try:
                with open(french_rules_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for rule_data in data.get("rules", []):
                        rule = RiskRule(
                            id=rule_data["id"],
                            description=rule_data["description"],
                            band=rule_data["band"],
                            conditions=rule_data["conditions"],
                            priority=rule_data.get("priority", 0),
                            level=rule_data.get("level", ""),
                            description_en=rule_data.get("description_en", rule_data["description"])
                        )
                        self.rules_fr.append(rule)
                print(f"  FRENCH triage rules loaded: {len(self.rules_fr)} rules")
            except Exception as e:
                print(f"Error loading French risk rules: {e}")

        if not self.rules_en:
            print("Warning: No English risk rules found")
            self._load_fallback_rules()

        if not self.rules_fr:
            print("Warning: No French risk rules found, using English rules as fallback")
            self.rules_fr = self.rules_en.copy()
    
    def _create_default_rules(self, rules_path: str):
        """Create default risk rules file."""
        os.makedirs(os.path.dirname(rules_path), exist_ok=True)
        # Will be created by config file
    
    def _load_fallback_rules(self):
        """Load minimal fallback rules if config fails."""
        self.rules = [
            RiskRule(
                id="age_infant",
                description="Patient is an infant (under 2 years)",
                band="red",
                conditions=[{"field": "demographics.age", "op": "<", "value": 2}],
                priority=100
            ),
            RiskRule(
                id="age_elderly",
                description="Patient is elderly (over 65 years)",
                band="amber",
                conditions=[{"field": "demographics.age", "op": ">", "value": 65}],
                priority=50
            ),
            RiskRule(
                id="default_green",
                description="No high-risk factors identified",
                band="green",
                conditions=[],
                priority=-100
            )
        ]
    
    def compute_risk(
        self,
        demographics: Dict[str, Any],
        answers: Dict[str, Any],
        language: str = "en"
    ) -> Dict[str, Any]:
        """
        Compute risk band from patient data.

        Args:
            demographics: Patient demographics (age, sex, pregnant)
            answers: Triage question answers
            language: "en" for English Manchester (5-level), "fr" for French FRENCH (6-level)

        Returns:
            For English (Manchester) system:
            {
                "band": "red" | "orange" | "yellow" | "green" | "blue",
                "level": "1" | "2" | "3" | "4" | "5",
                "level_info": {...},
                "triggered_rules": [{"id": ..., "description": ..., "level": ...}, ...]
            }

            For French (FRENCH) system:
            {
                "band": "red" | "amber" | "green",
                "level": "1" | "2" | "3A" | "3B" | "4" | "5",
                "level_info": {...},
                "triggered_rules": [{"id": ..., "description": ..., "level": ...}, ...]
            }
        """
        # Combine all data for rule evaluation
        data = {
            "demographics": demographics,
            "answers": answers
        }

        # Select rules based on language
        rules_to_use = self.rules_fr if language == "fr" else self.rules_en
        if not rules_to_use:
            rules_to_use = self.rules

        triggered_rules = []
        highest_level = "5"  # Default level (lowest priority) for both systems

        # Level priority mapping - higher number = more urgent
        if language == "fr":
            level_priority = {"5": 0, "4": 1, "3B": 2, "3A": 3, "2": 4, "1": 5}
        else:
            # Manchester system levels
            level_priority = {"5": 0, "4": 1, "3": 2, "2": 3, "1": 4}

        # Sort rules by priority (highest first)
        sorted_rules = sorted(rules_to_use, key=lambda r: r.priority, reverse=True)

        for rule in sorted_rules:
            if self._evaluate_rule(rule, data):
                rule_info = {
                    "id": rule.id,
                    "description": rule.description if language == "fr" else rule.description_en or rule.description,
                    "band": rule.band
                }

                if rule.level:
                    rule_info["level"] = rule.level

                triggered_rules.append(rule_info)

                # Update highest level
                if rule.level and level_priority.get(rule.level, 0) > level_priority.get(highest_level, 0):
                    highest_level = rule.level

        # Get level info based on system
        if language == "fr":
            level_info = self.french_levels.get(highest_level, {})
        else:
            level_info = self.manchester_levels.get(highest_level, {})

        result = {
            "level": highest_level,
            "level_info": level_info,
            "band": level_info.get("band", "green"),
            "color": level_info.get("color", "#28a745"),
            "max_wait_minutes": level_info.get("max_wait_minutes", 120),
            "triggered_rules": triggered_rules
        }

        return result
    
    def _evaluate_rule(self, rule: RiskRule, data: Dict[str, Any]) -> bool:
        """
        Evaluate if a rule's conditions are met.
        
        All conditions must be true for the rule to trigger.
        """
        if not rule.conditions:
            return True  # Rules with no conditions always match (e.g., default green)
        
        for condition in rule.conditions:
            if not self._evaluate_condition(condition, data):
                return False
        
        return True
    
    def _evaluate_condition(self, condition: Dict[str, Any], data: Dict[str, Any]) -> bool:
        """Evaluate a single condition."""
        field_path = condition.get("field", "")
        op = condition.get("op", "==")
        expected = condition.get("value")
        
        # Get actual value from data using dot notation
        actual = self._get_nested_value(data, field_path)
        
        if actual is None:
            return False
        
        # Perform comparison
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
                return actual in expected
            elif op == "contains":
                return expected in str(actual).lower()
            elif op == "exists":
                return actual is not None
            elif op == "is_true":
                return actual in [True, "yes", "Yes", "YES", 1, "true", "True"]
            elif op == "is_false":
                return actual in [False, "no", "No", "NO", 0, "false", "False"]
            else:
                return False
        except (ValueError, TypeError):
            return False
    
    def _get_nested_value(self, data: Dict, path: str) -> Any:
        """Get value from nested dict using dot notation (e.g., 'demographics.age')."""
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
    
    def get_rule_explanation(self, rule_id: str) -> Optional[str]:
        """Get explanation for a specific rule."""
        for rule in self.rules:
            if rule.id == rule_id:
                return rule.description
        return None
