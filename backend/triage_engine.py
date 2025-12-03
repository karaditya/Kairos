"""
Triage Engine - Decision Trees and Risk Assessment

Handles:
- Loading triage question trees from JSON
- Navigating through questions based on answers
- Deterministic risk band calculation (green/amber/red)

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
    """

    def __init__(self, config_dir: str = "../config"):
        self.config_dir = config_dir
        self.trees: Dict[str, Dict] = {}
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
                        self.trees[tree_id] = tree
                except Exception as e:
                    print(f"Error loading triage tree {filename}: {e}")
    
    def _create_default_trees(self, trees_dir: str):
        """Create default triage trees if none exist."""
        os.makedirs(trees_dir, exist_ok=True)
        
        # These will be created by the config files we write later
        print(f"Creating default triage trees in {trees_dir}")
    
    def get_available_complaints(self, language: str = "en") -> List[Dict[str, str]]:
        """Get list of available chief complaints."""
        complaints = []
        for tree_id, tree in self.trees.items():
            complaints.append({
                "id": tree_id,
                "name": tree.get("name", tree_id),
                "description": tree.get("description", "")
            })
        return complaints
    
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

class RiskEngine:
    """
    Deterministic risk band calculator.
    
    The model CANNOT override these rules - they are the source of truth.
    Risk is computed purely from patient data and rule conditions.
    """

    def __init__(self, config_dir: str = "../config"):
        self.config_dir = config_dir
        self.rules: List[RiskRule] = []
        self._load_rules()
    
    def _load_rules(self):
        """Load risk rules from config/risk_rules.json."""
        rules_path = os.path.join(self.config_dir, "risk_rules.json")
        
        if not os.path.exists(rules_path):
            print(f"Warning: Risk rules not found: {rules_path}")
            self._create_default_rules(rules_path)
        
        try:
            with open(rules_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for rule_data in data.get("rules", []):
                    self.rules.append(RiskRule(
                        id=rule_data["id"],
                        description=rule_data["description"],
                        band=rule_data["band"],
                        conditions=rule_data["conditions"],
                        priority=rule_data.get("priority", 0)
                    ))
        except Exception as e:
            print(f"Error loading risk rules: {e}")
            self._load_fallback_rules()
    
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
        answers: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compute risk band from patient data.
        
        Returns:
            {
                "band": "red" | "amber" | "green",
                "triggered_rules": [{"id": ..., "description": ...}, ...]
            }
        """
        # Combine all data for rule evaluation
        data = {
            "demographics": demographics,
            "answers": answers
        }
        
        triggered_rules = []
        highest_band = "green"
        band_priority = {"green": 0, "amber": 1, "red": 2}
        
        # Sort rules by priority (highest first)
        sorted_rules = sorted(self.rules, key=lambda r: r.priority, reverse=True)
        
        for rule in sorted_rules:
            if self._evaluate_rule(rule, data):
                triggered_rules.append({
                    "id": rule.id,
                    "description": rule.description,
                    "band": rule.band
                })
                
                # Update highest band
                if band_priority.get(rule.band, 0) > band_priority.get(highest_band, 0):
                    highest_band = rule.band
        
        return {
            "band": highest_band,
            "triggered_rules": triggered_rules
        }
    
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
