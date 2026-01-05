"""
Test suite for Output Format Consistency improvements.

Tests:
1. Response validator functionality
2. Context-aware fallback questions
3. Grammar strategy configuration for all models
4. Few-shot examples loading
"""

import sys
import os

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from response_validator import TriageResponseValidator, validate_triage_response
from retry_handler import GenerationRetryHandler, FallbackManager
from model_registry import (
    SUPPORTED_MODELS, GrammarStrategy, PromptStyle, get_model_config
)
from prompts.few_shot_examples import (
    TRIAGE_EXAMPLES, get_few_shot_prompt, get_example_for_complaint, COT_FRAMEWORK
)

# Test schemas
TRIAGE_CHAT_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "answer": {"type": "string"},
        "follow_up_questions": {"type": "array", "items": {"type": "string"}}
    },
    "required": ["reasoning", "answer", "follow_up_questions"]
}


def test_validator_valid_response():
    """Test validator with a valid response."""
    print("\n[TEST] Validator with valid response...")

    validator = TriageResponseValidator()
    valid_response = {
        "reasoning": "Patient presents with chest pain radiating to left arm. This constellation of symptoms raises concern for acute coronary syndrome.",
        "answer": "I understand you're experiencing chest pain. A doctor will see you right away to check your heart.",
        "follow_up_questions": [
            "Have you had heart problems before?",
            "When did the pain start?",
            "Are you taking any medications?"
        ]
    }

    result = validator.validate(valid_response, TRIAGE_CHAT_SCHEMA)

    assert result.is_valid, f"Expected valid, got errors: {result.format_errors}, {result.quality_issues}"
    assert result.severity == "pass", f"Expected 'pass' severity, got: {result.severity}"
    print("  PASSED: Valid response accepted")


def test_validator_missing_fields():
    """Test validator catches missing required fields."""
    print("\n[TEST] Validator with missing fields...")

    validator = TriageResponseValidator()
    invalid_response = {
        "reasoning": "Some reasoning here.",
        # Missing "answer" and "follow_up_questions"
    }

    result = validator.validate(invalid_response, TRIAGE_CHAT_SCHEMA)

    assert not result.is_valid or result.repaired_response is not None, "Expected invalid or repaired"
    assert len(result.format_errors) > 0 or result.repaired_response is not None
    print(f"  PASSED: Detected issues: {result.format_errors[:2]}...")


def test_validator_repairs_questions():
    """Test validator repairs insufficient questions."""
    print("\n[TEST] Validator repairs insufficient questions...")

    validator = TriageResponseValidator()
    response_with_few_questions = {
        "reasoning": "Clinical assessment of patient with headache and no red flags.",
        "answer": "Your headache symptoms have been noted. A provider will see you.",
        "follow_up_questions": ["Any fever?"]  # Only 1 question
    }

    context = {"chief_complaint": "headache", "risk_band": "green"}
    result = validator.validate(response_with_few_questions, TRIAGE_CHAT_SCHEMA, context)

    # Should repair by adding context-appropriate questions
    repaired = result.repaired_response or response_with_few_questions
    assert len(repaired.get("follow_up_questions", [])) >= 3, "Should have at least 3 questions"
    print(f"  PASSED: Questions repaired to {len(repaired['follow_up_questions'])}")


def test_validator_detects_instruction_echo():
    """Test validator detects instruction echoing."""
    print("\n[TEST] Validator detects instruction echoes...")

    validator = TriageResponseValidator()
    echoing_response = {
        "reasoning": "[Your clinical analysis] Based on the patient data above...",
        "answer": "YOUR PATIENT RESPONSE here is the answer.",
        "follow_up_questions": [
            "Question 1?",
            "[Your first question]?",
            "What symptoms do you have?"
        ]
    }

    result = validator.validate(echoing_response, TRIAGE_CHAT_SCHEMA)

    assert len(result.quality_issues) > 0, "Should detect forbidden patterns"
    print(f"  PASSED: Detected {len(result.quality_issues)} quality issues")


def test_context_aware_fallbacks():
    """Test context-aware fallback question generation."""
    print("\n[TEST] Context-aware fallback questions...")

    validator = TriageResponseValidator()

    test_cases = [
        ({"chief_complaint": "chest pain", "risk_band": "red"}, "chest"),
        ({"chief_complaint": "severe headache", "risk_band": "amber"}, "head"),
        ({"chief_complaint": "abdominal pain", "risk_band": "green"}, "abdomen"),
        ({"chief_complaint": "shortness of breath", "risk_band": "amber"}, "breath"),
    ]

    for context, expected_category in test_cases:
        # Get fallback questions for this context
        questions = validator._get_context_appropriate_question(context, [])
        assert questions is not None, f"Should return question for {expected_category}"
        print(f"  - {expected_category}: '{questions[:50]}...'")

    print("  PASSED: All context categories return appropriate questions")


def test_grammar_strategy_all_models():
    """Test that all models have grammar_strategy configured."""
    print("\n[TEST] Grammar strategy for all models...")

    strategy_counts = {
        GrammarStrategy.STRICT_JSON: [],
        GrammarStrategy.THINK_THEN_JSON: [],
        GrammarStrategy.GUIDED_JSON: [],
        GrammarStrategy.FALLBACK: [],
    }

    for model_id, config in SUPPORTED_MODELS.items():
        strategy = config.grammar_strategy
        assert isinstance(strategy, GrammarStrategy), f"{model_id} has invalid strategy"
        strategy_counts[strategy].append(model_id)

    print("  Grammar Strategy Distribution:")
    for strategy, models in strategy_counts.items():
        if models:
            print(f"    {strategy.value}: {', '.join(models)}")

    # Verify expected assignments
    deepseek_models = [m for m in SUPPORTED_MODELS if "deepseek" in m]
    for model_id in deepseek_models:
        assert SUPPORTED_MODELS[model_id].grammar_strategy == GrammarStrategy.THINK_THEN_JSON, \
            f"{model_id} should use THINK_THEN_JSON"

    llama_models = [m for m in SUPPORTED_MODELS if "llama-3.2" in m]
    for model_id in llama_models:
        assert SUPPORTED_MODELS[model_id].grammar_strategy == GrammarStrategy.GUIDED_JSON, \
            f"{model_id} should use GUIDED_JSON"

    print("  PASSED: All models have correct grammar strategy")


def test_few_shot_examples_loaded():
    """Test that few-shot examples are properly defined."""
    print("\n[TEST] Few-shot examples...")

    assert len(TRIAGE_EXAMPLES) >= 5, f"Expected at least 5 examples, got {len(TRIAGE_EXAMPLES)}"

    for i, example in enumerate(TRIAGE_EXAMPLES):
        assert "context" in example, f"Example {i} missing context"
        assert "response" in example, f"Example {i} missing response"

        ctx = example["context"]
        resp = example["response"]

        assert "chief_complaint" in ctx, f"Example {i} missing chief_complaint"
        assert "reasoning" in resp, f"Example {i} missing reasoning"
        assert "answer" in resp, f"Example {i} missing answer"
        assert "follow_up_questions" in resp, f"Example {i} missing follow_up_questions"
        assert len(resp["follow_up_questions"]) >= 3, f"Example {i} has too few questions"

    print(f"  PASSED: {len(TRIAGE_EXAMPLES)} examples loaded and validated")


def test_few_shot_prompt_generation():
    """Test few-shot prompt generation."""
    print("\n[TEST] Few-shot prompt generation...")

    prompt = get_few_shot_prompt(n_examples=2)

    assert len(prompt) > 100, "Prompt should have substantial content"
    assert "Example 1" in prompt, "Should contain Example 1"
    assert "Example 2" in prompt, "Should contain Example 2"
    assert "Reasoning:" in prompt, "Should contain Reasoning section"
    assert "Answer:" in prompt, "Should contain Answer section"

    print(f"  PASSED: Generated prompt with {len(prompt)} characters")


def test_retry_handler_fallback():
    """Test retry handler graceful fallback."""
    print("\n[TEST] Retry handler fallback...")

    handler = GenerationRetryHandler(max_retries=3)

    # Test fallback response generation
    context = {"chief_complaint": "chest pain", "risk_band": "red"}
    fallback = handler._graceful_fallback(context, TRIAGE_CHAT_SCHEMA)

    assert "answer" in fallback or "patient_response" in fallback
    assert "follow_up_questions" in fallback
    assert len(fallback["follow_up_questions"]) >= 3
    assert fallback.get("confidence_score", 1.0) < 0.5  # Low confidence for fallback

    print(f"  PASSED: Fallback generates valid response with {len(fallback['follow_up_questions'])} questions")


def test_cot_framework():
    """Test chain-of-thought framework is defined."""
    print("\n[TEST] Chain-of-thought framework...")

    assert len(COT_FRAMEWORK) > 100, "CoT framework should be substantial"
    assert "IDENTIFY" in COT_FRAMEWORK
    assert "ASSESS" in COT_FRAMEWORK
    assert "RISK STRATIFY" in COT_FRAMEWORK

    print("  PASSED: CoT framework properly defined")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("OUTPUT CONSISTENCY IMPROVEMENT TESTS")
    print("=" * 60)

    tests = [
        test_validator_valid_response,
        test_validator_missing_fields,
        test_validator_repairs_questions,
        test_validator_detects_instruction_echo,
        test_context_aware_fallbacks,
        test_grammar_strategy_all_models,
        test_few_shot_examples_loaded,
        test_few_shot_prompt_generation,
        test_retry_handler_fallback,
        test_cot_framework,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"  FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
