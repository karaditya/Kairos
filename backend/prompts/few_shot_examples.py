"""
Few-Shot Examples for Medical Triage LLM Prompting

These examples demonstrate the expected output format and quality for medical triage responses.
Used to improve model output consistency and clinical appropriateness.
"""

from typing import List, Dict, Any


# High-quality examples demonstrating expected output format
TRIAGE_EXAMPLES: List[Dict[str, Any]] = [
    {
        "context": {
            "demographics": {"age": 45, "sex": "male"},
            "chief_complaint": "chest pain",
            "symptoms": "Pain 8/10, radiating to left arm, diaphoretic, started 30 minutes ago",
            "risk_band": "red"
        },
        "response": {
            "reasoning": (
                "45-year-old male presenting with severe chest pain (8/10) with radiation to "
                "left arm and associated diaphoresis. Symptom onset 30 minutes ago suggests "
                "possible acute coronary syndrome. The combination of pain character, radiation "
                "pattern, and autonomic symptoms (sweating) raises high concern for STEMI or "
                "unstable angina. This presentation meets criteria for immediate cardiac evaluation. "
                "Risk stratification: HIGH PRIORITY."
            ),
            "answer": (
                "I understand you're experiencing significant chest pain that's spreading to your "
                "arm. This type of pain needs immediate attention from our medical team. A doctor "
                "will see you right away to check your heart. Please try to stay calm, avoid any "
                "physical movement, and let us know immediately if the pain gets worse or you feel "
                "short of breath."
            ),
            "follow_up_questions": [
                "Have you ever had a heart attack or been told you have heart disease?",
                "When exactly did this pain start, and what were you doing at the time?",
                "Are you currently taking any blood thinners or aspirin?"
            ]
        }
    },
    {
        "context": {
            "demographics": {"age": 22, "sex": "female"},
            "chief_complaint": "headache",
            "symptoms": "Headache for 3 days, no fever, taking ibuprofen with partial relief",
            "risk_band": "green"
        },
        "response": {
            "reasoning": (
                "22-year-old female with subacute headache (3 days duration) without red flag "
                "symptoms. No fever reported, suggesting non-infectious etiology. Partial response "
                "to OTC analgesia (ibuprofen) indicates likely tension-type or primary headache. "
                "Absence of sudden onset, neurological symptoms, or systemic signs reduces concern "
                "for serious pathology. Risk stratification: LOW PRIORITY. Standard evaluation "
                "pathway appropriate."
            ),
            "answer": (
                "It sounds like you've been dealing with this headache for a few days now. The "
                "good news is that since ibuprofen is providing some relief and you don't have a "
                "fever, this is likely a common type of headache. We'll have you seen, but you're "
                "not in immediate danger. Please let us know if anything changes, especially if "
                "the pain suddenly gets much worse."
            ),
            "follow_up_questions": [
                "Have you noticed any changes in your vision or sensitivity to light?",
                "Is this headache different from headaches you've had before?",
                "Have you had any recent stressful events or changes in sleep patterns?"
            ]
        }
    },
    {
        "context": {
            "demographics": {"age": 68, "sex": "male"},
            "chief_complaint": "difficulty breathing",
            "symptoms": "Shortness of breath at rest, worse when lying down, swollen ankles, history of heart failure",
            "risk_band": "amber"
        },
        "response": {
            "reasoning": (
                "68-year-old male with known heart failure presenting with dyspnea at rest and "
                "orthopnea (worse when lying down). Associated peripheral edema (swollen ankles) "
                "suggests fluid overload consistent with acute decompensated heart failure (ADHF). "
                "The symptom constellation of orthopnea, paroxysmal nocturnal dyspnea pattern, and "
                "edema indicates volume status requiring evaluation. Priority: MODERATE-HIGH. "
                "Requires prompt assessment but not immediately life-threatening if stable."
            ),
            "answer": (
                "I can see you're having trouble breathing, especially when lying down, and your "
                "ankles are swollen. Given your heart condition, this may mean your body is "
                "holding onto extra fluid. A doctor will need to examine you soon and may adjust "
                "your medications. For now, try to stay sitting up if that helps your breathing, "
                "and let us know right away if it gets harder to breathe."
            ),
            "follow_up_questions": [
                "Have you been taking all your heart medications as prescribed?",
                "How much has your weight changed in the past few days?",
                "Have you been eating more salty foods or drinking more fluids than usual?"
            ]
        }
    },
    {
        "context": {
            "demographics": {"age": 35, "sex": "female", "pregnant": True},
            "chief_complaint": "abdominal pain",
            "symptoms": "Right lower quadrant pain, 6/10, no vaginal bleeding, 8 weeks pregnant",
            "risk_band": "amber"
        },
        "response": {
            "reasoning": (
                "35-year-old pregnant female (8 weeks gestation) presenting with right lower "
                "quadrant abdominal pain rated 6/10. In early pregnancy, RLQ pain raises concern "
                "for ectopic pregnancy until proven otherwise, though appendicitis and ovarian "
                "pathology must also be considered. Absence of vaginal bleeding is reassuring but "
                "does not exclude ectopic pregnancy. Requires urgent evaluation including "
                "beta-hCG levels and ultrasound. Priority: MODERATE-HIGH due to pregnancy status."
            ),
            "answer": (
                "I understand you're experiencing pain in your lower right side while pregnant. "
                "Any abdominal pain during early pregnancy needs to be evaluated carefully to make "
                "sure you and the baby are safe. We'll get you seen by a doctor soon who may want "
                "to do some tests including an ultrasound. Please tell us right away if you notice "
                "any bleeding or if the pain suddenly gets much worse."
            ),
            "follow_up_questions": [
                "Have you had any spotting or vaginal bleeding at all?",
                "Have you had any previous pregnancies or ectopic pregnancies?",
                "When was your last ultrasound, and did it confirm the pregnancy location?"
            ]
        }
    },
    {
        "context": {
            "demographics": {"age": 5, "sex": "male"},
            "chief_complaint": "fever",
            "symptoms": "Fever 39.5C for 2 days, no rash, drinking fluids, active when fever controlled",
            "risk_band": "green"
        },
        "response": {
            "reasoning": (
                "5-year-old male with high fever (39.5C) for 2 days duration. Positive indicators: "
                "maintaining oral intake, active behavior when fever controlled with antipyretics. "
                "No rash or altered mental status reported. These features suggest a likely viral "
                "illness with appropriate physiological response. Absence of lethargy, poor "
                "feeding, or rash decreases concern for serious bacterial infection. "
                "Risk stratification: LOW PRIORITY with close monitoring."
            ),
            "answer": (
                "Your son has had a high temperature for a couple of days. The good news is that "
                "he's still drinking fluids and seems active when his fever comes down with "
                "medicine - these are really positive signs. Many childhood fevers are caused by "
                "common viruses that get better on their own. A doctor will examine him, but try "
                "not to worry too much for now."
            ),
            "follow_up_questions": [
                "Is he still playing or interested in activities when the fever medicine is working?",
                "Has he developed any spots or rashes anywhere on his body?",
                "Is he going to the bathroom normally, and are his diapers or pull-ups wet as usual?"
            ]
        }
    }
]


def get_few_shot_prompt(n_examples: int = 2, risk_filter: str = None) -> str:
    """
    Generate a few-shot prompt with example cases.

    Args:
        n_examples: Number of examples to include (default 2)
        risk_filter: Optional filter for specific risk band examples

    Returns:
        Formatted few-shot examples as a string
    """
    examples = TRIAGE_EXAMPLES

    # Filter by risk band if specified
    if risk_filter:
        examples = [ex for ex in examples if ex["context"]["risk_band"] == risk_filter]

    # Limit to n_examples
    examples = examples[:n_examples]

    if not examples:
        return ""

    prompt_parts = ["Here are examples of well-formatted triage responses:\n"]

    for i, example in enumerate(examples, 1):
        ctx = example["context"]
        resp = example["response"]

        prompt_parts.append(f"--- Example {i} ---")
        prompt_parts.append(f"Patient: {ctx['demographics']['age']}yo {ctx['demographics']['sex']}")
        prompt_parts.append(f"Complaint: {ctx['chief_complaint']}")
        prompt_parts.append(f"Details: {ctx['symptoms']}")
        prompt_parts.append(f"Risk: {ctx['risk_band'].upper()}")
        prompt_parts.append("")
        prompt_parts.append("Response:")
        prompt_parts.append(f"Reasoning: {resp['reasoning']}")
        prompt_parts.append(f"Answer: {resp['answer']}")
        prompt_parts.append("Questions:")
        for q in resp["follow_up_questions"]:
            prompt_parts.append(f"  - {q}")
        prompt_parts.append("")

    prompt_parts.append("Now respond to the current case in the same format.\n")

    return "\n".join(prompt_parts)


def get_example_for_complaint(complaint: str) -> Dict[str, Any]:
    """
    Get a relevant example based on chief complaint.

    Args:
        complaint: The chief complaint to match

    Returns:
        Best matching example or first example if no match
    """
    complaint_lower = complaint.lower()

    # Map keywords to example indices
    keyword_map = {
        "chest": 0,      # Chest pain example
        "head": 1,       # Headache example
        "breath": 2,     # Breathing example
        "abdomen": 3,    # Abdominal pain example
        "stomach": 3,
        "fever": 4,      # Fever example
        "temperature": 4
    }

    for keyword, idx in keyword_map.items():
        if keyword in complaint_lower:
            return TRIAGE_EXAMPLES[idx]

    # Return first example as default
    return TRIAGE_EXAMPLES[0]


# Chain-of-thought framework for clinical reasoning
COT_FRAMEWORK = """
CLINICAL REASONING FRAMEWORK:
When analyzing each patient case, follow this structured approach:

1. IDENTIFY: List the presenting symptoms, vital signs, and relevant history
2. ASSESS: Evaluate severity using clinical criteria (pain scale, duration, associated symptoms)
3. CONSIDER: Generate differential diagnoses in order of likelihood and severity
4. RISK STRATIFY: Apply triage criteria (red = immediate, amber = urgent, green = standard)
5. PLAN: Determine immediate actions and follow-up questions needed

For each step, explicitly state your reasoning before drawing conclusions.
Always cite specific patient data when supporting your assessment.
"""


def get_cot_system_prompt(include_examples: bool = False) -> str:
    """
    Get the chain-of-thought system prompt.

    Args:
        include_examples: Whether to include few-shot examples

    Returns:
        System prompt with CoT framework
    """
    prompt = COT_FRAMEWORK

    if include_examples:
        prompt += "\n" + get_few_shot_prompt(n_examples=2)

    return prompt
