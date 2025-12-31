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


# =============================================================================
# ENHANCED EXAMPLES - Optimized for empathy + clinical detail
# These examples demonstrate warmer, more patient-centered communication
# while maintaining clinical accuracy and protocol adherence
# =============================================================================

ENHANCED_EXAMPLES: List[Dict[str, Any]] = [
    {
        "id": "empathetic_cardiac",
        "context": {
            "demographics": {"age": 45, "sex": "female"},
            "chief_complaint": "chest pain",
            "symptoms": "Pain 7/10, radiating to jaw, sweating, feeling anxious",
            "risk_band": "amber",
            "protocol": "Chest Pain - Manchester Triage System"
        },
        "response": {
            "reasoning": (
                "Per MTS Chest Pain protocol, this 45-year-old female meets ORANGE criteria: "
                "severe pain (7/10) with radiation to jaw suggests possible acute coronary syndrome. "
                "Sweating (diaphoresis) is a red flag for cardiac ischemia. Differential diagnosis: "
                "1) Acute Coronary Syndrome - most urgent to rule out given presentation, "
                "2) Anxiety/panic attack - patient reports feeling anxious, "
                "3) Musculoskeletal - less likely given radiation pattern and diaphoresis. "
                "Priority: ORANGE (Very Urgent) - cardiac workup required."
            ),
            "answer": (
                "I can hear that you're feeling worried about your chest pain, and I want you to know "
                "we're taking this seriously. Based on what you've told me:\n\n"
                "• You have chest pain rated 7 out of 10\n"
                "• The pain is spreading to your jaw\n"
                "• You're experiencing sweating\n\n"
                "These symptoms need prompt evaluation. A doctor will see you shortly to run some tests "
                "and make sure your heart is okay. I know it's scary, but try to stay calm and rest "
                "while you wait - you're in the right place, and we're here to help."
            ),
            "follow_up_questions": [
                "Have you ever had heart problems or a heart attack before?",
                "Are you taking any medications, especially blood thinners or aspirin?",
                "When exactly did the pain start, and were you doing anything at the time?"
            ]
        }
    },
    {
        "id": "detailed_respiratory",
        "context": {
            "demographics": {"age": 58, "sex": "male"},
            "chief_complaint": "shortness of breath",
            "symptoms": "Difficulty breathing for 2 hours, worse with activity, mild wheeze, history of COPD",
            "risk_band": "amber",
            "protocol": "Shortness of Breath - Manchester Triage System"
        },
        "response": {
            "reasoning": (
                "Per MTS Shortness of Breath protocol, 58-year-old male with COPD presenting with "
                "acute dyspnea (2 hours duration). Wheeze present indicates bronchospasm component. "
                "Worsening with activity suggests significant respiratory compromise. "
                "Differential diagnosis: "
                "1) COPD exacerbation - most likely given history and wheeze, "
                "2) Pneumonia - possible superinfection, need to assess for fever/productive cough, "
                "3) Cardiac decompensation - COPD patients at risk for cor pulmonale, "
                "4) Pulmonary embolism - always consider with acute dyspnea. "
                "Priority: ORANGE (Very Urgent) - requires SpO2 monitoring and bronchodilator assessment."
            ),
            "answer": (
                "I understand breathing difficulties can feel really frightening, especially with your "
                "lung condition. Let me tell you what we know about your situation:\n\n"
                "• You've been having trouble breathing for about 2 hours\n"
                "• It's getting harder when you move around\n"
                "• There's a bit of wheezing\n\n"
                "Given your COPD history, your lungs might need some extra help right now. We're going to "
                "check your oxygen levels and a doctor will listen to your chest soon. For now, try to "
                "sit up comfortably and take slow, easy breaths. We'll get you feeling better."
            ),
            "follow_up_questions": [
                "Have you used your rescue inhaler today, and did it help at all?",
                "Have you noticed any fever, or is your cough bringing up anything different than usual?",
                "Is this breathing difficulty similar to past flare-ups, or does it feel different?"
            ]
        }
    },
    {
        "id": "pediatric_fever",
        "context": {
            "demographics": {"age": 3, "sex": "female"},
            "chief_complaint": "fever",
            "symptoms": "Temperature 39.2C for 1 day, irritable, drinking less than usual, no rash",
            "risk_band": "amber",
            "protocol": "Unwell Child - Manchester Triage System"
        },
        "response": {
            "reasoning": (
                "Per MTS Unwell Child protocol, 3-year-old female with high fever (39.2°C) for 1 day. "
                "Concerning features: reduced fluid intake and irritability - both amber flags in "
                "pediatric triage. Absence of rash is reassuring for meningococcal disease but doesn't "
                "exclude it entirely. Differential diagnosis: "
                "1) Viral illness - most common cause in this age group, "
                "2) Otitis media or tonsillitis - common bacterial causes to examine for, "
                "3) UTI - consider especially in young girls, "
                "4) Early serious bacterial infection - cannot exclude without examination. "
                "Priority: ORANGE (Very Urgent) due to age, reduced intake, and irritability."
            ),
            "answer": (
                "I know it's worrying when your little one isn't feeling well. Let me reassure you about "
                "what we're seeing:\n\n"
                "• She has a high temperature (39.2°C)\n"
                "• She's been a bit grumpy - which is normal when children feel unwell\n"
                "• She's drinking less than usual - this is something we want to watch\n\n"
                "High fevers in young children are common, often from viruses that pass on their own. "
                "Because she's not drinking as much as normal, a doctor will want to check her over soon "
                "to make sure she stays well hydrated. Try offering small sips of water or her favorite "
                "drink while you wait. You're doing the right thing by bringing her in."
            ),
            "follow_up_questions": [
                "Has she been tugging at her ears or seem bothered when swallowing?",
                "When was her last wet diaper, and was it a normal amount?",
                "Has she been around anyone who's been sick recently, like at daycare?"
            ]
        }
    },
    {
        "id": "abdominal_red_flags",
        "context": {
            "demographics": {"age": 72, "sex": "male"},
            "chief_complaint": "abdominal pain",
            "symptoms": "Severe pain 9/10, rigid abdomen, no bowel movement for 3 days, vomiting",
            "risk_band": "red",
            "protocol": "Abdominal Pain - Manchester Triage System"
        },
        "response": {
            "reasoning": (
                "Per MTS Abdominal Pain protocol, 72-year-old male with RED FLAG presentation: "
                "severe pain (9/10) with rigid abdomen - classic signs of peritonitis. Additional "
                "concerning features: 3-day constipation with vomiting suggests possible bowel "
                "obstruction. At age 72, must also consider: "
                "1) Perforated viscus (peptic ulcer, diverticular) - rigidity strongly suggests, "
                "2) Complete bowel obstruction - vomiting + no BM pattern, "
                "3) Mesenteric ischemia - must consider in elderly with acute abdomen, "
                "4) AAA rupture - less likely without hypotension but must exclude. "
                "Priority: RED (Immediate) - surgical emergency until proven otherwise."
            ),
            "answer": (
                "I can see you're in a lot of pain, and I want you to know we're treating this as "
                "very urgent. Based on what you've told me:\n\n"
                "• You have severe belly pain rated 9 out of 10\n"
                "• Your abdomen feels very tender and rigid\n"
                "• You haven't had a bowel movement in 3 days\n"
                "• You've been vomiting\n\n"
                "A doctor is going to see you immediately. We're going to make you as comfortable as "
                "possible while we figure out what's causing this pain. Please don't eat or drink "
                "anything for now, as you may need tests or treatment very quickly. You're our priority."
            ),
            "follow_up_questions": [
                "When did the pain start, and has it been getting steadily worse?",
                "Have you had any surgeries on your belly before?",
                "Have you noticed any blood when you've vomited or in the toilet?"
            ]
        }
    },
    {
        "id": "mental_health_sensitive",
        "context": {
            "demographics": {"age": 28, "sex": "female"},
            "chief_complaint": "feeling overwhelmed",
            "symptoms": "Anxiety, not sleeping for 3 days, racing thoughts, feeling like can't cope",
            "risk_band": "amber",
            "protocol": "Mental Health - Manchester Triage System"
        },
        "response": {
            "reasoning": (
                "Per MTS Mental Health protocol, 28-year-old female presenting with acute anxiety "
                "and 3 days of insomnia - significant sleep deprivation. Racing thoughts may indicate "
                "anxiety disorder exacerbation or requires screening for hypomanic features. "
                "Important to assess: "
                "1) Suicidal ideation - must screen directly (not indicated in chief complaint), "
                "2) Precipitating stressors - what triggered this episode, "
                "3) Substance use - can exacerbate anxiety and insomnia, "
                "4) Previous mental health history - baseline and prior episodes. "
                "Priority: ORANGE (Very Urgent) - mental health crisis requiring prompt assessment. "
                "Safety assessment is the first priority."
            ),
            "answer": (
                "Thank you for coming in and telling us how you're feeling - that takes courage. "
                "It sounds like things have been really difficult lately:\n\n"
                "• You've been feeling anxious and overwhelmed\n"
                "• You haven't been able to sleep for three days\n"
                "• Your mind has been racing\n\n"
                "Three days without proper sleep is exhausting, and it's no wonder you're feeling like "
                "you can't cope. That's your mind and body telling you they need help - and that's "
                "exactly what we're here for. Someone from our team who specializes in this will talk "
                "with you soon. You don't have to figure this out alone. Is there anything that would "
                "help you feel more comfortable while you wait?"
            ),
            "follow_up_questions": [
                "Sometimes when we feel this overwhelmed, we might have thoughts of hurting ourselves - have you had any thoughts like that?",
                "Has anything specific happened recently that's made things harder for you?",
                "Have you experienced something like this before, and if so, what helped then?"
            ]
        }
    }
]


def get_enhanced_example(complaint: str) -> Dict[str, Any]:
    """
    Get an enhanced (empathetic) example based on chief complaint.

    Args:
        complaint: The chief complaint to match

    Returns:
        Best matching enhanced example or first example if no match
    """
    complaint_lower = complaint.lower()

    keyword_map = {
        "chest": "empathetic_cardiac",
        "heart": "empathetic_cardiac",
        "breath": "detailed_respiratory",
        "breathing": "detailed_respiratory",
        "wheeze": "detailed_respiratory",
        "fever": "pediatric_fever",
        "temperature": "pediatric_fever",
        "child": "pediatric_fever",
        "abdomen": "abdominal_red_flags",
        "stomach": "abdominal_red_flags",
        "belly": "abdominal_red_flags",
        "mental": "mental_health_sensitive",
        "anxiety": "mental_health_sensitive",
        "overwhelmed": "mental_health_sensitive",
        "depressed": "mental_health_sensitive",
        "sleep": "mental_health_sensitive"
    }

    for keyword, example_id in keyword_map.items():
        if keyword in complaint_lower:
            for ex in ENHANCED_EXAMPLES:
                if ex["id"] == example_id:
                    return ex

    return ENHANCED_EXAMPLES[0]


def get_empathetic_prompt(n_examples: int = 2, complaint_filter: str = None) -> str:
    """
    Generate an empathy-focused few-shot prompt.

    Args:
        n_examples: Number of examples to include (default 2)
        complaint_filter: Optional filter for specific complaint type

    Returns:
        Formatted few-shot examples emphasizing empathy
    """
    examples = ENHANCED_EXAMPLES

    if complaint_filter:
        # Get the most relevant example first
        relevant = get_enhanced_example(complaint_filter)
        others = [ex for ex in examples if ex["id"] != relevant["id"]]
        examples = [relevant] + others

    examples = examples[:n_examples]

    if not examples:
        return ""

    prompt_parts = [
        "EXAMPLES OF HIGH-QUALITY EMPATHETIC TRIAGE RESPONSES:",
        "(Notice how each response: acknowledges concerns, lists symptoms as bullets, explains simply, and reassures)\n"
    ]

    for i, example in enumerate(examples, 1):
        ctx = example["context"]
        resp = example["response"]

        prompt_parts.append(f"=== Example {i} ===")
        prompt_parts.append(f"Patient: {ctx['demographics']['age']}yo {ctx['demographics']['sex']}")
        prompt_parts.append(f"Complaint: {ctx['chief_complaint']}")
        prompt_parts.append(f"Protocol: {ctx.get('protocol', 'Standard Triage')}")
        prompt_parts.append(f"Risk: {ctx['risk_band'].upper()}")
        prompt_parts.append("")
        prompt_parts.append("REASONING (Clinical):")
        prompt_parts.append(resp['reasoning'])
        prompt_parts.append("")
        prompt_parts.append("ANSWER (Patient-friendly, warm):")
        prompt_parts.append(resp['answer'])
        prompt_parts.append("")
        prompt_parts.append("FOLLOW-UP QUESTIONS (Conversational):")
        for q in resp["follow_up_questions"]:
            prompt_parts.append(f"  - {q}")
        prompt_parts.append("")
        prompt_parts.append("")

    prompt_parts.append("Now respond to the current case with the same warmth and clinical precision.\n")

    return "\n".join(prompt_parts)


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
