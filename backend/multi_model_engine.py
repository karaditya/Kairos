"""
Multi-Model Reasoning Engine - Professional LLM Engine for Medical Triage

HYBRID ARCHITECTURE: ADAPTIVE PROMPTING + JSON GRAMMAR
1. Input: Grounded & Sorted Patient Data.
2. Prompting: Role-Based (Senior MD vs Patient vs User) & Adaptive (XML/Brackets/JSON).
3. Output: JSON grammar for compatible models, regex parsing for others.
4. Safety: Factual Grounding Enforced.
"""

import os
import gc
import threading
import logging
import re
import random
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)
LLM_LOG_PATH = os.path.join(os.path.dirname(__file__), "backend.log")

from model_registry import (
    SUPPORTED_MODELS, ModelConfig, get_model_config, get_all_models, model_to_dict, DEFAULT_MODEL_ID, PromptStyle,
)

# =============================================================================
# JSON SCHEMAS (for models that support JSON grammar)
# =============================================================================

TRIAGE_CHAT_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "answer": {"type": "string"},
        "follow_up_questions": {"type": "array", "items": {"type": "string"}}
    },
    "required": ["reasoning", "answer", "follow_up_questions"]
}

# French Pivot: Schema for symptom translation to French medical terminology
FRENCH_PIVOT_SCHEMA = {
    "type": "object",
    "properties": {
        "french_terms": {"type": "string", "description": "Symptoms in French medical terminology"},
        "medical_category": {"type": "string", "description": "Medical category (e.g., cardiac, respiratory)"},
        "urgency_indicators": {"type": "array", "items": {"type": "string"}}
    },
    "required": ["french_terms", "medical_category"]
}

# RAG-grounded response schema
RAG_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "answer": {"type": "string"},
        "protocol_applied": {"type": "string"},
        "follow_up_questions": {"type": "array", "items": {"type": "string"}}
    },
    "required": ["reasoning", "answer", "protocol_applied", "follow_up_questions"]
}

MEDICAL_REPORT_SCHEMA = {
    "type": "object",
    "properties": {
        "clinical_summary": {"type": "string"},
        "differential_analysis": {"type": "string"},
        "risk_factors_identified": {"type": "array", "items": {"type": "string"}},
        "recommended_actions": {"type": "string"}
    },
    "required": ["clinical_summary", "differential_analysis", "risk_factors_identified", "recommended_actions"]
}

# =============================================================================
# LOGGING
# =============================================================================

def log_llm_output(raw_output: str, context: str = ""):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_msg = f"\n{'='*60}\n[{timestamp}] {context}\n{'-'*60}\n{raw_output}\n{'='*60}\n"
    try:
        with open(LLM_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(log_msg)
    except Exception:
        pass
    print(f"[{context}] Generated {len(raw_output)} chars.")

# =============================================================================
# ENGINE CLASS
# =============================================================================

try:
    from llama_cpp import Llama
    LLAMA_AVAILABLE = True
except ImportError:
    LLAMA_AVAILABLE = False

@dataclass
class LoadedModel:
    model_id: str
    config: ModelConfig
    instance: Any
    loaded_at: datetime
    inference_count: int = 0

class MultiModelEngine:
    def __init__(self, models_dir: str, default_model_id: str = DEFAULT_MODEL_ID, auto_load: bool = True):
        self.models_dir = models_dir
        self.default_model_id = default_model_id
        self._current_model: Optional[LoadedModel] = None
        self._lock = threading.RLock()
        self._total_inferences = 0
        self.n_threads = int(os.environ.get("N_THREADS", "6"))

        # DrBERT engine reference for RAG (set externally)
        self._drbert_engine: Optional[Any] = None

        # Auto-detect GPU layers if not explicitly set
        env_gpu_layers = os.environ.get("N_GPU_LAYERS")
        if env_gpu_layers is not None:
            self.n_gpu_layers = int(env_gpu_layers)
            self._auto_gpu = False
        else:
            self.n_gpu_layers = self._detect_optimal_gpu_layers()
            self._auto_gpu = True

        if auto_load:
            self._try_load_default_model()

    def set_drbert_engine(self, drbert_engine):
        """Set the DrBERT engine for RAG operations."""
        self._drbert_engine = drbert_engine

    @property
    def rag_available(self) -> bool:
        """Check if RAG is available (DrBERT loaded and vector store ready)."""
        return (
            self._drbert_engine is not None and
            self._drbert_engine.rag_ready
        )

    def _detect_optimal_gpu_layers(self) -> int:
        """Auto-detect optimal GPU layers based on available VRAM."""
        try:
            import torch
            if not torch.cuda.is_available():
                return 0

            torch.cuda.init()
            props = torch.cuda.get_device_properties(0)
            total_mb = props.total_memory // (1024 * 1024)
            reserved = torch.cuda.memory_reserved(0) // (1024 * 1024)
            free_mb = total_mb - reserved

            print(f"  GPU detected: {props.name} ({free_mb}MB free / {total_mb}MB total)")

            # Conservative layer allocation based on free VRAM
            if free_mb > 6000:
                return -1  # All layers on GPU
            elif free_mb > 4000:
                return 35
            elif free_mb > 2000:
                return 20
            elif free_mb > 1000:
                return 10
            elif free_mb > 500:
                return 5
            else:
                return 0

        except ImportError:
            return 0
        except Exception as e:
            logger.warning(f"GPU detection failed: {e}")
            return 0

    # =========================================================================
    # MODEL LOADING
    # =========================================================================

    def _try_load_default_model(self):
        if self._model_exists(self.default_model_id):
            self.load_model(self.default_model_id)
            return
        for model_id in SUPPORTED_MODELS:
            if self._model_exists(model_id):
                self.load_model(model_id)
                return

    def _model_exists(self, model_id: str) -> bool:
        config = get_model_config(model_id)
        return bool(config and os.path.exists(os.path.join(self.models_dir, config.filename)))

    def load_model(self, model_id: str) -> bool:
        if not LLAMA_AVAILABLE: return False
        config = get_model_config(model_id)
        if not config: return False
        path = os.path.join(self.models_dir, config.filename)
        with self._lock:
            if self._current_model and self._current_model.model_id == model_id: return True
            self.unload_model()
            try:
                # Determine GPU layers: use auto-detected if available, else config recommendation
                if self._auto_gpu and self.n_gpu_layers > 0:
                    # Auto mode: use detected layers, but cap to model's recommendation
                    gpu_layers = min(self.n_gpu_layers, config.recommended_gpu_layers) if config.recommended_gpu_layers > 0 else self.n_gpu_layers
                else:
                    # Manual mode or no GPU: use config or env setting
                    gpu_layers = config.recommended_gpu_layers if config.recommended_gpu_layers > 0 else self.n_gpu_layers

                # Handle -1 (all layers)
                if self.n_gpu_layers == -1:
                    gpu_layers = -1

                print(f"Loading {config.name}...")
                print(f"  GPU layers: {gpu_layers}, CPU threads: {self.n_threads}")

                instance = Llama(
                    model_path=path, n_ctx=4096, n_threads=self.n_threads,
                    n_gpu_layers=gpu_layers, verbose=False
                )
                self._current_model = LoadedModel(model_id, config, instance, datetime.now())
                self._current_gpu_layers = gpu_layers
                return True
            except Exception as e:
                print(f"Load failed: {e}")
                return False

    def unload_model(self):
        with self._lock:
            if self._current_model:
                try:
                    del self._current_model.instance
                    gc.collect()
                except: pass
                self._current_model = None

    def switch_model(self, model_id: str) -> bool: return self.load_model(model_id)

    @property
    def is_loaded(self) -> bool: return self._current_model is not None

    @property
    def _prompt_style(self) -> PromptStyle:
        if self._current_model:
            return self._current_model.config.prompt_style
        return PromptStyle.STRUCTURED

    @property
    def _supports_json_grammar(self) -> bool:
        """
        Determine if current model works well with JSON grammar.
        DeepSeek (THINK_TAGS) uses <think> internally - JSON grammar breaks its reasoning.
        Llama/SmolLM (SIMPLE) can struggle with strict JSON.
        STRUCTURED models (Gemma, Phi, etc.) handle JSON grammar well.
        """
        if not self._current_model:
            return False
        style = self._current_model.config.prompt_style
        # Only STRUCTURED models reliably support JSON grammar
        return style == PromptStyle.STRUCTURED

    def get_current_model(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            if not self._current_model: return None
            return {
                "model_id": self._current_model.model_id,
                "id": self._current_model.model_id,
                "name": self._current_model.config.name,
                "loaded_at": self._current_model.loaded_at.isoformat(),
                "inference_count": self._current_model.inference_count
            }

    # =========================================================================
    # DATA FLATTENER
    # =========================================================================

    def _flatten_patient_data(self, data: Dict[str, Any]) -> str:
        lines = []
        rules = data.get("triggered_rules", [])
        priority_map = {"red": 3, "amber": 2, "green": 1, "check": 0}

        sorted_rules = sorted(
            rules,
            key=lambda x: priority_map.get(str(x.get("band")).lower(), 0),
            reverse=True
        )

        highest_band = "LOW"
        if sorted_rules:
            highest_band = sorted_rules[0].get("band", "check").upper()

        lines.append(f"*** TRIAGE PRIORITY: {highest_band} ***")
        lines.append("-" * 30)

        demo = data.get("demographics", {})
        lines.append(f"PATIENT: {demo.get('age', '?')} year old {demo.get('sex', 'Unknown')}")
        if demo.get("pregnant"): lines.append("STATUS: Pregnant")

        lines.append("\nREPORTED SYMPTOMS:")
        answers = data.get("answers", {})
        if not answers: lines.append(" - No specific symptoms recorded.")
        for key, val in answers.items():
            readable_key = key.replace("_", " ").title()
            if isinstance(val, dict):
                display_val = val.get("label", val.get("value", str(val)))
            elif isinstance(val, bool):
                display_val = "Yes" if val else "No"
            else:
                display_val = val
            lines.append(f" - {readable_key}: {display_val}")

        lines.append("\nCLINICAL ALERTS:")
        if not sorted_rules:
            lines.append(" - No automated alerts.")
        else:
            for rule in sorted_rules:
                band = rule.get('band', 'check').upper()
                desc = rule.get('description', '')
                prefix = f"!! {band} !!" if band == "RED" else band
                lines.append(f" - [RISK: {prefix}] {desc}")

        return "\n".join(lines)

    # =========================================================================
    # ADAPTIVE PROMPTING (Cross-Model Styles + Role-Based Personas)
    # =========================================================================

    def _construct_adaptive_system_prompt(self, patient_text_card: str, use_json: bool = False) -> str:
        """
        Model-aware prompting with ROLE-BASED PERSONAS:
        - Reasoning: Senior physician's clinical thinking (technical, diagnostic)
        - Answer: Patient-friendly explanation (warm, simple, reassuring, non-technical)
        - Questions: Triage nurse's focused assessment questions (practical, systematic)

        CRITICAL: Personas are embedded via TONE descriptions, not "As a..." to prevent echoing.
        """
        grounding = (
            "You are a clinical triage assistant.\n\n"
            f"{patient_text_card}\n\n"
            "STRICT RULES:\n"
            "- ONLY use facts from the patient data above.\n"
            "- NEVER invent symptoms, history, or values.\n"
            "- Quote exact values (e.g., 'pain 8/10').\n"
            "- DO NOT repeat instructions or the patient card in your response.\n"
            "- DO NOT start with phrases like 'Based on...', 'Looking at...', 'I can see...'.\n"
            "- Jump straight into the content for each section.\n\n"
        )

        # Role-based tone guidance (embedded, not explicit "As a...")
        role_guidance = (
            "WRITING STYLE:\n"
            "- For reasoning: Use medical terminology, discuss differential diagnosis, "
            "risk factors, and clinical significance.\n"
            "- For answer: Use simple, reassuring language without medical jargon. "
            "Be empathetic and explain what happens next.\n"
            "- For questions: Ask specific, practical questions to assess urgency.\n\n"
        )

        # If using JSON grammar, request JSON format
        if use_json:
            return grounding + role_guidance + (
                "Respond with ONLY valid JSON, nothing else:\n"
                '{"reasoning": "...", "answer": "...", "follow_up_questions": ["...", "...", "..."]}'
            )

        # DeepSeek / Qwen: XML works naturally, they use <think> internally
        if self._prompt_style == PromptStyle.THINK_TAGS:
            return grounding + role_guidance + (
                "Respond using ONLY these XML tags with your actual content:\n"
                "<reasoning>YOUR CLINICAL ANALYSIS</reasoning>\n"
                "<answer>YOUR PATIENT RESPONSE</answer>\n"
                "<questions>\n- YOUR FIRST QUESTION?\n- YOUR SECOND QUESTION?\n- YOUR THIRD QUESTION?\n</questions>"
            )

        # Llama / SmolLM: Simple numbered format
        elif self._prompt_style == PromptStyle.SIMPLE:
            return grounding + role_guidance + (
                "Respond using ONLY these headers with your actual content:\n\n"
                "##REASONING##\n"
                "[Your clinical analysis]\n\n"
                "##ANSWER##\n"
                "[Your patient response]\n\n"
                "##QUESTIONS##\n"
                "- [Your first question]?\n"
                "- [Your second question]?\n"
                "- [Your third question]?"
            )

        # Mistral / Gemma / Phi: Structured markdown
        else:
            return grounding + role_guidance + (
                "Respond using ONLY these headers with your actual content:\n\n"
                "## Reasoning\n"
                "[Your clinical analysis]\n\n"
                "## Answer\n"
                "[Your patient response]\n\n"
                "## Questions\n"
                "- [Your first question]?\n"
                "- [Your second question]?\n"
                "- [Your third question]?"
            )

    # =========================================================================
    # OUTPUT PARSING (Regex-based for styled output)
    # =========================================================================

    def _strip_instruction_echoes(self, text: str) -> str:
        """
        Remove common instruction echo patterns from LLM output.
        Prevents the model from repeating prompts/instructions in the response.
        """
        if not text:
            return text

        result = text.strip()

        # Direct replacements for common echoed placeholders
        echoed_placeholders = [
            "Patient-friendly response here",
            "Clinical analysis here",
            "YOUR CLINICAL ANALYSIS",
            "YOUR PATIENT RESPONSE",
            "YOUR FIRST QUESTION",
            "YOUR SECOND QUESTION",
            "YOUR THIRD QUESTION",
            "[Your clinical analysis]",
            "[Your patient response]",
            "[Your first question]",
            "[Your second question]",
            "[Your third question]",
            "Question 1",
            "Question 2",
            "Question 3",
            "your clinical analysis here",
            "your patient-friendly response here",
        ]

        for placeholder in echoed_placeholders:
            result = result.replace(placeholder, "").strip()

        # Patterns that indicate echoed instructions (at start of text)
        echo_patterns = [
            r'^(?:Based on|Looking at|I can see|According to|From the)[\s\S]{0,50}(?:data|information|card|above)',
            r'^(?:Here is|Here\'s|Below is|The following)',
            r'^(?:As (?:a|an|the) (?:nurse|doctor|physician|clinician))',
            r'^(?:In my (?:clinical|medical|professional))',
            r'^(?:TONE FOR|OUTPUT FORMAT|STRICT RULES|FACTUAL GROUNDING|WRITING STYLE)',
            r'^\s*[-•]\s*(?:For )?(?:reasoning|answer|questions):\s*',
            r'^\[Your [^\]]+\]\s*',
        ]

        for pattern in echo_patterns:
            match = re.match(pattern, result, re.IGNORECASE)
            if match:
                remainder = result[match.end():]
                content_start = re.search(r'[A-Z][a-z]', remainder)
                if content_start:
                    result = remainder[content_start.start():]

        # Clean up any remaining bracket placeholders anywhere in text
        result = re.sub(r'\[Your [^\]]+\]\??', '', result)

        return result.strip()

    def _parse_adaptive_output(self, raw_str: str) -> Dict[str, Any]:
        """
        Universal Parser with 'Assimilation' fallback for creative headers.
        Includes echo-stripping to remove instruction leakage.
        """
        data = {"reasoning": "", "answer": "", "follow_up_questions": []}

        # 1. Remove <think> blocks (DeepSeek artifact)
        think_content = ""
        think_match = re.search(r'<think(?:ing)?>(.*?)</think(?:ing)?>', raw_str, flags=re.DOTALL)
        if think_match:
            think_content = think_match.group(1).strip()
        clean_str = re.sub(r'<think(?:ing)?>.*?</think(?:ing)?>', '', raw_str, flags=re.DOTALL)

        # 1b. STRIP ECHOED PATIENT CARD (Llama issue)
        response_start = re.search(
            r'(?:^|\n)\s*(?:##REASONING##|REASONING:|1\.?\s*REASONING|##\s*Reasoning|<reasoning>)',
            clean_str, re.IGNORECASE
        )
        if response_start:
            clean_str = clean_str[response_start.start():]

        # 2. Try XML tags (DeepSeek/THINK_TAGS)
        r_match = re.search(r'<reasoning>(.*?)(?:</reasoning>|<answer>|$)', clean_str, re.DOTALL | re.IGNORECASE)
        a_match = re.search(r'<answer>(.*?)(?:</answer>|<questions>|$)', clean_str, re.DOTALL | re.IGNORECASE)
        q_match = re.search(r'<questions>(.*?)(?:</questions>|$)', clean_str, re.DOTALL | re.IGNORECASE)

        # 3. Try ##HEADER## format (Llama/SIMPLE)
        if not a_match:
            r_match = r_match or re.search(r'##REASONING##\s*\n?(.*?)(?=##ANSWER##|$)', clean_str, re.DOTALL | re.IGNORECASE)
            a_match = re.search(r'##ANSWER##\s*\n?(.*?)(?=##QUESTIONS##|$)', clean_str, re.DOTALL | re.IGNORECASE)
            q_match = q_match or re.search(r'##QUESTIONS##\s*\n?(.*?)$', clean_str, re.DOTALL | re.IGNORECASE)

        # 4. Try numbered format fallback: "1. REASONING" or just "REASONING:"
        if not a_match:
            r_match = r_match or re.search(r'(?:^|\n)\s*(?:1\.?\s*)?REASONING[:\s]*\n(.*?)(?=\n\s*(?:2\.?\s*)?ANSWER|$)', clean_str, re.DOTALL | re.IGNORECASE)
            a_match = re.search(r'(?:^|\n)\s*(?:2\.?\s*)?ANSWER[:\s]*\n(.*?)(?=\n\s*(?:3\.?\s*)?QUESTIONS|$)', clean_str, re.DOTALL | re.IGNORECASE)
            q_match = q_match or re.search(r'(?:^|\n)\s*(?:3\.?\s*)?QUESTIONS[:\s]*\n(.*?)$', clean_str, re.DOTALL | re.IGNORECASE)

        # 5. Try markdown headers (Mistral/Gemma)
        if not a_match:
            r_match = r_match or re.search(r'#{1,3}\s*Reasoning\s*\n(.*?)(?=#{1,3}\s*Answer|$)', clean_str, re.DOTALL | re.IGNORECASE)
            a_match = re.search(r'#{1,3}\s*Answer\s*\n(.*?)(?=#{1,3}\s*Questions|$)', clean_str, re.DOTALL | re.IGNORECASE)
            q_match = q_match or re.search(r'#{1,3}\s*Questions\s*\n(.*?)$', clean_str, re.DOTALL | re.IGNORECASE)

        # 6. Extract values if matching succeeded (with echo stripping)
        if r_match: data["reasoning"] = self._strip_instruction_echoes(r_match.group(1).strip())
        if a_match: data["answer"] = self._strip_instruction_echoes(a_match.group(1).strip())
        if q_match: data["follow_up_questions"] = self._extract_questions(q_match.group(1))

        # 7. FILL GAPS USING THINKING (DeepSeek specific)
        if think_content and len(data["reasoning"]) < 50:
            data["reasoning"] = self._strip_instruction_echoes(think_content[:500])

        # 8. THE "ASSIMILATION" FIX (Creative Header Catch-all)
        if not data["answer"]:
            fallback = clean_str
            fallback = re.sub(r'<[^>]+>', '', fallback)
            fallback = re.sub(r'^\s*##[A-Z]+##\s*$', '', fallback, flags=re.MULTILINE)
            fallback = re.sub(r'^\s*\d+\.?\s*REASONING\s*$', '', fallback, flags=re.MULTILINE)
            fallback = re.sub(r'^\s*\d+\.?\s*ANSWER\s*$', '', fallback, flags=re.MULTILINE)
            fallback = re.sub(r'^\s*\d+\.?\s*QUESTIONS\s*$', '', fallback, flags=re.MULTILINE)
            data["answer"] = self._strip_instruction_echoes(fallback.strip())

            potential_questions = re.findall(r'(?:^|\n)\s*[-*]\s*(.+?)\??\s*$', data["answer"])
            if potential_questions and len(potential_questions) <= 5:
                qs_candidates = [q for q in potential_questions if "?" in q or "question" in q.lower()]
                if qs_candidates:
                    data["follow_up_questions"].extend(qs_candidates[:3])

        # 9. Failsafe
        if not data["answer"]:
            data["answer"] = "Analysis complete. Please review the patient data."

        return data

    def _extract_questions(self, raw_qs: str) -> List[str]:
        """Helper to clean bulleted questions and remove echoed placeholders."""
        clean_qs = []
        for line in raw_qs.split('\n'):
            cleaned = line.strip()

            # Skip empty lines
            if not cleaned:
                continue

            # Remove bracket placeholders
            cleaned = re.sub(r'\[Your [^\]]+\]\??', '', cleaned)
            cleaned = re.sub(r'\[.*?\]', '', cleaned)

            # Remove common prefixes: "- Question 1:", "1.", "- ", etc.
            cleaned = re.sub(
                r'^[\d\-\.\)\*•]*\s*(?:Question\s*\d*[:\.]?\s*)?',
                '',
                cleaned,
                flags=re.IGNORECASE
            )

            # Remove echoed placeholders
            echoed_q_placeholders = [
                "YOUR FIRST QUESTION",
                "YOUR SECOND QUESTION",
                "YOUR THIRD QUESTION",
                "First question",
                "Second question",
                "Third question",
            ]
            for placeholder in echoed_q_placeholders:
                cleaned = cleaned.replace(placeholder, "").strip()

            # Only keep if it's a real question (has substance)
            cleaned = cleaned.strip()
            if len(cleaned) > 10 and cleaned not in ["?", "-", ""]:
                # Ensure it ends with a question mark if it looks like a question
                if not cleaned.endswith("?") and any(w in cleaned.lower() for w in ["what", "how", "when", "where", "why", "do you", "are you", "have you", "is the", "does"]):
                    cleaned += "?"
                clean_qs.append(cleaned)

        return clean_qs

    def _clean_json_output(self, raw_output: str) -> str:
        """Robust JSON sanitizer for models using JSON grammar."""
        text = raw_output.strip()

        # Strip <think>...</think> blocks (DeepSeek-R1)
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

        # Remove Markdown code blocks
        if text.startswith("```"):
            text = re.sub(r"^```(json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)

        # Emergency Truncation (If loop detected)
        if len(text) > 2000 and not text.endswith("}"):
            last_valid = text.rfind('",')
            if last_valid > 100:
                text = text[:last_valid+2] + ' "answer": "Response truncated.", "follow_up_questions": []}'
                return text

        # Fix cut-off JSON
        if text.count('"') % 2 != 0:
            text += '"'

        if not text.endswith('}'):
            open_braces = text.count('{')
            close_braces = text.count('}')
            if open_braces > close_braces:
                text += '}' * (open_braces - close_braces)

        return text

    def _validate_response_schema(self, data: Dict[str, Any]) -> Dict[str, Any]:
        defaults = ["Is the pain worsening?", "Any fever?", "History of heart issues?", "Any nausea?", "Short of breath?"]
        questions = data.get("follow_up_questions", [])
        if not isinstance(questions, list): questions = []
        valid_qs = [q for q in questions if isinstance(q, str) and len(q) > 5]
        while len(valid_qs) < 3:
            choice = random.choice(defaults)
            if choice not in valid_qs: valid_qs.append(choice)
        data["follow_up_questions"] = valid_qs[:3]
        return data

    # =========================================================================
    # CORE GENERATION (Hybrid: JSON Grammar or Styled Prompting)
    # =========================================================================

    def _generate_structured(self, patient_context: Dict[str, Any], user_query: str) -> Dict[str, Any]:
        """
        HYBRID GENERATION:
        - STRUCTURED models (Gemma, Phi): Use JSON grammar for reliable output
        - THINK_TAGS models (DeepSeek): Use XML styled prompting + regex parsing
        - SIMPLE models (Llama, SmolLM): Use ##HEADER## prompting + regex parsing
        """
        with self._lock:
            if not self._current_model:
                return {"answer": "Engine Error: Model not loaded.", "reasoning": "", "follow_up_questions": []}

            card = self._flatten_patient_data(patient_context)
            use_json = self._supports_json_grammar
            sys_prompt = self._construct_adaptive_system_prompt(card, use_json=use_json)

            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_query}
            ]

            try:
                # Build generation kwargs
                gen_kwargs = {
                    "messages": messages,
                    "max_tokens": 1500,
                    "temperature": 0.1,
                    "top_p": 0.90,
                    "repeat_penalty": 1.1,
                    "stop": ["<|im_end|>", "<|im_start|>", "</s>", "<|eot_id|>"],
                }

                # Add JSON grammar for compatible models
                if use_json:
                    gen_kwargs["response_format"] = {
                        "type": "json_object",
                        "schema": TRIAGE_CHAT_SCHEMA
                    }

                output = self._current_model.instance.create_chat_completion(**gen_kwargs)
                raw = output['choices'][0]['message']['content']
                log_llm_output(raw, context=f"OUTPUT ({'JSON Grammar' if use_json else 'Styled'})")

                # Parse based on mode
                if use_json:
                    # JSON grammar path
                    clean_json = self._clean_json_output(raw)
                    try:
                        parsed_data = json.loads(clean_json)
                    except json.JSONDecodeError:
                        # Fallback to regex parsing if JSON fails
                        parsed_data = self._parse_adaptive_output(raw)
                else:
                    # Styled prompting path (DeepSeek, Llama, SmolLM)
                    parsed_data = self._parse_adaptive_output(raw)

                self._total_inferences += 1
                return self._validate_response_schema(parsed_data)

            except Exception as e:
                logger.error(f"Inference error: {e}")
                return {"answer": "Error during generation.", "reasoning": str(e), "follow_up_questions": []}

    # =========================================================================
    # RAW TEXT GENERATION (PDF)
    # =========================================================================

    def _generate_raw_text(self, system_prompt: str, user_prompt: str, force_prefix: str = "") -> str:
        with self._lock:
            if not self._current_model:
                return "Error: Model not loaded."

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]

            if force_prefix:
                messages[1]["content"] += f"\n\nIMPORTANT: Start your response immediately with the header: '{force_prefix}'"

            try:
                output = self._current_model.instance.create_chat_completion(
                    messages=messages,
                    max_tokens=2500,
                    temperature=0.2,
                    top_p=0.95,
                    stop=["<|im_end|>", "<|im_start|>", "<|eot_id|>"]
                )

                raw_response = output['choices'][0]['message']['content']
                # DEEPSEEK FIX: Aggressive think tag removal
                clean_response = re.sub(r'<think(?:ing)?>.*?</think(?:ing)?>', '', raw_response, flags=re.DOTALL)

                return clean_response.strip()

            except Exception as e:
                logger.error(f"PDF Gen Error: {e}")
                return "Error generating clinical text. Please try again."

    def generate_pdf_summary(self, clinical_state: Dict[str, Any], model_id: Optional[str] = None) -> Dict[str, str]:
        """
        Generate PDF sections with ROLE-BASED PERSONAS:
        - Clinical Summary: Triage nurse's systematic observation (factual, procedural, organized)
        - Diagnosis: Senior physician's differential analysis (clinical, analytical, evidence-based)
        - Conclusion: Attending physician's recommendations (authoritative, actionable, clear)

        CRITICAL: Personas embedded via TONE, not explicit "As a..." to prevent echoing.
        """
        if model_id: self.switch_model(model_id)

        card = self._flatten_patient_data(clinical_state)

        system_prompt = (
            "You are a medical documentation specialist writing a Clinical Referral Letter.\n\n"
            "FACTUAL GROUNDING:\n"
            "- ONLY include facts explicitly stated in PATIENT DATA.\n"
            "- Use exact values (e.g., 'pain severity 8/10').\n"
            "- NEVER invent symptoms, history, or test results.\n\n"
            "TONE FOR EACH SECTION:\n"
            "- CLINICAL SUMMARY: Write systematic, factual observations. Document the presenting "
            "complaint, vital information, and reported symptoms in organized prose. "
            "Focus on what was observed and reported, not interpretation.\n"
            "- DIAGNOSIS: Write analytical clinical assessment. Discuss differential considerations, "
            "risk factors, and clinical significance. Use appropriate medical terminology. "
            "Provide evidence-based reasoning for the assessment.\n"
            "- CONCLUSION: Write clear, actionable recommendations. State the triage priority, "
            "immediate actions needed, and follow-up requirements. Be direct and authoritative.\n\n"
            "FORMATTING RULES:\n"
            "- Use EXACT headers: 'CLINICAL SUMMARY:', 'DIAGNOSIS:', 'CONCLUSION:'.\n"
            "- Do NOT use bold (**), italics, or markdown formatting.\n"
            "- Write in continuous paragraphs (prose).\n"
            "- DO NOT echo these instructions or start with 'Based on the data...'.\n"
            "- Jump straight into the clinical content."
        )

        user_prompt = (
            f"PATIENT DATA:\n{card}\n\n"
            "Write the Clinical Referral Letter now. Start with CLINICAL SUMMARY:"
        )

        raw_text = self._generate_raw_text(
            system_prompt,
            user_prompt,
            force_prefix=""
        )

        return self._parse_pdf_sections(raw_text)

    def _parse_pdf_sections(self, text: str) -> Dict[str, str]:
        """Robust parser for PDF sections with echo stripping."""
        sections = {"summary": "", "diagnosis": "", "conclusion": ""}

        p_summary = re.search(r'(?:1\.|[\#\*]+)?\s*CLINICAL SUMMARY[:\s]*', text, re.IGNORECASE)
        p_diagnosis = re.search(r'(?:2\.|[\#\*]+)?\s*(?:DIAGNOSIS|ASSESSMENT)[:\s]*', text, re.IGNORECASE)
        p_conclusion = re.search(r'(?:3\.|[\#\*]+)?\s*(?:CONCLUSION|PLAN|RECOMMENDATION)[:\s]*', text, re.IGNORECASE)

        idx_s = p_summary.end() if p_summary else 0
        idx_d = p_diagnosis.start() if p_diagnosis else len(text)
        idx_c = p_conclusion.start() if p_conclusion else len(text)

        def clean_chunk(s):
            # Remove markdown artifacts
            cleaned = s.replace("**", "").replace("##", "").strip()
            # Apply echo stripping
            return self._strip_instruction_echoes(cleaned)

        if p_summary: sections["summary"] = clean_chunk(text[idx_s:idx_d])
        else: sections["summary"] = clean_chunk(text[:idx_d])

        if p_diagnosis: sections["diagnosis"] = clean_chunk(text[p_diagnosis.end():idx_c])
        if p_conclusion: sections["conclusion"] = clean_chunk(text[p_conclusion.end():])

        return sections

    # =========================================================================
    # FRENCH PIVOT RAG PIPELINE
    # =========================================================================

    def _pivot_to_french(self, patient_context: Dict[str, Any], user_query: str) -> Dict[str, Any]:
        """
        Step 1 of French Pivot: Translate symptoms to French medical terminology.

        Uses the LLM to extract and translate symptoms from any language
        to precise French medical terms for DrBERT semantic search.

        Args:
            patient_context: Patient clinical state
            user_query: Original user question (any language)

        Returns:
            Dict with french_terms, medical_category, urgency_indicators
        """
        if not self._current_model:
            return {"french_terms": "", "medical_category": "general", "urgency_indicators": []}

        # Build a concise patient summary
        card = self._flatten_patient_data(patient_context)

        system_prompt = (
            "You are a medical terminology expert. Your task is to extract symptoms "
            "from the patient data and translate them to PRECISE French medical terminology.\n\n"
            "RULES:\n"
            "- Output ONLY the French medical terms, not translations of general text\n"
            "- Use standard French medical vocabulary (as used in SFMU/HAS protocols)\n"
            "- Be specific: 'Douleur thoracique constrictive' not just 'douleur'\n"
            "- Include relevant modifiers (acute, chronic, severity, location)\n\n"
            "Examples:\n"
            "- 'chest pain' → 'Douleur thoracique'\n"
            "- 'difficulty breathing' → 'Dyspnée'\n"
            "- 'high fever with chills' → 'Fièvre élevée avec frissons'\n"
            "- 'sudden severe headache' → 'Céphalée brutale intense'\n\n"
        )

        # Construct user prompt
        user_prompt = (
            f"PATIENT DATA:\n{card}\n\n"
            f"USER QUERY: {user_query}\n\n"
            "Extract the key symptoms and translate to French medical terms. "
            "Also identify the medical category (cardiac, respiratory, neurological, etc.).\n\n"
            "Respond with:\n"
            "FRENCH_TERMS: [French medical terminology for symptoms]\n"
            "CATEGORY: [medical category]\n"
            "URGENCY: [any urgency indicators, comma-separated]"
        )

        with self._lock:
            try:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]

                output = self._current_model.instance.create_chat_completion(
                    messages=messages,
                    max_tokens=300,
                    temperature=0.1,
                    stop=["<|im_end|>", "<|im_start|>", "</s>", "<|eot_id|>"]
                )

                raw = output['choices'][0]['message']['content']
                log_llm_output(raw, context="FRENCH_PIVOT")

                # Parse the response
                french_terms = ""
                category = "general"
                urgency = []

                for line in raw.split('\n'):
                    line = line.strip()
                    if line.upper().startswith("FRENCH_TERMS:"):
                        french_terms = line.split(":", 1)[1].strip()
                    elif line.upper().startswith("CATEGORY:"):
                        category = line.split(":", 1)[1].strip().lower()
                    elif line.upper().startswith("URGENCY:"):
                        urgency_str = line.split(":", 1)[1].strip()
                        urgency = [u.strip() for u in urgency_str.split(",") if u.strip()]

                return {
                    "french_terms": french_terms,
                    "medical_category": category,
                    "urgency_indicators": urgency
                }

            except Exception as e:
                logger.error(f"French pivot error: {e}")
                return {"french_terms": "", "medical_category": "general", "urgency_indicators": []}

    def _build_rag_system_prompt(self,
                                  patient_card: str,
                                  protocol_text: str,
                                  protocol_metadata: Dict[str, Any],
                                  user_language: str = "en") -> str:
        """
        Build system prompt with retrieved French protocol as ground truth.

        Args:
            patient_card: Flattened patient data
            protocol_text: Retrieved French protocol text
            protocol_metadata: Protocol metadata (title, source, etc.)
            user_language: User's language for response

        Returns:
            Complete system prompt with protocol injection
        """
        language_instruction = {
            "en": "Respond in English.",
            "fr": "Répondez en français.",
            "es": "Responda en español.",
            "ar": "أجب بالعربية."
        }.get(user_language, "Respond in English.")

        protocol_title = protocol_metadata.get("title", "Medical Protocol")
        protocol_source = protocol_metadata.get("source", "SFMU/HAS")

        return (
            "You are a medical triage assistant. You MUST use the FRENCH PROTOCOL below "
            "as your SOURCE OF TRUTH for medical decisions.\n\n"
            "═══════════════════════════════════════════════════════════════\n"
            f"📋 OFFICIAL PROTOCOL: {protocol_title}\n"
            f"📚 Source: {protocol_source}\n"
            "═══════════════════════════════════════════════════════════════\n"
            f"{protocol_text}\n"
            "═══════════════════════════════════════════════════════════════\n\n"
            f"PATIENT DATA:\n{patient_card}\n\n"
            "STRICT RULES:\n"
            "1. Base your reasoning ONLY on the protocol above.\n"
            "2. Quote specific protocol text when relevant.\n"
            "3. Apply the protocol's triage criteria to this patient.\n"
            "4. If the protocol doesn't cover this case, say so explicitly.\n"
            "5. DO NOT invent information not in the protocol or patient data.\n\n"
            "WRITING STYLE:\n"
            "- For reasoning: Use medical terminology, reference the protocol.\n"
            "- For answer: Use simple, reassuring language for the patient.\n"
            "- For questions: Ask specific questions to assess urgency.\n\n"
            f"LANGUAGE: {language_instruction}\n"
        )

    def query_with_rag(self,
                       patient_context: Dict[str, Any],
                       user_query: str,
                       user_language: str = "en",
                       n_protocols: int = 1) -> Dict[str, Any]:
        """
        Query with French Pivot RAG Pipeline.

        The full pipeline:
        1. PIVOT: LLM translates symptoms → French medical terms
        2. RETRIEVE: DrBERT searches ChromaDB with French terms
        3. SYNTHESIZE: LLM generates response grounded in French protocol

        Args:
            patient_context: Clinical state dictionary
            user_query: User's question (any language)
            user_language: User's preferred response language
            n_protocols: Number of protocols to retrieve

        Returns:
            Response with answer, reasoning, protocol_applied, citations
        """
        # Check if RAG is available
        if not self.rag_available:
            logger.warning("RAG not available. Falling back to standard generation.")
            return self._generate_structured_fallback(patient_context, user_query)

        # === STEP 1: PIVOT TO FRENCH ===
        pivot_result = self._pivot_to_french(patient_context, user_query)
        french_query = pivot_result.get("french_terms", "")
        medical_category = pivot_result.get("medical_category", "general")

        if not french_query:
            # Fallback: use original query if pivot fails
            french_query = user_query
            logger.warning("French pivot produced empty result, using original query")

        log_llm_output(
            f"French Terms: {french_query}\nCategory: {medical_category}",
            context="PIVOT_RESULT"
        )

        # === STEP 2: RETRIEVE FROM DRBERT ===
        protocols = self._drbert_engine.search_protocols(
            french_query=french_query,
            n_results=n_protocols,
            category=medical_category if medical_category != "general" else None
        )

        if not protocols:
            logger.warning("No protocols found. Falling back to standard generation.")
            return self._generate_structured_fallback(patient_context, user_query)

        best_protocol = protocols[0]
        protocol_text = best_protocol.get("text", "")
        protocol_metadata = best_protocol.get("metadata", {})

        log_llm_output(
            f"Protocol: {protocol_metadata.get('title', 'Unknown')}\n"
            f"Score: {best_protocol.get('relevance_score', 0):.3f}\n"
            f"Text Preview: {protocol_text[:200]}...",
            context="RETRIEVED_PROTOCOL"
        )

        # === STEP 3: SYNTHESIZE WITH GROUNDED PROMPT ===
        patient_card = self._flatten_patient_data(patient_context)
        rag_system_prompt = self._build_rag_system_prompt(
            patient_card=patient_card,
            protocol_text=protocol_text,
            protocol_metadata=protocol_metadata,
            user_language=user_language
        )

        # Use appropriate output format based on model
        use_json = self._supports_json_grammar

        if use_json:
            rag_system_prompt += (
                "\n\nRespond with ONLY valid JSON:\n"
                '{"reasoning": "...", "answer": "...", "protocol_applied": "...", "follow_up_questions": ["...", "..."]}'
            )
        else:
            # Styled output for non-JSON models
            rag_system_prompt += (
                "\n\nRespond using ONLY these headers:\n\n"
                "## Reasoning\n[Your clinical analysis based on the protocol]\n\n"
                "## Answer\n[Your response to the patient]\n\n"
                "## Protocol Applied\n[Which protocol section you used]\n\n"
                "## Questions\n- [Follow-up question 1]\n- [Follow-up question 2]"
            )

        with self._lock:
            try:
                messages = [
                    {"role": "system", "content": rag_system_prompt},
                    {"role": "user", "content": user_query}
                ]

                gen_kwargs = {
                    "messages": messages,
                    "max_tokens": 1500,
                    "temperature": 0.1,
                    "top_p": 0.90,
                    "repeat_penalty": 1.1,
                    "stop": ["<|im_end|>", "<|im_start|>", "</s>", "<|eot_id|>"]
                }

                if use_json:
                    gen_kwargs["response_format"] = {
                        "type": "json_object",
                        "schema": RAG_RESPONSE_SCHEMA
                    }

                output = self._current_model.instance.create_chat_completion(**gen_kwargs)
                raw = output['choices'][0]['message']['content']
                log_llm_output(raw, context="RAG_SYNTHESIS")

                # Extract reasoning from think tags before stripping
                _, think_reasoning = self._strip_think_tags(raw)

                # Parse response
                if use_json:
                    clean_json = self._clean_json_output(raw)
                    try:
                        parsed = json.loads(clean_json)
                        # Use think content as reasoning if not in JSON
                        if not parsed.get("reasoning") and think_reasoning:
                            parsed["reasoning"] = think_reasoning
                    except json.JSONDecodeError:
                        parsed = self._parse_rag_output(raw)
                else:
                    parsed = self._parse_rag_output(raw)

                self._total_inferences += 1

                # Build final response
                return {
                    "answer": parsed.get("answer", ""),
                    "reasoning": parsed.get("reasoning", ""),
                    "protocol_applied": parsed.get("protocol_applied", protocol_metadata.get("title", "")),
                    "follow_up_questions": parsed.get("follow_up_questions", [])[:3],
                    "french_query_used": french_query,
                    "protocol_relevance": best_protocol.get("relevance_score", 0),
                    "protocol_source": protocol_metadata.get("source", "SFMU/HAS"),
                    "rag_used": True,
                    "model_used": self._current_model.config.name if self._current_model else "Unknown"
                }

            except Exception as e:
                logger.error(f"RAG synthesis error: {e}")
                return self._generate_structured_fallback(patient_context, user_query)

    def _strip_think_tags(self, text: str) -> tuple:
        """Strip <think>...</think> tags from DeepSeek-R1 output, return (cleaned, reasoning)."""
        reasoning = ""
        think_match = re.search(r'<think>(.*?)</think>', text, flags=re.DOTALL)
        if think_match:
            reasoning = think_match.group(1).strip()
        cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
        return cleaned, reasoning

    def _parse_rag_output(self, raw_str: str) -> Dict[str, Any]:
        """Parse RAG output with protocol_applied field."""
        # Strip think tags first, extract reasoning
        cleaned, think_reasoning = self._strip_think_tags(raw_str)

        data = self._parse_adaptive_output(cleaned)

        # Use think block as reasoning if no explicit reasoning found
        if not data.get("reasoning") and think_reasoning:
            data["reasoning"] = think_reasoning

        # Extract protocol_applied if present
        protocol_match = re.search(
            r'#{1,3}\s*Protocol(?:\s+Applied)?\s*\n(.*?)(?=#{1,3}|$)',
            cleaned, re.DOTALL | re.IGNORECASE
        )
        if protocol_match:
            data["protocol_applied"] = protocol_match.group(1).strip()
        else:
            data["protocol_applied"] = ""

        return data

    def _generate_structured_fallback(self, patient_context: Dict[str, Any], user_query: str) -> Dict[str, Any]:
        """Fallback to standard generation when RAG is unavailable."""
        result = self._generate_structured(patient_context, user_query)
        result["rag_used"] = False
        result["protocol_applied"] = ""
        result["french_query_used"] = ""
        result["protocol_relevance"] = 0
        result["protocol_source"] = ""
        return result

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def _extract_key_flags(self, clinical_state: Dict[str, Any]) -> List[str]:
        flags = []
        for rule in clinical_state.get("triggered_rules", []):
            if rule.get("band") in ["red", "amber"]:
                flags.append(f"{rule.get('description')} ({rule.get('band').upper()})")
        return flags

    def _extract_citations(self, text: str, data: Dict[str, Any]) -> List[str]:
        citations = []
        t_low = text.lower()
        for k, v in data.get("answers", {}).items():
            if str(v).lower() in t_low: citations.append(f"{k}: {v}")
        for rule in data.get("triggered_rules", []):
            if rule.get("description", "").lower() in t_low: citations.append(f"Rule: {rule.get('description')}")
        return citations[:5]

    def answer_staff_question(self,
                               question: str,
                               clinical_state: Dict[str, Any],
                               model_id: Optional[str] = None,
                               use_rag: bool = True,
                               user_language: str = "en") -> Dict[str, Any]:
        """
        Answer a staff question about a case.

        Args:
            question: The question to answer
            clinical_state: Patient clinical data
            model_id: Optional model to switch to
            use_rag: Whether to use RAG if available (default: True)
            user_language: Language for response

        Returns:
            Response dictionary with answer, reasoning, citations, etc.
        """
        if model_id:
            self.switch_model(model_id)

        # Use RAG if available and requested
        if use_rag and self.rag_available:
            response = self.query_with_rag(
                patient_context=clinical_state,
                user_query=question,
                user_language=user_language
            )
            citations = self._extract_citations(response.get("answer", ""), clinical_state)

            return {
                "answer": response.get("answer", ""),
                "reasoning": response.get("reasoning", ""),
                "suggested_questions": response.get("follow_up_questions", []),
                "has_reasoning": True,
                "cited_data": citations,
                "model_used": response.get("model_used", "Unknown"),
                "rag_used": response.get("rag_used", False),
                "protocol_applied": response.get("protocol_applied", ""),
                "protocol_source": response.get("protocol_source", ""),
                "french_query_used": response.get("french_query_used", "")
            }
        else:
            # Standard generation without RAG
            response = self._generate_structured(clinical_state, question)
            citations = self._extract_citations(response.get("answer", ""), clinical_state)

            return {
                "answer": response.get("answer", ""),
                "reasoning": response.get("reasoning", ""),
                "suggested_questions": response.get("follow_up_questions", []),
                "has_reasoning": True,
                "cited_data": citations,
                "model_used": self._current_model.config.name if self._current_model else "System",
                "rag_used": False,
                "protocol_applied": "",
                "protocol_source": "",
                "french_query_used": ""
            }

    def get_available_models(self) -> List[Dict]:
        return [{**model_to_dict(c), "is_available": self._model_exists(c.id), "is_loaded": (self._current_model and self._current_model.model_id == c.id)} for c in get_all_models()]

    def get_engine_stats(self) -> Dict:
        config = self._current_model.config if self._current_model else None
        gpu_layers = getattr(self, '_current_gpu_layers', 0)

        stats = {
            "llama_available": LLAMA_AVAILABLE,
            "model_loaded": config.name if config else "None",
            "model_id": config.id if config else None,
            "current_model": config.name if config else None,
            "total_inferences": self._total_inferences,
            "uses_json_grammar": self._supports_json_grammar,
            "gpu_layers": gpu_layers,
            "auto_gpu": self._auto_gpu,
            "detected_gpu_layers": self.n_gpu_layers,
            "cpu_threads": self.n_threads,
            # RAG status
            "rag_available": self.rag_available,
            "drbert_connected": self._drbert_engine is not None
        }

        # Add DrBERT stats if connected
        if self._drbert_engine:
            drbert_stats = self._drbert_engine.get_engine_stats()
            stats["drbert_model"] = drbert_stats.get("model_loaded")
            stats["vector_store"] = drbert_stats.get("vector_store", {})

        return stats

# =============================================================================
# SINGLETON
# =============================================================================

_engine_instance: Optional[MultiModelEngine] = None

def get_engine(
    models_dir: str = "../models",
    default_model_id: str = DEFAULT_MODEL_ID,
    reinitialize: bool = False
) -> MultiModelEngine:
    global _engine_instance
    if _engine_instance is None or reinitialize:
        if _engine_instance and reinitialize:
            _engine_instance.unload_model()
        _engine_instance = MultiModelEngine(models_dir, default_model_id)
    return _engine_instance
