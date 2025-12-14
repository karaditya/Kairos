"""
Multi-Model Reasoning Engine - Professional LLM Engine for Medical Triage

FINAL ARCHITECTURE: ADAPTIVE + ROLE-BASED
1. Input: Grounded & Sorted Patient Data.
2. Prompting: Role-Based (Senior MD vs Patient vs User) & Adaptive (XML/Brackets).
3. Parsing: Universal Regex with "Assimilation" Fallback.
4. Safety: Factual Grounding Enforced.
"""

import os
import gc
import threading
import logging
import re
import random
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)
LLM_LOG_PATH = os.path.join(os.path.dirname(__file__), "backend.log")

from model_registry import (
    SUPPORTED_MODELS, ModelConfig, get_model_config, get_all_models, model_to_dict, DEFAULT_MODEL_ID, PromptStyle,
)

# =============================================================================
# 1. LOGGING
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
# 2. ENGINE CLASS
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
        self.n_gpu_layers = int(os.environ.get("N_GPU_LAYERS", "-1"))
        self.n_threads = int(os.environ.get("N_THREADS", "6"))

        if auto_load:
            self._try_load_default_model()

    # --- Model Loading Logic ---
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
                print(f"Loading {config.name}...")
                instance = Llama(
                    model_path=path, n_ctx=4096, n_threads=self.n_threads,
                    n_gpu_layers=config.recommended_gpu_layers or self.n_gpu_layers, verbose=False
                )
                self._current_model = LoadedModel(model_id, config, instance, datetime.now())
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
    # 1. ADAPTIVE + ROLE-BASED PROMPTING
    # =========================================================================

    def _construct_adaptive_system_prompt(self, patient_text_card: str) -> str:
        """
        Model-aware prompting: Each model family gets its natural format.
        """
        grounding = (
            "You are a clinical triage assistant.\n\n"
            f"{patient_text_card}\n\n"
            "STRICT RULES:\n"
            "- ONLY use facts from the patient data above.\n"
            "- NEVER invent symptoms, history, or values.\n"
            "- Quote exact values (e.g., 'pain 8/10').\n"
            "- DO NOT just repeat the patient card. You must analyze it.\n\n"
        )

        # DeepSeek / Qwen: XML works naturally, they use <think> internally
        if self._prompt_style == PromptStyle.THINK_TAGS:
            return grounding + (
                "Respond using these XML tags:\n"
                "<reasoning>Brief clinical analysis (2-3 sentences)</reasoning>\n"
                "<answer>Your response to the user</answer>\n"
                "<questions>\n1. First question\n2. Second question\n3. Third question\n</questions>"
            )

        # Llama / SmolLM: Simple numbered format
        # Added explicit instruction not to repeat the card
        elif self._prompt_style == PromptStyle.SIMPLE:
            return grounding + (
                "Respond in exactly this format:\n\n"
                "1. REASONING\n"
                "Brief clinical analysis here.\n\n"
                "2. ANSWER\n"
                "Your response to the user here. If listing symptoms, use bullets.\n\n"
                "3. QUESTIONS\n"
                "- First follow-up question\n"
                "- Second follow-up question\n"
                "- Third follow-up question"
            )

        # Mistral / Gemma / Phi: Structured markdown
        else:
            return grounding + (
                "Respond in exactly this format:\n\n"
                "## Reasoning\n"
                "Brief clinical analysis here.\n\n"
                "## Answer\n"
                "Your response to the user here.\n\n"
                "## Questions\n"
                "1. First follow-up question\n"
                "2. Second follow-up question\n"
                "3. Third follow-up question"
            )
        
    def _parse_adaptive_output(self, raw_str: str) -> Dict[str, Any]:
        """
        Universal Parser with 'Assimilation' fallback for creative headers.
        """
        data = {"reasoning": "", "answer": "", "follow_up_questions": []}

        # 1. Remove <think> blocks (DeepSeek artifact)
        think_content = ""
        think_match = re.search(r'<think(?:ing)?>(.*?)</think(?:ing)?>', raw_str, flags=re.DOTALL)
        if think_match:
            think_content = think_match.group(1).strip()
        clean_str = re.sub(r'<think(?:ing)?>.*?</think(?:ing)?>', '', raw_str, flags=re.DOTALL)

        # 1b. STRIP ECHOED PATIENT CARD (Llama issue)
        # Find where actual response starts (REASONING: or ANSWER: or ## or 1. REASONING)
        response_start = re.search(
            r'(?:^|\n)\s*(?:REASONING:|ANSWER:|1\.?\s*REASONING|##\s*Reasoning|<reasoning>)',
            clean_str, re.IGNORECASE
        )
        if response_start:
            clean_str = clean_str[response_start.start():]

        # 2. Try XML tags (DeepSeek/THINK_TAGS)
        r_match = re.search(r'<reasoning>(.*?)(?:</reasoning>|<answer>|$)', clean_str, re.DOTALL | re.IGNORECASE)
        a_match = re.search(r'<answer>(.*?)(?:</answer>|<questions>|$)', clean_str, re.DOTALL | re.IGNORECASE)
        q_match = re.search(r'<questions>(.*?)(?:</questions>|$)', clean_str, re.DOTALL | re.IGNORECASE)

        # 3. Try numbered format (Llama/SIMPLE): "1. REASONING" or just "REASONING:"
        if not a_match:
            r_match = r_match or re.search(r'(?:^|\n)\s*(?:1\.?\s*)?REASONING[:\s]*\n(.*?)(?=\n\s*(?:2\.?\s*)?ANSWER|$)', clean_str, re.DOTALL | re.IGNORECASE)
            a_match = re.search(r'(?:^|\n)\s*(?:2\.?\s*)?ANSWER[:\s]*\n(.*?)(?=\n\s*(?:3\.?\s*)?QUESTIONS|$)', clean_str, re.DOTALL | re.IGNORECASE)
            q_match = q_match or re.search(r'(?:^|\n)\s*(?:3\.?\s*)?QUESTIONS[:\s]*\n(.*?)$', clean_str, re.DOTALL | re.IGNORECASE)

        # 4. Try markdown headers (Mistral/Gemma)
        if not a_match:
            r_match = r_match or re.search(r'#{1,3}\s*Reasoning\s*\n(.*?)(?=#{1,3}\s*Answer|$)', clean_str, re.DOTALL | re.IGNORECASE)
            a_match = re.search(r'#{1,3}\s*Answer\s*\n(.*?)(?=#{1,3}\s*Questions|$)', clean_str, re.DOTALL | re.IGNORECASE)
            q_match = q_match or re.search(r'#{1,3}\s*Questions\s*\n(.*?)$', clean_str, re.DOTALL | re.IGNORECASE)

        # 5. Extract values if strict matching succeeded
        if r_match: data["reasoning"] = r_match.group(1).strip()
        if a_match: data["answer"] = a_match.group(1).strip()
        if q_match: data["follow_up_questions"] = self._extract_questions(q_match.group(1))

        # 6. FILL GAPS USING THINKING (DeepSeek specific)
        if think_content and len(data["reasoning"]) < 50:
            data["reasoning"] = think_content[:500] 

        # 7. THE LLAMA "ASSIMILATION" FIX (Creative Header Catch-all)
        # If no specific "Answer" section was found, or it's empty...
        if not data["answer"]:
            # Treat the whole cleaned string as the answer, but remove known artifacts
            fallback = clean_str
            
            # Remove XML tags if they exist but were empty/broken
            fallback = re.sub(r'<[^>]+>', '', fallback)
            
            # Remove "System" style headers that the model might echo
            # e.g. "1. REASONING" (but empty)
            fallback = re.sub(r'^\s*\d+\.?\s*REASONING\s*$', '', fallback, flags=re.MULTILINE)
            fallback = re.sub(r'^\s*\d+\.?\s*ANSWER\s*$', '', fallback, flags=re.MULTILINE)
            fallback = re.sub(r'^\s*\d+\.?\s*QUESTIONS\s*$', '', fallback, flags=re.MULTILINE)

            data["answer"] = fallback.strip()
            
            # If we assimilated everything into answer, try to find questions inside it
            # Look for lines starting with "- " or "1. " at the end of the text
            potential_questions = re.findall(r'(?:^|\n)\s*[-*]\s*(.+?)\??\s*$', data["answer"])
            if potential_questions and len(potential_questions) <= 5:
                # If these look like questions, extract them
                qs_candidates = [q for q in potential_questions if "?" in q or "question" in q.lower()]
                if qs_candidates:
                    data["follow_up_questions"].extend(qs_candidates[:3])

        # 8. Failsafe
        if not data["answer"]:
            data["answer"] = "Analysis complete. Please review the patient data."

        return data

    def _extract_questions(self, raw_qs: str) -> List[str]:
        """Helper to clean bulleted questions."""
        clean_qs = []
        for line in raw_qs.split('\n'):
            cleaned = re.sub(r'\[.*?\]', '', line) 
            cleaned = re.sub(r'^[\d\-\.\)\*]*(?:Question\s*\d*[:\.]?)?\s*', '', cleaned.strip(), flags=re.IGNORECASE)
            if len(cleaned) > 5:
                clean_qs.append(cleaned)
        return clean_qs

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

    def _generate_structured(self, patient_context: Dict[str, Any], user_query: str) -> Dict[str, Any]:
        with self._lock:
            if not self._current_model:
                return {"answer": "Engine Error: Model not loaded.", "reasoning": "", "follow_up_questions": []}

            card = self._flatten_patient_data(patient_context)
            sys_prompt = self._construct_adaptive_system_prompt(card)
            
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_query}
            ]

            try:
                output = self._current_model.instance.create_chat_completion(
                    messages=messages,
                    max_tokens=1500,
                    temperature=0.1,
                    top_p=0.90,
                    repeat_penalty=1.1,
                    stop=["<|im_end|>", "<|im_start|>", "</s>", "<|eot_id|>"], 
                )
                
                raw = output['choices'][0]['message']['content']
                log_llm_output(raw, context="OUTPUT (Chat)")
                
                parsed_data = self._parse_adaptive_output(raw)
                return self._validate_response_schema(parsed_data)
                
            except Exception as e:
                logger.error(f"Inference error: {e}")
                return {"answer": "Error during generation.", "reasoning": str(e), "follow_up_questions": []}
            
    # =========================================================================
    # 2. RAW TEXT GENERATION (PDF)
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
        if model_id: self.switch_model(model_id)
        
        card = self._flatten_patient_data(clinical_state)
        
        system_prompt = (
            "You are a professional Medical Scribe AI. "
            "Write a formal Clinical Referral Letter based ONLY on the provided data.\n"
            "FACTUAL GROUNDING:\n"
            "- ONLY include facts explicitly stated in PATIENT DATA.\n"
            "- Use exact values (e.g., 'pain severity 8/10').\n"
            "FORMATTING RULES:\n"
            "1. Use EXACT headers: 'CLINICAL SUMMARY:', 'DIAGNOSIS:', 'CONCLUSION:'.\n"
            "2. Do NOT use bold (**), italics, or hashes (#). Write in plain text.\n"
            "3. Write in continuous paragraphs (prose)."
        )
        
        user_prompt = (
            f"PATIENT DATA:\n{card}\n\n"
            "TASK: Write the Clinical Referral Letter.\n"
            "Start immediately with the Summary."
        )
        
        raw_text = self._generate_raw_text(
            system_prompt, 
            user_prompt, 
            force_prefix="CLINICAL SUMMARY:\n"
        )
        
        return self._parse_pdf_sections(raw_text)

    def _parse_pdf_sections(self, text: str) -> Dict[str, str]:
        """
        Robust parser for PDF sections. 
        FIX: Handles DeepSeek bold headers (**CLINICAL SUMMARY**) and casing variations.
        """
        sections = {"summary": "", "diagnosis": "", "conclusion": ""}
        
        # Don't strip ** yet, just normalize strict headers logic
        # We search for the keywords regardless of surrounding punctuation
        
        # Regex explanation:
        # (?:1\.|[\#\*]+)? -> Optional Numbering (1.) or Markdown (## / **)
        # \s* -> Optional whitespace
        # KEYWORD -> The header
        # [:\s]* -> Optional colon and whitespace
        
        p_summary = re.search(r'(?:1\.|[\#\*]+)?\s*CLINICAL SUMMARY[:\s]*', text, re.IGNORECASE)
        p_diagnosis = re.search(r'(?:2\.|[\#\*]+)?\s*(?:DIAGNOSIS|ASSESSMENT)[:\s]*', text, re.IGNORECASE)
        p_conclusion = re.search(r'(?:3\.|[\#\*]+)?\s*(?:CONCLUSION|PLAN|RECOMMENDATION)[:\s]*', text, re.IGNORECASE)

        # Calculate Indices
        idx_s = p_summary.end() if p_summary else 0
        idx_d = p_diagnosis.start() if p_diagnosis else len(text)
        idx_c = p_conclusion.start() if p_conclusion else len(text)

        # Extract and Clean
        def clean_chunk(s):
            return s.replace("**", "").replace("##", "").strip()

        if p_summary: sections["summary"] = clean_chunk(text[idx_s:idx_d])
        else: sections["summary"] = clean_chunk(text[:idx_d]) # Fallback: Start of text

        if p_diagnosis: sections["diagnosis"] = clean_chunk(text[p_diagnosis.end():idx_c])
        if p_conclusion: sections["conclusion"] = clean_chunk(text[p_conclusion.end():])

        return sections

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

    def answer_staff_question(self, question: str, clinical_state: Dict[str, Any], model_id: Optional[str] = None) -> Dict[str, Any]:
        if model_id: self.switch_model(model_id)
        
        response = self._generate_structured(clinical_state, question)
        citations = self._extract_citations(response.get("answer", ""), clinical_state)
        
        return {
            "answer": response.get("answer", ""),
            "reasoning": response.get("reasoning", ""),
            "suggested_questions": response.get("follow_up_questions", []),
            "has_reasoning": True,
            "cited_data": citations,
            "model_used": self._current_model.config.name if self._current_model else "System"
        }

    def get_available_models(self) -> List[Dict]:
        return [{**model_to_dict(c), "is_available": self._model_exists(c.id), "is_loaded": (self._current_model and self._current_model.model_id == c.id)} for c in get_all_models()]

    def get_engine_stats(self) -> Dict:
        return {
            "llama_available": LLAMA_AVAILABLE,
            "current_model": self._current_model.config.name if self._current_model else None,
            "total_inferences": self._total_inferences
        }

# Singleton
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