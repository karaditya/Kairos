"""
Multi-Model Reasoning Engine - Professional LLM Engine for Medical Triage

FINAL ARCHITECTURE: XML TAGGING (The "Universal" Format)
1. Input: Grounded & Sorted.
2. Prompting: XML-Based (No JSON fragility).
3. Parsing: Regex Tag Extraction (Robust to formatting errors).
4. Features: PDF Generation & Citation preserved.
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
    SUPPORTED_MODELS, ModelConfig, get_model_config, get_all_models, model_to_dict, DEFAULT_MODEL_ID,
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

        # NOTE: We removed self.json_grammar because XML does not need it.
        # This makes the engine lighter and faster.

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
            lines.append(f" - {readable_key}: {val}")

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
    # 1. XML-BASED GENERATION (The "Universal Adapter")
    # =========================================================================

    def _construct_xml_system_prompt(self, patient_text_card: str) -> str:
        """
        The Secret Sauce: XML Instructions.
        This prompt works on Deepseek, Llama, Mistral, and even old models.
        """
        return (
            "You are a helpful Medical Assistant.\n"
            "Your Goal: Answer the user's question using the Patient Card below.\n\n"
            "PATIENT CARD:\n"
            f"{patient_text_card}\n\n"
            "INSTRUCTIONS:\n"
            "1. **Context:** The 'TRIAGE PRIORITY' is correct. Do not re-assess it.\n"
            "2. **Format:** You MUST use the following XML tags for your response:\n\n"
            "<reasoning>\n"
            "(Briefly explain your clinical thinking here)\n"
            "</reasoning>\n"
            "<answer>\n"
            "(Direct answer to the user)\n"
            "</answer>\n"
            "<questions>\n"
            "- Question 1\n"
            "- Question 2\n"
            "- Question 3\n"
            "</questions>"
        )

    def _parse_xml_output(self, raw_str: str) -> Dict[str, Any]:
        """
        Robust XML Parser.
        It doesn't care about commas, quotes, or JSON syntax errors.
        It just finds the tags.
        """
        data = {
            "reasoning": "",
            "answer": "Processing error. Raw output captured.",
            "follow_up_questions": []
        }
        
        # 1. Clean up "think" tags (Deepseek specific artifact)
        clean_str = re.sub(r'<think>.*?</think>', '', raw_str, flags=re.DOTALL)

        # 2. Extract <reasoning>
        # DOTALL allows the dot (.) to match newlines, capturing multi-line text
        r_match = re.search(r'<reasoning>(.*?)</reasoning>', clean_str, re.DOTALL | re.IGNORECASE)
        if r_match:
            data["reasoning"] = r_match.group(1).strip()

        # 3. Extract <answer>
        a_match = re.search(r'<answer>(.*?)</answer>', clean_str, re.DOTALL | re.IGNORECASE)
        if a_match:
            data["answer"] = a_match.group(1).strip()
        else:
            # Fallback: If no tags found, assume the whole text is the answer (excluding tags)
            # This handles models that forget the tags completely.
            fallback_text = re.sub(r'<.*?>', '', clean_str).strip()
            if fallback_text:
                data["answer"] = fallback_text

        # 4. Extract <questions>
        q_match = re.search(r'<questions>(.*?)</questions>', clean_str, re.DOTALL | re.IGNORECASE)
        if q_match:
            raw_qs = q_match.group(1).strip()
            # Split by newlines or hyphens
            lines = raw_qs.split('\n')
            clean_qs = []
            for line in lines:
                line = line.strip().lstrip('-').lstrip('*').strip()
                if len(line) > 5:
                    clean_qs.append(line)
            data["follow_up_questions"] = clean_qs

        return data

    def _validate_response_schema(self, data: Dict[str, Any]) -> Dict[str, Any]:
        defaults = ["Is the pain worsening?", "Any fever?", "History of heart issues?", "Any nausea?", "Short of breath?"]
        
        # Ensure list structure
        questions = data.get("follow_up_questions", [])
        if not isinstance(questions, list): questions = []
        
        # Filter junk
        valid_qs = [q for q in questions if isinstance(q, str) and len(q) > 5]
        
        # Fill gaps
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
            # Use the new XML System Prompt
            sys_prompt = self._construct_xml_system_prompt(card)
            
            prompt = (
                f"<|im_start|>system\n{sys_prompt}<|im_end|>\n"
                f"<|im_start|>user\n{user_query}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )

            try:
                # No grammar needed! Pure text generation.
                output = self._current_model.instance(
                    prompt, max_tokens=1500, temperature=0.1, 
                    top_p=0.90, repeat_penalty=1.1, 
                    stop=["<|im_end|>", "<|im_start|>", "</s>"], 
                    echo=False
                )
                raw = output['choices'][0]['text']
                log_llm_output(raw, context="OUTPUT (Chat XML)")
                
                # Use the robust XML parser
                parsed_data = self._parse_xml_output(raw)
                return self._validate_response_schema(parsed_data)
                
            except Exception as e:
                logger.error(f"Inference error: {e}")
                return {"answer": "Error.", "reasoning": str(e), "follow_up_questions": []}

    # =========================================================================
    # 2. RAW TEXT GENERATION (PDF)
    # =========================================================================

    def _generate_raw_text(self, system_prompt: str, user_prompt: str, force_prefix: str = "") -> str:
        with self._lock:
            if not self._current_model: return ""
            
            full_prompt = (
                f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
                f"<|im_start|>assistant\n{force_prefix}" 
            )
            
            try:
                output = self._current_model.instance(
                    full_prompt, max_tokens=2000, temperature=0.2, 
                    top_p=0.95, repeat_penalty=1.2, 
                    stop=["<|im_end|>", "<|im_start|>", "User:"], echo=False
                )
                
                raw_continuation = output['choices'][0]['text']
                full_response = force_prefix + raw_continuation
                
                full_response = re.sub(r'<think>.*?</think>', '', full_response, flags=re.DOTALL)
                full_response = re.sub(r'<.*?>', '', full_response) 
                    
                if "PATIENT DATA:" in full_response: full_response = full_response.split("PATIENT DATA:")[0]
                if "\nUser:" in full_response: full_response = full_response.split("\nUser:")[0]
                
                log_llm_output(full_response, context="OUTPUT (PDF Raw)")
                return full_response.strip()
            except Exception as e:
                logger.error(f"PDF Gen Error: {e}")
                return ""

    def generate_pdf_summary(self, clinical_state: Dict[str, Any], model_id: Optional[str] = None) -> Dict[str, str]:
        if model_id: self.switch_model(model_id)
        
        card = self._flatten_patient_data(clinical_state)
        
        system_prompt = (
            "You are a professional Medical Scribe AI. "
            "Write a formal Clinical Referral Letter. "
            "STRICT FORMATTING RULES:\n"
            "1. Use EXACT headers: 'CLINICAL SUMMARY:', 'DIAGNOSIS:', 'CONCLUSION:'.\n"
            "2. Do NOT use markdown bold (**), italics, or hashes (#).\n"
            "3. Write in plain paragraphs."
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
        sections = {"summary": "", "diagnosis": "", "conclusion": ""}
        
        clean_text = text.replace("**", "").replace("##", "").replace("__", "")
        
        p_summary = re.search(r'(?:1\.\s*)?CLINICAL SUMMARY[:\s]*', clean_text, re.IGNORECASE)
        p_diagnosis = re.search(r'(?:2\.\s*)?(?:DIAGNOSIS|ASSESSMENT)[:\s]*', clean_text, re.IGNORECASE)
        p_conclusion = re.search(r'(?:3\.\s*)?(?:CONCLUSION|PLAN|RECOMMENDATION)[:\s]*', clean_text, re.IGNORECASE)

        idx_s = p_summary.end() if p_summary else 0
        idx_d = p_diagnosis.start() if p_diagnosis else len(clean_text)
        idx_c = p_conclusion.start() if p_conclusion else len(clean_text)

        if p_summary: sections["summary"] = clean_text[idx_s:idx_d].strip()
        else: sections["summary"] = clean_text[:idx_d].strip()

        if p_diagnosis: sections["diagnosis"] = clean_text[p_diagnosis.end():idx_c].strip()
        if p_conclusion: sections["conclusion"] = clean_text[p_conclusion.end():].strip()

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