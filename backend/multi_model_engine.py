"""
Multi-Model Reasoning Engine - Professional LLM Engine for Medical Triage

FINAL STABILITY FIXES:
1. "Garbage Cutter": Instantly chops off hallucinations (</think>, <|im_start|>).
2. Lazy Question Guard: Fills in valid questions if the model gets lazy.
3. Force-Feed PDF: Ensures report starts correctly.
"""

import os
import gc
import threading
import logging
import json
import re
import random
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict

logger = logging.getLogger(__name__)
LLM_LOG_PATH = os.path.join(os.path.dirname(__file__), "backend.log")

from model_registry import (
    SUPPORTED_MODELS, ModelConfig, get_model_config, get_all_models, model_to_dict, DEFAULT_MODEL_ID,
)

# =============================================================================
# 1. LOGGING & SCHEMA
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

class TriageResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=()) 
    
    reasoning: str = Field(..., description="Brief clinical thinking.")
    answer: str = Field(..., description="Direct answer to the user.")
    follow_up_questions: List[str] = Field(..., description="3 specific clinical questions", min_items=1, max_items=3)

# =============================================================================
# 2. ENGINE CLASS
# =============================================================================

try:
    from llama_cpp import Llama, LlamaGrammar
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

        self.json_grammar = None
        if LLAMA_AVAILABLE:
            try:
                schema = TriageResponse.model_json_schema()
                self.json_grammar = LlamaGrammar.from_json_schema(json.dumps(schema))
            except Exception as e:
                print(f"Grammar error: {e}")

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
        rules = data.get("triggered_rules", [])
        if not rules: lines.append(" - No automated alerts.")
        for rule in rules:
            lines.append(f" - [RISK: {rule.get('band', 'check').upper()}] {rule.get('description')}")

        return "\n".join(lines)

    # =========================================================================
    # 1. STRUCTURED GENERATION (CHAT)
    # =========================================================================

    def _construct_system_prompt(self, patient_text_card: str) -> str:
        return (
            "You are a Medical Assistant. Use the Patient Card below to answer questions.\n"
            "PATIENT CARD:\n"
            f"{patient_text_card}\n\n"
            "INSTRUCTIONS:\n"
            "1. Answer the user's question directly based on facts.\n"
            "2. 'follow_up_questions': Generate 3 RELEVANT clinical questions.\n"
            "3. Output valid JSON only."
        )

    def _repair_json(self, json_str: str) -> Dict[str, Any]:
        # 1. Strip Pre-amble / Hallucinations
        clean_str = re.sub(r'<think>.*?</think>', '', json_str, flags=re.DOTALL)
        clean_str = clean_str.replace("||im_start|>", "").replace("<|im_start|>", "")
        
        # Stop processing if we hit hallucinated user input
        if "<|im_start|>user" in clean_str:
            clean_str = clean_str.split("<|im_start|>user")[0]

        data = {}
        try:
            data = json.loads(clean_str)
        except:
            r_match = re.search(r'"reasoning"\s*:\s*"(.*)"\s*,\s*"answer"', clean_str, re.DOTALL)
            a_match = re.search(r'"answer"\s*:\s*"(.*)"\s*,\s*"follow_up_questions"', clean_str, re.DOTALL)
            data["reasoning"] = r_match.group(1).strip() if r_match else "Processing..."
            data["answer"] = a_match.group(1).strip() if a_match else "Error parsing response."
            data["follow_up_questions"] = []

        # 2. Fix Lazy/Empty Questions
        defaults = [
            "Has this happened before?",
            "Is the pain getting worse?",
            "Do you have a fever?",
            "Any nausea or vomiting?",
            "Does it hurt to breathe?"
        ]
        questions = data.get("follow_up_questions", [])
        if not isinstance(questions, list): questions = []
        
        valid_qs = []
        for q in questions:
            if isinstance(q, str) and len(q) > 5 and "Question 1" not in q:
                valid_qs.append(q)
        
        while len(valid_qs) < 3:
            choice = random.choice(defaults)
            if choice not in valid_qs:
                valid_qs.append(choice)
        
        data["follow_up_questions"] = valid_qs[:3]
        return data

    def _generate_structured(self, patient_context: Dict[str, Any], user_query: str) -> Dict[str, Any]:
        with self._lock:
            if not self._current_model or not self.json_grammar:
                return {"answer": "Engine Error: Model not loaded.", "reasoning": "", "follow_up_questions": []}

            card = self._flatten_patient_data(patient_context)
            sys_prompt = self._construct_system_prompt(card)
            
            prompt = (
                f"<|im_start|>system\n{sys_prompt}<|im_end|>\n"
                f"<|im_start|>user\n{user_query}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )

            try:
                output = self._current_model.instance(
                    prompt, max_tokens=1000, temperature=0.2,
                    top_p=0.92, repeat_penalty=1.1, 
                    grammar=self.json_grammar, stop=["<|im_end|>", "<|im_start|>"], echo=False
                )
                raw = output['choices'][0]['text']
                log_llm_output(raw, context="OUTPUT (Chat)")
                return self._repair_json(raw)
            except Exception as e:
                logger.error(f"Inference error: {e}")
                return {"answer": "Error.", "reasoning": str(e), "follow_up_questions": []}

    # =========================================================================
    # 2. RAW TEXT GENERATION (PDF) - WITH GARBAGE CUTTER
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
                    top_p=0.95, repeat_penalty=1.2, # Higher penalty to prevent loops
                    stop=["<|im_end|>", "<|im_start|>", "</think>"], echo=False
                )
                
                raw_continuation = output['choices'][0]['text']
                full_response = force_prefix + raw_continuation
                
                # --- THE GARBAGE CUTTER ---
                # 1. Stop if it prints the user prompt again
                if "PATIENT DATA:" in full_response:
                    full_response = full_response.split("PATIENT DATA:")[0]
                    
                # 2. Stop if it prints tags
                for tag in ["</think>", "<|im_start|>", "<|im_end|>"]:
                    full_response = full_response.split(tag)[0]
                
                # 3. Stop if it tries to roleplay "User:"
                if "\nUser:" in full_response:
                    full_response = full_response.split("\nUser:")[0]
                    
                log_llm_output(full_response, context="OUTPUT (PDF Raw)")
                return full_response
            except Exception as e:
                logger.error(f"PDF Gen Error: {e}")
                return ""

    def generate_pdf_summary(self, clinical_state: Dict[str, Any], model_id: Optional[str] = None) -> Dict[str, str]:
        if model_id: self.switch_model(model_id)
        
        card = self._flatten_patient_data(clinical_state)
        
        system_prompt = (
            "You are a professional Medical Scribe AI. "
            "Write a formal Clinical Referral Letter. "
            "Do NOT use markdown bolding. Use plain text."
        )
        
        user_prompt = (
            f"PATIENT DATA:\n{card}\n\n"
            "TASK: Finish the report starting immediately with the Clinical Summary.\n"
            "Follow this structure:\n"
            "1. CLINICAL SUMMARY (3-4 sentences)\n"
            "2. DIAGNOSIS (2-3 sentences)\n"
            "3. CONCLUSION (Recommendation)"
        )
        
        raw_text = self._generate_raw_text(
            system_prompt, 
            user_prompt, 
            force_prefix="CLINICAL SUMMARY:\n"
        )
        
        return self._parse_pdf_sections(raw_text)

    def _parse_pdf_sections(self, text: str) -> Dict[str, str]:
        sections = {"summary": "", "diagnosis": "", "conclusion": ""}
        
        clean_text = text.replace("**", "").replace("__", "").replace("##", "")
        
        p_summary = re.search(r'CLINICAL SUMMARY[:\s]*', clean_text, re.IGNORECASE)
        p_diagnosis = re.search(r'(?:DIAGNOSIS|ASSESSMENT)[:\s]*', clean_text, re.IGNORECASE)
        p_conclusion = re.search(r'(?:CONCLUSION|PLAN)[:\s]*', clean_text, re.IGNORECASE)

        if not p_summary:
            return {"summary": clean_text.strip(), "diagnosis": "", "conclusion": ""}

        idx_s = p_summary.end()
        idx_d = p_diagnosis.start() if p_diagnosis else len(clean_text)
        idx_c = p_conclusion.start() if p_conclusion else len(clean_text)

        if p_diagnosis:
            sections["summary"] = clean_text[idx_s:idx_d].strip()
            if p_conclusion:
                sections["diagnosis"] = clean_text[p_diagnosis.end():idx_c].strip()
                sections["conclusion"] = clean_text[p_conclusion.end():].strip()
            else:
                sections["diagnosis"] = clean_text[p_diagnosis.end():].strip()
        else:
            sections["summary"] = clean_text[idx_s:].strip()

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