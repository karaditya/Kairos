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
            # Extract label from dict values, handle booleans nicely
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
    # 1. XML-BASED GENERATION (The "Universal Adapter")
    # =========================================================================

    def _construct_xml_system_prompt(self, patient_text_card: str) -> str:
        return (
            "You are an expert Clinical Triage AI. "
            "Your goal is to answer the user's request based strictly on the PATIENT CARD.\n\n"
            "PATIENT CARD:\n"
            f"{patient_text_card}\n\n"
            "--- FACTUAL GROUNDING (CRITICAL) ---\n"
            "- ONLY state facts explicitly listed in the PATIENT CARD above.\n"
            "- NEVER invent, assume, or infer symptoms, values, or history not provided.\n"
            "- If data is missing, say 'not reported' - do NOT guess.\n"
            "- Quote exact values (e.g., 'severity 8/10', 'left side') when referencing data.\n\n"
            "--- INSTRUCTIONS ---\n"
            "You must generate a response in valid XML using the exact tags below.\n\n"
            "1. <reasoning>\n"
            "   - **Audience:** Senior Doctor (Internal Note).\n"
            "   - **Content:** Explain the risk factors. Why is this Red/Amber? Connect symptoms to the protocol.\n"
            "   - **Rule:** Do NOT show this to the user.\n\n"
            "2. <answer>\n"
            "   - **Audience:** The User (The final display text).\n"
            "   - **Content:** The direct answer to the prompt. If asking for a list, provide a Markdown bullet list here.\n"
            "   - **Rule:** Do NOT summarize what you just wrote in reasoning. Just give the answer.\n\n"
            "3. <questions>\n"
            "   - **Audience:** The Patient (Follow-up).\n"
            "   - **Content:** 3 short, specific questions to clarify the condition.\n\n"
            "--- REQUIRED OUTPUT FORMAT ---\n"
            "<reasoning>\n"
            "... (Clinical logic goes here) ...\n"
            "</reasoning>\n"
            "<answer>\n"
            "... (The detailed response/list goes here) ...\n"
            "</answer>\n"
            "<questions>\n"
            "- Question 1\n"
            "- Question 2\n"
            "- Question 3\n"
            "</questions>"
        )
    
    def _parse_xml_output(self, raw_str: str) -> Dict[str, Any]:
        data = {
            "reasoning": "",
            "answer": "",
            "follow_up_questions": []
        }
        
        # 1. Clean Deepseek/Think tags first (remove completely)
        clean_str = re.sub(r'<think>.*?</think>', '', raw_str, flags=re.DOTALL)
        
        # 2. Extract Reasoning (Non-greedy)
        r_match = re.search(r'<reasoning>(.*?)</reasoning>', clean_str, re.DOTALL | re.IGNORECASE)
        if r_match:
            data["reasoning"] = r_match.group(1).strip()

        # 3. Extract Answer (The Priority)
        # We look for the tag explicitly.
        a_match = re.search(r'<answer>(.*?)</answer>', clean_str, re.DOTALL | re.IGNORECASE)
        if a_match:
            data["answer"] = a_match.group(1).strip()
        else:
            # FAILSAFE: If no <answer> tag is found, we assume the model failed formatting.
            # We take the whole string but remove the <reasoning> and <questions> parts to clean it.
            fallback = clean_str
            fallback = re.sub(r'<reasoning>.*?</reasoning>', '', fallback, flags=re.DOTALL)
            fallback = re.sub(r'<questions>.*?</questions>', '', fallback, flags=re.DOTALL)
            data["answer"] = fallback.strip()

        # 4. Extract Questions
        q_match = re.search(r'<questions>(.*?)</questions>', clean_str, re.DOTALL | re.IGNORECASE)
        if q_match:
            raw_qs = q_match.group(1).strip()
            # Split by newlines
            lines = raw_qs.split('\n')
            clean_qs = []
            for line in lines:
                # Cleaning Logic:
                # Remove markdown bullets (- or *)
                # Remove numbering (1. or 1))
                # Remove "Question 1:" prefixes
                cleaned_line = re.sub(r'^[\d\-\.\)\*]*(?:Question\s*\d*[:\.]?)?\s*', '', line.strip(), flags=re.IGNORECASE)
                
                if len(cleaned_line) > 5:
                    clean_qs.append(cleaned_line)
            data["follow_up_questions"] = clean_qs

        # Final Cleanup: If "Answer" is empty (rare), put a default message
        if not data["answer"]:
            data["answer"] = "Analysis complete. Please review the reasoning or questions."

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
            sys_prompt = self._construct_xml_system_prompt(card)
            
            # --- CHANGE: Use Message List, not Raw String ---
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_query}
            ]

            try:
                # --- CHANGE: Use create_chat_completion ---
                # This automatically applies the correct template (Llama-3, ChatML, Mistral, etc.)
                output = self._current_model.instance.create_chat_completion(
                    messages=messages,
                    max_tokens=1500,
                    temperature=0.1,
                    top_p=0.90,
                    repeat_penalty=1.1,
                    stop=["<|im_end|>", "<|im_start|>", "</s>", "<|eot_id|>"], # Covers most models
                    # stream=False is default
                )
                
                # Extract content safely
                raw = output['choices'][0]['message']['content']
                log_llm_output(raw, context="OUTPUT (Chat XML)")
                
                parsed_data = self._parse_xml_output(raw)
                return self._validate_response_schema(parsed_data)
                
            except Exception as e:
                logger.error(f"Inference error: {e}")
                return {"answer": "Error.", "reasoning": str(e), "follow_up_questions": []}
            
    # =========================================================================
    # 2. RAW TEXT GENERATION (PDF)
    # =========================================================================

    def _generate_raw_text(self, system_prompt: str, user_prompt: str, force_prefix: str = "") -> str:
        """
        Generates plain text (non-XML) for PDF reports.
        Uses create_chat_completion to ensure compatibility with Llama-3/Deepseek.
        """
        with self._lock:
            if not self._current_model: 
                return "Error: Model not loaded."
            
            # 1. Use the Message Format (Fixes "Garbage Output" on Llama-3)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # 2. Handle the "Force Prefix" (e.g., "CLINICAL SUMMARY:")
            # Prefilling is tricky with Chat APIs, so we append it to the user prompt instructions.
            if force_prefix:
                messages[1]["content"] += f"\n\nIMPORTANT: Start your response immediately with the header: '{force_prefix}'"

            try:
                # 3. Call the Engine
                output = self._current_model.instance.create_chat_completion(
                    messages=messages,
                    max_tokens=2500,  # PDFs need more space
                    temperature=0.2,  # Keep it professional/dry
                    top_p=0.95,
                    stop=["<|im_end|>", "<|im_start|>", "<|eot_id|>"]
                )
                
                # 4. Extract and Clean
                raw_response = output['choices'][0]['message']['content']
                
                # Deepseek Cleaner (Just in case it "thinks" during a PDF gen)
                clean_response = re.sub(r'<think>.*?</think>', '', raw_response, flags=re.DOTALL)
                
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
            "FACTUAL GROUNDING (CRITICAL):\n"
            "- ONLY include facts explicitly stated in PATIENT DATA.\n"
            "- NEVER invent symptoms, history, or values not provided.\n"
            "- Use exact values from the data (e.g., 'pain severity 8/10').\n"
            "FORMATTING RULES:\n"
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