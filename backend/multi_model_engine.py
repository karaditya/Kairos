# """
# Multi-Model Reasoning Engine - Professional LLM Engine for Medical Triage

# Supports multiple offline quantized models with:
# - Dynamic model switching at runtime
# - TRM-style iterative reasoning
# - Automatic fallback on errors
# - Memory-efficient model management
# - Thread-safe model loading

# Supported model families:
# - Llama (Meta)
# - Gemma (Google)
# - DeepSeek
# - Phi (Microsoft)
# - Qwen (Alibaba)
# - SmolLM (HuggingFace)
# - MedLlama (Medical specialized)

# CRITICAL: The engine CANNOT override deterministic risk bands.
# """

# import os
# import gc
# import threading
# import logging
# from typing import Dict, List, Any, Tuple, Optional
# from dataclasses import dataclass
# from datetime import datetime

# import re
# import json

# logger = logging.getLogger(__name__)

# # Path to LLM output log file (append mode)
# LLM_LOG_PATH = os.path.join(os.path.dirname(__file__), "backend.log")


# def log_llm_output(raw_output: str, context: str = ""):
#     """Log raw LLM output to backend.log in append mode."""
#     try:
#         with open(LLM_LOG_PATH, "a", encoding="utf-8") as f:
#             timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#             f.write(f"\n{'=' * 60}\n")
#             f.write(f"[{timestamp}] RAW MODEL OUTPUT")
#             if context:
#                 f.write(f" ({context})")
#             f.write(f"\n{'=' * 60}\n")
#             f.write(raw_output)
#             f.write(f"\n{'=' * 60}\n")
#     except Exception as e:
#         print(f"Warning: Could not write to LLM log: {e}")

# from model_registry import (
#     SUPPORTED_MODELS,
#     ModelConfig,
#     get_model_config,
#     get_all_models,
#     model_to_dict,
#     DEFAULT_MODEL_ID,
# )


# # =============================================================================
# # Response Parsing (Chain-of-Thought Models)
# # =============================================================================

# def _clean_output(text: str) -> str:
#     """Remove instruction bleed-through and clean text generally."""
#     if not text:
#         return ""
#     text = text.strip()
#     # Remove bracketed placeholders [like this]
#     text = re.sub(r'\[.*?\]', '', text)
#     # Remove stray XML-like tags
#     text = re.sub(r'<[^>]+>', '', text)
#     # Remove instruction-like prefixes
#     text = re.sub(r'^(?:Please|Provide|List|Write|Your|The)\s+(?:your\s+)?(?:answer|response|question).*?(?:here|below)?[.:]?\s*', '', text, flags=re.IGNORECASE)
#     # Normalize spaces (but preserve newlines for markdown)
#     text = re.sub(r'[ \t]+', ' ', text)  # Only collapse spaces/tabs, not newlines
#     text = re.sub(r'\n{3,}', '\n\n', text)  # Max 2 newlines
#     return text.strip()


# def parse_reasoning_response(text: str) -> dict:
#     """
#     Parse model response in markdown format.

#     Expected format:
#     ## Answer
#     The answer text here...

#     ## Follow-up Questions
#     1. First question?
#     2. Second question?

#     Also handles <think>...</think> for chain-of-thought reasoning.

#     Returns:
#         {
#             "reasoning": str or None,
#             "answer": str,
#             "suggested_questions": list[str],
#             "has_reasoning": bool
#         }
#     """
#     if not text:
#         return {"reasoning": None, "answer": "", "suggested_questions": [], "has_reasoning": False}

#     text = text.strip()
#     reasoning = None
#     answer = ""
#     suggested_questions = []
#     has_reasoning = False

#     # =================================================================
#     # STEP 1: Extract <think>...</think> reasoning (DeepSeek R1 style)
#     # =================================================================

#     think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL | re.IGNORECASE)
#     if think_match:
#         reasoning = think_match.group(1).strip()
#         text = text[:think_match.start()] + text[think_match.end():]
#         text = text.strip()
#         has_reasoning = True
#     else:
#         # Handle malformed: content followed by </think> without opening tag
#         close_match = re.search(r'^(.*?)</think>', text, re.DOTALL | re.IGNORECASE)
#         if close_match:
#             reasoning = close_match.group(1).strip()
#             text = text[close_match.end():].strip()
#             has_reasoning = True

#     # =================================================================
#     # STEP 2: Extract markdown sections
#     # =================================================================

#     # Try markdown format: ## Answer ... ## Follow-up Questions
#     answer_match = re.search(
#         r'##\s*Answer\s*\n(.*?)(?=##\s*Follow-up|##\s*Questions|$)',
#         text, re.DOTALL | re.IGNORECASE
#     )
#     if answer_match:
#         answer = answer_match.group(1).strip()

#     questions_match = re.search(
#         r'##\s*(?:Follow-up\s*)?Questions\s*\n(.*?)(?=##|$)',
#         text, re.DOTALL | re.IGNORECASE
#     )
#     if questions_match:
#         questions_text = questions_match.group(1).strip()
#         # Extract numbered items: "1.", "2.", etc.
#         items = re.findall(r'^\s*\d+[.\)]\s*(.+?)$', questions_text, re.MULTILINE)
#         for item in items:
#             q = item.strip()
#             if len(q) > 10:  # Must be substantial
#                 suggested_questions.append(q)

#     # =================================================================
#     # STEP 3: Fallback - try legacy formats if markdown didn't work
#     # =================================================================

#     if not answer:
#         # Try XML format: <answer>...</answer>
#         xml_answer = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL | re.IGNORECASE)
#         if xml_answer:
#             answer = xml_answer.group(1).strip()

#     if not suggested_questions:
#         # Try XML format: <questions>...</questions>
#         xml_questions = re.search(r'<questions>(.*?)</questions>', text, re.DOTALL | re.IGNORECASE)
#         if xml_questions:
#             questions_text = xml_questions.group(1).strip()
#             items = re.findall(r'^\s*\d+[.\)]\s*(.+?)$', questions_text, re.MULTILINE)
#             if not items:
#                 # Try line-by-line for unnumbered
#                 for line in questions_text.split('\n'):
#                     line = line.strip()
#                     if line and '?' in line and len(line) > 10:
#                         line = re.sub(r'^[-•*]\s*', '', line)
#                         suggested_questions.append(line)
#             else:
#                 for item in items:
#                     q = item.strip()
#                     if len(q) > 10:
#                         suggested_questions.append(q)

#         # Try JSON format: {"questions": [...]}
#         if not suggested_questions:
#             json_match = re.search(r'\{["\']questions["\']\s*:\s*\[([^\]]+)\]', text, re.DOTALL)
#             if json_match:
#                 try:
#                     json_str = '{"questions": [' + json_match.group(1) + ']}'
#                     parsed = json.loads(json_str)
#                     for q in parsed.get("questions", []):
#                         if isinstance(q, str) and len(q) > 10:
#                             suggested_questions.append(q.strip())
#                 except json.JSONDecodeError:
#                     pass

#     # =================================================================
#     # STEP 4: Final fallback - use remaining text as answer
#     # =================================================================

#     if not answer:
#         # Remove any format markers and use remaining text
#         cleaned = re.sub(r'<[^>]+>', '', text)  # Remove XML tags
#         cleaned = re.sub(r'\{["\']questions["\'].*?\}', '', cleaned, flags=re.DOTALL)  # Remove JSON
#         cleaned = re.sub(r'##.*?\n', '', cleaned)  # Remove markdown headers
#         answer = cleaned.strip()

#     # =================================================================
#     # STEP 5: Final sanitization
#     # =================================================================

#     return {
#         "reasoning": _sanitize_text(reasoning) if reasoning else None,
#         "answer": _sanitize_text(answer),
#         "suggested_questions": [_sanitize_text(q) for q in suggested_questions],
#         "has_reasoning": has_reasoning
#     }


# def _sanitize_text(text: str) -> str:
#     """Clean up text for display."""
#     if not text:
#         return text
#     # Normalize whitespace and clean up
#     text = re.sub(r' {2,}', ' ', text)
#     text = re.sub(r'\n{3,}', '\n\n', text)
#     return text.strip()


# # =============================================================================
# # LLM Backend
# # =============================================================================

# try:
#     from llama_cpp import Llama
#     LLAMA_AVAILABLE = True
# except ImportError:
#     LLAMA_AVAILABLE = False
#     print("Warning: llama-cpp-python not installed. Using fallback mode.")


# # =============================================================================
# # System Prompts (Model-Agnostic)
# # =============================================================================

# SYSTEM_PROMPT = """You are a medical triage assistant. You do NOT diagnose or prescribe.

# Your role is ONLY to:
# 1. Summarize patient-reported symptoms and data
# 2. Explain why certain risk flags were triggered
# 3. Suggest follow-up questions based on clinical protocols
# 4. Help staff understand the triage logic

# CRITICAL RULES:
# - NEVER provide medical diagnosis
# - NEVER recommend treatments or medications
# - NEVER override the risk band assigned by the system
# - ALWAYS remind users that a healthcare professional must review all cases
# - Cite specific data points when explaining decisions

# The risk band (red/amber/green) is determined by deterministic rules, not by you."""


# SUMMARY_PROMPT_TEMPLATE = """Based on the following patient data, provide a brief clinical summary.

# PATIENT DATA:
# - Age: {age}
# - Sex: {sex}
# - Pregnant: {pregnant}
# - Chief Complaint: {chief_complaint}

# REPORTED SYMPTOMS:
# {symptoms}

# RISK ASSESSMENT:
# - Risk Band: {risk_band}
# - Triggered Rules: {triggered_rules}

# TASK: Write a 2-4 sentence summary for the receiving healthcare provider. Include:
# 1. Brief patient description
# 2. Main presenting concerns
# 3. Key risk factors identified

# Do NOT diagnose. Do NOT recommend treatment. Just summarize the data.

# SUMMARY:"""


# REASONING_PROMPT_TEMPLATE = """You are refining your understanding of this case.

# CURRENT REASONING STATE:
# {z}

# PATIENT DATA:
# {patient_data}

# CURRENT SUMMARY DRAFT:
# {y}

# Update your reasoning. What patterns do you notice? What's most important for the clinician to know?
# Keep your reasoning notes brief (2-3 sentences).

# UPDATED REASONING:"""


# # =============================================================================
# # Staff Q&A Prompt (Markdown Format)
# # =============================================================================

# STAFF_QA_PROMPT = """You are a medical triage assistant. Answer in the same language as the question.

# {case_data}

# QUESTION: {question}

# Respond using EXACTLY this markdown format:

# ## Answer
# Write your answer here. Use markdown formatting (bullets, bold, etc.) as needed.

# ## Follow-up Questions
# 1. First follow-up question?
# 2. Second follow-up question?
# 3. Third follow-up question?

# IMPORTANT: Always include both sections. List 2-5 relevant follow-up questions."""


# # =============================================================================
# # Model Instance Data
# # =============================================================================

# @dataclass
# class LoadedModel:
#     """Represents a loaded model instance."""
#     model_id: str
#     config: ModelConfig
#     instance: Any  # Llama instance
#     loaded_at: datetime
#     inference_count: int = 0


# # =============================================================================
# # Multi-Model Reasoning Engine
# # =============================================================================

# class MultiModelEngine:
#     """
#     Professional multi-model reasoning engine with dynamic model switching.

#     Features:
#     - Load/unload models dynamically
#     - TRM-style iterative reasoning
#     - Thread-safe operations
#     - Automatic memory management
#     - Fallback on errors
#     """

#     def __init__(
#         self,
#         models_dir: str,
#         default_model_id: str = DEFAULT_MODEL_ID,
#         n_iterations: int = 3,
#         auto_load: bool = True
#     ):
#         """
#         Initialize the multi-model engine.

#         Args:
#             models_dir: Directory containing GGUF model files
#             default_model_id: Model to load by default
#             n_iterations: Number of TRM reasoning iterations
#             auto_load: Whether to auto-load default model on init
#         """
#         self.models_dir = models_dir
#         self.default_model_id = default_model_id
#         self.n_iterations = n_iterations

#         # Current loaded model
#         self._current_model: Optional[LoadedModel] = None
#         self._lock = threading.RLock()

#         # GPU configuration
#         self.n_gpu_layers = int(os.environ.get("N_GPU_LAYERS", "0"))
#         self.n_threads = int(os.environ.get("N_THREADS", "4"))

#         # Statistics
#         self._total_inferences = 0
#         self._model_switches = 0

#         if auto_load:
#             self._try_load_default_model()

#     # =========================================================================
#     # Model Management
#     # =========================================================================

#     def _try_load_default_model(self):
#         """Attempt to load the default model."""
#         # Try default model first
#         if self._model_exists(self.default_model_id):
#             self.load_model(self.default_model_id)
#             return

#         # Try any available model
#         for model_id in SUPPORTED_MODELS:
#             if self._model_exists(model_id):
#                 print(f"Default model not found. Loading {model_id} instead.")
#                 self.load_model(model_id)
#                 return

#         print("No models found. Running in fallback mode.")

#     def _model_exists(self, model_id: str) -> bool:
#         """Check if model file exists."""
#         config = get_model_config(model_id)
#         if not config:
#             return False
#         path = os.path.join(self.models_dir, config.filename)
#         return os.path.exists(path)

#     def _get_model_path(self, model_id: str) -> Optional[str]:
#         """Get full path to model file."""
#         config = get_model_config(model_id)
#         if not config:
#             return None
#         return os.path.join(self.models_dir, config.filename)

#     def load_model(self, model_id: str) -> bool:
#         """
#         Load a specific model, unloading current if needed.

#         Args:
#             model_id: ID of model to load

#         Returns:
#             True if successful, False otherwise
#         """
#         if not LLAMA_AVAILABLE:
#             print("llama-cpp-python not available")
#             return False

#         config = get_model_config(model_id)
#         if not config:
#             print(f"Unknown model: {model_id}")
#             return False

#         model_path = self._get_model_path(model_id)
#         if not model_path or not os.path.exists(model_path):
#             print(f"Model file not found: {model_path}")
#             return False

#         with self._lock:
#             # Unload current model if different
#             if self._current_model and self._current_model.model_id != model_id:
#                 self.unload_model()
#             elif self._current_model and self._current_model.model_id == model_id:
#                 print(f"Model {model_id} already loaded")
#                 return True

#             try:
#                 print(f"Loading model: {config.name} ({model_id})...")

#                 # Determine GPU layers
#                 gpu_layers = self.n_gpu_layers
#                 if gpu_layers == -1:
#                     gpu_layers = config.recommended_gpu_layers or 0

#                 # Determine threads
#                 threads = max(self.n_threads, config.recommended_threads)

#                 instance = Llama(
#                     model_path=model_path,
#                     n_ctx=config.context_length,
#                     n_threads=threads,
#                     n_gpu_layers=gpu_layers,
#                     verbose=False,
#                 )

#                 self._current_model = LoadedModel(
#                     model_id=model_id,
#                     config=config,
#                     instance=instance,
#                     loaded_at=datetime.now(),
#                 )

#                 self._model_switches += 1
#                 print(f"Model loaded: {config.name}")
#                 print(f"  Context: {config.context_length} tokens")
#                 print(f"  GPU layers: {gpu_layers}")
#                 print(f"  Threads: {threads}")

#                 return True

#             except Exception as e:
#                 import traceback
#                 print(f"Failed to load model {model_id}: {e}")
#                 traceback.print_exc()
#                 return False

#     def unload_model(self):
#         """Unload current model to free memory."""
#         with self._lock:
#             if self._current_model:
#                 model_name = self._current_model.config.name
#                 del self._current_model.instance
#                 self._current_model = None
#                 gc.collect()
#                 print(f"Unloaded model: {model_name}")

#     def switch_model(self, model_id: str) -> bool:
#         """
#         Switch to a different model.

#         Args:
#             model_id: ID of model to switch to

#         Returns:
#             True if successful
#         """
#         return self.load_model(model_id)

#     # =========================================================================
#     # Model Information
#     # =========================================================================

#     def get_current_model(self) -> Optional[Dict]:
#         """Get information about currently loaded model."""
#         with self._lock:
#             if not self._current_model:
#                 return None
#             return {
#                 "model_id": self._current_model.model_id,
#                 "name": self._current_model.config.name,
#                 "family": self._current_model.config.family,
#                 "loaded_at": self._current_model.loaded_at.isoformat(),
#                 "inference_count": self._current_model.inference_count,
#                 "is_loaded": True,
#             }

#     def get_available_models(self) -> List[Dict]:
#         """Get list of all available models with availability status."""
#         models = []
#         for config in get_all_models():
#             model_dict = model_to_dict(config)
#             model_dict["is_available"] = self._model_exists(config.id)
#             model_dict["is_loaded"] = (
#                 self._current_model is not None and
#                 self._current_model.model_id == config.id
#             )
#             models.append(model_dict)
#         return models

#     def get_engine_stats(self) -> Dict:
#         """Get engine statistics."""
#         return {
#             "llama_available": LLAMA_AVAILABLE,
#             "models_dir": self.models_dir,
#             "n_gpu_layers": self.n_gpu_layers,
#             "n_threads": self.n_threads,
#             "total_inferences": self._total_inferences,
#             "model_switches": self._model_switches,
#             "current_model": self.get_current_model(),
#         }

#     @property
#     def is_loaded(self) -> bool:
#         """Check if any model is loaded."""
#         return self._current_model is not None

#     # =========================================================================
#     # Text Generation
#     # =========================================================================

#     def _generate(
#         self,
#         prompt: str,
#         max_tokens: int = 256*2,
#         temperature: float = 0.3,
#         stop: Optional[List[str]] = None,
#         repeat_penalty: float = 1.15
#     ) -> str:
#         """
#         Generate text using the loaded model.

#         Args:
#             prompt: Input prompt
#             max_tokens: Maximum tokens to generate
#             temperature: Sampling temperature
#             stop: Stop sequences
#             repeat_penalty: Penalty for repeating tokens (1.0 = no penalty, >1.0 = discourage repetition)

#         Returns:
#             Generated text
#         """
#         if stop is None:
#             stop = ["</s>", "\n\n\n", "PATIENT DATA:", "TASK:"]

#         with self._lock:
#             if not self._current_model:
#                 return self._fallback_generate(prompt)

#             try:
#                 response = self._current_model.instance(
#                     prompt,
#                     max_tokens=max_tokens,
#                     temperature=temperature,
#                     stop=stop,
#                     echo=False,
#                     repeat_penalty=repeat_penalty,
#                 )

#                 self._current_model.inference_count += 1
#                 self._total_inferences += 1

#                 # Post-process to catch any remaining repetition
#                 text = response["choices"][0]["text"].strip()
#                 text = self._truncate_repetition(text)
#                 return text

#             except Exception as e:
#                 print(f"Generation error: {e}")
#                 return self._fallback_generate(prompt)

#     def _truncate_repetition(self, text: str) -> str:
#         """Truncate text at the point where it starts repeating."""
#         if not text or len(text) < 100:
#             return text

#         # Split into sentences
#         sentences = re.split(r'(?<=[.!?])\s+', text)
#         if len(sentences) < 3:
#             return text

#         # Find first repeated sentence and cut there
#         seen = {}
#         for i, sent in enumerate(sentences):
#             # Normalize for comparison
#             normalized = ' '.join(sent.lower().split())
#             if len(normalized) < 20:
#                 continue

#             if normalized in seen:
#                 # Found repetition - keep only up to first occurrence
#                 return ' '.join(sentences[:seen[normalized] + 1])
#             seen[normalized] = i

#         return text

#     def _fallback_generate(self, prompt: str) -> str:
#         """Fallback generation when model not available."""
#         if "SUMMARY:" in prompt:
#             return "Patient presents with reported symptoms requiring clinical evaluation. Risk assessment completed per protocol. Healthcare provider review recommended."
#         elif "REASONING:" in prompt:
#             return "Reviewing symptom patterns and risk factors. Key data points identified for clinical handoff."
#         elif "RESPONSE:" in prompt:
#             return "Based on the patient data provided, I can help explain the triage logic. Please note that clinical decisions must be made by a qualified healthcare provider."
#         else:
#             return "Information processed. Healthcare provider review required for clinical decisions."

#     # =========================================================================
#     # TRM-Style Reasoning
#     # =========================================================================

#     def generate_summary(
#         self,
#         clinical_state: Dict[str, Any],
#         model_id: Optional[str] = None
#     ) -> Tuple[str, List[str]]:
#         """
#         Generate patient summary using TRM-style iterative reasoning.

#         Args:
#             clinical_state: Patient clinical data
#             model_id: Optional model to use (switches if different)

#         Returns:
#             (summary_text, list_of_key_flags)
#         """
#         # Switch model if requested
#         if model_id and model_id != (self._current_model.model_id if self._current_model else None):
#             if not self.switch_model(model_id):
#                 print(f"Failed to switch to {model_id}, using current model")

#         # Initialize TRM states
#         z = ""  # Latent reasoning state
#         y = ""  # Output draft

#         # Format patient data
#         patient_data = self._format_patient_data(clinical_state)

#         # === TRM Reasoning Loop ===
#         for iteration in range(self.n_iterations):
#             reasoning_prompt = REASONING_PROMPT_TEMPLATE.format(
#                 z=z if z else "Initial analysis.",
#                 patient_data=patient_data,
#                 y=y if y else "No draft yet."
#             )
#             z = self._generate(reasoning_prompt, max_tokens=128, temperature=0.3)

#         # === Final Summary Generation ===
#         demographics = clinical_state.get("demographics", {})
#         answers = clinical_state.get("answers", {})

#         # Format symptoms
#         symptoms_list = []
#         for key, value in answers.items():
#             if not key.startswith("_") and key not in ["chief_complaint", "chief_complaint_text"]:
#                 symptoms_list.append(f"- {key}: {value}")
#         symptoms_text = "\n".join(symptoms_list) if symptoms_list else "No specific symptoms recorded"

#         # Format triggered rules
#         triggered = clinical_state.get("triggered_rules", [])
#         rules_text = ", ".join([r.get("description", r.get("id", "Unknown")) for r in triggered]) if triggered else "None"

#         summary_prompt = SUMMARY_PROMPT_TEMPLATE.format(
#             age=demographics.get("age", "Unknown"),
#             sex=demographics.get("sex", "Unknown"),
#             pregnant=demographics.get("pregnant", "N/A"),
#             chief_complaint=clinical_state.get("chief_complaint", "Unknown"),
#             symptoms=symptoms_text,
#             risk_band=clinical_state.get("risk_band", "Unknown").upper(),
#             triggered_rules=rules_text
#         )

#         summary = self._generate(summary_prompt, max_tokens=256, temperature=0.3)

#         # Log raw summary output to file (append mode)
#         log_llm_output(summary, context="Patient Summary Generation")

#         key_flags = self._extract_key_flags(clinical_state)

#         return summary, key_flags

#     def answer_staff_question(
#         self,
#         question: str,
#         clinical_state: Dict[str, Any],
#         model_id: Optional[str] = None
#     ) -> Dict[str, Any]:
#         """
#         Answer a staff question about a case.

#         Args:
#             question: Staff member's question
#             clinical_state: Full case data
#             model_id: Optional model to use

#         Returns:
#             {
#                 "answer": str,           # Clean final answer
#                 "reasoning": str | None, # Chain-of-thought (if available)
#                 "has_reasoning": bool,   # Whether CoT was present
#                 "cited_data": list,      # Referenced data points
#                 "model_used": str        # Model that generated response
#             }
#         """
#         # Switch model if requested
#         if model_id and model_id != (self._current_model.model_id if self._current_model else None):
#             if not self.switch_model(model_id):
#                 print(f"Failed to switch to {model_id}, using current model")

#         case_text = self._format_case_for_staff(clinical_state)

#         prompt = STAFF_QA_PROMPT.format(
#             case_data=case_text,
#             question=question
#         )

#         raw_response = self._generate(prompt, max_tokens=512, temperature=0.5, repeat_penalty=1.2)

#         # Log raw output to file (append mode)
#         log_llm_output(raw_response, context=f"Staff Q&A: {question[:50]}...")

#         # Parse to separate reasoning from answer
#         parsed = parse_reasoning_response(raw_response)

#         cited_data = self._extract_citations(parsed["answer"], clinical_state)

#         return {
#             "answer": parsed["answer"],
#             "reasoning": parsed["reasoning"],
#             "has_reasoning": parsed["has_reasoning"],
#             "suggested_questions": parsed["suggested_questions"],
#             "cited_data": cited_data,
#             "model_used": self._current_model.config.name if self._current_model else "Fallback"
#         }

#     # =========================================================================
#     # Data Formatting Helpers
#     # =========================================================================

#     def _format_patient_data(self, clinical_state: Dict[str, Any]) -> str:
#         """Format clinical state as readable text."""
#         lines = []

#         demo = clinical_state.get("demographics", {})
#         lines.append(f"Patient: {demo.get('age', '?')} y/o {demo.get('sex', '?')}")
#         if demo.get("pregnant"):
#             lines.append("Currently pregnant")

#         lines.append(f"Chief complaint: {clinical_state.get('chief_complaint', 'Unknown')}")
#         lines.append(f"Risk band: {clinical_state.get('risk_band', 'Unknown').upper()}")

#         answers = clinical_state.get("answers", {})
#         if answers:
#             lines.append("Responses:")
#             for k, v in answers.items():
#                 if not k.startswith("_"):
#                     lines.append(f"  - {k}: {v}")

#         return "\n".join(lines)

#     def _format_case_for_staff(self, clinical_state: Dict[str, Any]) -> str:
#         """Format case data for staff view."""
#         lines = []

#         demo = clinical_state.get("demographics", {})
#         lines.append("=== PATIENT DEMOGRAPHICS ===")
#         lines.append(f"Age: {demo.get('age', 'Unknown')}")
#         lines.append(f"Sex: {demo.get('sex', 'Unknown')}")
#         lines.append(f"Pregnant: {demo.get('pregnant', 'N/A')}")

#         lines.append("\n=== CHIEF COMPLAINT ===")
#         lines.append(clinical_state.get("chief_complaint", "Unknown"))

#         lines.append("\n=== PATIENT RESPONSES ===")
#         for key, value in clinical_state.get("answers", {}).items():
#             if not key.startswith("_"):
#                 lines.append(f"{key}: {value}")

#         lines.append("\n=== RISK ASSESSMENT ===")
#         lines.append(f"Risk Band: {clinical_state.get('risk_band', 'Unknown').upper()}")

#         lines.append("\nTriggered Rules:")
#         for rule in clinical_state.get("triggered_rules", []):
#             lines.append(f"- [{rule.get('band', '?').upper()}] {rule.get('description', 'Unknown rule')}")

#         if clinical_state.get("summary"):
#             lines.append("\n=== AUTO-GENERATED SUMMARY ===")
#             lines.append(clinical_state["summary"])

#         return "\n".join(lines)

#     def _extract_key_flags(self, clinical_state: Dict[str, Any]) -> List[str]:
#         """Extract key clinical flags from the data."""
#         flags = []

#         demo = clinical_state.get("demographics", {})
#         answers = clinical_state.get("answers", {})

#         # Age flags
#         age = demo.get("age", 0)
#         if age < 2:
#             flags.append("Infant patient (under 2 years)")
#         elif age > 65:
#             flags.append("Elderly patient (over 65 years)")

#         # Pregnancy flag
#         if demo.get("pregnant"):
#             flags.append("Patient is pregnant")

#         # Check answers for concerning patterns
#         for key, value in answers.items():
#             key_lower = key.lower()

#             if "fever" in key_lower or "temperature" in key_lower:
#                 if isinstance(value, (int, float)) and value > 39.5:
#                     flags.append(f"High fever ({value}C)")
#                 elif isinstance(value, (int, float)) and value > 38:
#                     flags.append(f"Fever present ({value}C)")

#             if "pain" in key_lower and "severity" in key_lower:
#                 if isinstance(value, (int, float)) and value >= 8:
#                     flags.append("Severe pain reported")

#             if "breathing" in key_lower or "breath" in key_lower:
#                 if value in [True, "yes", "Yes"]:
#                     flags.append("Breathing difficulty reported")

#             if "chest" in key_lower and "pain" in key_lower:
#                 if value in [True, "yes", "Yes"]:
#                     flags.append("Chest pain reported")

#             if "duration" in key_lower or "days" in key_lower:
#                 if isinstance(value, (int, float)) and value > 7:
#                     flags.append(f"Prolonged symptoms ({value} days)")

#         # Add triggered rule flags
#         for rule in clinical_state.get("triggered_rules", []):
#             if rule.get("band") == "red":
#                 flags.append(f"RED FLAG: {rule.get('description', 'Red flag rule triggered')}")

#         return flags

#     def _extract_citations(self, answer: str, clinical_state: Dict[str, Any]) -> List[str]:
#         """Extract data points cited in the answer."""
#         citations = []
#         answer_lower = answer.lower()

#         demo = clinical_state.get("demographics", {})
#         if str(demo.get("age", "")) in answer:
#             citations.append(f"Age: {demo.get('age')}")

#         for key, value in clinical_state.get("answers", {}).items():
#             if str(value).lower() in answer_lower or key.lower() in answer_lower:
#                 citations.append(f"{key}: {value}")

#         for rule in clinical_state.get("triggered_rules", []):
#             desc = rule.get("description", "")
#             if desc.lower() in answer_lower:
#                 citations.append(f"Rule: {desc}")

#         return citations[:5]


# # =============================================================================
# # Singleton Instance Factory
# # =============================================================================

# _engine_instance: Optional[MultiModelEngine] = None


# def get_engine(
#     models_dir: str = "../models",
#     default_model_id: str = DEFAULT_MODEL_ID,
#     reinitialize: bool = False
# ) -> MultiModelEngine:
#     """
#     Get or create the multi-model engine singleton.

#     Args:
#         models_dir: Directory containing model files
#         default_model_id: Default model to load
#         reinitialize: Force re-initialization

#     Returns:
#         MultiModelEngine instance
#     """
#     global _engine_instance

#     if _engine_instance is None or reinitialize:
#         _engine_instance = MultiModelEngine(
#             models_dir=models_dir,
#             default_model_id=default_model_id,
#         )

#     return _engine_instance


# # =============================================================================
# # Standalone Testing
# # =============================================================================

# if __name__ == "__main__":
#     print("=== Multi-Model Engine Test ===\n")

#     engine = MultiModelEngine("../models", auto_load=True)

#     print("\n--- Available Models ---")
#     for model in engine.get_available_models():
#         status = "LOADED" if model["is_loaded"] else ("available" if model["is_available"] else "not downloaded")
#         print(f"  {model['id']}: {model['name']} [{status}]")

#     print("\n--- Engine Stats ---")
#     stats = engine.get_engine_stats()
#     for k, v in stats.items():
#         print(f"  {k}: {v}")

#     if engine.is_loaded:
#         print("\n--- Test Summary Generation ---")
#         test_case = {
#             "demographics": {"age": 45, "sex": "male", "pregnant": None},
#             "chief_complaint": "chest_discomfort",
#             "answers": {
#                 "chest_pain": True,
#                 "pain_severity": 7,
#                 "shortness_of_breath": True,
#                 "pain_radiating": "left arm",
#                 "duration_hours": 2
#             },
#             "risk_band": "red",
#             "triggered_rules": [
#                 {"id": "chest_pain_sob", "description": "Chest pain with shortness of breath", "band": "red"}
#             ]
#         }

#         summary, flags = engine.generate_summary(test_case)
#         print(f"\nSummary:\n{summary}")
#         print(f"\nKey Flags:\n{flags}")



"""
Multi-Model Reasoning Engine - Professional LLM Engine for Medical Triage

Supports multiple offline quantized models with:
- Real-time LLM inference (Llama, DeepSeek, Mistral, etc.)
- Structured JSON output via GBNF Grammars
- Full Raw Output Logging (Backend.log)
- Context-aware generation (No hardcoded responses)

Usage:
    engine = get_engine()
    response = engine.answer_staff_question("Why is this high risk?", patient_data)
"""

import os
import gc
import threading
import logging
import json
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

# Pydantic is used to define the strict schema for the model's output
from pydantic import BaseModel, Field, ConfigDict

logger = logging.getLogger(__name__)

# Path to LLM output log file (append mode)
LLM_LOG_PATH = os.path.join(os.path.dirname(__file__), "backend.log")

# Import your existing registry
from model_registry import (
    SUPPORTED_MODELS,
    ModelConfig,
    get_model_config,
    get_all_models,
    model_to_dict,
    DEFAULT_MODEL_ID,
)

# =============================================================================
# 1. LOGGING UTILITY (Minimalistic & Robust)
# =============================================================================

def log_llm_output(raw_output: str, context: str = ""):
    """Logs the raw text generated by the LLM to backend.log."""
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = (
            f"\n{'='*60}\n"
            f"[{timestamp}] RAW MODEL OUTPUT ({context})\n"
            f"{'-'*60}\n"
            f"{raw_output}\n"
            f"{'='*60}\n"
        )
        with open(LLM_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(log_entry)
    except Exception as e:
        print(f"Warning: Logging failed: {e}")

# =============================================================================
# 2. DEFINE OUTPUT SCHEMA
# =============================================================================

class TriageResponse(BaseModel):
    """
    Strict schema for frontend integration.
    """
    model_config = ConfigDict(protected_namespaces=())

    reasoning: str = Field(..., description="Clinical reasoning process.")
    answer: str = Field(..., description="Direct answer to the user.")
    follow_up_questions: List[str] = Field(..., min_items=1, max_items=5)

# =============================================================================
# 3. ENGINE SETUP
# =============================================================================

try:
    from llama_cpp import Llama, LlamaGrammar
    LLAMA_AVAILABLE = True
except ImportError:
    LLAMA_AVAILABLE = False
    print("CRITICAL WARNING: llama-cpp-python not installed. Engine will run in fallback mode.")

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
        
        # Inference Config
        self.n_gpu_layers = int(os.environ.get("N_GPU_LAYERS", "-1")) 
        self.n_threads = int(os.environ.get("N_THREADS", "6")) 

        # Grammar Compilation
        self.json_grammar = None
        if LLAMA_AVAILABLE:
            try:
                schema = TriageResponse.model_json_schema()
                self.json_grammar = LlamaGrammar.from_json_schema(json.dumps(schema))
                print("GBNF Grammar compiled successfully.")
            except Exception as e:
                print(f"Failed to compile grammar: {e}")

        if auto_load:
            self._try_load_default_model()

    # =========================================================================
    # Model Management
    # =========================================================================

    def _try_load_default_model(self):
        if self._model_exists(self.default_model_id):
            self.load_model(self.default_model_id)
            return
        for model_id in SUPPORTED_MODELS:
            if self._model_exists(model_id):
                self.load_model(model_id)
                return
        print("No models found. Running in fallback.")

    def _model_exists(self, model_id: str) -> bool:
        config = get_model_config(model_id)
        return bool(config and os.path.exists(os.path.join(self.models_dir, config.filename)))

    def load_model(self, model_id: str) -> bool:
        if not LLAMA_AVAILABLE: return False
        config = get_model_config(model_id)
        if not config: return False
        
        path = os.path.join(self.models_dir, config.filename)

        with self._lock:
            if self._current_model and self._current_model.model_id == model_id:
                return True 

            self.unload_model()
            try:
                print(f"Loading model: {config.name}...")
                instance = Llama(
                    model_path=path, n_ctx=4096, n_threads=self.n_threads,
                    n_gpu_layers=self.n_gpu_layers if self.n_gpu_layers != -1 else config.recommended_gpu_layers or -1,
                    verbose=False
                )
                self._current_model = LoadedModel(model_id, config, instance, datetime.now())
                return True
            except Exception as e:
                print(f"Failed to load {model_id}: {e}")
                return False

    def unload_model(self):
        with self._lock:
            if self._current_model:
                try:
                    del self._current_model.instance
                    gc.collect()
                except: pass
                self._current_model = None

    def switch_model(self, model_id: str) -> bool:
        return self.load_model(model_id)

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
    # Generation Logic
    # =========================================================================

    def _generate_structured(self, patient_context: Dict[str, Any], user_query: str, temperature: float = 0.7) -> Dict[str, Any]:
        with self._lock:
            if not self._current_model or not self.json_grammar:
                return self._fallback_response("Engine not ready.")

            # Construct Prompt
            data_str = json.dumps(patient_context, indent=2)
            system_msg = (
                "You are an expert medical triage assistant. Analyze the patient data below.\n"
                f"### PATIENT DATA\n{data_str}\n\n"
                "### INSTRUCTIONS\n"
                "1. Answer the User Query based strictly on the Patient Data.\n"
                "2. Provide detailed medical reasoning.\n"
                "3. Suggest 3 relevant follow-up questions.\n"
                "4. Output must be valid JSON."
            )
            
            prompt = (
                f"<|im_start|>system\n{system_msg}<|im_end|>\n"
                f"<|im_start|>user\n{user_query}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )

            try:
                # Inference
                output = self._current_model.instance(
                    prompt, max_tokens=1500, temperature=temperature, top_p=0.9,
                    repeat_penalty=1.1, grammar=self.json_grammar,
                    stop=["<|im_end|>", "}"], echo=False
                )
                
                self._current_model.inference_count += 1
                raw_text = output['choices'][0]['text']

                # --- LOGGING CALL ADDED HERE ---
                log_llm_output(raw_text, context="Structured Generation")
                # -------------------------------

                # Cleaning
                clean_text = raw_text.strip()
                if not clean_text.endswith("}"): clean_text += "}"
                if "```" in clean_text: 
                    clean_text = clean_text.replace("```json", "").replace("```", "").strip()

                return json.loads(clean_text)

            except Exception as e:
                print(f"Generation error: {e}")
                return self._fallback_response(str(e))

    def _fallback_response(self, error_msg: str) -> Dict[str, Any]:
        return {
            "reasoning": f"System Alert: {error_msg}",
            "answer": "Unable to process request. Consult clinical protocols.",
            "follow_up_questions": ["Check system logs."]
        }

    # =========================================================================
    # Public API
    # =========================================================================

    def generate_summary(self, clinical_state: Dict[str, Any], model_id: Optional[str] = None) -> Tuple[str, List[str]]:
        if model_id: self.switch_model(model_id)
        response = self._generate_structured(clinical_state, "Provide a clinical triage summary.")
        return response.get("answer", "No summary."), self._extract_key_flags(clinical_state)

    def answer_staff_question(self, question: str, clinical_state: Dict[str, Any], model_id: Optional[str] = None) -> Dict[str, Any]:
        if model_id: self.switch_model(model_id)
        response = self._generate_structured(clinical_state, question)
        return {
            "answer": response["answer"],
            "reasoning": response["reasoning"],
            "suggested_questions": response["follow_up_questions"],
            "has_reasoning": True,
            "cited_data": self._extract_citations(response["answer"], clinical_state),
            "model_used": self._current_model.config.name if self._current_model else "Fallback"
        }

    def _extract_key_flags(self, clinical_state: Dict[str, Any]) -> List[str]:
        flags = []
        demo = clinical_state.get("demographics", {})
        answers = clinical_state.get("answers", {})
        if demo.get("age", 0) < 2: flags.append("Infant (< 2y)")
        elif demo.get("age", 0) > 65: flags.append("Elderly (> 65y)")
        if demo.get("pregnant"): flags.append("Pregnant")
        for k, v in answers.items():
            if "fever" in k and v > 38: flags.append(f"Fever ({v})")
            if "pain" in k and v >= 8: flags.append("Severe Pain")
        for rule in clinical_state.get("triggered_rules", []):
            if rule.get("band") == "red": flags.append(f"RED: {rule.get('description')}")
        return flags

    def _extract_citations(self, text: str, data: Dict[str, Any]) -> List[str]:
        citations = []
        t_low = text.lower()
        for k, v in data.get("answers", {}).items():
            if str(v).lower() in t_low: citations.append(f"{k}: {v}")
        for rule in data.get("triggered_rules", []):
            if rule.get("description", "").lower() in t_low: citations.append(f"Rule: {rule.get('description')}")
        return citations[:5]

    def get_available_models(self) -> List[Dict]:
        return [{**model_to_dict(c), "is_available": self._model_exists(c.id), "is_loaded": (self._current_model and self._current_model.model_id == c.id)} for c in get_all_models()]

    def get_engine_stats(self) -> Dict:
        return {
            "llama_available": LLAMA_AVAILABLE,
            "current_model": self._current_model.config.name if self._current_model else None,
            "total_inferences": self._total_inferences
        }
    
    @property
    def is_loaded(self) -> bool: return self._current_model is not None


_engine_instance: Optional[MultiModelEngine] = None

def get_engine(models_dir: str = "../models", default_model_id: str = DEFAULT_MODEL_ID, reinitialize: bool = False) -> MultiModelEngine:
    global _engine_instance
    if _engine_instance is None or reinitialize:
        if _engine_instance and reinitialize: _engine_instance.unload_model()
        _engine_instance = MultiModelEngine(models_dir, default_model_id)
    return _engine_instance

if __name__ == "__main__":
    print("=== Engine Test ===")
    eng = MultiModelEngine("../models", auto_load=True)
    if eng.is_loaded:
        res = eng.answer_staff_question("Why red risk?", {"risk_band": "red", "answers": {"pain": 8}})
        print(json.dumps(res, indent=2))