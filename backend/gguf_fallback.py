"""
GGUF Fallback Engine - Local Model Fallback via llama-cpp-python

Used when Parlant/Ollama is unavailable.
Provides reliable local LLM inference as a backup.
"""

import os
import gc
import asyncio
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Lazy import llama-cpp-python
LLAMA_CPP_AVAILABLE = False
Llama = None

try:
    from llama_cpp import Llama as _Llama

    Llama = _Llama
    LLAMA_CPP_AVAILABLE = True
    logger.info("llama-cpp-python available for fallback")
except ImportError:
    logger.info("llama-cpp-python not installed - GGUF fallback unavailable")


# =============================================================================
# GGUF MODEL CONFIGURATION
# =============================================================================

# These match the files in /models/ directory
GGUF_MODELS = {
    "qwen2.5-1.5b": {
        "filename": "qwen2.5-1.5b-instruct-q4_k_m.gguf",
        "context_size": 4096,
        "description": "Multilingual fallback (good for French/English)",
    },
    "llama-3.2-1b": {
        "filename": "llama-3.2-1b-instruct-q4_k_m.gguf",
        "context_size": 4096,
        "description": "Compact, fast fallback model",
    },
    "deepseek-r1-1.5b": {
        "filename": "deepseek-r1-distill-qwen-1.5b-q4_k_m.gguf",
        "context_size": 4096,
        "description": "Reasoning-focused fallback",
    },
}

DEFAULT_GGUF_MODEL = "qwen2.5-1.5b"  # Good for EN/FR
GGUF_RESPONSE_TIMEOUT = 60.0  # seconds


@dataclass
class GGUFModelInfo:
    """Info about a GGUF model."""

    model_id: str
    filename: str
    context_size: int
    description: str
    file_exists: bool = False


class GGUFFallback:
    """
    GGUF model fallback for when Parlant/Ollama is unavailable.

    Features:
    - Automatic GPU/CPU layer detection
    - Simple prompt-based generation
    - Proper async timeout handling
    """

    def __init__(self, models_dir: str = None):
        if models_dir is None:
            # Default to ../models relative to this file
            base_dir = os.path.dirname(os.path.abspath(__file__))
            models_dir = os.path.join(base_dir, "..", "models")
        self.models_dir = os.path.abspath(models_dir)
        self._model: Optional[Any] = None
        self._model_id: Optional[str] = None
        self._n_gpu_layers = self._detect_gpu_layers()
        self._n_threads = int(os.environ.get("N_THREADS", "4"))

    @property
    def is_available(self) -> bool:
        """Check if llama-cpp-python is available."""
        return LLAMA_CPP_AVAILABLE

    @property
    def is_loaded(self) -> bool:
        """Check if a model is loaded."""
        return self._model is not None

    @property
    def model_id(self) -> Optional[str]:
        return self._model_id

    def _detect_gpu_layers(self) -> int:
        """Auto-detect optimal GPU layers."""
        try:
            import torch

            if not torch.cuda.is_available():
                logger.info("CUDA not available, using CPU only")
                return 0

            props = torch.cuda.get_device_properties(0)
            free_mb = (props.total_memory - torch.cuda.memory_reserved(0)) // (
                1024 * 1024
            )

            # Conservative allocation for small models
            if free_mb > 4000:
                layers = -1  # All layers on GPU
            elif free_mb > 2000:
                layers = 20
            elif free_mb > 1000:
                layers = 10
            else:
                layers = 0

            logger.info(f"GPU memory: {free_mb}MB, using {layers} GPU layers")
            return layers
        except Exception as e:
            logger.debug(f"GPU detection failed: {e}")
            return 0

    def get_available_models(self) -> List[GGUFModelInfo]:
        """Get list of available GGUF models (downloaded ones)."""
        available = []
        for model_id, config in GGUF_MODELS.items():
            path = os.path.join(self.models_dir, config["filename"])
            exists = os.path.exists(path)
            available.append(
                GGUFModelInfo(
                    model_id=model_id,
                    filename=config["filename"],
                    context_size=config["context_size"],
                    description=config["description"],
                    file_exists=exists,
                )
            )
        return available

    def get_downloaded_models(self) -> List[GGUFModelInfo]:
        """Get only downloaded models."""
        return [m for m in self.get_available_models() if m.file_exists]

    def load_model(self, model_id: str = None) -> bool:
        """Load a GGUF model."""
        if not LLAMA_CPP_AVAILABLE:
            logger.error("llama-cpp-python not available")
            return False

        model_id = model_id or DEFAULT_GGUF_MODEL

        if model_id not in GGUF_MODELS:
            logger.error(f"Unknown model: {model_id}")
            return False

        config = GGUF_MODELS[model_id]
        path = os.path.join(self.models_dir, config["filename"])

        if not os.path.exists(path):
            logger.error(f"Model file not found: {path}")
            return False

        try:
            self.unload()

            logger.info(f"Loading GGUF model: {model_id}")
            logger.info(f"  Path: {path}")
            logger.info(f"  GPU layers: {self._n_gpu_layers}, Threads: {self._n_threads}")

            self._model = Llama(
                model_path=path,
                n_ctx=config["context_size"],
                n_threads=self._n_threads,
                n_gpu_layers=self._n_gpu_layers,
                verbose=False,
            )
            self._model_id = model_id

            logger.info(f"GGUF model loaded successfully: {model_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to load GGUF model: {e}")
            return False

    def unload(self):
        """Unload current model to free memory."""
        if self._model:
            del self._model
            self._model = None
            self._model_id = None
            gc.collect()
            logger.info("GGUF model unloaded")

    async def generate_response(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.1,
        timeout: float = None,
    ) -> str:
        """
        Generate response with timeout.

        Args:
            prompt: Full prompt text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            timeout: Timeout in seconds

        Returns:
            Generated text
        """
        if not self.is_loaded:
            if not self.load_model():
                raise RuntimeError("Failed to load GGUF model")

        timeout = timeout or GGUF_RESPONSE_TIMEOUT

        def _generate():
            result = self._model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=["</s>", "<|endoftext|>", "Human:", "User:", "<|im_end|>"],
                echo=False,
            )
            return result["choices"][0]["text"].strip()

        # Run in thread pool with timeout
        try:
            loop = asyncio.get_event_loop()
            return await asyncio.wait_for(
                loop.run_in_executor(None, _generate), timeout=timeout
            )
        except asyncio.TimeoutError:
            logger.warning(f"GGUF generation timed out after {timeout}s")
            raise

    async def answer_question(
        self, question: str, patient_context: str, language: str = "en"
    ) -> Dict[str, Any]:
        """
        Answer a triage question.

        Args:
            question: Staff question
            patient_context: Formatted patient data from PatientDataRAG
            language: 'en' or 'fr'

        Returns:
            Response dictionary
        """
        if language == "fr":
            lang_instruction = "Reponds en francais. Utilise la terminologie SFMU."
        else:
            lang_instruction = "Respond in English. Use Manchester Triage terminology."

        prompt = f"""<|im_start|>system
You are a medical triage assistant. {lang_instruction}

Rules:
- Only use information from the patient data provided
- Quote exact values when citing data
- Never invent symptoms or history
- You are a triage assistant, not a diagnosing physician
<|im_end|>
<|im_start|>user
PATIENT DATA:
{patient_context}

QUESTION: {question}

Provide:
1. ANSWER: Clear response
2. REASONING: Brief clinical reasoning citing patient data
3. FOLLOW-UP QUESTIONS: 3 specific questions for this patient
<|im_end|>
<|im_start|>assistant
"""

        try:
            raw = await self.generate_response(prompt)
            return self._parse_response(raw, language)
        except asyncio.TimeoutError:
            return self._timeout_response(language)
        except Exception as e:
            logger.error(f"GGUF generation error: {e}")
            return self._error_response(str(e), language)

    async def generate_pdf_section(
        self, section_type: str, patient_context: str, language: str = "en"
    ) -> str:
        """
        Generate a single PDF section.

        Args:
            section_type: 'summary', 'diagnosis', or 'conclusion'
            patient_context: Formatted patient data
            language: 'en' or 'fr'

        Returns:
            Generated section text
        """
        lang = "French" if language == "fr" else "English"

        prompts = {
            "summary": f"""<|im_start|>system
You are a triage nurse writing a clinical summary.
<|im_end|>
<|im_start|>user
Write a CLINICAL SUMMARY in {lang} based on this patient data. Document objectively:
- Presenting complaint
- Demographics
- All symptoms with exact values
- Time course
Be factual. Do not interpret or diagnose.

PATIENT DATA:
{patient_context}
<|im_end|>
<|im_start|>assistant
""",
            "diagnosis": f"""<|im_start|>system
You are a senior physician writing a diagnostic assessment.
<|im_end|>
<|im_start|>user
Write a DIAGNOSTIC ASSESSMENT in {lang} based on this patient data:
- Differential considerations
- Triage protocol criteria met
- Risk factors identified
- Red flags present/ruled out
Use medical terminology appropriately.

PATIENT DATA:
{patient_context}
<|im_end|>
<|im_start|>assistant
""",
            "conclusion": f"""<|im_start|>system
You are an attending physician writing recommendations.
<|im_end|>
<|im_start|>user
Write RECOMMENDATIONS in {lang} based on this patient data:
- Triage priority with justification
- Immediate actions needed
- Suggested investigations
- Disposition recommendation
Be decisive and clear.

PATIENT DATA:
{patient_context}
<|im_end|>
<|im_start|>assistant
""",
        }

        prompt = prompts.get(section_type, prompts["summary"])

        try:
            return await self.generate_response(prompt, max_tokens=512, timeout=30.0)
        except Exception as e:
            logger.error(f"GGUF PDF section generation error: {e}")
            return f"Error generating {section_type}: {e}"

    def _parse_response(self, raw: str, language: str) -> Dict[str, Any]:
        """Parse structured response from GGUF output."""
        import re

        answer = ""
        reasoning = ""
        follow_ups = []

        # Extract sections
        answer_match = re.search(
            r"ANSWER:\s*(.*?)(?=REASONING:|FOLLOW-UP|$)", raw, re.DOTALL | re.IGNORECASE
        )
        reasoning_match = re.search(
            r"REASONING:\s*(.*?)(?=FOLLOW-UP|$)", raw, re.DOTALL | re.IGNORECASE
        )
        followup_match = re.search(
            r"FOLLOW-UP\s*(?:QUESTIONS)?:\s*(.*?)$", raw, re.DOTALL | re.IGNORECASE
        )

        if answer_match:
            answer = answer_match.group(1).strip()
        else:
            # Use entire response if no structure found
            answer = raw.strip()

        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()

        if followup_match:
            questions = re.findall(
                r"\d+\.\s*(.+?)(?=\d+\.|$)", followup_match.group(1), re.DOTALL
            )
            follow_ups = [q.strip() for q in questions if q.strip()][:3]

        if len(follow_ups) < 3:
            follow_ups = self._default_questions(language)

        return {
            "answer": answer,
            "reasoning": reasoning,
            "follow_up_questions": follow_ups,
            "suggested_questions": follow_ups,
            "model_used": f"GGUF ({self._model_id})",
            "has_reasoning": bool(reasoning),
            "rag_used": False,
            "protocol_applied": "",
            "is_fallback": True,
        }

    def _default_questions(self, language: str) -> List[str]:
        """Get default follow-up questions."""
        if language == "fr":
            return [
                "Depuis combien de temps avez-vous ces symptomes?",
                "Avez-vous pris des medicaments?",
                "Y a-t-il d'autres symptomes?",
            ]
        return [
            "How long have you had these symptoms?",
            "Have you taken any medications?",
            "Are there any other symptoms?",
        ]

    def _timeout_response(self, language: str) -> Dict[str, Any]:
        """Generate timeout response."""
        if language == "fr":
            answer = "Le traitement a pris trop de temps. Veuillez reessayer."
        else:
            answer = "Processing timed out. Please try again."

        return {
            "answer": answer,
            "reasoning": "Timeout",
            "follow_up_questions": self._default_questions(language),
            "suggested_questions": self._default_questions(language),
            "model_used": f"GGUF ({self._model_id}) - Timeout",
            "has_reasoning": False,
            "is_fallback": True,
            "error": "timeout",
        }

    def _error_response(self, error: str, language: str) -> Dict[str, Any]:
        """Generate error response."""
        if language == "fr":
            answer = "Une erreur s'est produite. Veuillez reessayer."
        else:
            answer = "An error occurred. Please try again."

        return {
            "answer": answer,
            "reasoning": f"Error: {error}",
            "follow_up_questions": self._default_questions(language),
            "suggested_questions": self._default_questions(language),
            "model_used": f"GGUF ({self._model_id}) - Error",
            "has_reasoning": False,
            "is_fallback": True,
            "error": error,
        }


# =============================================================================
# SINGLETON
# =============================================================================

_gguf_instance: Optional[GGUFFallback] = None


def get_gguf_fallback(models_dir: str = None) -> GGUFFallback:
    """Get singleton GGUF fallback instance."""
    global _gguf_instance
    if _gguf_instance is None:
        _gguf_instance = GGUFFallback(models_dir)
    return _gguf_instance
