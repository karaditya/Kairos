"""
Translation Engine for Medical Triage RAG

Uses Helsinki-NLP/opus-mt-en-fr model (~300MB) to translate
English medical queries to French for DrBERT vector search.

The French Pivot Architecture requires queries in French to match
against French medical protocols stored in the vector database.
"""

import os
import logging
from typing import Optional
import torch

logger = logging.getLogger(__name__)

# Model will be downloaded to this directory
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "translation")


class TranslationEngine:
    """
    Translates English text to French using MarianMT.

    Used in the RAG pipeline to convert English patient symptoms
    into French medical terminology for vector search.
    """

    MODEL_NAME = "Helsinki-NLP/opus-mt-en-fr"

    def __init__(self, device: Optional[str] = None):
        """
        Initialize translation engine.

        Args:
            device: "cuda", "cpu", or None for auto-detect
        """
        self._model = None
        self._tokenizer = None
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded

    def load(self) -> bool:
        """
        Load the MarianMT model and tokenizer.

        Returns:
            True if loaded successfully, False otherwise
        """
        if self._loaded:
            return True

        try:
            from transformers import MarianMTModel, MarianTokenizer

            logger.info(f"Loading translation model: {self.MODEL_NAME}")

            # Create cache directory
            os.makedirs(MODELS_DIR, exist_ok=True)

            # Load tokenizer and model
            self._tokenizer = MarianTokenizer.from_pretrained(
                self.MODEL_NAME,
                cache_dir=MODELS_DIR
            )
            self._model = MarianMTModel.from_pretrained(
                self.MODEL_NAME,
                cache_dir=MODELS_DIR
            )

            # Move to device
            self._model.to(self._device)
            self._model.eval()

            self._loaded = True
            logger.info(f"Translation model loaded on {self._device}")
            return True

        except ImportError:
            logger.error("transformers library not installed. Run: pip install transformers")
            return False
        except Exception as e:
            logger.error(f"Failed to load translation model: {e}")
            return False

    def unload(self):
        """Unload model to free memory."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        self._loaded = False

        # Clear GPU cache if using CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Translation model unloaded")

    def translate_to_french(self, text: str, max_length: int = 512) -> str:
        """
        Translate English text to French.

        Args:
            text: English text to translate
            max_length: Maximum output length

        Returns:
            French translation, or original text if translation fails
        """
        if not text or not text.strip():
            return text

        # Auto-load if not loaded
        if not self._loaded:
            if not self.load():
                logger.warning("Translation model not available, returning original text")
                return text

        try:
            # Tokenize input
            inputs = self._tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            )

            # Move to device
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            # Generate translation
            with torch.no_grad():
                translated = self._model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=4,
                    early_stopping=True
                )

            # Decode output
            result = self._tokenizer.decode(translated[0], skip_special_tokens=True)

            logger.debug(f"Translated: '{text[:50]}...' -> '{result[:50]}...'")
            return result

        except Exception as e:
            logger.error(f"Translation failed: {e}")
            return text

    def translate_medical_query(self, query: str) -> str:
        """
        Translate a medical triage query to French.

        Optimized for medical terminology - adds context hints
        to improve translation accuracy for medical terms.

        Args:
            query: English medical query (symptoms, complaints)

        Returns:
            French medical query suitable for DrBERT search
        """
        if not query or not query.strip():
            return query

        # Clean up the query - remove dict representations
        cleaned = self._clean_query(query)

        # Translate
        french_query = self.translate_to_french(cleaned)

        return french_query

    def _clean_query(self, query: str) -> str:
        """
        Clean up a query string before translation.

        Removes Python dict representations and formats for translation.
        """
        import re

        # Remove dict-like patterns: {'value': 'x', 'label': 'y'} -> y
        # Extract just the label values
        def extract_label(match):
            content = match.group(0)
            label_match = re.search(r"'label':\s*'([^']*)'", content)
            if label_match:
                return label_match.group(1)
            value_match = re.search(r"'value':\s*'([^']*)'", content)
            if value_match:
                return value_match.group(1)
            return ""

        cleaned = re.sub(r"\{[^}]+\}", extract_label, query)

        # Remove underscores
        cleaned = cleaned.replace("_", " ")

        # Remove duplicate words
        words = cleaned.split()
        seen = set()
        unique_words = []
        for word in words:
            word_lower = word.lower()
            if word_lower not in seen:
                seen.add(word_lower)
                unique_words.append(word)

        cleaned = " ".join(unique_words)

        # Clean up extra spaces
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        return cleaned


# Singleton instance for reuse
_translation_engine: Optional[TranslationEngine] = None


def get_translation_engine() -> TranslationEngine:
    """Get the singleton translation engine instance."""
    global _translation_engine
    if _translation_engine is None:
        _translation_engine = TranslationEngine()
    return _translation_engine


def translate_query_to_french(query: str) -> str:
    """
    Convenience function to translate a query to French.

    Args:
        query: English query string

    Returns:
        French translation
    """
    engine = get_translation_engine()
    return engine.translate_medical_query(query)
