"""
Centralized Configuration for Parlant-Native Medical Triage Backend

All configuration values in one place for easy management.
"""

import os
from typing import Optional
from dataclasses import dataclass

# =============================================================================
# OLLAMA / LLM CONFIGURATION
# =============================================================================

# Default Ollama model (can be changed at runtime via API)
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "mistral")

# Active model - can be changed dynamically during runtime
# This is the model that will be used for generation
ACTIVE_OLLAMA_MODEL = OLLAMA_MODEL

# Ollama server URL
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

# Ollama API timeout (seconds)
OLLAMA_TIMEOUT = int(os.environ.get("OLLAMA_TIMEOUT", "300"))

# Parlant server ports
PARLANT_PORT = int(os.environ.get("PARLANT_PORT", "8800"))
PARLANT_TOOL_PORT = int(os.environ.get("PARLANT_TOOL_PORT", "8818"))

# =============================================================================
# GGUF FALLBACK CONFIGURATION
# =============================================================================

# GGUF models available for fallback (llama-cpp-python)
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

# Default GGUF model for fallback
DEFAULT_GGUF_MODEL = os.environ.get("DEFAULT_GGUF_MODEL", "qwen2.5-1.5b")

# =============================================================================
# TIMEOUT CONFIGURATION
# =============================================================================

# Response timeouts (seconds)
PARLANT_RESPONSE_TIMEOUT = float(os.environ.get("PARLANT_RESPONSE_TIMEOUT", "30.0"))
GGUF_RESPONSE_TIMEOUT = float(os.environ.get("GGUF_RESPONSE_TIMEOUT", "60.0"))
OVERALL_REQUEST_TIMEOUT = float(os.environ.get("OVERALL_REQUEST_TIMEOUT", "90.0"))

# =============================================================================
# SUPPORTED OLLAMA MODELS
# =============================================================================

SUPPORTED_OLLAMA_MODELS = {
    "mistral": {
        "name": "Mistral 7B",
        "size_gb": 4.1,
        "description": "Fast, balanced performance - good default choice",
    },
    "mistral:7b-instruct": {
        "name": "Mistral 7B Instruct",
        "size_gb": 4.1,
        "description": "Instruction-tuned Mistral for better chat",
    },
    "llama3.2": {
        "name": "Llama 3.2 3B",
        "size_gb": 2.0,
        "description": "Meta's compact model - fast responses",
    },
    "llama3.2:1b": {
        "name": "Llama 3.2 1B",
        "size_gb": 1.3,
        "description": "Ultra-compact for low resources",
    },
    "llama3.1": {
        "name": "Llama 3.1 8B",
        "size_gb": 4.7,
        "description": "Meta's capable 8B model",
    },
    "deepseek-r1:7b": {
        "name": "DeepSeek R1 7B",
        "size_gb": 4.7,
        "description": "Strong reasoning capabilities",
    },
    "deepseek-r1:14b": {
        "name": "DeepSeek R1 14B",
        "size_gb": 9.0,
        "description": "Larger DeepSeek for complex reasoning",
    },
    "deepseek-r1:1.5b": {
        "name": "DeepSeek R1 1.5B",
        "size_gb": 1.1,
        "description": "Compact DeepSeek - fast reasoning",
    },
    "qwen2.5": {
        "name": "Qwen 2.5 7B",
        "size_gb": 4.4,
        "description": "Alibaba's multilingual model - good for French",
    },
    "qwen2.5:3b": {
        "name": "Qwen 2.5 3B",
        "size_gb": 1.9,
        "description": "Compact Qwen - efficient multilingual",
    },
    "gemma2": {
        "name": "Gemma 2 9B",
        "size_gb": 5.4,
        "description": "Google's efficient model",
    },
    "gemma2:2b": {
        "name": "Gemma 2 2B",
        "size_gb": 1.6,
        "description": "Compact Gemma for fast inference",
    },
    "phi3": {
        "name": "Phi-3 Mini",
        "size_gb": 2.2,
        "description": "Microsoft's compact reasoning model",
    },
    "codellama": {
        "name": "Code Llama 7B",
        "size_gb": 3.8,
        "description": "Code-focused (for technical queries)",
    },
}

# =============================================================================
# FEATURE FLAGS
# =============================================================================

# Enable strict guideline enforcement
STRICT_GUIDELINES = os.environ.get("STRICT_GUIDELINES", "true").lower() == "true"

# Enable RAG (DrBERT + ChromaDB)
RAG_ENABLED = os.environ.get("RAG_ENABLED", "true").lower() == "true"

# Enable debug logging
DEBUG_MODE = os.environ.get("DEBUG_MODE", "false").lower() == "true"

# =============================================================================
# PATHS
# =============================================================================

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.environ.get("CONFIG_DIR", os.path.join(BASE_DIR, "..", "config"))
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(BASE_DIR, "..", "data"))
MODELS_DIR = os.environ.get("MODELS_DIR", os.path.join(BASE_DIR, "..", "models"))

# Database
DB_PATH = os.environ.get("DB_PATH", os.path.join(DATA_DIR, "triage.db"))

# ChromaDB persistence
CHROMA_PERSIST_DIR = os.environ.get("CHROMA_PERSIST_DIR", os.path.join(DATA_DIR, "chroma"))

# =============================================================================
# STAFF AUTHENTICATION
# =============================================================================

STAFF_PIN = os.environ.get("STAFF_PIN", "1234")  # Default PIN for demo

# =============================================================================
# DRBERT CONFIGURATION
# =============================================================================

@dataclass
class DrBERTConfig:
    """Configuration for DrBERT models."""
    model_id: str
    name: str
    hf_repo: str
    size_gb: float
    description: str


DRBERT_MODELS = {
    "drbert-4gb": DrBERTConfig(
        model_id="drbert-4gb",
        name="DrBERT 4GB",
        hf_repo="Dr-BERT/DrBERT-4GB-CP-PubMedBERT",
        size_gb=4.0,
        description="French medical BERT trained on 4GB NACHOS corpus"
    ),
    "drbert-7gb": DrBERTConfig(
        model_id="drbert-7gb",
        name="DrBERT 7GB (Recommended)",
        hf_repo="Dr-BERT/DrBERT-7GB-CP-PubMedBERT",
        size_gb=7.0,
        description="French medical BERT trained on 7GB NACHOS corpus"
    ),
    "drbert-4gb-pubmed": DrBERTConfig(
        model_id="drbert-4gb-pubmed",
        name="DrBERT-PubMedBERT Hybrid",
        hf_repo="Dr-BERT/DrBERT-4GB-CP-PubMedBERT",
        size_gb=4.5,
        description="Hybrid French BERT + PubMedBERT for bilingual support"
    ),
}

DEFAULT_DRBERT_MODEL = "drbert-7gb"

# =============================================================================
# TRIAGE SYSTEM CONFIGURATION
# =============================================================================

# Supported languages
SUPPORTED_LANGUAGES = ["en", "fr"]
DEFAULT_LANGUAGE = "en"

# Risk band mappings (English - Manchester Triage)
ENGLISH_RISK_BANDS = {
    "red": {"priority": 1, "color": "#dc3545", "label": "Immediate", "wait_time": "0 min"},
    "amber": {"priority": 2, "color": "#fd7e14", "label": "Very Urgent", "wait_time": "10 min"},
    "yellow": {"priority": 3, "color": "#ffc107", "label": "Urgent", "wait_time": "60 min"},
    "green": {"priority": 4, "color": "#28a745", "label": "Standard", "wait_time": "120 min"},
    "blue": {"priority": 5, "color": "#17a2b8", "label": "Non-urgent", "wait_time": "240 min"},
}

# Risk band mappings (French - SFMU/CIMU)
FRENCH_RISK_BANDS = {
    "1": {"priority": 1, "color": "#dc3545", "label": "Urgence vitale", "wait_time": "0 min"},
    "2": {"priority": 2, "color": "#fd7e14", "label": "Très urgent", "wait_time": "20 min"},
    "3A": {"priority": 3, "color": "#ffc107", "label": "Urgent", "wait_time": "60 min"},
    "3B": {"priority": 4, "color": "#ffc107", "label": "Moins urgent", "wait_time": "120 min"},
    "4": {"priority": 5, "color": "#28a745", "label": "Non urgent", "wait_time": "180 min"},
    "5": {"priority": 6, "color": "#17a2b8", "label": "Consultations", "wait_time": "240 min"},
}

# =============================================================================
# LLM GENERATION PARAMETERS
# =============================================================================

# Default generation parameters
DEFAULT_MAX_TOKENS = 1500
DEFAULT_TEMPERATURE = 0.1
DEFAULT_TOP_P = 0.9

# PDF generation (slightly higher temperature for natural prose)
PDF_MAX_TOKENS = 2500
PDF_TEMPERATURE = 0.2
PDF_TOP_P = 0.95

# =============================================================================
# CHROMADB COLLECTION NAMES
# =============================================================================

ENGLISH_PROTOCOLS_COLLECTION = "english_protocols"
FRENCH_PROTOCOLS_COLLECTION = "french_protocols"
PATIENT_CASES_COLLECTION = "patient_cases"

# =============================================================================
# LOCALIZED STRINGS
# =============================================================================

WAITING_INSTRUCTIONS = {
    "en": {
        "red": "A healthcare provider will see you immediately. Please stay calm.",
        "amber": "You will be seen very soon. Please alert staff if symptoms worsen.",
        "yellow": "You will be seen within the hour. Please wait in the designated area.",
        "green": "Please wait to be called. This may take some time.",
        "blue": "Please wait to be called. Consider if emergency care is necessary.",
    },
    "fr": {
        "1": "Un soignant va vous voir immédiatement. Restez calme.",
        "2": "Vous serez vu très rapidement. Alertez le personnel si vos symptômes s'aggravent.",
        "3A": "Vous serez vu dans l'heure. Patientez dans la zone d'attente.",
        "3B": "Patientez en salle d'attente. L'attente peut être longue.",
        "4": "Patientez en salle d'attente. L'attente sera longue.",
        "5": "Une consultation en ville serait plus adaptée à votre situation.",
    },
}

DISCLAIMER = {
    "en": (
        "DISCLAIMER: This is a pre-screening tool and does not provide medical diagnosis. "
        "A qualified healthcare professional will review your case. If you experience "
        "worsening symptoms or a medical emergency, please alert staff immediately."
    ),
    "fr": (
        "AVERTISSEMENT: Cet outil de pré-tri ne fournit pas de diagnostic médical. "
        "Un professionnel de santé qualifié examinera votre cas. Si vos symptômes "
        "s'aggravent ou en cas d'urgence, alertez immédiatement le personnel."
    ),
}
