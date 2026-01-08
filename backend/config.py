"""
Clinical Admin Edge (CAE) System - Centralized Configuration

All configuration for the Ambient Admin Assistant system.
100% local/offline - all models run on localhost.
"""

import os
from pathlib import Path

# =============================================================================
# PATHS
# =============================================================================

BASE_DIR = Path(__file__).parent.absolute()
DATA_DIR = BASE_DIR / "data"
CONFIG_DIR = BASE_DIR / "config"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
(DATA_DIR / "qdrant").mkdir(exist_ok=True)
(DATA_DIR / "sqlite").mkdir(exist_ok=True)
(DATA_DIR / "coordinate_maps").mkdir(exist_ok=True)
(CONFIG_DIR / "protocols").mkdir(parents=True, exist_ok=True)

# =============================================================================
# DATABASE
# =============================================================================

DATABASE_PATH = os.environ.get("DATABASE_PATH", str(DATA_DIR / "sqlite" / "cae.db"))

# =============================================================================
# OLLAMA / LLM
# =============================================================================

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "mistral")
OLLAMA_TIMEOUT = int(os.environ.get("OLLAMA_TIMEOUT", "300"))
OLLAMA_EMBEDDING_MODEL = os.environ.get("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")

# Supported models for UI selection
SUPPORTED_OLLAMA_MODELS = {
    "mistral": {
        "name": "Mistral 7B",
        "size_gb": 4.1,
        "description": "Fast, balanced - good default choice",
    },
    "mistral:7b-instruct": {
        "name": "Mistral 7B Instruct",
        "size_gb": 4.1,
        "description": "Instruction-tuned for better chat",
    },
    "llama3.2": {
        "name": "Llama 3.2 3B",
        "size_gb": 2.0,
        "description": "Meta's compact model - fast",
    },
    "llama3.2:1b": {
        "name": "Llama 3.2 1B",
        "size_gb": 1.3,
        "description": "Ultra-compact for low resources",
    },
    "deepseek-r1:7b": {
        "name": "DeepSeek R1 7B",
        "size_gb": 4.7,
        "description": "Strong reasoning capabilities",
    },
    "qwen2.5": {
        "name": "Qwen 2.5 7B",
        "size_gb": 4.4,
        "description": "Multilingual - good for French",
    },
    "qwen2.5:3b": {
        "name": "Qwen 2.5 3B",
        "size_gb": 1.9,
        "description": "Compact multilingual",
    },
}

# =============================================================================
# PARLANT
# =============================================================================

PARLANT_PORT = int(os.environ.get("PARLANT_PORT", "8800"))
PARLANT_RESPONSE_TIMEOUT = float(os.environ.get("PARLANT_RESPONSE_TIMEOUT", "60.0"))
PARLANT_STARTUP_TIMEOUT = int(os.environ.get("PARLANT_STARTUP_TIMEOUT", "120"))

# =============================================================================
# VISION / DEEPSEEK-VL2
# =============================================================================

DEEPSEEK_VL2_MODEL = os.environ.get("DEEPSEEK_VL2_MODEL", "deepseek-ai/deepseek-vl2-tiny")
DEEPSEEK_DEVICE = os.environ.get("DEEPSEEK_DEVICE", "auto")  # "cuda", "cpu", "auto"
VISION_MAX_IMAGE_SIZE = int(os.environ.get("VISION_MAX_IMAGE_SIZE", "2048"))
VISION_ENABLED = os.environ.get("VISION_ENABLED", "true").lower() == "true"

# =============================================================================
# AUDIO / FASTER-WHISPER
# =============================================================================

WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "large-v3")
WHISPER_DEVICE = os.environ.get("WHISPER_DEVICE", "auto")  # "cuda", "cpu", "auto"
WHISPER_COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE_TYPE", "float16")
WHISPER_LANGUAGE = os.environ.get("WHISPER_LANGUAGE", "fr")
AUDIO_SAMPLE_RATE = int(os.environ.get("AUDIO_SAMPLE_RATE", "16000"))
AUDIO_CHUNK_DURATION_MS = int(os.environ.get("AUDIO_CHUNK_DURATION_MS", "3000"))
AUDIO_ENABLED = os.environ.get("AUDIO_ENABLED", "true").lower() == "true"

# Medical keywords to tag in transcripts
MEDICAL_KEYWORDS = {
    "fr": [
        "douleur", "fievre", "temperature", "tension", "pouls", "saturation",
        "allergie", "antecedent", "traitement", "medicament", "symptome",
        "diagnostic", "examen", "analyse", "radiographie", "scanner",
        "urgence", "hospitalisation", "chirurgie", "intervention",
    ],
    "en": [
        "pain", "fever", "temperature", "blood pressure", "pulse", "saturation",
        "allergy", "history", "treatment", "medication", "symptom",
        "diagnosis", "exam", "test", "x-ray", "scan",
        "emergency", "hospitalization", "surgery", "procedure",
    ],
}

# =============================================================================
# RPA / PYAUTOGUI
# =============================================================================

RPA_ENABLED = os.environ.get("RPA_ENABLED", "true").lower() == "true"
RPA_TYPING_INTERVAL = float(os.environ.get("RPA_TYPING_INTERVAL", "0.02"))
RPA_CLICK_PAUSE = float(os.environ.get("RPA_CLICK_PAUSE", "0.1"))
RPA_FAILSAFE = os.environ.get("RPA_FAILSAFE", "true").lower() == "true"

# Human-in-the-loop is ALWAYS required - non-configurable safety
REQUIRE_HUMAN_VERIFICATION = True

# =============================================================================
# QDRANT VECTOR DB
# =============================================================================

QDRANT_PATH = os.environ.get("QDRANT_PATH", str(DATA_DIR / "qdrant"))
QDRANT_COLLECTION_PROTOCOLS = "triage_protocols"
QDRANT_COLLECTION_PATIENTS = "patient_history"
EMBEDDING_DIMENSION = int(os.environ.get("EMBEDDING_DIMENSION", "768"))
RAG_ENABLED = os.environ.get("RAG_ENABLED", "true").lower() == "true"

# =============================================================================
# SUPPORTED LANGUAGES
# =============================================================================

SUPPORTED_LANGUAGES = ["fr", "en"]
DEFAULT_LANGUAGE = "fr"

# =============================================================================
# LLM GENERATION PARAMETERS
# =============================================================================

DEFAULT_MAX_TOKENS = 2000
DEFAULT_TEMPERATURE = 0.1
DEFAULT_TOP_P = 0.9

# =============================================================================
# STAFF AUTHENTICATION
# =============================================================================

STAFF_PIN = os.environ.get("STAFF_PIN", "1234")

# =============================================================================
# FEATURE FLAGS
# =============================================================================

DEBUG_MODE = os.environ.get("DEBUG_MODE", "false").lower() == "true"

# =============================================================================
# COMPTE RENDU SECTIONS (French Clinical Report)
# =============================================================================

COMPTE_RENDU_SECTIONS = [
    "motif_consultation",      # Reason for visit
    "anamnese",                # Patient history
    "antecedents",             # Medical history
    "allergies",               # Allergies
    "traitements_actuels",     # Current treatments
    "examen_clinique",         # Clinical examination
    "examens_complementaires", # Additional tests
    "hypotheses_diagnostiques", # Diagnostic hypotheses
    "plan_therapeutique",      # Treatment plan
]

# Required fields that trigger MISSING alerts
REQUIRED_CR_FIELDS = [
    "motif_consultation",
    "allergies",
    "examen_clinique",
]

# =============================================================================
# DISCLAIMER
# =============================================================================

ADMINISTRATIVE_DISCLAIMER = {
    "fr": (
        "Ce document a ete genere par un assistant administratif automatise. "
        "Il ne constitue pas un avis medical. Toutes les sections marquees "
        "[A VALIDER] necessitent une verification par un professionnel de sante."
    ),
    "en": (
        "This document was generated by an automated administrative assistant. "
        "It does not constitute medical advice. All sections marked "
        "[TO VALIDATE] require verification by a healthcare professional."
    ),
}
