"""
Model Registry - Centralized configuration for supported offline LLM models.

Supports quantized GGUF models optimized for local/on-device inference.
All models are selected for:
- Small size (< 5GB)
- Medical/clinical reasoning capability
- Fast inference on CPU/GPU
- Privacy-preserving (fully offline)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum


class ModelSize(Enum):
    """Model size categories."""
    TINY = "tiny"       # < 500MB
    SMALL = "small"     # 500MB - 1GB
    MEDIUM = "medium"   # 1GB - 3GB
    LARGE = "large"     # 3GB - 5GB


class ModelQuality(Enum):
    """Model quality/capability tiers."""
    BASIC = "basic"         # Fast, simple responses
    STANDARD = "standard"   # Balanced quality/speed
    ADVANCED = "advanced"   # Best quality, slower


class PromptStyle(Enum):
    """How the model prefers to receive/format prompts."""
    THINK_TAGS = "think_tags"   # DeepSeek R1: uses <think>...</think> for reasoning
    STRUCTURED = "structured"   # Llama/Phi: follows structured instructions well
    SIMPLE = "simple"           # Basic models: need simple, direct prompts


class GrammarStrategy(Enum):
    """Strategy for constrained JSON output generation."""
    STRICT_JSON = "strict_json"       # Full JSON grammar support (Gemma, Phi, Qwen)
    THINK_THEN_JSON = "think_then_json"  # Extract <think> first, then parse JSON (DeepSeek)
    GUIDED_JSON = "guided_json"       # JSON grammar with simpler prompts (Llama, SmolLM)
    FALLBACK = "fallback"             # No grammar, regex parsing only


@dataclass
class ModelConfig:
    """Configuration for a supported model."""
    id: str                                 # Unique identifier
    name: str                               # Display name
    family: str                             # Model family (llama, gemma, deepseek, etc.)
    description: str                        # User-facing description
    filename: str                           # GGUF filename
    download_url: str                       # HuggingFace download URL
    size_bytes: int                         # Approximate size in bytes
    size_category: ModelSize                # Size tier
    quality: ModelQuality                   # Quality tier
    context_length: int                     # Max context window
    prompt_style: PromptStyle = PromptStyle.STRUCTURED  # How model handles prompts
    grammar_strategy: GrammarStrategy = GrammarStrategy.STRICT_JSON  # JSON output strategy
    recommended_threads: int = 4            # Recommended CPU threads
    recommended_gpu_layers: int = 0         # Recommended GPU layers (0 = CPU only)
    supports_medical: bool = True           # Suitable for medical summarization
    quantization: str = "Q4_K_M"            # Quantization type
    speed_rating: int = 5                   # 1-10 speed rating (10 = fastest)
    quality_rating: int = 5                 # 1-10 quality rating (10 = best)
    memory_mb: int = 1000                   # Approximate RAM usage in MB
    tags: List[str] = field(default_factory=list)


# =============================================================================
# Supported Models Registry
# =============================================================================

SUPPORTED_MODELS: Dict[str, ModelConfig] = {

    # =========================================================================
    # LLAMA FAMILY (Meta)
    # =========================================================================

    "llama-3.2-1b": ModelConfig(
        id="llama-3.2-1b",
        name="Llama 3.2 1B",
        family="llama",
        description="Meta's compact 1B model. Fast, efficient, good for basic summaries.",
        filename="llama-3.2-1b-instruct-q4_k_m.gguf",
        download_url="https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        size_bytes=750_000_000,
        size_category=ModelSize.SMALL,
        quality=ModelQuality.BASIC,
        context_length=2048,
        prompt_style=PromptStyle.SIMPLE,
        grammar_strategy=GrammarStrategy.GUIDED_JSON,
        recommended_threads=4,
        recommended_gpu_layers=0,
        quantization="Q4_K_M",
        speed_rating=9,
        quality_rating=5,
        memory_mb=800,
        tags=["fast", "lightweight", "recommended-cpu"]
    ),

    "llama-3.2-3b": ModelConfig(
        id="llama-3.2-3b",
        name="Llama 3.2 3B",
        family="llama",
        description="Meta's 3B model. Better reasoning, moderate speed.",
        filename="llama-3.2-3b-instruct-q4_k_m.gguf",
        download_url="https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        size_bytes=2_000_000_000,
        size_category=ModelSize.MEDIUM,
        quality=ModelQuality.STANDARD,
        context_length=4096,
        prompt_style=PromptStyle.SIMPLE,
        grammar_strategy=GrammarStrategy.GUIDED_JSON,
        recommended_threads=6,
        recommended_gpu_layers=20,
        quantization="Q4_K_M",
        speed_rating=7,
        quality_rating=7,
        memory_mb=2500,
        tags=["balanced", "medical-capable"]
    ),

    # =========================================================================
    # GEMMA FAMILY (Google)
    # =========================================================================

    "gemma-2-2b": ModelConfig(
        id="gemma-2-2b",
        name="Gemma 2 2B",
        family="gemma",
        description="Google's efficient 2B model. Excellent instruction following.",
        filename="gemma-2-2b-it-q4_k_m.gguf",
        download_url="https://huggingface.co/bartowski/gemma-2-2b-it-GGUF/resolve/main/gemma-2-2b-it-Q4_K_M.gguf",
        size_bytes=1_500_000_000,
        size_category=ModelSize.MEDIUM,
        quality=ModelQuality.STANDARD,
        context_length=8192,
        recommended_threads=4,
        recommended_gpu_layers=15,
        quantization="Q4_K_M",
        speed_rating=8,
        quality_rating=7,
        memory_mb=1800,
        tags=["google", "instruction-tuned", "long-context"]
    ),

    # =========================================================================
    # DEEPSEEK FAMILY
    # =========================================================================

    "deepseek-r1-1.5b": ModelConfig(
        id="deepseek-r1-1.5b",
        name="DeepSeek R1 1.5B",
        family="deepseek",
        description="DeepSeek's reasoning model. Strong logical thinking.",
        filename="deepseek-r1-distill-qwen-1.5b-q4_k_m.gguf",
        download_url="https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF/resolve/main/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf",
        size_bytes=1_100_000_000,
        size_category=ModelSize.SMALL,
        quality=ModelQuality.STANDARD,
        context_length=4096,
        prompt_style=PromptStyle.THINK_TAGS,
        grammar_strategy=GrammarStrategy.THINK_THEN_JSON,
        recommended_threads=4,
        recommended_gpu_layers=10,
        quantization="Q4_K_M",
        speed_rating=8,
        quality_rating=7,
        memory_mb=1200,
        tags=["reasoning", "chain-of-thought", "recommended"]
    ),

    "deepseek-r1-7b": ModelConfig(
        id="deepseek-r1-7b",
        name="DeepSeek R1 7B",
        family="deepseek",
        description="DeepSeek's larger reasoning model. Excellent for complex analysis.",
        filename="deepseek-r1-distill-qwen-7b-q4_k_m.gguf",
        download_url="https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF/resolve/main/DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf",
        size_bytes=4_400_000_000,
        size_category=ModelSize.LARGE,
        quality=ModelQuality.ADVANCED,
        context_length=8192,
        prompt_style=PromptStyle.THINK_TAGS,
        grammar_strategy=GrammarStrategy.THINK_THEN_JSON,
        recommended_threads=8,
        recommended_gpu_layers=35,
        quantization="Q4_K_M",
        speed_rating=5,
        quality_rating=9,
        memory_mb=5000,
        tags=["reasoning", "advanced", "gpu-recommended"]
    ),

    # =========================================================================
    # PHI FAMILY (Microsoft)
    # =========================================================================

    "phi-3.5-mini": ModelConfig(
        id="phi-3.5-mini",
        name="Phi 3.5 Mini",
        family="phi",
        description="Microsoft's compact model. Strong reasoning in small package.",
        filename="phi-3.5-mini-instruct-q4_k_m.gguf",
        download_url="https://huggingface.co/bartowski/Phi-3.5-mini-instruct-GGUF/resolve/main/Phi-3.5-mini-instruct-Q4_K_M.gguf",
        size_bytes=2_300_000_000,
        size_category=ModelSize.MEDIUM,
        quality=ModelQuality.STANDARD,
        context_length=4096,
        recommended_threads=4,
        recommended_gpu_layers=20,
        quantization="Q4_K_M",
        speed_rating=7,
        quality_rating=8,
        memory_mb=2800,
        tags=["microsoft", "reasoning", "medical-capable"]
    ),

    # =========================================================================
    # QWEN FAMILY (Alibaba)
    # =========================================================================

    "qwen2.5-1.5b": ModelConfig(
        id="qwen2.5-1.5b",
        name="Qwen 2.5 1.5B",
        family="qwen",
        description="Alibaba's efficient model. Good multilingual support.",
        filename="qwen2.5-1.5b-instruct-q4_k_m.gguf",
        download_url="https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf",
        size_bytes=1_000_000_000,
        size_category=ModelSize.SMALL,
        quality=ModelQuality.BASIC,
        context_length=4096,
        recommended_threads=4,
        recommended_gpu_layers=10,
        quantization="Q4_K_M",
        speed_rating=8,
        quality_rating=6,
        memory_mb=1100,
        tags=["multilingual", "fast", "efficient"]
    ),

    "qwen2.5-3b": ModelConfig(
        id="qwen2.5-3b",
        name="Qwen 2.5 3B",
        family="qwen",
        description="Alibaba's balanced model. Strong reasoning and multilingual.",
        filename="qwen2.5-3b-instruct-q4_k_m.gguf",
        download_url="https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf",
        size_bytes=2_000_000_000,
        size_category=ModelSize.MEDIUM,
        quality=ModelQuality.STANDARD,
        context_length=8192,
        recommended_threads=6,
        recommended_gpu_layers=20,
        quantization="Q4_K_M",
        speed_rating=7,
        quality_rating=7,
        memory_mb=2400,
        tags=["multilingual", "long-context", "balanced"]
    ),

    # =========================================================================
    # SMOLLM FAMILY (HuggingFace)
    # =========================================================================

    "smollm2-1.7b": ModelConfig(
        id="smollm2-1.7b",
        name="SmolLM2 1.7B",
        family="smollm",
        description="HuggingFace's tiny powerhouse. Ultra-fast on any hardware.",
        filename="smollm2-1.7b-instruct-q4_k_m.gguf",
        download_url="https://huggingface.co/bartowski/SmolLM2-1.7B-Instruct-GGUF/resolve/main/SmolLM2-1.7B-Instruct-Q4_K_M.gguf",
        size_bytes=1_100_000_000,
        size_category=ModelSize.SMALL,
        quality=ModelQuality.BASIC,
        context_length=2048,
        prompt_style=PromptStyle.SIMPLE,
        grammar_strategy=GrammarStrategy.GUIDED_JSON,
        recommended_threads=4,
        recommended_gpu_layers=0,
        quantization="Q4_K_M",
        speed_rating=9,
        quality_rating=5,
        memory_mb=1200,
        tags=["ultra-fast", "lightweight", "cpu-friendly"]
    ),

    # =========================================================================
    # MEDICAL SPECIALIZED (If available)
    # =========================================================================

    "medllama3-v20": ModelConfig(
        id="medllama3-v20",
        name="MedLlama3 v2.0 8B",
        family="medllama",
        description="Medical-specialized Llama. Best for clinical summaries (requires GPU).",
        filename="medllama3-v20-q4_k_m.gguf",
        download_url="https://huggingface.co/bartowski/medllama3-v20-GGUF/resolve/main/medllama3-v20-Q4_K_M.gguf",
        size_bytes=4_600_000_000,
        size_category=ModelSize.LARGE,
        quality=ModelQuality.ADVANCED,
        context_length=8192,
        recommended_threads=8,
        recommended_gpu_layers=35,
        quantization="Q4_K_M",
        speed_rating=4,
        quality_rating=10,
        memory_mb=5500,
        tags=["medical", "specialized", "clinical", "gpu-required"]
    ),
}


# =============================================================================
# Default Model Selection
# =============================================================================

DEFAULT_MODEL_ID = "deepseek-r1-1.5b"  # Best quality on CPU, strong reasoning
RECOMMENDED_CPU_MODEL = "deepseek-r1-1.5b"  # Best quality on CPU
RECOMMENDED_GPU_MODEL = "deepseek-r1-7b"    # Best quality with GPU
FALLBACK_MODEL_ID = "llama-3.2-1b"  # Fast fallback option


# =============================================================================
# Helper Functions
# =============================================================================

def get_model_config(model_id: str) -> Optional[ModelConfig]:
    """Get configuration for a specific model."""
    return SUPPORTED_MODELS.get(model_id)


def get_all_models() -> List[ModelConfig]:
    """Get all supported models."""
    return list(SUPPORTED_MODELS.values())


def get_models_by_family(family: str) -> List[ModelConfig]:
    """Get all models from a specific family."""
    return [m for m in SUPPORTED_MODELS.values() if m.family == family]


def get_models_by_size(size: ModelSize) -> List[ModelConfig]:
    """Get all models of a specific size category."""
    return [m for m in SUPPORTED_MODELS.values() if m.size_category == size]


def get_models_by_quality(quality: ModelQuality) -> List[ModelConfig]:
    """Get all models of a specific quality tier."""
    return [m for m in SUPPORTED_MODELS.values() if m.quality == quality]


def get_recommended_model(has_gpu: bool = False, max_memory_mb: int = 4000) -> ModelConfig:
    """Get recommended model based on hardware capabilities."""
    if has_gpu and max_memory_mb >= 5000:
        return SUPPORTED_MODELS.get(RECOMMENDED_GPU_MODEL, SUPPORTED_MODELS[DEFAULT_MODEL_ID])
    elif max_memory_mb >= 2000:
        return SUPPORTED_MODELS.get(RECOMMENDED_CPU_MODEL, SUPPORTED_MODELS[DEFAULT_MODEL_ID])
    else:
        return SUPPORTED_MODELS[DEFAULT_MODEL_ID]


def get_model_families() -> List[str]:
    """Get list of unique model families."""
    return list(set(m.family for m in SUPPORTED_MODELS.values()))


def model_to_dict(config: ModelConfig) -> Dict:
    """Convert ModelConfig to dictionary for API responses."""
    return {
        "id": config.id,
        "name": config.name,
        "family": config.family,
        "description": config.description,
        "filename": config.filename,
        "download_url": config.download_url,
        "size_mb": round(config.size_bytes / 1_000_000),
        "size_category": config.size_category.value,
        "quality": config.quality.value,
        "context_length": config.context_length,
        "prompt_style": config.prompt_style.value,
        "grammar_strategy": config.grammar_strategy.value,
        "quantization": config.quantization,
        "speed_rating": config.speed_rating,
        "quality_rating": config.quality_rating,
        "memory_mb": config.memory_mb,
        "tags": config.tags,
        "recommended_threads": config.recommended_threads,
        "recommended_gpu_layers": config.recommended_gpu_layers,
    }
