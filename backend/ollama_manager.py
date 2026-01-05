"""
Ollama Model Manager - Clean model detection and management for Medical Triage MVP

Handles:
- Detecting models in local models/ directory (Ready to use)
- Listing models available in Ollama (with pulled status)
- Pulling models with progress streaming
- Switching active model for Parlant agents
"""

import os
import asyncio
import logging
import httpx
from typing import Dict, List, Optional, AsyncGenerator
from dataclasses import dataclass, asdict
from pathlib import Path

from config import (
    OLLAMA_BASE_URL,
    OLLAMA_TIMEOUT,
    MODELS_DIR,
)

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ModelInfo:
    """Information about a model."""
    id: str
    name: str
    size_gb: float
    description: str
    status: str  # 'ready', 'available', 'downloading', 'not_found'
    source: str  # 'local', 'ollama'
    is_active: bool = False

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class PullProgress:
    """Progress information for model pulling."""
    model_id: str
    status: str  # 'downloading', 'verifying', 'complete', 'error'
    progress: float  # 0.0 to 1.0
    downloaded_gb: float
    total_gb: float
    message: str

    def to_dict(self) -> Dict:
        return asdict(self)


# =============================================================================
# SUPPORTED MODELS REGISTRY
# =============================================================================

SUPPORTED_MODELS = {
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
        "description": "Alibaba's multilingual model",
    },
    "qwen2.5:3b": {
        "name": "Qwen 2.5 3B",
        "size_gb": 1.9,
        "description": "Compact Qwen - good for French",
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
# OLLAMA MANAGER CLASS
# =============================================================================

class OllamaManager:
    """
    Manages Ollama models for the Medical Triage system.

    Responsibilities:
    - Detect local GGUF models in models/ directory
    - List Ollama models with their pulled/available status
    - Pull new models with progress streaming
    - Track and switch active model
    """

    def __init__(self):
        self._active_model: str = "mistral"
        self._ollama_available: bool = False
        self._pulled_models: set = set()
        self._http_client: Optional[httpx.AsyncClient] = None

    async def initialize(self) -> bool:
        """
        Initialize the manager and check Ollama availability.

        Returns:
            True if Ollama is available, False otherwise
        """
        self._http_client = httpx.AsyncClient(timeout=30.0)
        self._ollama_available = await self._check_ollama_available()

        if self._ollama_available:
            await self._refresh_pulled_models()
            logger.info(f"Ollama available with {len(self._pulled_models)} models pulled")
        else:
            logger.warning("Ollama not available - model features limited")

        return self._ollama_available

    async def shutdown(self):
        """Clean up resources."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

    # =========================================================================
    # MODEL LISTING
    # =========================================================================

    async def get_all_models(self) -> List[ModelInfo]:
        """
        Get all models - both local and Ollama - with status.

        Returns list with:
        - Local models in models/ dir -> status='ready', source='local'
        - Pulled Ollama models -> status='ready', source='ollama'
        - Available (not pulled) -> status='available', source='ollama'
        """
        models = []

        # Get local GGUF models
        local_models = self._get_local_models()
        models.extend(local_models)

        # Get Ollama models
        ollama_models = await self._get_ollama_models()
        models.extend(ollama_models)

        # Mark active model
        for model in models:
            model.is_active = (model.id == self._active_model)

        return models

    async def get_ready_models(self) -> List[ModelInfo]:
        """Get only models that are ready to use (local + pulled Ollama)."""
        all_models = await self.get_all_models()
        return [m for m in all_models if m.status == 'ready']

    async def get_available_models(self) -> List[ModelInfo]:
        """Get models available for download (not yet pulled)."""
        all_models = await self.get_all_models()
        return [m for m in all_models if m.status == 'available']

    def _get_local_models(self) -> List[ModelInfo]:
        """Scan models/ directory for local GGUF files."""
        models = []
        models_path = Path(MODELS_DIR)

        if not models_path.exists():
            logger.debug(f"Models directory not found: {models_path}")
            return models

        # Look for GGUF files
        for file in models_path.glob("*.gguf"):
            size_gb = file.stat().st_size / (1024 ** 3)
            model_id = file.stem  # filename without extension

            models.append(ModelInfo(
                id=f"local:{model_id}",
                name=model_id.replace("-", " ").replace("_", " ").title(),
                size_gb=round(size_gb, 1),
                description=f"Local GGUF model",
                status="ready",
                source="local",
            ))

        # Look for subdirectories (HuggingFace style)
        for subdir in models_path.iterdir():
            if subdir.is_dir() and not subdir.name.startswith('.'):
                # Check if it has model files
                has_model = any(
                    subdir.glob("*.bin") or
                    subdir.glob("*.safetensors") or
                    subdir.glob("*.gguf")
                )
                if has_model:
                    # Estimate size from all files
                    total_size = sum(f.stat().st_size for f in subdir.rglob("*") if f.is_file())
                    size_gb = total_size / (1024 ** 3)

                    models.append(ModelInfo(
                        id=f"local:{subdir.name}",
                        name=subdir.name.replace("-", " ").replace("_", " ").title(),
                        size_gb=round(size_gb, 1),
                        description=f"Local model directory",
                        status="ready",
                        source="local",
                    ))

        return models

    async def _get_ollama_models(self) -> List[ModelInfo]:
        """Get Ollama models with their status."""
        models = []

        # Refresh pulled models list
        if self._ollama_available:
            await self._refresh_pulled_models()

        for model_id, info in SUPPORTED_MODELS.items():
            is_pulled = model_id in self._pulled_models or self._is_model_variant_pulled(model_id)

            models.append(ModelInfo(
                id=model_id,
                name=info["name"],
                size_gb=info["size_gb"],
                description=info["description"],
                status="ready" if is_pulled else "available",
                source="ollama",
            ))

        return models

    def _is_model_variant_pulled(self, model_id: str) -> bool:
        """Check if model or a variant of it is pulled."""
        base_name = model_id.split(":")[0]
        for pulled in self._pulled_models:
            if pulled.startswith(base_name):
                return True
        return False

    # =========================================================================
    # MODEL PULLING
    # =========================================================================

    async def pull_model(self, model_id: str) -> AsyncGenerator[PullProgress, None]:
        """
        Pull a model from Ollama with progress streaming.

        Yields PullProgress objects with download status.
        """
        if not self._ollama_available:
            yield PullProgress(
                model_id=model_id,
                status="error",
                progress=0.0,
                downloaded_gb=0.0,
                total_gb=0.0,
                message="Ollama is not available"
            )
            return

        try:
            url = f"{OLLAMA_BASE_URL}/api/pull"

            async with self._http_client.stream(
                "POST",
                url,
                json={"name": model_id},
                timeout=None  # No timeout for large downloads
            ) as response:
                total_size = 0
                completed_size = 0

                async for line in response.aiter_lines():
                    if not line:
                        continue

                    import json
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    status = data.get("status", "")

                    if "total" in data:
                        total_size = data["total"]
                    if "completed" in data:
                        completed_size = data["completed"]

                    progress = (completed_size / total_size) if total_size > 0 else 0.0

                    if "error" in data:
                        yield PullProgress(
                            model_id=model_id,
                            status="error",
                            progress=progress,
                            downloaded_gb=completed_size / (1024**3),
                            total_gb=total_size / (1024**3),
                            message=data["error"]
                        )
                        return

                    # Determine status
                    if "pulling" in status.lower():
                        pull_status = "downloading"
                    elif "verifying" in status.lower():
                        pull_status = "verifying"
                    elif "success" in status.lower() or data.get("status") == "success":
                        pull_status = "complete"
                    else:
                        pull_status = "downloading"

                    yield PullProgress(
                        model_id=model_id,
                        status=pull_status,
                        progress=progress,
                        downloaded_gb=completed_size / (1024**3),
                        total_gb=total_size / (1024**3),
                        message=status
                    )

                    if pull_status == "complete":
                        self._pulled_models.add(model_id)
                        return

            # If we get here without complete, mark as complete
            self._pulled_models.add(model_id)
            yield PullProgress(
                model_id=model_id,
                status="complete",
                progress=1.0,
                downloaded_gb=0.0,
                total_gb=0.0,
                message="Model pulled successfully"
            )

        except Exception as e:
            logger.error(f"Error pulling model {model_id}: {e}")
            yield PullProgress(
                model_id=model_id,
                status="error",
                progress=0.0,
                downloaded_gb=0.0,
                total_gb=0.0,
                message=str(e)
            )

    # =========================================================================
    # MODEL SELECTION
    # =========================================================================

    def get_current_model(self) -> str:
        """Get the currently active model ID."""
        return self._active_model

    async def set_model(self, model_id: str) -> bool:
        """
        Set the active model for generation.

        Args:
            model_id: Model ID to activate

        Returns:
            True if model was set successfully
        """
        # Refresh pulled models
        if self._ollama_available:
            await self._refresh_pulled_models()

        # Check if model is available
        if model_id.startswith("local:"):
            # Local model - just set it
            self._active_model = model_id
            logger.info(f"Active model set to: {model_id}")
            return True

        # Ollama model - check if pulled
        if model_id in self._pulled_models or self._is_model_variant_pulled(model_id):
            self._active_model = model_id
            logger.info(f"Active model set to: {model_id}")
            return True

        logger.warning(f"Model {model_id} not available - not pulled")
        return False

    async def check_model_available(self, model_id: str) -> bool:
        """Check if a model is available for use (pulled or local)."""
        if model_id.startswith("local:"):
            # Check local path
            local_name = model_id.replace("local:", "")
            local_path = Path(MODELS_DIR) / f"{local_name}.gguf"
            local_dir = Path(MODELS_DIR) / local_name
            return local_path.exists() or local_dir.exists()

        # Check Ollama
        await self._refresh_pulled_models()
        return model_id in self._pulled_models or self._is_model_variant_pulled(model_id)

    # =========================================================================
    # OLLAMA COMMUNICATION
    # =========================================================================

    async def _check_ollama_available(self) -> bool:
        """Check if Ollama server is running and responsive."""
        try:
            response = await self._http_client.get(
                f"{OLLAMA_BASE_URL}/api/tags",
                timeout=5.0
            )
            return response.status_code == 200
        except Exception as e:
            logger.debug(f"Ollama not available: {e}")
            return False

    async def _refresh_pulled_models(self):
        """Refresh the list of pulled models from Ollama."""
        if not self._ollama_available:
            return

        try:
            response = await self._http_client.get(
                f"{OLLAMA_BASE_URL}/api/tags",
                timeout=10.0
            )

            if response.status_code == 200:
                data = response.json()
                self._pulled_models = {
                    model["name"] for model in data.get("models", [])
                }
                logger.debug(f"Pulled models: {self._pulled_models}")
        except Exception as e:
            logger.error(f"Error refreshing pulled models: {e}")

    async def generate_test(self, model_id: str, prompt: str = "Hello") -> Optional[str]:
        """
        Test model generation with a simple prompt.

        Returns generated text or None if failed.
        """
        if not self._ollama_available:
            return None

        try:
            response = await self._http_client.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": model_id,
                    "prompt": prompt,
                    "stream": False,
                },
                timeout=60.0
            )

            if response.status_code == 200:
                data = response.json()
                return data.get("response", "")
            return None
        except Exception as e:
            logger.error(f"Error testing model {model_id}: {e}")
            return None

    # =========================================================================
    # PROPERTIES
    # =========================================================================

    @property
    def is_available(self) -> bool:
        """Check if Ollama is available."""
        return self._ollama_available

    @property
    def active_model(self) -> str:
        """Get active model ID."""
        return self._active_model

    @property
    def pulled_models(self) -> set:
        """Get set of pulled model IDs."""
        return self._pulled_models.copy()


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_manager_instance: Optional[OllamaManager] = None


async def get_ollama_manager() -> OllamaManager:
    """Get singleton OllamaManager instance."""
    global _manager_instance

    if _manager_instance is None:
        _manager_instance = OllamaManager()
        await _manager_instance.initialize()

    return _manager_instance


def get_ollama_manager_sync() -> OllamaManager:
    """Get OllamaManager instance synchronously (must call initialize separately)."""
    global _manager_instance

    if _manager_instance is None:
        _manager_instance = OllamaManager()

    return _manager_instance
