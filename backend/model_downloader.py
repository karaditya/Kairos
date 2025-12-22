"""
Model Download Service - HTTP API for downloading models with progress tracking.

Supports:
- GGUF models (Llama, DeepSeek, Gemma, etc.)
- HuggingFace Transformers models (DrBERT)
- Progress streaming via Server-Sent Events
- Automatic GPU/CPU detection
"""

import os
import gc
import threading
import asyncio
from pathlib import Path
from typing import Dict, Optional, Generator, Any
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# =============================================================================
# GPU Detection Utilities
# =============================================================================

def detect_gpu_capabilities() -> Dict[str, Any]:
    """
    Detect GPU capabilities for automatic model layer distribution.
    Returns info about available VRAM and recommended settings.
    """
    result = {
        "cuda_available": False,
        "gpu_name": None,
        "total_vram_mb": 0,
        "free_vram_mb": 0,
        "recommended_gpu_layers": 0,
        "can_use_gpu": False
    }

    try:
        import torch
        if torch.cuda.is_available():
            result["cuda_available"] = True
            result["gpu_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            total_mb = props.total_memory // (1024 * 1024)

            # Get free memory
            torch.cuda.init()
            reserved = torch.cuda.memory_reserved(0) // (1024 * 1024)
            free_mb = total_mb - reserved

            result["total_vram_mb"] = total_mb
            result["free_vram_mb"] = free_mb
            result["can_use_gpu"] = free_mb > 500  # At least 500MB free

            # Calculate recommended GPU layers based on VRAM
            # Rough estimate: ~100MB per layer for 7B model, ~50MB for smaller
            if free_mb > 6000:
                result["recommended_gpu_layers"] = -1  # All layers
            elif free_mb > 4000:
                result["recommended_gpu_layers"] = 35
            elif free_mb > 2000:
                result["recommended_gpu_layers"] = 20
            elif free_mb > 1000:
                result["recommended_gpu_layers"] = 10
            elif free_mb > 500:
                result["recommended_gpu_layers"] = 5
            else:
                result["recommended_gpu_layers"] = 0

    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"GPU detection error: {e}")

    return result


def compute_optimal_gpu_layers(model_size_mb: int, vram_free_mb: int) -> int:
    """
    Compute optimal GPU layers for a given model size and available VRAM.

    Strategy:
    - Reserve 300MB for system overhead
    - Estimate layers based on model size
    - Return -1 if full model fits, else partial layers
    """
    if vram_free_mb < 500:
        return 0  # Not enough VRAM

    usable_vram = vram_free_mb - 300  # Safety margin

    if usable_vram >= model_size_mb * 1.2:
        return -1  # Full model fits on GPU

    # Estimate: assume 32 layers for typical model
    # Each layer takes roughly model_size / 32
    layer_size_estimate = model_size_mb / 32

    if layer_size_estimate > 0:
        layers = int(usable_vram / layer_size_estimate)
        return min(layers, 35)  # Cap at 35 layers

    return 0


# =============================================================================
# Download Progress Tracking
# =============================================================================

@dataclass
class DownloadProgress:
    """Track download progress for a model."""
    model_id: str
    model_type: str  # "gguf" or "drbert"
    status: str  # "pending", "downloading", "completed", "failed", "loading"
    progress: float  # 0.0 to 1.0
    downloaded_mb: float
    total_mb: float
    speed_mbps: float
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


# Global download state
_download_state: Dict[str, DownloadProgress] = {}
_download_lock = threading.Lock()


def get_download_progress(model_id: str) -> Optional[DownloadProgress]:
    """Get current download progress for a model."""
    with _download_lock:
        return _download_state.get(model_id)


def get_all_downloads() -> Dict[str, DownloadProgress]:
    """Get all download progress states."""
    with _download_lock:
        return dict(_download_state)


# =============================================================================
# GGUF Model Download
# =============================================================================

def download_gguf_model(
    model_id: str,
    download_url: str,
    filename: str,
    expected_size_bytes: int,
    models_dir: str
) -> Generator[Dict[str, Any], None, bool]:
    """
    Download a GGUF model with progress updates.
    Yields progress dictionaries for SSE streaming.
    Returns True on success.
    """
    import requests

    dest_path = Path(models_dir) / filename
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize progress
    progress = DownloadProgress(
        model_id=model_id,
        model_type="gguf",
        status="downloading",
        progress=0.0,
        downloaded_mb=0.0,
        total_mb=expected_size_bytes / (1024 * 1024),
        speed_mbps=0.0,
        started_at=datetime.now()
    )

    with _download_lock:
        _download_state[model_id] = progress

    yield {"type": "start", "model_id": model_id, "total_mb": progress.total_mb}

    try:
        response = requests.get(download_url, stream=True, timeout=30)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', expected_size_bytes))
        progress.total_mb = total_size / (1024 * 1024)

        downloaded = 0
        last_update = datetime.now()
        last_downloaded = 0
        chunk_size = 1024 * 1024  # 1MB chunks

        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)

                    # Calculate speed
                    now = datetime.now()
                    elapsed = (now - last_update).total_seconds()
                    if elapsed >= 0.5:  # Update every 0.5s
                        bytes_since = downloaded - last_downloaded
                        speed_mbps = (bytes_since / elapsed) / (1024 * 1024)

                        progress.downloaded_mb = downloaded / (1024 * 1024)
                        progress.progress = downloaded / total_size
                        progress.speed_mbps = speed_mbps

                        with _download_lock:
                            _download_state[model_id] = progress

                        yield {
                            "type": "progress",
                            "model_id": model_id,
                            "progress": progress.progress,
                            "downloaded_mb": progress.downloaded_mb,
                            "total_mb": progress.total_mb,
                            "speed_mbps": round(speed_mbps, 2)
                        }

                        last_update = now
                        last_downloaded = downloaded

        progress.status = "completed"
        progress.progress = 1.0
        progress.completed_at = datetime.now()

        with _download_lock:
            _download_state[model_id] = progress

        yield {"type": "complete", "model_id": model_id, "success": True}
        return True

    except Exception as e:
        progress.status = "failed"
        progress.error = str(e)

        with _download_lock:
            _download_state[model_id] = progress

        # Clean up partial download
        if dest_path.exists():
            dest_path.unlink()

        yield {"type": "error", "model_id": model_id, "error": str(e)}
        return False


# =============================================================================
# DrBERT Model Download
# =============================================================================

def download_drbert_model(
    model_id: str,
    hf_model_id: str,
    cache_dir: str
) -> Generator[Dict[str, Any], None, bool]:
    """
    Download a DrBERT model from HuggingFace.
    Yields progress dictionaries for SSE streaming.
    """
    progress = DownloadProgress(
        model_id=model_id,
        model_type="drbert",
        status="downloading",
        progress=0.0,
        downloaded_mb=0.0,
        total_mb=440.0,  # Approximate size
        speed_mbps=0.0,
        started_at=datetime.now()
    )

    with _download_lock:
        _download_state[model_id] = progress

    yield {"type": "start", "model_id": model_id, "total_mb": progress.total_mb}

    try:
        from transformers import AutoModel, AutoTokenizer

        # Download tokenizer
        progress.status = "downloading"
        progress.progress = 0.2
        yield {"type": "progress", "model_id": model_id, "progress": 0.2, "status": "Downloading tokenizer..."}

        AutoTokenizer.from_pretrained(hf_model_id, cache_dir=cache_dir)

        # Download model
        progress.progress = 0.5
        yield {"type": "progress", "model_id": model_id, "progress": 0.5, "status": "Downloading model weights..."}

        AutoModel.from_pretrained(hf_model_id, cache_dir=cache_dir)

        progress.status = "completed"
        progress.progress = 1.0
        progress.completed_at = datetime.now()

        with _download_lock:
            _download_state[model_id] = progress

        yield {"type": "complete", "model_id": model_id, "success": True}
        return True

    except Exception as e:
        progress.status = "failed"
        progress.error = str(e)

        with _download_lock:
            _download_state[model_id] = progress

        yield {"type": "error", "model_id": model_id, "error": str(e)}
        return False


# =============================================================================
# Unified Download & Load
# =============================================================================

def download_and_load_model(
    model_id: str,
    model_type: str,  # "gguf" or "drbert"
    config: Dict[str, Any],
    models_dir: str,
    auto_load: bool = True
) -> Generator[Dict[str, Any], None, Dict[str, Any]]:
    """
    Download a model and optionally load it with auto GPU/CPU distribution.

    Args:
        model_id: Model identifier
        model_type: "gguf" or "drbert"
        config: Model configuration dict
        models_dir: Directory for model files
        auto_load: Whether to load after download

    Yields:
        Progress updates for streaming

    Returns:
        Final result dict with success status and GPU info
    """
    result = {
        "success": False,
        "model_id": model_id,
        "model_type": model_type,
        "gpu_info": None,
        "error": None
    }

    # Detect GPU capabilities first
    gpu_info = detect_gpu_capabilities()
    result["gpu_info"] = gpu_info

    yield {"type": "gpu_detected", "gpu_info": gpu_info}

    # Download based on type
    if model_type == "gguf":
        download_gen = download_gguf_model(
            model_id=model_id,
            download_url=config.get("download_url"),
            filename=config.get("filename"),
            expected_size_bytes=config.get("size_bytes", 1_000_000_000),
            models_dir=models_dir
        )
    elif model_type == "drbert":
        download_gen = download_drbert_model(
            model_id=model_id,
            hf_model_id=config.get("hf_model_id"),
            cache_dir=os.path.join(models_dir, "drbert_cache")
        )
    else:
        yield {"type": "error", "error": f"Unknown model type: {model_type}"}
        result["error"] = f"Unknown model type: {model_type}"
        return result

    # Stream download progress
    download_success = False
    for update in download_gen:
        yield update
        if update.get("type") == "complete":
            download_success = update.get("success", False)

    if not download_success:
        result["error"] = "Download failed"
        return result

    # Load model if requested
    if auto_load:
        yield {"type": "loading", "model_id": model_id, "status": "Loading model..."}

        try:
            if model_type == "gguf":
                # Import and load GGUF model
                from multi_model_engine import get_engine

                engine = get_engine(models_dir=models_dir)

                # Calculate optimal GPU layers
                model_size_mb = config.get("size_bytes", 0) / (1024 * 1024)
                optimal_layers = compute_optimal_gpu_layers(
                    model_size_mb,
                    gpu_info.get("free_vram_mb", 0)
                )

                # Update environment for this load
                if optimal_layers != 0:
                    os.environ["N_GPU_LAYERS"] = str(optimal_layers)

                yield {
                    "type": "loading",
                    "model_id": model_id,
                    "status": f"Loading with {optimal_layers} GPU layers..."
                }

                success = engine.load_model(model_id)
                result["success"] = success
                result["gpu_layers_used"] = optimal_layers

            elif model_type == "drbert":
                from drbert_engine import get_drbert_engine

                engine = get_drbert_engine()
                success = engine.load_model(model_id)
                result["success"] = success

                if success:
                    current = engine.get_current_model()
                    result["device"] = current.get("device") if current else "unknown"

            if result["success"]:
                yield {"type": "loaded", "model_id": model_id, "result": result}
            else:
                yield {"type": "error", "model_id": model_id, "error": "Failed to load model"}
                result["error"] = "Failed to load model"

        except Exception as e:
            result["error"] = str(e)
            yield {"type": "error", "model_id": model_id, "error": str(e)}
    else:
        result["success"] = True

    return result


# =============================================================================
# Model Status Check
# =============================================================================

def check_model_downloaded(model_id: str, model_type: str, config: Dict, models_dir: str) -> bool:
    """Check if a model is already downloaded."""
    if model_type == "gguf":
        path = Path(models_dir) / config.get("filename", "")
        return path.exists()
    elif model_type == "drbert":
        # Check HuggingFace cache
        cache_dir = Path(models_dir) / "drbert_cache"
        hf_model_id = config.get("hf_model_id", "")
        # HF cache structure: models--org--name
        cache_name = f"models--{hf_model_id.replace('/', '--')}"
        return (cache_dir / cache_name).exists()
    return False
