#!/usr/bin/env python3
"""
Model Download Utility for Medical AI Triage System

Download and manage quantized GGUF models for offline inference.
All models are optimized for local CPU/GPU execution.

Usage:
    python download_models.py                    # Interactive mode
    python download_models.py --list             # List all available models
    python download_models.py --download MODEL   # Download specific model
    python download_models.py --all              # Download all models
    python download_models.py --recommended      # Download recommended models

Author: Medical AI Triage Team
"""

import os
import sys
import argparse
import hashlib
from pathlib import Path
from typing import Optional

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

try:
    from model_registry import (
        SUPPORTED_MODELS,
        ModelConfig,
        get_all_models,
        get_model_config,
        DEFAULT_MODEL_ID,
        RECOMMENDED_CPU_MODEL,
        RECOMMENDED_GPU_MODEL,
    )
except ImportError:
    print("Error: Could not import model_registry. Run from project root.")
    sys.exit(1)

# Try to import progress bar library
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


# =============================================================================
# Configuration
# =============================================================================

MODELS_DIR = Path(__file__).parent.parent / "models"
CHUNK_SIZE = 8192  # Download chunk size


# =============================================================================
# Helper Functions
# =============================================================================

def format_size(size_bytes: int) -> str:
    """Format bytes as human readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def print_header():
    """Print script header."""
    print()
    print("=" * 60)
    print("   Medical AI Triage - Model Download Utility")
    print("=" * 60)
    print()


def print_model_info(config: ModelConfig, is_downloaded: bool = False):
    """Print detailed model information."""
    status = "[DOWNLOADED]" if is_downloaded else "[NOT DOWNLOADED]"
    status_color = "\033[92m" if is_downloaded else "\033[93m"
    reset = "\033[0m"

    print(f"\n{status_color}{status}{reset} {config.name} ({config.id})")
    print(f"   Family: {config.family.capitalize()}")
    print(f"   Description: {config.description}")
    print(f"   Size: {format_size(config.size_bytes)}")
    print(f"   Quality: {config.quality_rating}/10 | Speed: {config.speed_rating}/10")
    print(f"   Context: {config.context_length} tokens")
    print(f"   Quantization: {config.quantization}")
    print(f"   Memory Required: ~{config.memory_mb} MB")
    print(f"   Tags: {', '.join(config.tags)}")


def model_exists(config: ModelConfig) -> bool:
    """Check if model file exists."""
    model_path = MODELS_DIR / config.filename
    return model_path.exists()


def get_downloaded_models() -> list:
    """Get list of downloaded model IDs."""
    return [
        model_id for model_id, config in SUPPORTED_MODELS.items()
        if model_exists(config)
    ]


# =============================================================================
# Download Functions
# =============================================================================

def download_with_progress(url: str, dest_path: Path, expected_size: int) -> bool:
    """Download file with progress bar."""
    if not HAS_REQUESTS:
        print("Error: 'requests' library not installed.")
        print("Install with: pip install requests")
        return False

    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', expected_size))

        # Create models directory if needed
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        # Download with progress
        if HAS_TQDM:
            progress = tqdm(
                total=total_size,
                unit='B',
                unit_scale=True,
                desc=dest_path.name[:30],
            )
        else:
            print(f"Downloading: {dest_path.name}")
            downloaded = 0

        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                    f.write(chunk)
                    if HAS_TQDM:
                        progress.update(len(chunk))
                    else:
                        downloaded += len(chunk)
                        pct = (downloaded / total_size) * 100
                        print(f"\r  Progress: {pct:.1f}% ({format_size(downloaded)} / {format_size(total_size)})", end="")

        if HAS_TQDM:
            progress.close()
        else:
            print()  # Newline after progress

        return True

    except requests.exceptions.RequestException as e:
        print(f"\nError downloading: {e}")
        if dest_path.exists():
            dest_path.unlink()  # Remove partial download
        return False
    except KeyboardInterrupt:
        print("\n\nDownload cancelled by user.")
        if dest_path.exists():
            dest_path.unlink()
        return False


def download_model(model_id: str, force: bool = False) -> bool:
    """Download a specific model."""
    config = get_model_config(model_id)
    if not config:
        print(f"Error: Unknown model '{model_id}'")
        return False

    dest_path = MODELS_DIR / config.filename

    if dest_path.exists() and not force:
        print(f"Model already downloaded: {config.name}")
        print(f"  Location: {dest_path}")
        return True

    print(f"\nDownloading: {config.name}")
    print(f"  Size: {format_size(config.size_bytes)}")
    print(f"  Source: HuggingFace")
    print()

    success = download_with_progress(config.download_url, dest_path, config.size_bytes)

    if success:
        print(f"\nDownload complete: {config.name}")
        print(f"  Location: {dest_path}")
        return True
    else:
        print(f"\nFailed to download: {config.name}")
        return False


# =============================================================================
# Interactive Mode
# =============================================================================

def interactive_mode():
    """Run interactive model selection."""
    print_header()

    downloaded = get_downloaded_models()
    models = get_all_models()

    print(f"Models Directory: {MODELS_DIR}")
    print(f"Downloaded: {len(downloaded)} / {len(models)}")
    print()

    # Group by family
    families = {}
    for config in models:
        if config.family not in families:
            families[config.family] = []
        families[config.family].append(config)

    # Print models grouped by family
    print("Available Models:")
    print("-" * 40)

    model_list = []
    idx = 1
    for family, configs in sorted(families.items()):
        print(f"\n{family.upper()} Family:")
        for config in configs:
            is_downloaded = model_exists(config)
            status = "+" if is_downloaded else " "
            rec = ""
            if config.id == DEFAULT_MODEL_ID:
                rec = " [DEFAULT]"
            elif config.id == RECOMMENDED_CPU_MODEL:
                rec = " [RECOMMENDED-CPU]"
            elif config.id == RECOMMENDED_GPU_MODEL:
                rec = " [RECOMMENDED-GPU]"

            print(f"  [{idx}] {status} {config.name} ({format_size(config.size_bytes)}){rec}")
            model_list.append(config)
            idx += 1

    print()
    print("Legend: [+] = Downloaded, [ ] = Not downloaded")
    print()
    print("Options:")
    print("  Enter number(s) to download (e.g., '1' or '1,3,5')")
    print("  'r' = Download recommended models")
    print("  'a' = Download all models")
    print("  'q' = Quit")
    print()

    while True:
        try:
            choice = input("Select: ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            return

        if choice == 'q':
            print("Exiting.")
            return
        elif choice == 'r':
            # Download recommended
            print("\nDownloading recommended models...")
            download_model(DEFAULT_MODEL_ID)
            download_model(RECOMMENDED_CPU_MODEL)
            print("\nDone!")
            return
        elif choice == 'a':
            # Download all
            confirm = input(f"Download all {len(models)} models? This may take a while. (y/N): ")
            if confirm.lower() == 'y':
                for config in models:
                    download_model(config.id)
                print("\nAll downloads complete!")
            return
        else:
            # Parse number selections
            try:
                selections = [int(x.strip()) for x in choice.split(',')]
                for sel in selections:
                    if 1 <= sel <= len(model_list):
                        download_model(model_list[sel - 1].id)
                    else:
                        print(f"Invalid selection: {sel}")
                return
            except ValueError:
                print("Invalid input. Enter number(s), 'r', 'a', or 'q'.")


# =============================================================================
# CLI Commands
# =============================================================================

def list_models():
    """List all available models."""
    print_header()

    downloaded = get_downloaded_models()
    models = get_all_models()

    print(f"Models Directory: {MODELS_DIR}")
    print(f"Downloaded: {len(downloaded)} / {len(models)}")

    for config in models:
        is_downloaded = model_exists(config)
        print_model_info(config, is_downloaded)

    print()


def download_recommended():
    """Download recommended models."""
    print_header()
    print("Downloading recommended models for optimal performance...\n")

    # Always download default (fast, works everywhere)
    print("1. Default Model (fast, CPU-friendly)")
    download_model(DEFAULT_MODEL_ID)

    # Download CPU recommended
    print("\n2. Recommended CPU Model (best quality on CPU)")
    download_model(RECOMMENDED_CPU_MODEL)

    print("\nRecommended models downloaded!")
    print("\nTo use GPU-accelerated models, also download:")
    print(f"  python download_models.py --download {RECOMMENDED_GPU_MODEL}")


def download_all():
    """Download all models."""
    print_header()
    models = get_all_models()

    total_size = sum(m.size_bytes for m in models)
    print(f"Downloading all {len(models)} models...")
    print(f"Total size: {format_size(total_size)}")
    print()

    confirm = input("Continue? (y/N): ")
    if confirm.lower() != 'y':
        print("Cancelled.")
        return

    for config in models:
        download_model(config.id)

    print("\nAll models downloaded!")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Download and manage AI models for Medical Triage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                        Interactive mode
  %(prog)s --list                 List all models
  %(prog)s --download llama-3.2-1b  Download specific model
  %(prog)s --recommended          Download recommended models
  %(prog)s --all                  Download all models
        """
    )

    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='List all available models'
    )
    parser.add_argument(
        '--download', '-d',
        type=str,
        metavar='MODEL_ID',
        help='Download a specific model by ID'
    )
    parser.add_argument(
        '--recommended', '-r',
        action='store_true',
        help='Download recommended models'
    )
    parser.add_argument(
        '--all', '-a',
        action='store_true',
        help='Download all models'
    )
    parser.add_argument(
        '--force', '-f',
        action='store_true',
        help='Force re-download even if model exists'
    )

    args = parser.parse_args()

    # Ensure models directory exists
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if args.list:
        list_models()
    elif args.download:
        print_header()
        download_model(args.download, force=args.force)
    elif args.recommended:
        download_recommended()
    elif args.all:
        download_all()
    else:
        interactive_mode()


if __name__ == "__main__":
    main()
