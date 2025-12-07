#!/bin/bash
# Model Download Script for Medical AI Triage
# Quick wrapper for the Python download utility

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run the download script
python3 scripts/download_models.py "$@"
