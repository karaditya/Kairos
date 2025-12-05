#!/bin/bash

echo "🏥 Starting Triage Backend..."
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Check if dependencies are installed
if ! python -c "import fastapi" 2>/dev/null; then
    echo "Installing Python dependencies..."
    pip install -r requirements.txt
fi

# Check if model exists
if [ ! -f "models/llama-3.2-1b-instruct-q4_k_m.gguf" ]; then
    echo ""
    echo "⚠️  Warning: LLM model not found at models/llama-3.2-1b-instruct-q4_k_m.gguf"
    echo "The system will run in fallback mode (template-based responses)"
    echo ""
    echo "To download the model, run:"
    echo "  mkdir -p models && cd models"
    echo "  wget https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf -O llama-3.2-1b-instruct-q4_k_m.gguf"
    echo ""
fi

# Start backend
echo "Starting FastAPI backend on http://localhost:8000..."
echo "GPU Acceleration: ${N_GPU_LAYERS:-0} layers (set N_GPU_LAYERS env var to change)"
cd backend
python main.py
