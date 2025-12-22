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

# Check if any models exist
MODEL_COUNT=$(find models -name "*.gguf" 2>/dev/null | wc -l)
if [ "$MODEL_COUNT" -eq 0 ]; then
    echo ""
    echo "⚠️  Warning: No LLM models found in models/ directory"
    echo "The system will run in fallback mode (template-based responses)"
    echo ""
    echo "📚 10+ models supported! Quick download options:"
    echo ""
    echo "Option 1: Llama 3.2 1B (Fastest, CPU-friendly, 738MB)"
    echo "  mkdir -p models && cd models"
    echo "  wget https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf -O llama-3.2-1b-instruct-q4_k_m.gguf"
    echo ""
    echo "Option 2: DeepSeek R1 1.5B (Chain-of-thought reasoning, 1GB)"
    echo "  wget https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF/resolve/main/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf -O deepseek-r1-1.5b-q4_k_m.gguf"
    echo ""
    echo "📖 See README.md for all 10 models and download instructions"
    echo ""
else
    echo "✅ Found $MODEL_COUNT model(s) in models/ directory"
    # List available models
    echo "Available models:"
    find models -name "*.gguf" -exec basename {} \; | sed 's/^/  - /'
    echo ""
fi

# Start backend
echo "Starting FastAPI backend on http://localhost:8000..."
if [ -z "$N_GPU_LAYERS" ]; then
    echo "GPU Acceleration: Auto-detect (will use optimal GPU/CPU split based on VRAM)"
else
    echo "GPU Acceleration: $N_GPU_LAYERS layers (manual override)"
fi
echo "LLM output will be appended to backend/backend.log"
cd backend
python main.py
