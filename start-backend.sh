#!/bin/bash

echo "==================================="
echo "  Medical Triage Backend (Parlant)"
echo "==================================="
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to check if a command exists
command_exists() {
    command -v "$1" &> /dev/null
}

# Function to check if a port is in use
port_in_use() {
    lsof -i :"$1" &> /dev/null
}

# =============================================================================
# Step 1: Check Python virtual environment
# =============================================================================
echo -e "${BLUE}[1/5]${NC} Checking Python environment..."

if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate

# Check dependencies
if ! python -c "import fastapi" 2>/dev/null; then
    echo "Installing Python dependencies..."
    pip install -r requirements.txt
fi
echo -e "${GREEN}  Python environment ready${NC}"

# =============================================================================
# Step 2: Check and start Ollama
# =============================================================================
echo -e "${BLUE}[2/5]${NC} Checking Ollama..."

if ! command_exists ollama; then
    echo -e "${RED}  Ollama not installed!${NC}"
    echo ""
    echo "Install Ollama from: https://ollama.ai/download"
    echo "  curl -fsSL https://ollama.ai/install.sh | sh"
    echo ""
    echo "The backend will run in FALLBACK MODE (rule-based only, no LLM)"
    OLLAMA_AVAILABLE=false
else
    # Check if Ollama is running
    if ! curl -s http://localhost:11434/api/tags &> /dev/null; then
        echo "  Starting Ollama server..."
        ollama serve &> /dev/null &
        sleep 3
    fi

    # Verify Ollama is responding
    if curl -s http://localhost:11434/api/tags &> /dev/null; then
        echo -e "${GREEN}  Ollama server running${NC}"
        OLLAMA_AVAILABLE=true
    else
        echo -e "${YELLOW}  Ollama not responding - will use fallback mode${NC}"
        OLLAMA_AVAILABLE=false
    fi
fi

# =============================================================================
# Step 3: Check for available models
# =============================================================================
echo -e "${BLUE}[3/5]${NC} Checking LLM models..."

if [ "$OLLAMA_AVAILABLE" = true ]; then
    # Get list of pulled models
    MODELS=$(ollama list 2>/dev/null | tail -n +2 | awk '{print $1}')
    MODEL_COUNT=$(echo "$MODELS" | grep -c .)

    if [ "$MODEL_COUNT" -gt 0 ]; then
        echo -e "${GREEN}  Found $MODEL_COUNT model(s):${NC}"
        echo "$MODELS" | while read model; do
            echo "    - $model"
        done
    else
        echo -e "${YELLOW}  No models found. Pull a model first:${NC}"
        echo ""
        echo "  Recommended models (run one of these):"
        echo "    ollama pull mistral        # 4.1GB - Good default"
        echo "    ollama pull llama3.2       # 2.0GB - Fast, compact"
        echo "    ollama pull deepseek-r1:7b # 4.7GB - Strong reasoning"
        echo "    ollama pull qwen2.5        # 4.4GB - Good for French"
        echo ""
        echo "  The backend will start but use fallback mode until a model is pulled."
    fi
else
    echo -e "${YELLOW}  Ollama not available - using rule-based fallback${NC}"
fi

# =============================================================================
# Step 4: Check ports
# =============================================================================
echo -e "${BLUE}[4/5]${NC} Checking ports..."

# Check port 8000 (FastAPI)
if port_in_use 8000; then
    echo -e "${YELLOW}  Port 8000 in use, killing existing process...${NC}"
    kill $(lsof -t -i:8000) 2>/dev/null
    sleep 1
fi

# Check port 8800 (Parlant)
if port_in_use 8800; then
    echo -e "${YELLOW}  Port 8800 in use, killing existing process...${NC}"
    kill $(lsof -t -i:8800) 2>/dev/null
    sleep 1
fi

echo -e "${GREEN}  Ports 8000 and 8800 available${NC}"

# =============================================================================
# Step 5: Start backend
# =============================================================================
echo -e "${BLUE}[5/5]${NC} Starting backend..."
echo ""

cd backend

echo "==================================="
echo "  Backend URLs:"
echo "    Patient Interface: http://localhost:8000/"
echo "    Staff Portal:      http://localhost:8000/staff"
echo "    API Docs:          http://localhost:8000/docs"
echo "==================================="
echo ""

# Start with uvicorn
exec uvicorn main:app --host 0.0.0.0 --port 8000 --reload
