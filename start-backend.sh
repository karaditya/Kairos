#!/bin/bash

echo "╔═══════════════════════════════════════════════════════════════════════════╗"
echo "║         Clinical Admin Edge (CAE) System - Backend                        ║"
echo "║         100% Local/Offline Ambient Admin Assistant                        ║"
echo "╚═══════════════════════════════════════════════════════════════════════════╝"
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# =============================================================================
# GPU/CPU Detection and Smart Allocation
# =============================================================================

detect_hardware() {
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BOLD}Hardware Detection & Smart Allocation${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""

    # Detect GPU
    GPU_AVAILABLE=false
    GPU_NAME=""
    GPU_VRAM_MB=0
    GPU_COMPUTE_CAP=""

    if command -v nvidia-smi &> /dev/null; then
        GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader,nounits 2>/dev/null)
        if [ ! -z "$GPU_INFO" ]; then
            GPU_AVAILABLE=true
            GPU_NAME=$(echo "$GPU_INFO" | cut -d',' -f1 | xargs)
            GPU_VRAM_MB=$(echo "$GPU_INFO" | cut -d',' -f2 | xargs)
            GPU_COMPUTE_CAP=$(echo "$GPU_INFO" | cut -d',' -f3 | xargs)
        fi
    fi

    # Detect CPU
    CPU_CORES=$(nproc 2>/dev/null || echo "4")
    CPU_NAME=$(grep -m1 "model name" /proc/cpuinfo 2>/dev/null | cut -d':' -f2 | xargs || echo "Unknown CPU")
    TOTAL_RAM_MB=$(free -m 2>/dev/null | awk '/^Mem:/{print $2}' || echo "8192")

    echo -e "${BLUE}CPU:${NC} $CPU_NAME"
    echo -e "${BLUE}CPU Cores:${NC} $CPU_CORES"
    echo -e "${BLUE}System RAM:${NC} ${TOTAL_RAM_MB} MB"
    echo ""

    if [ "$GPU_AVAILABLE" = true ]; then
        echo -e "${GREEN}GPU Detected:${NC} $GPU_NAME"
        echo -e "${GREEN}GPU VRAM:${NC} ${GPU_VRAM_MB} MB"
        echo -e "${GREEN}Compute Capability:${NC} $GPU_COMPUTE_CAP"

        # Determine allocation strategy based on VRAM
        if [ "$GPU_VRAM_MB" -ge 8000 ]; then
            ALLOCATION_TIER="high"
            echo -e "${GREEN}Allocation Tier:${NC} HIGH (8GB+ VRAM)"
        elif [ "$GPU_VRAM_MB" -ge 4000 ]; then
            ALLOCATION_TIER="medium"
            echo -e "${YELLOW}Allocation Tier:${NC} MEDIUM (4-8GB VRAM)"
        else
            ALLOCATION_TIER="low"
            echo -e "${YELLOW}Allocation Tier:${NC} LOW (<4GB VRAM)"
        fi
    else
        echo -e "${YELLOW}GPU:${NC} Not detected - CPU-only mode"
        ALLOCATION_TIER="cpu_only"
    fi
    echo ""
}

show_allocation_plan() {
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BOLD}Service Allocation Plan${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""

    printf "%-25s %-15s %-20s\n" "SERVICE" "DEVICE" "NOTES"
    echo "─────────────────────────────────────────────────────────────────────────"

    case "$ALLOCATION_TIER" in
        "high")
            # 8GB+ VRAM: Everything on GPU
            printf "%-25s ${GREEN}%-15s${NC} %-20s\n" "Faster-Whisper (Audio)" "GPU" "Large-v3 model"
            printf "%-25s ${GREEN}%-15s${NC} %-20s\n" "Ollama LLM" "GPU" "All layers on GPU"
            printf "%-25s ${GREEN}%-15s${NC} %-20s\n" "Sentence Transformers" "GPU" "Embedding model"
            printf "%-25s ${BLUE}%-15s${NC} %-20s\n" "Qdrant Vector DB" "CPU" "Optimized for CPU"
            printf "%-25s ${BLUE}%-15s${NC} %-20s\n" "FastAPI Server" "CPU" "Async I/O bound"
            printf "%-25s ${BLUE}%-15s${NC} %-20s\n" "SQLite Database" "CPU" "Disk I/O bound"

            # Set environment variables for high tier
            export WHISPER_DEVICE="cuda"
            export WHISPER_COMPUTE_TYPE="float16"
            export WHISPER_MODEL_SIZE="large-v3"
            export OLLAMA_GPU_LAYERS="-1"
            export EMBEDDING_DEVICE="cuda"
            ;;
        "medium")
            # 4-8GB VRAM: Prioritize Whisper and LLM on GPU
            printf "%-25s ${GREEN}%-15s${NC} %-20s\n" "Faster-Whisper (Audio)" "GPU" "Medium model"
            printf "%-25s ${YELLOW}%-15s${NC} %-20s\n" "Ollama LLM" "GPU+CPU" "Partial offload"
            printf "%-25s ${BLUE}%-15s${NC} %-20s\n" "Sentence Transformers" "CPU" "Save GPU memory"
            printf "%-25s ${BLUE}%-15s${NC} %-20s\n" "Qdrant Vector DB" "CPU" "Optimized for CPU"
            printf "%-25s ${BLUE}%-15s${NC} %-20s\n" "FastAPI Server" "CPU" "Async I/O bound"
            printf "%-25s ${BLUE}%-15s${NC} %-20s\n" "SQLite Database" "CPU" "Disk I/O bound"

            # Set environment variables for medium tier
            export WHISPER_DEVICE="cuda"
            export WHISPER_COMPUTE_TYPE="int8"
            export WHISPER_MODEL_SIZE="medium"
            export OLLAMA_GPU_LAYERS="20"
            export EMBEDDING_DEVICE="cpu"
            ;;
        "low")
            # <4GB VRAM: Only Whisper on GPU
            printf "%-25s ${GREEN}%-15s${NC} %-20s\n" "Faster-Whisper (Audio)" "GPU" "Small model"
            printf "%-25s ${BLUE}%-15s${NC} %-20s\n" "Ollama LLM" "CPU" "GPU memory limited"
            printf "%-25s ${BLUE}%-15s${NC} %-20s\n" "Sentence Transformers" "CPU" "GPU memory limited"
            printf "%-25s ${BLUE}%-15s${NC} %-20s\n" "Qdrant Vector DB" "CPU" "Optimized for CPU"
            printf "%-25s ${BLUE}%-15s${NC} %-20s\n" "FastAPI Server" "CPU" "Async I/O bound"
            printf "%-25s ${BLUE}%-15s${NC} %-20s\n" "SQLite Database" "CPU" "Disk I/O bound"

            # Set environment variables for low tier
            export WHISPER_DEVICE="cuda"
            export WHISPER_COMPUTE_TYPE="int8"
            export WHISPER_MODEL_SIZE="small"
            export OLLAMA_GPU_LAYERS="0"
            export EMBEDDING_DEVICE="cpu"
            ;;
        "cpu_only")
            # No GPU: Everything on CPU
            printf "%-25s ${BLUE}%-15s${NC} %-20s\n" "Faster-Whisper (Audio)" "CPU" "Base model"
            printf "%-25s ${BLUE}%-15s${NC} %-20s\n" "Ollama LLM" "CPU" "CPU inference"
            printf "%-25s ${BLUE}%-15s${NC} %-20s\n" "Sentence Transformers" "CPU" "CPU embeddings"
            printf "%-25s ${BLUE}%-15s${NC} %-20s\n" "Qdrant Vector DB" "CPU" "Optimized for CPU"
            printf "%-25s ${BLUE}%-15s${NC} %-20s\n" "FastAPI Server" "CPU" "Async I/O bound"
            printf "%-25s ${BLUE}%-15s${NC} %-20s\n" "SQLite Database" "CPU" "Disk I/O bound"

            # Set environment variables for CPU-only
            export WHISPER_DEVICE="cpu"
            export WHISPER_COMPUTE_TYPE="int8"
            export WHISPER_MODEL_SIZE="base"
            export OLLAMA_GPU_LAYERS="0"
            export EMBEDDING_DEVICE="cpu"
            ;;
    esac

    echo ""
    echo -e "${MAGENTA}Legend:${NC} ${GREEN}GPU${NC} = CUDA accelerated | ${BLUE}CPU${NC} = CPU processing"
    echo ""
}

# Function to check if a command exists
command_exists() {
    command -v "$1" &> /dev/null
}

# Function to check if a port is in use
port_in_use() {
    lsof -i :"$1" &> /dev/null
}

# =============================================================================
# Run Hardware Detection
# =============================================================================
detect_hardware
show_allocation_plan

# =============================================================================
# Step 1: Check Python virtual environment
# =============================================================================
echo -e "${BLUE}[1/6]${NC} Checking Python environment..."

if [ ! -d "venv" ]; then
    echo "  Creating Python virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate

# Check dependencies
if ! python -c "import fastapi" 2>/dev/null; then
    echo "  Installing Python dependencies..."
    pip install -r backend/requirements.txt
fi
echo -e "${GREEN}  ✓ Python environment ready${NC}"

# =============================================================================
# Step 2: Check and start Ollama
# =============================================================================
echo -e "${BLUE}[2/6]${NC} Checking Ollama..."

if ! command_exists ollama; then
    echo -e "${RED}  ✗ Ollama not installed!${NC}"
    echo ""
    echo "  Install Ollama from: https://ollama.ai/download"
    echo "    curl -fsSL https://ollama.ai/install.sh | sh"
    echo ""
    echo "  The backend will run in FALLBACK MODE (rule-based only, no LLM)"
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
        echo -e "${GREEN}  ✓ Ollama server running${NC}"
        OLLAMA_AVAILABLE=true
    else
        echo -e "${YELLOW}  ⚠ Ollama not responding - will use fallback mode${NC}"
        OLLAMA_AVAILABLE=false
    fi
fi

# =============================================================================
# Step 3: Check for LLM models
# =============================================================================
echo -e "${BLUE}[3/6]${NC} Checking LLM models..."

DEFAULT_MODEL="${OLLAMA_MODEL:-mistral}"

if [ "$OLLAMA_AVAILABLE" = true ]; then
    MODELS=$(ollama list 2>/dev/null | tail -n +2 | awk '{print $1}')
    MODEL_COUNT=$(echo "$MODELS" | grep -v '^$' | wc -l)

    if [ "$MODEL_COUNT" -gt 0 ]; then
        echo -e "${GREEN}  ✓ Found $MODEL_COUNT model(s)${NC}"
    else
        echo -e "${YELLOW}  ⚠ No models found. Pulling default model...${NC}"
        echo "  Pulling $DEFAULT_MODEL (this may take a few minutes)..."
        if ollama pull "$DEFAULT_MODEL"; then
            echo -e "${GREEN}  ✓ Successfully pulled $DEFAULT_MODEL${NC}"
        else
            echo -e "${RED}  ✗ Failed to pull $DEFAULT_MODEL${NC}"
        fi
    fi

    # Check for embedding model
    EMBEDDING_MODEL="nomic-embed-text"
    if ! echo "$MODELS" | grep -q "$EMBEDDING_MODEL"; then
        echo "  Pulling embedding model ($EMBEDDING_MODEL)..."
        ollama pull "$EMBEDDING_MODEL" 2>/dev/null
    fi
else
    echo -e "${YELLOW}  ⚠ Ollama not available - using rule-based fallback${NC}"
fi

# =============================================================================
# Step 4: Check Whisper model
# =============================================================================
echo -e "${BLUE}[4/6]${NC} Checking Faster-Whisper..."

echo -e "  Model: ${WHISPER_MODEL_SIZE}, Device: ${WHISPER_DEVICE}, Compute: ${WHISPER_COMPUTE_TYPE}"
echo -e "${GREEN}  ✓ Whisper will auto-download on first use${NC}"

# =============================================================================
# Step 5: Check ports
# =============================================================================
echo -e "${BLUE}[5/6]${NC} Checking ports..."

# Check port 8000 (FastAPI)
if port_in_use 8000; then
    echo -e "${YELLOW}  ⚠ Port 8000 in use, killing existing process...${NC}"
    kill $(lsof -t -i:8000) 2>/dev/null
    sleep 1
fi

echo -e "${GREEN}  ✓ Port 8000 available${NC}"

# =============================================================================
# Step 6: Start backend
# =============================================================================
echo -e "${BLUE}[6/6]${NC} Starting CAE backend..."
echo ""

cd backend

echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BOLD}CAE System URLs${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "  ${BLUE}API Server:${NC}      http://localhost:8000"
echo -e "  ${BLUE}API Docs:${NC}        http://localhost:8000/docs"
echo -e "  ${BLUE}Health Check:${NC}    http://localhost:8000/admin/status"
echo ""
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Export allocation settings for the Python backend
export CAE_ALLOCATION_TIER="$ALLOCATION_TIER"
export CAE_GPU_AVAILABLE="$GPU_AVAILABLE"
export CAE_GPU_VRAM_MB="$GPU_VRAM_MB"

# Start with uvicorn
exec uvicorn main:app --host 0.0.0.0 --port 8000 --reload
