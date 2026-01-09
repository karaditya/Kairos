#!/bin/bash

echo "╔═══════════════════════════════════════════════════════════════════════════╗"
echo "║         Clinical Admin Edge (CAE) System - Shutdown                       ║"
echo "╚═══════════════════════════════════════════════════════════════════════════╝"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

STOPPED_SOMETHING=false

# =============================================================================
# Step 1: Stop uvicorn processes (FastAPI backend)
# =============================================================================
echo -e "${BLUE}[1/4]${NC} Stopping FastAPI backend..."

if pgrep -f "uvicorn main:app" > /dev/null; then
    pkill -f "uvicorn main:app"
    sleep 1

    if pgrep -f "uvicorn main:app" > /dev/null; then
        echo -e "${YELLOW}  Force killing uvicorn...${NC}"
        pkill -9 -f "uvicorn main:app"
        sleep 1
    fi
    echo -e "${GREEN}  ✓ FastAPI backend stopped${NC}"
    STOPPED_SOMETHING=true
else
    echo "  No uvicorn process found"
fi

# =============================================================================
# Step 2: Stop Whisper processes (if running separately)
# =============================================================================
echo -e "${BLUE}[2/4]${NC} Checking audio services..."

# Check for any orphaned faster-whisper processes
if pgrep -f "faster_whisper" > /dev/null; then
    pkill -f "faster_whisper"
    echo -e "${GREEN}  ✓ Whisper processes stopped${NC}"
    STOPPED_SOMETHING=true
else
    echo "  No separate Whisper processes found"
fi

# =============================================================================
# Step 3: Free up ports
# =============================================================================
echo -e "${BLUE}[3/4]${NC} Freeing ports..."

# Check port 8000 (FastAPI)
if lsof -i :8000 &> /dev/null; then
    echo -e "${YELLOW}  Port 8000 in use, killing...${NC}"
    kill $(lsof -t -i:8000) 2>/dev/null
    sleep 1

    if lsof -i :8000 &> /dev/null; then
        kill -9 $(lsof -t -i:8000) 2>/dev/null
        sleep 1
    fi
    echo -e "${GREEN}  ✓ Port 8000 freed${NC}"
    STOPPED_SOMETHING=true
else
    echo "  Port 8000 is free"
fi

# =============================================================================
# Step 4: Show GPU/memory status
# =============================================================================
echo -e "${BLUE}[4/4]${NC} Resource status..."

# Show GPU memory status if available
if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null)
    if [ ! -z "$GPU_INFO" ]; then
        GPU_NAME=$(echo "$GPU_INFO" | cut -d',' -f1 | xargs)
        GPU_MEM_USED=$(echo "$GPU_INFO" | cut -d',' -f2 | xargs)
        GPU_MEM_TOTAL=$(echo "$GPU_INFO" | cut -d',' -f3 | xargs)
        echo ""
        echo -e "${CYAN}GPU Status:${NC}"
        echo -e "  Device: $GPU_NAME"
        echo -e "  Memory: ${GPU_MEM_USED} / ${GPU_MEM_TOTAL} MiB"

        # Check if GPU memory is mostly free
        if [ "$GPU_MEM_USED" -lt 500 ]; then
            echo -e "  ${GREEN}✓ GPU memory released${NC}"
        else
            echo -e "  ${YELLOW}⚠ Some GPU memory still in use (may be Ollama)${NC}"
        fi
    fi
else
    echo "  No NVIDIA GPU detected"
fi

# Show system memory
MEM_INFO=$(free -m 2>/dev/null | awk '/^Mem:/{print $3, $2}')
if [ ! -z "$MEM_INFO" ]; then
    MEM_USED=$(echo "$MEM_INFO" | cut -d' ' -f1)
    MEM_TOTAL=$(echo "$MEM_INFO" | cut -d' ' -f2)
    echo ""
    echo -e "${CYAN}System Memory:${NC}"
    echo -e "  Used: ${MEM_USED} / ${MEM_TOTAL} MB"
fi

# =============================================================================
# Summary
# =============================================================================
echo ""
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
if [ "$STOPPED_SOMETHING" = true ]; then
    echo -e "${GREEN}CAE backend stopped successfully${NC}"
else
    echo "No CAE backend processes were running"
fi
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Note about Ollama
echo -e "${YELLOW}Note:${NC} Ollama server is not stopped (may be used by other applications)"
echo "  To stop Ollama manually: pkill ollama"
echo ""
