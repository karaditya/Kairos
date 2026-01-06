#!/bin/bash

echo "==================================="
echo "  Stopping Triage Backend"
echo "==================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

STOPPED_SOMETHING=false

# =============================================================================
# Stop uvicorn processes (FastAPI backend)
# =============================================================================
echo -e "${BLUE}[1/3]${NC} Stopping FastAPI backend..."

if pgrep -f "uvicorn main:app" > /dev/null; then
    pkill -f "uvicorn main:app"
    sleep 1

    if pgrep -f "uvicorn main:app" > /dev/null; then
        echo -e "${YELLOW}  Force killing uvicorn...${NC}"
        pkill -9 -f "uvicorn main:app"
        sleep 1
    fi
    echo -e "${GREEN}  FastAPI backend stopped${NC}"
    STOPPED_SOMETHING=true
else
    echo "  No uvicorn process found"
fi

# Also check for old-style python main.py
if pgrep -f "python main.py" > /dev/null; then
    pkill -f "python main.py"
    sleep 1
    echo -e "${GREEN}  Legacy backend process stopped${NC}"
    STOPPED_SOMETHING=true
fi

# =============================================================================
# Stop Parlant server (port 8800)
# =============================================================================
echo -e "${BLUE}[2/3]${NC} Stopping Parlant server..."

if lsof -i :8800 &> /dev/null; then
    kill $(lsof -t -i:8800) 2>/dev/null
    sleep 1
    echo -e "${GREEN}  Parlant server stopped${NC}"
    STOPPED_SOMETHING=true
else
    echo "  No Parlant process found on port 8800"
fi

# =============================================================================
# Free up port 8000 if still in use
# =============================================================================
echo -e "${BLUE}[3/3]${NC} Checking port 8000..."

if lsof -i :8000 &> /dev/null; then
    echo -e "${YELLOW}  Port 8000 still in use, killing...${NC}"
    kill $(lsof -t -i:8000) 2>/dev/null
    sleep 1

    if lsof -i :8000 &> /dev/null; then
        kill -9 $(lsof -t -i:8000) 2>/dev/null
        sleep 1
    fi
    echo -e "${GREEN}  Port 8000 freed${NC}"
    STOPPED_SOMETHING=true
else
    echo "  Port 8000 is free"
fi

# =============================================================================
# Summary
# =============================================================================
echo ""
if [ "$STOPPED_SOMETHING" = true ]; then
    echo -e "${GREEN}Backend stopped successfully${NC}"
else
    echo "No backend processes were running"
fi

# Show GPU memory status if available
if command -v nvidia-smi &> /dev/null; then
    GPU_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)
    if [ ! -z "$GPU_MEM" ]; then
        echo ""
        echo "GPU memory in use: ${GPU_MEM} MiB"
    fi
fi

echo ""
