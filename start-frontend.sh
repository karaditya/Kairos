#!/bin/bash

echo "╔═══════════════════════════════════════════════════════════════════════════╗"
echo "║         Clinical Admin Edge (CAE) System - Frontend                       ║"
echo "║         Next.js + React Dashboard Interface                               ║"
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
NC='\033[0m' # No Color
BOLD='\033[1m'

# =============================================================================
# Step 1: Check Node.js
# =============================================================================
echo -e "${BLUE}[1/4]${NC} Checking Node.js..."

if ! command -v node &> /dev/null; then
    echo -e "${RED}  ✗ Node.js not installed!${NC}"
    echo ""
    echo "  Install Node.js from: https://nodejs.org/"
    echo "  Or use nvm:"
    echo "    curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash"
    echo "    nvm install 20"
    exit 1
fi

NODE_VERSION=$(node -v)
echo -e "${GREEN}  ✓ Node.js $NODE_VERSION${NC}"

# =============================================================================
# Step 2: Install dependencies
# =============================================================================
echo -e "${BLUE}[2/4]${NC} Checking dependencies..."

cd frontend

if [ ! -d "node_modules" ]; then
    echo "  Installing Node.js dependencies..."
    npm install
else
    echo -e "${GREEN}  ✓ Dependencies already installed${NC}"
fi

# =============================================================================
# Step 3: Check if backend is running
# =============================================================================
echo -e "${BLUE}[3/4]${NC} Checking backend connection..."

BACKEND_URL="http://localhost:8000"

if curl -s "${BACKEND_URL}/admin/status" &> /dev/null; then
    echo -e "${GREEN}  ✓ Backend is running on port 8000${NC}"

    # Try to get service status
    STATUS=$(curl -s "${BACKEND_URL}/admin/status" 2>/dev/null)
    if [ ! -z "$STATUS" ]; then
        echo ""
        echo -e "${CYAN}  Backend Services Status:${NC}"
        echo "$STATUS" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    services = data.get('services', {})
    for name, info in services.items():
        status = info.get('status', 'unknown')
        symbol = '✓' if status == 'healthy' else '⚠' if status == 'degraded' else '✗'
        color = '\033[0;32m' if status == 'healthy' else '\033[1;33m' if status == 'degraded' else '\033[0;31m'
        print(f'    {color}{symbol}\033[0m {name}: {status}')
except:
    pass
" 2>/dev/null
    fi
else
    echo -e "${YELLOW}  ⚠ Backend not detected on port 8000${NC}"
    echo ""
    echo "  Run ./start-backend.sh in another terminal first"
    echo ""
fi

# =============================================================================
# Step 4: Start frontend
# =============================================================================
echo ""
echo -e "${BLUE}[4/4]${NC} Starting CAE frontend..."
echo ""

echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BOLD}CAE Dashboard URLs${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "  ${BLUE}Home Page:${NC}            http://localhost:3000"
echo ""
echo -e "  ${GREEN}Receptionist Portal:${NC}  http://localhost:3000/receptionist"
echo -e "    └─ Start sessions, record audio, view live transcript"
echo ""
echo -e "  ${GREEN}Clinician Portal:${NC}     http://localhost:3000/clinician"
echo -e "    └─ Review sessions, edit Compte Rendu, approve RPA sync"
echo ""
echo -e "  ${GREEN}Admin Portal:${NC}         http://localhost:3000/admin"
echo -e "    └─ System status, protocols, model management"
echo ""
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Start Next.js dev server
exec npm run dev
