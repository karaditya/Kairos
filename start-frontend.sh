#!/bin/bash

echo "==================================="
echo "  Medical Triage Frontend (Next.js)"
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

# =============================================================================
# Step 1: Check Node.js
# =============================================================================
echo -e "${BLUE}[1/3]${NC} Checking Node.js..."

if ! command -v node &> /dev/null; then
    echo -e "${RED}  Node.js not installed!${NC}"
    echo ""
    echo "Install Node.js from: https://nodejs.org/"
    echo "  Or use nvm: curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash"
    exit 1
fi

NODE_VERSION=$(node -v)
echo -e "${GREEN}  Node.js $NODE_VERSION${NC}"

# =============================================================================
# Step 2: Install dependencies
# =============================================================================
echo -e "${BLUE}[2/3]${NC} Checking dependencies..."

cd frontend

if [ ! -d "node_modules" ]; then
    echo "  Installing Node.js dependencies..."
    npm install
else
    echo -e "${GREEN}  Dependencies already installed${NC}"
fi

# =============================================================================
# Step 3: Check if backend is running
# =============================================================================
echo -e "${BLUE}[3/3]${NC} Checking backend..."

if curl -s http://localhost:8000/health &> /dev/null || curl -s http://localhost:8000/ &> /dev/null; then
    echo -e "${GREEN}  Backend is running on port 8000${NC}"
else
    echo -e "${YELLOW}  Backend not detected on port 8000${NC}"
    echo "  Run ./start-backend.sh in another terminal first"
fi

# =============================================================================
# Start frontend
# =============================================================================
echo ""
echo "==================================="
echo "  Frontend URLs:"
echo "    Main App:    http://localhost:3000/"
echo "    Triage:      http://localhost:3000/triage"
echo "    Staff:       http://localhost:3000/staff"
echo "==================================="
echo ""

# Start Next.js dev server
exec npm run dev
