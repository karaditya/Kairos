#!/bin/bash

echo "🌐 Starting Triage Frontend..."
echo ""

# Navigate to frontend directory
cd frontend

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "Installing Node.js dependencies..."
    npm install
fi

# Start frontend
echo "Starting Next.js frontend on http://localhost:3000..."
npm run dev
