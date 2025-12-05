#!/bin/bash

echo "🛑 Stopping Triage Backend..."
echo ""

# Find and kill the backend process
if pgrep -f "python main.py" > /dev/null; then
    pkill -f "python main.py"
    sleep 2

    # Check if it stopped
    if pgrep -f "python main.py" > /dev/null; then
        echo "⚠️  Process still running, forcing shutdown..."
        pkill -9 -f "python main.py"
        sleep 1
    fi

    echo "✅ Backend stopped successfully"

    # Show GPU memory if nvidia-smi is available
    if command -v nvidia-smi &> /dev/null; then
        GPU_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)
        if [ ! -z "$GPU_MEM" ]; then
            echo "📊 GPU memory: ${GPU_MEM} MiB"
        fi
    fi
else
    echo "ℹ️  Backend is not running"
fi

echo ""
