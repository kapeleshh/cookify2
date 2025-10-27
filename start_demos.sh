#!/bin/bash

# Start all Cookify demos
# Comparison landing page on port 5000
# VLM demo on port 5001
# Traditional demo on port 5002

echo "🍳 Starting Cookify Demo Servers..."
echo "=================================="

# Activate virtual environment
source venv/bin/activate

# Kill any existing Python processes
echo "Cleaning up existing processes..."
killall -9 python 2>/dev/null
sleep 2

# Start comparison landing page (port 5000)
echo "🌐 Starting Comparison Landing Page on port 5000..."
python src/ui/app_compare.py > logs/compare.log 2>&1 &
sleep 2

# Start VLM demo (port 5001)
echo "🤖 Starting VLM-Enhanced Demo on port 5001..."
python src/ui/app_video.py > logs/vlm.log 2>&1 &
sleep 2

# Start traditional demo (port 5002)
echo "💻 Starting Traditional Demo on port 5002..."
python src/ui/app_traditional.py > logs/traditional.log 2>&1 &
sleep 3

echo ""
echo "✅ All servers started!"
echo "=================================="
echo ""
echo "📊 Open in your browser:"
echo "   Comparison Page:  http://localhost:5000"
echo "   VLM Demo:         http://localhost:5001"
echo "   Traditional Demo: http://localhost:5002"
echo ""
echo "📝 Logs available in logs/ directory"
echo ""
echo "Press Ctrl+C to stop all servers"
echo ""

# Wait for user interrupt
tail -f logs/compare.log

