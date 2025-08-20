#!/bin/bash

# Script to monitor the Portrait Enhancement Pipeline on vast.ai
# Usage: ./monitor.sh

echo "📊 Monitoring Portrait Enhancement Pipeline on vast.ai..."

# Check if SSH connection is available
if ! ssh -p 18826 -o ConnectTimeout=10 root@ssh4.vast.ai "echo 'SSH connection successful'" 2>/dev/null; then
    echo "❌ Failed to connect to vast.ai. Please check your SSH connection."
    exit 1
fi

echo "✅ SSH connection established"
echo ""
echo "🔍 Checking system status..."

# Check system resources
echo "📊 System Resources:"
ssh -p 18826 root@ssh4.vast.ai "free -h && echo '---' && df -h /workspace && echo '---' && nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits"

echo ""
echo "🎬 Checking WebUI status..."

# Check if WebUI is running
if ssh -p 18826 root@ssh4.vast.ai "curl -s http://127.0.0.1:7860/sdapi/v1/sd-models >/dev/null 2>&1"; then
    echo "✅ Stable Diffusion WebUI is running"
    echo "🌐 Available at: http://localhost:8080 (via SSH tunnel)"
else
    echo "❌ Stable Diffusion WebUI is not running"
fi

echo ""
echo "📁 Checking directories:"
ssh -p 18826 root@ssh4.vast.ai "ls -la /workspace/portrait-enhancer/ && echo '---' && ls -la /workspace/portrait-enhancer/input/ && echo '---' && ls -la /workspace/portrait-enhancer/output/"

echo ""
echo "🔧 To attach to tmux session:"
echo "   ssh -p 18826 root@ssh4.vast.ai 'tmux attach -t webui'"
echo ""
echo "🛑 To stop the pipeline:"
echo "   ssh -p 18826 root@ssh4.vast.ai 'tmux kill-session -t webui'"
