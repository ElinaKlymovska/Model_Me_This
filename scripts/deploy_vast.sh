#!/bin/bash

# Script to deploy Portrait Enhancement Pipeline on vast.ai
# Usage: ./deploy_vast.sh

set -e

echo "🚀 Deploying Portrait Enhancement Pipeline on vast.ai..."

# Check if SSH connection is available
echo "📡 Testing SSH connection to vast.ai..."
if ! ssh -p 18826 -o ConnectTimeout=10 root@ssh4.vast.ai "echo 'SSH connection successful'" 2>/dev/null; then
    echo "❌ Failed to connect to vast.ai. Please check your SSH connection."
    exit 1
fi

echo "✅ SSH connection established"

# Create remote directories
echo "📁 Creating remote directories..."
ssh -p 18826 root@ssh4.vast.ai << 'EOF'
mkdir -p /workspace/portrait-enhancer/{input,work,output}
mkdir -p /workspace/models
EOF

# Copy project files
echo "📤 Copying project files..."
rsync -avz -e "ssh -p 18826" \
    --exclude '.git' \
    --exclude 'portrait-enhancer/input/*' \
    --exclude 'portrait-enhancer/output/*' \
    --exclude 'portrait-enhancer/work/*' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude 'adetailer_2cn_plus/.venv' \
    --exclude 'adetailer_2cn_plus/__pycache__' \
    ./ root@ssh4.vast.ai:/workspace/

# Set permissions
echo "🔐 Setting permissions..."
ssh -p 18826 root@ssh4.vast.ai "chmod +x /workspace/bootstrap.sh"

# Start the pipeline
echo "🎬 Starting Portrait Enhancement Pipeline..."
ssh -p 18826 root@ssh4.vast.ai "cd /workspace && ./bootstrap.sh"

echo "✅ Deployment complete!"
echo "🌐 WebUI should be available at: http://localhost:8080 (via SSH tunnel)"
echo "📁 Place images in: portrait-enhancer/input/"
echo "📁 Results will be in: portrait-enhancer/output/"
echo ""
echo "To monitor the process:"
echo "  ssh -p 18826 root@ssh4.vast.ai 'tmux attach -t webui'"
echo ""
echo "To stop the pipeline:"
echo "  ssh -p 18826 root@ssh4.vast.ai 'tmux kill-session -t webui'"
