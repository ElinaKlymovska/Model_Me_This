#!/bin/bash

# Script to download results from vast.ai
# Usage: ./download_results.sh

set -e

echo "📥 Downloading results from vast.ai..."

# Check if SSH connection is available
if ! ssh -p 18826 -o ConnectTimeout=10 root@ssh4.vast.ai "echo 'SSH connection successful'" 2>/dev/null; then
    echo "❌ Failed to connect to vast.ai. Please check your SSH connection."
    exit 1
fi

# Create local output directory
mkdir -p portrait-enhancer/output

# Download results
echo "📁 Downloading results..."
rsync -avz -e "ssh -p 18826" \
    --progress \
    root@ssh4.vast.ai:/workspace/portrait-enhancer/output/ portrait-enhancer/output/

echo "✅ Results downloaded successfully!"
echo "📁 Check portrait-enhancer/output/ for your enhanced images"
