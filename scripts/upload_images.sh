#!/bin/bash

# Script to upload images to vast.ai for processing
# Usage: ./upload_images.sh [image_directory]

set -e

IMAGE_DIR="${1:-portrait-enhancer/input}"

if [ ! -d "$IMAGE_DIR" ]; then
    echo "❌ Directory $IMAGE_DIR not found!"
    echo "Usage: $0 [image_directory]"
    exit 1
fi

echo "📤 Uploading images from $IMAGE_DIR to vast.ai..."

# Check if SSH connection is available
if ! ssh -p 18826 -o ConnectTimeout=10 root@ssh4.vast.ai "echo 'SSH connection successful'" 2>/dev/null; then
    echo "❌ Failed to connect to vast.ai. Please check your SSH connection."
    exit 1
fi

# Create remote input directory
ssh -p 18826 root@ssh4.vast.ai "mkdir -p /workspace/portrait-enhancer/input"

# Upload images
echo "📁 Uploading images..."
rsync -avz -e "ssh -p 18826" \
    --progress \
    "$IMAGE_DIR"/ root@ssh4.vast.ai:/workspace/portrait-enhancer/input/

echo "✅ Images uploaded successfully!"
echo "🎬 You can now run the processing pipeline:"
echo "   ssh -p 18826 root@ssh4.vast.ai 'cd /workspace && python portrait-enhancer/batch.py'"
