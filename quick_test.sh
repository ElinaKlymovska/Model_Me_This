#!/bin/bash

# Quick test script for Portrait Enhancement Pipeline
# Usage: ./quick_test.sh

set -e

echo "🧪 Quick Test for Portrait Enhancement Pipeline"
echo "================================================"

# Check if we're in the right directory
if [ ! -f "bootstrap.sh" ]; then
    echo "❌ Error: bootstrap.sh not found. Please run this script from the project root."
    exit 1
fi

echo "✅ Project structure check passed"

# Check if required files exist
echo "📁 Checking required files..."
required_files=(
    "bootstrap.sh"
    "models_auto.py"
    "portrait-enhancer/config.yaml"
    "portrait-enhancer/batch.py"
    "portrait-enhancer/run_a_pass.py"
    "portrait-enhancer/run_b_pass.py"
)

for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✅ $file"
    else
        echo "  ❌ $file (missing)"
        exit 1
    fi
done

# Check if directories exist
echo "📁 Checking directories..."
required_dirs=(
    "portrait-enhancer/input"
    "portrait-enhancer/work"
    "portrait-enhancer/output"
)

for dir in "${required_dirs[@]}"; do
    if [ -d "$dir" ]; then
        echo "  ✅ $dir"
    else
        echo "  ❌ $dir (missing)"
        exit 1
    fi
done

# Check if scripts are executable
echo "🔧 Checking script permissions..."
scripts=(
    "deploy_vast.sh"
    "upload_images.sh"
    "download_results.sh"
    "monitor.sh"
)

for script in "${scripts[@]}"; do
    if [ -x "$script" ]; then
        echo "  ✅ $script (executable)"
    else
        echo "  ❌ $script (not executable)"
        chmod +x "$script"
        echo "  ✅ $script (made executable)"
    fi
done

# Test SSH connection to vast.ai
echo "🔌 Testing SSH connection to vast.ai..."
if make test-connection >/dev/null 2>&1; then
    echo "  ✅ SSH connection to vast.ai successful"
else
    echo "  ⚠️  SSH connection to vast.ai failed (this is normal if not deployed yet)"
fi

echo ""
echo "🎉 Quick test completed successfully!"
echo ""
echo "Next steps:"
echo "1. Add images to portrait-enhancer/input/"
echo "2. Deploy to vast.ai: make deploy"
echo "3. Upload images: make upload"
echo "4. Monitor progress: make monitor"
echo "5. Download results: make download"
echo ""
echo "For help: make help"
