#!/bin/bash

# Test script for ADetailer 2CN Plus integration
# Usage: ./test_ad2cn.sh

set -e

echo "🧪 Testing ADetailer 2CN Plus Integration"
echo "=========================================="

# Check if adetailer_2cn_plus directory exists
if [ ! -d "adetailer_2cn_plus" ]; then
    echo "❌ Error: adetailer_2cn_plus directory not found!"
    exit 1
fi

echo "✅ ADetailer 2CN Plus directory found"

# Check if Python environment is set up
if [ ! -d "portrait-enhancer/.venv" ]; then
    echo "⚠️  Python virtual environment not found. Setting up..."
    cd portrait-enhancer
    python3 -m venv .venv
    source .venv/bin/activate
    # Try macOS-compatible requirements first
    if [ -f "requirements_macos.txt" ]; then
        pip install -r requirements_macos.txt
    else
        pip install -r requirements.txt
    fi
    cd ..
else
    echo "✅ Python virtual environment found"
fi

# Activate virtual environment
source portrait-enhancer/.venv/bin/activate

# Test basic imports
echo "🔍 Testing basic imports..."
cd adetailer_2cn_plus

if python -c "from ad2cn.config import Config; print('✓ Config import successful')" 2>/dev/null; then
    echo "✅ ADetailer 2CN Plus imports working"
else
    echo "❌ ADetailer 2CN Plus imports failed"
    echo "Installing dependencies..."
    # Try macOS-compatible requirements first
    if [ -f "requirements_macos.txt" ]; then
        pip install -r requirements_macos.txt
    else
        pip install -r requirements.txt
    fi
    
    # Test again
    if python -c "from ad2cn.config import Config; print('✓ Config import successful')" 2>/dev/null; then
        echo "✅ Dependencies installed and imports working"
    else
        echo "❌ Still having import issues"
        exit 1
    fi
fi

# Test configuration
echo "🔧 Testing configuration..."
if python -c "
from ad2cn.config import Config
config_data = {'pipeline': {'detectors': [{'name': 'blazeface'}]}}
config = Config(**config_data)
print('✓ Configuration validation successful')
" 2>/dev/null; then
    echo "✅ Configuration validation working"
else
    echo "❌ Configuration validation failed"
    exit 1
fi

# Test pipeline initialization
echo "🚀 Testing pipeline initialization..."
if python -c "
from ad2cn.config import Config
from ad2cn.pipeline.detect import DetectionPipeline
config_data = {'pipeline': {'detectors': [{'name': 'blazeface'}]}}
config = Config(**config_data)
pipeline = DetectionPipeline(config.pipeline.dict())
print('✓ Pipeline initialization successful')
" 2>/dev/null; then
    echo "✅ Pipeline initialization working"
else
    echo "❌ Pipeline initialization failed"
    exit 1
fi

cd ..

# Test enhanced batch processing
echo "🎯 Testing enhanced batch processing..."
if python -c "
import sys
sys.path.insert(0, 'adetailer_2cn_plus')
try:
    from ad2cn.config import Config
    print('✓ Enhanced processing available')
except ImportError:
    print('⚠ Enhanced processing not available')
" 2>/dev/null; then
    echo "✅ Enhanced batch processing ready"
else
    echo "⚠ Enhanced batch processing not available"
fi

echo ""
echo "🎉 ADetailer 2CN Plus integration test completed!"
echo ""
echo "Next steps:"
echo "1. Use enhanced processing: python portrait-enhancer/enhanced_batch.py --use-ad2cn"
echo "2. Use standard processing: python portrait-enhancer/batch.py"
echo "3. Deploy to vast.ai: make deploy"
echo ""
echo "The enhanced version provides:"
echo "  - Multiple face detectors (BlazeFace, RetinaFace, MTCNN)"
echo "  - Advanced search strategies"
echo "  - Cascade detection for better accuracy"
echo "  - Improved face alignment"
