# Portrait-Enhancer Integration Guide

This document explains how ADetailer 2CN Plus integrates with your existing portrait-enhancer pipeline.

## Overview

ADetailer 2CN Plus is designed to **extend** your existing portrait-enhancer workflow, not replace it. It adds advanced face detection capabilities while using your existing A-Pass and B-Pass scripts.

## How It Works

### 1. **A-Pass Integration**
- **Input**: Your original image
- **Process**: Runs your existing `run_a_pass.py` script
- **Output**: Face masks, contour maps, and base enhanced image
- **Files Created**:
  - `work/masks/face_mask.png` - Face mask for B-Pass
  - `work/a_pass/base_enhanced.png` - A-Pass enhanced image
  - `work/a_pass/contour_map.png` - ControlNet2 contour map

### 2. **B-Pass Integration**
- **Input**: A-Pass results + face masks
- **Process**: Runs your existing `run_b_pass.py` script
- **Output**: Final enhanced image using A1111
- **Files Created**:
  - `output/final_enhanced.png` - Final B-Pass result

### 3. **Seamless Operation**
- No changes to your existing code
- Uses your existing configuration files
- Maintains your existing workflow
- Adds advanced face detection capabilities

## Directory Structure

```
your-project/
├── portrait-enhancer/          # Your existing installation
│   ├── run_a_pass.py          # Your A-Pass script
│   ├── run_b_pass.py          # Your B-Pass script
│   ├── config.yaml            # Your configuration
│   └── requirements.txt       # Your dependencies
│
└── adetailer_2cn_plus/        # New detection pipeline
    ├── ad2cn/                 # Core detection code
    ├── scripts/               # CLI scripts
    ├── config.yaml            # Detection configuration
    └── examples/              # Integration examples
```

## Setup Instructions

### 1. **Check Current Installation**

```bash
# Navigate to your project directory
cd your-project

# Check if portrait-enhancer exists
ls -la portrait-enhancer/
```

### 2. **Install ADetailer 2CN Plus**

```bash
# Clone the repository
git clone https://github.com/adetailer2cn/adetailer-2cn-plus.git

# Navigate to the directory
cd adetailer-2cn-plus

# Install dependencies
make setup
```

### 3. **Verify Integration**

```bash
# Check integration status
make check-integration

# Setup integration (creates symlinks)
make setup-integration
```

### 4. **Test Integration**

```bash
# Test with a sample image
make test-portrait-enhancer
```

## Configuration

### **Detection Configuration** (`adetailer_2cn_plus/config.yaml`)

```yaml
# A-Pass integration (uses your existing run_a_pass.py)
a_pass:
  enabled: true
  workdir: "work"  # Working directory for A-Pass processing

# B-Pass integration (uses your existing run_b_pass.py)
b_pass:
  enabled: true
  workdir: "work"  # Working directory for B-Pass processing
  output_dir: "output"  # Output directory for final results
  config_path: "config.yaml"  # Path to portrait-enhancer config

# Portrait-enhancer integration settings
portrait_enhancer:
  enabled: true
  auto_detect: true  # Auto-detect portrait-enhancer installation
  path: "../portrait-enhancer"  # Path to portrait-enhancer directory
  use_existing_config: true  # Use existing portrait-enhancer config
  fallback_to_basic: true  # Fallback to basic enhancement if portrait-enhancer fails
```

### **Your Existing Configuration** (`portrait-enhancer/config.yaml`)

Your existing configuration remains unchanged. ADetailer 2CN Plus will use:

- A1111 endpoint settings
- ControlNet2 model configurations
- ADetailer settings
- Prompt configurations
- All other existing settings

## Usage Examples

### **Single Image Processing**

```bash
# Full pipeline (detection + A-Pass + B-Pass)
python scripts/enhance.py -i image.jpg -o output/ -c config.yaml --enhance

# Detection only (no enhancement)
python scripts/enhance.py -i image.jpg -o output/ -c config.yaml --no-enhance
```

### **Batch Processing**

```bash
# Process multiple images
python scripts/enhance.py -i input_dir/ -o output_dir/ -c config.yaml --enhance --batch
```

### **Python API**

```python
from ad2cn.config import Config
from ad2cn.pipeline.detect import DetectionPipeline

# Load configuration
config = Config(**config_data)

# Initialize pipeline
pipeline = DetectionPipeline(config.pipeline.dict())

# Process image with enhancement
detections = pipeline.detect_faces(image, enable_enhancement=True)

# Save results
pipeline.save_enhanced_results(detections, "output/")
```

## Workflow Comparison

### **Before (Portrait-Enhancer Only)**
```
Input Image → run_a_pass.py → run_b_pass.py → Final Result
```

### **After (With ADetailer 2CN Plus)**
```
Input Image → Face Detection → run_a_pass.py → run_b_pass.py → Final Result
                ↓
        Multiple detectors
        Cascade detection
        Advanced search strategies
        Face alignment
        Performance profiling
```

## Benefits of Integration

### **Enhanced Detection**
- **Multiple Detectors**: BlazeFace (fast), RetinaFace (accurate), MTCNN (balanced)
- **Cascade Detection**: Multi-stage detection for better accuracy
- **Search Strategies**: Sliding window and multi-scale approaches
- **Face Alignment**: Automatic facial landmark detection

### **Performance Improvements**
- **Profiling**: Built-in timing and memory profiling
- **Optimization**: Configurable detector parameters
- **Batch Processing**: Efficient processing of multiple images

### **Maintained Compatibility**
- **No Code Changes**: Your existing scripts work unchanged
- **Same Output**: Same high-quality results
- **Same Configuration**: Use your existing settings
- **Same Workflow**: Maintain your current process

## Troubleshooting

### **Integration Issues**

```bash
# Check integration status
make check-integration

# Verify portrait-enhancer installation
ls -la ../portrait-enhancer/

# Check configuration files
cat ../portrait-enhancer/config.yaml
```

### **Common Problems**

1. **Portrait-Enhancer Not Found**
   ```
   Warning: portrait-enhancer not available, using fallback A-Pass
   ```
   - Solution: Verify the path in `config.yaml`
   - Check: `portrait_enhancer.path: "../portrait-enhancer"`

2. **A-Pass Script Not Found**
   ```
   ImportError: No module named 'run_a_pass'
   ```
   - Solution: Run `make setup-integration`
   - Check: Symlinks in `scripts/` directory

3. **B-Pass Configuration Issues**
   ```
   Error: A1111 endpoint not configured
   ```
   - Solution: Check your `portrait-enhancer/config.yaml`
   - Verify: `a1111_endpoint` setting

### **Fallback Mode**

If portrait-enhancer integration fails, the system automatically falls back to:

- **Basic Face Detection**: Using multiple detectors
- **Simple Enhancement**: Basic brightness/contrast adjustments
- **Local Processing**: No external dependencies

## Advanced Configuration

### **Custom Detector Settings**

```yaml
pipeline:
  detectors:
    - name: blazeface
      confidence_threshold: 0.3  # Lower threshold for more detections
      nms_threshold: 0.2         # Stricter NMS
      max_faces: 200             # Higher face limit
      input_size: 256            # Larger input size
```

### **Search Strategy Tuning**

```yaml
pipeline:
  search:
    strategy: sliding_window
    window_size: 1024            # Larger windows for high-res images
    stride: 512                  # Smaller stride for better coverage
```

### **Cascade Configuration**

```yaml
pipeline:
  enable_cascade: true
  cascade_order: [blazeface, retinaface, mtcnn]  # Custom order
```

## Performance Tuning

### **GPU Memory Optimization**

```yaml
use_gpu: true
gpu_memory_fraction: 0.6        # Use 60% of GPU memory
num_workers: 2                   # Reduce parallel processing
```

### **Detector Selection**

```yaml
# Fast processing
cascade_order: [blazeface]

# High accuracy
cascade_order: [retinaface, mtcnn]

# Balanced approach
cascade_order: [blazeface, retinaface]
```

## Monitoring and Debugging

### **Enable Debug Logging**

```yaml
log_level: DEBUG
log_file: "logs/debug.log"
```

### **Performance Profiling**

```python
from ad2cn.utils.timing import Timer

with Timer("custom_operation"):
    # Your code here
    pass

print(f"Operation took: {Timer.get_last('custom_operation').elapsed_time:.3f}s")
```

### **Pipeline Information**

```python
# Get detailed pipeline info
pipeline_info = pipeline.get_pipeline_info()
print("Detectors:", pipeline_info['detectors'])
print("Search Strategy:", pipeline_info['search_strategy'])
print("A-Pass Status:", pipeline_info.get('a_pass', {}).get('enabled'))
print("B-Pass Status:", pipeline_info.get('b_pass', {}).get('enabled'))
```

## Migration Guide

### **From Portrait-Enhancer Only**

1. **Install ADetailer 2CN Plus**
   ```bash
   git clone https://github.com/adetailer2cn/adetailer-2cn-plus.git
   cd adetailer-2cn-plus
   make setup
   ```

2. **Verify Integration**
   ```bash
   make check-integration
   make setup-integration
   ```

3. **Test Integration**
   ```bash
   make test-portrait-enhancer
   ```

4. **Update Your Workflow**
   - Use `scripts/enhance.py` instead of direct script calls
   - Benefit from enhanced face detection
   - Maintain same output quality

### **Configuration Migration**

Your existing `portrait-enhancer/config.yaml` remains unchanged. Only add detection-specific settings to `adetailer_2cn_plus/config.yaml`.

## Support and Community

- **Issues**: [GitHub Issues](https://github.com/adetailer2cn/adetailer-2cn-plus/issues)
- **Discussions**: [GitHub Discussions](https://github.com/adetailer2cn/adetailer-2cn-plus/discussions)
- **Wiki**: [Integration Wiki](https://github.com/adetailer2cn/adetailer-2cn-plus/wiki)

## Conclusion

ADetailer 2CN Plus enhances your existing portrait-enhancer pipeline by adding:

- **Advanced face detection** with multiple backends
- **Performance profiling** and optimization
- **Seamless integration** with your existing code
- **Fallback support** for independent operation

The integration is designed to be **non-intrusive** and **backward-compatible**, allowing you to benefit from enhanced detection capabilities while maintaining your proven enhancement workflow.
