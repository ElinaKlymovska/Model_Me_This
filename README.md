# ADetailer 2CN Plus

> Advanced face detection and enhancement pipeline for Automatic1111 integration

## Overview

ADetailer 2CN Plus is a sophisticated facial detection and enhancement system designed for high-quality image processing workflows. It combines multiple state-of-the-art face detection algorithms with advanced facial enhancement techniques to provide precise and customizable results.

### Key Features

- **Multi-Algorithm Detection**: MediaPipe, OpenCV, RetinaFace, BlazeFace support
- **Cascade Detection**: Intelligent fallback between detection methods
- **Facial Enhancement**: Advanced contouring and makeup application
- **Person-Specific Profiles**: Customized settings for different individuals
- **A1111 Integration**: Seamless integration with Automatic1111
- **Batch Processing**: Efficient processing of multiple images
- **GPU Acceleration**: ONNX runtime with GPU support

## Architecture

```
ADetailer 2CN Plus
├── Detection Pipeline
│   ├── MediaPipe Detector
│   ├── OpenCV Detector  
│   ├── RetinaFace Detector
│   ├── BlazeFace Detector
│   └── MTCNN Detector
├── Processing Pipelines
│   ├── A-Pass Pipeline
│   └── B-Pass Pipeline
├── Enhancement System
│   ├── Facial Contouring
│   ├── Expression Lines
│   └── Lip Enhancement
└── Search Strategies
    ├── Sliding Window
    └── Multi-Scale Search
```

## Installation

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (recommended)
- 8GB+ RAM
- 2GB+ VRAM

### Setup

1. **Clone the repository:**
```bash
git clone <repository-url>
cd ADetailer_2CN
```

2. **Install dependencies:**
```bash
# Using pip
pip install -r adetailer_2cn_plus/requirements.txt

# Or using poetry (recommended)
cd adetailer_2cn_plus
poetry install
```

3. **Download models:**
```bash
# Models will be automatically downloaded on first run
# Or manually download from releases page
```

## Configuration

The system uses a YAML-based configuration system. See `adetailer_2cn_plus/config.yaml` for the main configuration file.

### Key Configuration Sections

#### Detection Pipeline
```yaml
pipeline:
  detectors:
    - name: mediapipe
      confidence_threshold: 0.7
      detection_confidence: 0.7
      model_selection: 0
      with_landmarks: true
      max_faces: 5
    - name: opencv
      confidence_threshold: 0.3
      max_faces: 5
      scale_factor: 1.05
      min_neighbors: 3
  enable_fallback: true
  min_faces_threshold: 1
```

#### Facial Enhancement
```yaml
facial_enhancement:
  contouring:
    enabled: true
    intensity: 1.5
    blend_strength: 0.8
  expression_lines:
    enabled: true
    neutral_intensity: 0.3
    smile_intensity: 0.6
```

#### Person Profiles
```yaml
person_profiles:
  massy:
    contour_strength: 1.2
    expression_strength: 0.8
    lip_enhancement:
      cupids_bow_intensity: 1.2
      lip_volume_boost: 1.0
      color_intensity: 1.5
  orbi:
    contour_strength: 1.0
    expression_strength: 0.8
  yana:
    contour_strength: 1.1
    expression_strength: 0.75
```

## Usage

### Command Line Interface

#### Single Image Processing
```bash
python adetailer_2cn_plus/scripts/detect.py \
  -i path/to/image.jpg \
  -o path/to/output.jpg \
  -c adetailer_2cn_plus/config.yaml
```

#### Batch Processing
```bash
python adetailer_2cn_plus/scripts/detect.py \
  -i input_directory/ \
  -o output_directory/ \
  -c adetailer_2cn_plus/config.yaml \
  --batch
```

#### Pipeline Information
```bash
python adetailer_2cn_plus/scripts/detect.py \
  -c adetailer_2cn_plus/config.yaml \
  --info
```

### Python API

```python
from ad2cn import Config, DetectionPipeline
from ad2cn.utils.io import load_image
import yaml

# Load configuration
with open('config.yaml', 'r') as f:
    config_data = yaml.safe_load(f)
config = Config(**config_data)

# Initialize pipeline
pipeline = DetectionPipeline(config.pipeline.dict())

# Process image
image = load_image('image.jpg')
detections = pipeline.detect(image)

print(f"Detected {len(detections)} faces")
for i, detection in enumerate(detections):
    bbox = detection['bbox']
    confidence = detection['confidence']
    print(f"Face {i+1}: confidence={confidence:.3f}, bbox={bbox}")
```

## Workflow Chain

### 1. **Input Processing**
- Image loading and validation
- Format conversion and preprocessing
- Batch organization

### 2. **Detection Phase**
```
Image → Primary Detector (MediaPipe) → Fallback (OpenCV) → Validation
  ↓
Search Strategy (Sliding Window/Multi-Scale) → Face Candidates
  ↓  
NMS Filtering → Final Detections
```

### 3. **Enhancement Phase**
```
Detected Faces → Person Profile Selection → Enhancement Pipeline
  ↓
Contouring Application → Expression Lines → Lip Enhancement
  ↓
Blending → Quality Assessment → Final Output
```

### 4. **Processing Pipelines**
- **A-Pass**: Initial detection and alignment
- **B-Pass**: Refinement and enhancement
- **Integration**: A1111 API communication

### 5. **Output Generation**
- Visualization overlays
- Enhanced image output
- Detection metadata
- Performance metrics

## Project Structure

```
ADetailer_2CN/
├── adetailer_2cn_plus/           # Main package
│   ├── ad2cn/                    # Core library
│   │   ├── alignment/            # Face alignment
│   │   ├── detectors/            # Detection algorithms
│   │   ├── pipeline/             # Processing pipelines
│   │   ├── postprocess/          # Enhancement modules
│   │   ├── prompts/              # Enhanced prompts
│   │   ├── recovery/             # Face recovery
│   │   ├── search/               # Search strategies
│   │   └── utils/                # Utilities
│   ├── scripts/                  # CLI scripts
│   ├── config.yaml               # Main configuration
│   ├── pyproject.toml            # Poetry configuration
│   └── requirements.txt          # Dependencies
└── data/                         # Sample data
    ├── samples/                  # Test images
    │   ├── Massy/               # Massy profile samples
    │   ├── Orbi/                # Orbi profile samples
    │   └── Yana/                # Yana profile samples
    └── facial_expressions_and_age_lines/  # Reference data
```

## Performance Optimization

### GPU Acceleration
- ONNX Runtime GPU support
- CUDA memory optimization
- Batch processing efficiency

### Memory Management
- Configurable GPU memory fraction
- Automatic garbage collection
- Streaming for large batches

### Detection Optimization
- Cascade detection for speed/accuracy balance
- Multi-scale search for precision
- NMS optimization for duplicate removal

## Integration

### Automatic1111 Integration
```yaml
a1111_base_url: "http://127.0.0.1:7860"
```

The system seamlessly integrates with A1111 through:
- API endpoint communication
- Image format compatibility
- Batch processing support
- Real-time detection feedback

### Custom Detector Integration
To add new detectors:

1. Inherit from `FaceDetector` base class
2. Implement required methods:
   - `load_model()`
   - `detect_faces()`
   - `preprocess_image()`
3. Register in `DetectionPipeline`

## Troubleshooting

### Common Issues

#### GPU Memory Errors
```yaml
# Reduce GPU memory usage
gpu_memory_fraction: 0.6
num_workers: 2
```

#### Detection Quality Issues
```yaml
# Adjust thresholds
detectors:
  - name: mediapipe
    confidence_threshold: 0.8  # Increase for higher precision
    detection_confidence: 0.8
```

#### Performance Issues
```yaml
# Optimize for speed
pipeline:
  enable_cascade: true
  cascade_order: ['mediapipe', 'opencv']  # Fast → Accurate
```

## Development

### Code Quality
- Black formatting (line length: 88)
- Ruff linting
- MyPy type checking
- Pre-commit hooks

### Testing
```bash
# Run tests
pytest tests/

# With coverage
pytest --cov=ad2cn tests/
```

### Contributing
1. Follow PEP 8 style guidelines
2. Add type hints to all functions
3. Include docstrings for public APIs
4. Add tests for new features
5. Update documentation

## License

MIT License - see LICENSE file for details

## Acknowledgments

- **MediaPipe**: Google's face detection framework
- **InsightFace**: Advanced face analysis toolkit
- **OpenCV**: Computer vision library
- **ONNX**: Open neural network exchange format

## Support

For issues and questions:
- GitHub Issues: Report bugs and feature requests
- Documentation: See inline code documentation
- Configuration: Reference `config.yaml` examples

---

**Version**: 0.1.0  
**Authors**: ADetailer 2CN Team  
**Last Updated**: $(date +%Y-%m-%d)