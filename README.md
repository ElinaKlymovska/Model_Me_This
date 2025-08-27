# ADetailer 2CN Plus

Face detection and enhancement pipeline for A1111.

## Installation

```bash
pip install -e .
```

## Usage

```bash
# Detect faces
detect --input image.jpg

# Enhance portraits  
enhance --input image.jpg --output enhanced.jpg
```

## Configuration

Edit `adetailer_2cn_plus/config.yaml` for settings.

## Structure

```
adetailer_2cn_plus/     # Main package
├── ad2cn/              # Core modules
└── config.yaml         # Settings

data/
├── samples/            # Test images
└── models/             # ML models
```

## License

MIT