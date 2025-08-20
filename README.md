# 🎨 Portrait Enhancer with ADetailer 2CN Plus

**Intelligent portrait enhancement pipeline with smart fallback logic and advanced face detection**

## ✨ Features

- **🎯 Smart Processing Logic**: Automatically chooses the best enhancement method based on face count
  - **2+ faces** → Basic enhancement for stability
  - **1 face** → ADetailer 2CN Plus for detailed processing
  - **0 faces** → Basic enhancement as fallback
- **🔍 Advanced Face Detection**: Multiple detectors (BlazeFace, RetinaFace, MTCNN)
- **📊 Comprehensive Analysis**: Detailed face counting and processing statistics
- **🚀 Cloud Deployment**: Optimized for vast.ai GPU instances
- **🐳 Docker Support**: Easy local development and deployment

## 🏗️ Project Structure

```
ADetailer_2CN/
├── adetailer_2cn_plus/          # ADetailer 2CN Plus library
├── portrait-enhancer/           # Main application
│   ├── input/                   # Input images
│   ├── output/                  # Enhanced images
│   ├── work/                    # Temporary files
│   ├── simple_enhance.py        # Smart enhancement script
│   ├── face_count_analyzer.py   # Face detection analyzer
│   ├── face_verification.py     # Result verification
│   └── requirements.txt         # Dependencies
├── docker/                      # Docker configuration
│   ├── Dockerfile              # Container configuration
│   ├── docker-compose.yml      # Docker orchestration
│   └── README.md               # Docker documentation
├── scripts/                    # Scripts for project management
│   ├── bootstrap.sh           # Setup and configuration
│   ├── deploy_vast.sh         # vast.ai deployment
│   ├── upload_images.sh       # Image upload
│   ├── download_results.sh    # Results download
│   ├── monitor.sh             # Instance monitoring
│   ├── models_auto.py         # AI models downloader
│   └── README.md              # Scripts documentation
├── config/                     # Configuration files
│   ├── config.yaml            # Main configuration
│   ├── models.yaml            # Models configuration
│   └── README.md              # Configuration documentation
├── tests/                      # Test suite
│   ├── test_config.py         # Configuration tests
│   ├── test_basic.py          # Basic functionality tests
│   └── README.md              # Testing documentation
├── Makefile                    # Simplified command management
└── README.md                   # This file
```

## 🚀 Quick Start

### 1. **Deploy to vast.ai**
```bash
make deploy
```

### 2. **Upload Images**
```bash
make upload
```

### 3. **Process Images with Smart Logic**
```bash
# SSH to vast.ai and run smart enhancement
ssh -p 18826 root@ssh4.vast.ai
cd /workspace/portrait-enhancer
python simple_enhance.py --smart --use-ad2cn
```

### 4. **Download Results**
```bash
make download
```

### 5. **Analyze Results**
```bash
cd portrait-enhancer
python face_count_analyzer.py --use-ad2cn
python face_verification.py
```

## 🎯 Smart Processing Logic

The system intelligently chooses the enhancement method:

| Face Count | Processing Method | Reason |
|------------|-------------------|---------|
| **0** | Basic Enhancement | No faces detected, use fallback |
| **1** | ADetailer 2CN Plus | Single face, detailed enhancement |
| **2+** | Basic Enhancement | Multiple faces, stability priority |

## 🛠️ Available Commands

```bash
# 🚀 Deployment
make deploy          # Deploy to vast.ai
make upload          # Upload images
make download        # Download results
make monitor         # Monitor instance

# 🏗️ Local Development
make build           # Build Docker image
make run             # Run locally
make clean           # Clean up resources

# 🔗 Testing
make test-connection # Test SSH connection
make help            # Show all commands
```

## 🔧 Local Development

### **Prerequisites**
- Docker & Docker Compose
- Python 3.8+
- SSH access to vast.ai

### **Setup**
```bash
# Clone repository
git clone https://github.com/ElinaKlymovska/Model_Me_This.git
cd Model_Me_This

# Build and run locally
make build
make run

# Check status
docker-compose ps
```

## 📊 Processing Results

### **Face Detection Analysis**
- **Total images**: 25
- **Success rate**: 100%
- **Face distribution**: 0-3 faces per image
- **Processing recommendations**: Automatic method selection

### **Output Structure**
```
portrait-enhancer/output/
├── smart_enhanced/              # Smart processing results
├── all_images/                  # Basic processing results
├── enhanced_face_count_analysis.txt
├── face_count_analysis.txt
└── face_verification_results.txt
```

## 🎨 ADetailer 2CN Plus Benefits

- **Multiple Detectors**: BlazeFace, RetinaFace, MTCNN
- **Advanced Search**: Sliding window, multi-scale detection
- **Face Alignment**: Automatic orientation correction
- **Cascade Detection**: Two-pass detection pipeline
- **Smart Fallback**: Automatic fallback to basic processing

## 🔍 Analysis Tools

### **Face Count Analyzer**
```bash
python face_count_analyzer.py --use-ad2cn
```
- Detects faces in all images
- Recommends processing method
- Provides detailed statistics

### **Face Verification**
```bash
python face_verification.py
```
- Compares original vs enhanced
- Reports face count changes
- Performance analysis

## 🐳 Docker Configuration

Всі Docker файли організовані в папці `docker/`:

- **`docker/Dockerfile`** - Основний образ з CUDA та Stable Diffusion
- **`docker/docker-compose.yml`** - Локальне розгортання
- **`docker/README.md`** - Детальна документація Docker

### **Локальний запуск**
```bash
# Збірка та запуск
make build
make run

# Або безпосередньо
cd docker
docker-compose up --build
```

## 📜 Scripts Organization

Всі bash скрипти організовані в папці `scripts/`:

- **`scripts/bootstrap.sh`** - Основна установка та налаштування
- **`scripts/deploy_vast.sh`** - Розгортання на vast.ai
- **`scripts/upload_images.sh`** - Завантаження зображень
- **`scripts/download_results.sh`** - Завантаження результатів
- **`scripts/monitor.sh`** - Моніторинг інстансу
- **`scripts/README.md`** - Детальна документація скриптів

### **Використання скриптів**
```bash
# Через Makefile (рекомендовано)
make deploy          # deploy_vast.sh
make upload          # upload_images.sh
make download        # download_results.sh
make monitor         # monitor.sh
make download-models # models_auto.py

# Або безпосередньо
./scripts/bootstrap.sh
./scripts/deploy_vast.sh
./scripts/models_auto.py
```

### **Завантаження AI моделей**
```bash
# Автоматичне завантаження всіх моделей
make download-models

# Або безпосередньо
cd scripts
python models_auto.py
```

## ⚙️ Configuration Management

Всі конфігураційні файли організовані в папці `config/`:

- **`config/config.yaml`** - Основна конфігурація проекту
- **`config/models.yaml`** - Конфігурація моделей
- **`config/README.md`** - Детальна документація конфігурації

### **Використання конфігурації**
```python
import yaml

# Завантаження конфігурації
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Отримання значень
webui_host = config['webui']['host']
webui_port = config['webui']['port']
```

## 🧪 Testing

Тестова підмножина організована в папці `tests/`:

- **`tests/test_config.py`** - Тести конфігураційних файлів
- **`tests/test_basic.py`** - Базові тести функціональності
- **`tests/README.md`** - Детальна документація тестування

### **Запуск тестів**
```bash
# Всі тести
make test

# Конкретний тест
python tests/test_config.py

# Через pytest
python -m pytest tests/ -v
```

## 🚀 vast.ai Deployment

### **SSH Connection**
```bash
ssh -p 18826 root@ssh4.vast.ai -L 8080:localhost:7860
```

### **Environment**
- **OS**: Ubuntu 22.04
- **GPU**: NVIDIA CUDA 11.8
- **Python**: 3.8+
- **Port**: 7860 (WebUI), 8080 (tunnel)

## 📝 Configuration

### **Docker Environment**
```yaml
# docker/docker-compose.yml
environment:
  - NVIDIA_VISIBLE_DEVICES=all
  - NVIDIA_DRIVER_CAPABILITIES=compute,utility
  - PORT=7860
```

### **Volume Mounts**
```yaml
volumes:
  - ./portrait-enhancer/input:/workspace/portrait-enhancer/input
  - ./portrait-enhancer/output:/workspace/portrait-enhancer/output
  - ./portrait-enhancer/work:/workspace/portrait-enhancer/work
```

## 🔧 Troubleshooting

### **Common Issues**
1. **SSH Connection Failed**: Check port and credentials
2. **Docker Build Error**: Ensure Docker is running
3. **GPU Not Available**: Verify NVIDIA drivers
4. **Processing Failed**: Check input image format

### **Debug Commands**
```bash
# Test connection
make test-connection

# Check logs
cd docker && docker-compose logs

# Monitor resources
make monitor
```

## 📚 Dependencies

### **Core Requirements**
- `opencv-python` - Face detection
- `numpy` - Numerical operations
- `PIL` - Image processing
- `pydantic` - Configuration management

### **ADetailer 2CN Plus**
- `onnxruntime-gpu` - AI model inference
- `insightface` - Advanced face detection
- `facenet-pytorch` - Face recognition

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Make changes
4. Test thoroughly
5. Submit pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **ADetailer 2CN Plus** - Advanced face detection
- **vast.ai** - Cloud GPU infrastructure
- **Stable Diffusion WebUI** - AI image processing foundation

---

**🎯 Smart. Efficient. Intelligent Portrait Enhancement.**
