# Docker Configuration

Ця папка містить всі Docker-пов'язані файли для Portrait Enhancement Pipeline.

## 📁 Файли

### `Dockerfile`
Основний Docker образ з:
- CUDA 11.8 runtime
- Stable Diffusion WebUI
- ControlNet та ADetailer розширення
- Python залежності
- Системні бібліотеки

### `docker-compose.yml`
Локальне розгортання з:
- Volume mapping для input/output/work
- GPU passthrough
- Port forwarding (7860)
- Автоматичний restart

## 🚀 Використання

### Локальний запуск
```bash
cd docker
docker-compose up --build
```

### Тільки збірка образу
```bash
cd docker
docker build -t portrait-enhancer .
```

### Запуск збірного образу
```bash
docker run -p 7860:7860 --gpus all portrait-enhancer
```

## 🔧 Налаштування

### GPU підтримка
Переконайтеся, що у вас встановлено:
- Docker з GPU підтримкою
- NVIDIA Container Runtime
- CUDA драйвери

### Volume mapping
- `./portrait-enhancer/input` → `/workspace/portrait-enhancer/input`
- `./portrait-enhancer/output` → `/workspace/portrait-enhancer/output`
- `./portrait-enhancer/work` → `/workspace/portrait-enhancer/work`
- `./models` → `/workspace/models`

## 📝 Примітки

- Образ базується на `nvidia/cuda:11.8-devel-ubuntu22.04`
- WebUI доступний на порту 7860
- Автоматично встановлюються всі необхідні розширення
- Bootstrap скрипт запускається автоматично при старті
