# Portrait Enhancement Pipeline

Автоматична система покращення портретів з використанням Stable Diffusion, ControlNet та ADetailer.

## Структура проєкту

```
ADetailer_2CN/
├── README.md                 # Документація проєкту
├── bootstrap.sh              # Автоматична установка та налаштування
├── models_auto.py            # Автоматичне завантаження моделей
├── Dockerfile                # Docker образ для розгортання
├── docker-compose.yml        # Docker Compose конфігурація
├── deploy_vast.sh            # Скрипт розгортання на vast.ai
├── upload_images.sh          # Скрипт завантаження зображень
├── download_results.sh       # Скрипт завантаження результатів
├── monitor.sh                # Скрипт моніторингу
├── portrait-enhancer/        # Основний модуль покращення
│   ├── config.yaml          # Конфігурація параметрів
│   ├── batch.py             # Пакетна обробка зображень
│   ├── run_a_pass.py        # Перший прохід - створення масок та контурів
│   ├── run_b_pass.py        # Другий прохід - AI покращення
│   └── requirements.txt     # Залежності Python
└── .gitignore               # Git ігнорування
```

## Основні функції

- **Автоматична установка** Stable Diffusion WebUI та розширень
- **Завантаження моделей** з CivitAI
- **Створення масок** обличчя та контурних карт
- **AI покращення** з використанням ControlNet та ADetailer
- **Пакетна обробка** зображень
- **Розгортання на vast.ai** з автоматизацією
- **🎯 ADetailer 2CN Plus** - розширена детекція облич з множинними детекторами

## Швидкий старт на vast.ai

### 1. Підготовка SSH тунелю
```bash
ssh -p 18826 root@ssh4.vast.ai -L 8080:localhost:8080
```

### 2. Розгортання проекту
```bash
chmod +x deploy_vast.sh
./deploy_vast.sh
```

### 3. Завантаження зображень
```bash
chmod +x upload_images.sh
./upload_images.sh [шлях_до_зображень]
```

### 4. Моніторинг процесу
```bash
chmod +x monitor.sh
./monitor.sh
```

### 5. Завантаження результатів
```bash
chmod +x download_results.sh
./download_results.sh
```

## Локальне використання

### 1. Запуск через Docker
```bash
docker-compose up --build
```

### 2. Ручна установка
```bash
chmod +x bootstrap.sh
./bootstrap.sh
```

## Використання

1. Помістіть зображення в папку `portrait-enhancer/input/`
2. Запустіть `python batch.py` для обробки
3. Результати зберігаються в `portrait-enhancer/output/`

## Технології

- **Stable Diffusion WebUI** - основа для генерації
- **ControlNet** - контроль структури зображення
- **ADetailer** - деталізація обличчя
- **PIL/Pillow** - обробка зображень
- **PyYAML** - конфігурація
- **Docker** - контейнеризація
- **vast.ai** - хмарне розгортання

## 🎯 ADetailer 2CN Plus

Проект включає розширену версію ADetailer з покращеною детекцією облич:

### Переваги:
- **Множинні детектори**: BlazeFace (швидкий), RetinaFace (точний), MTCNN (збалансований)
- **Cascade detection**: багатоетапна детекція для кращої точності
- **Розширені стратегії пошуку**: sliding window та multi-scale підходи
- **Автоматичне вирівнювання облич**: детекція ключових точок
- **Повна інтеграція** з існуючим portrait-enhancer pipeline

### Використання:
```bash
# Тестування інтеграції
make test-ad2cn

# Розширена обробка зображень
python portrait-enhancer/enhanced_batch.py --use-ad2cn

# Завантаження та розширена обробка на vast.ai
make enhanced-upload
```

## Моніторинг та управління

### Перевірка статусу WebUI
```bash
ssh -p 18826 root@ssh4.vast.ai 'curl -s http://127.0.0.1:7860/sdapi/v1/sd-models'
```

### Приєднання до tmux сесії
```bash
ssh -p 18826 root@ssh4.vast.ai 'tmux attach -t webui'
```

### Зупинка процесу
```bash
ssh -p 18826 root@ssh4.vast.ai 'tmux kill-session -t webui'
```

## Конфігурація

Основні параметри налаштовуються в `portrait-enhancer/config.yaml`:

- **ControlNet моделі** - для контролю структури
- **ADetailer налаштування** - для деталізації обличчя
- **Промпти** - для керування результатом
- **Параметри генерації** - steps, CFG, denoise

## Підтримувані формати

- **Вхідні**: PNG, JPG, JPEG, WebP
- **Вихідні**: PNG (збереження якості)
- **Моделі**: SafeTensors, CKPT
