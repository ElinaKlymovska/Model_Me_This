# Deployment Guide - Portrait Enhancement Pipeline

Детальна інструкція по розгортанню Portrait Enhancement Pipeline на vast.ai.

## 🚀 Швидкий старт

### 1. Підготовка SSH тунелю
```bash
# Відкрийте новий термінал і створіть SSH тунель
ssh -p 18826 root@ssh4.vast.ai -L 8080:localhost:8080
```

### 2. Розгортання проекту
```bash
# З основного терміналу
make deploy
```

### 3. Завантаження зображень
```bash
# Додайте зображення в portrait-enhancer/input/
make upload
```

### 4. Моніторинг
```bash
make monitor
```

### 5. Завантаження результатів
```bash
make download
```

## 📋 Детальні інструкції

### Перевірка з'єднання
```bash
make test-connection
```

### Ручне розгортання
```bash
# Створення директорій на vast.ai
ssh -p 18826 root@ssh4.vast.ai "mkdir -p /workspace/portrait-enhancer/{input,work,output}"

# Копіювання файлів
rsync -avz -e "ssh -p 18826" \
    --exclude '.git' \
    --exclude 'portrait-enhancer/input/*' \
    --exclude 'portrait-enhancer/output/*' \
    --exclude 'portrait-enhancer/work/*' \
    ./ root@ssh4.vast.ai:/workspace/

# Запуск bootstrap
ssh -p 18826 root@ssh4.vast.ai "cd /workspace && chmod +x bootstrap.sh && ./bootstrap.sh"
```

### Моніторинг процесу

#### Перевірка статусу WebUI
```bash
ssh -p 18826 root@ssh4.vast.ai 'curl -s http://127.0.0.1:7860/sdapi/v1/sd-models'
```

#### Перевірка ресурсів
```bash
ssh -p 18826 root@ssh4.vast.ai 'nvidia-smi && free -h && df -h'
```

#### Перегляд логів
```bash
ssh -p 18826 root@ssh4.vast.ai 'tmux attach -t webui'
```

### Управління процесом

#### Зупинка
```bash
ssh -p 18826 root@ssh4.vast.ai 'tmux kill-session -t webui'
```

#### Перезапуск
```bash
ssh -p 18826 root@ssh4.vast.ai 'cd /workspace && ./bootstrap.sh'
```

## 🔧 Конфігурація

### Змінні середовища
Основні параметри налаштовуються в `portrait-enhancer/config.yaml`:

```yaml
general:
  backend: "a1111"
  a1111_endpoint: "http://127.0.0.1:7860"
  model_checkpoint: ""

b_pass:
  prompt: "ultra sharp contouring, very strong cheekbone definition..."
  denoise: 0.18
  cfg: 5.0
  steps: 32
```

### Моделі
Проект автоматично завантажує:
- SDXL checkpoint з CivitAI
- ControlNet SoftEdge модель
- ControlNet Canny модель
- ADetailer face detection

## 📁 Структура файлів

```
/workspace/
├── stable-diffusion-webui/     # A1111 WebUI
├── portrait-enhancer/          # Основний модуль
│   ├── input/                  # Вхідні зображення
│   ├── work/                   # Проміжні файли
│   └── output/                 # Результати
└── models/                     # Завантажені моделі
```

## 🐛 Вирішення проблем

### Проблема: SSH з'єднання не встановлюється
```bash
# Перевірте правильність порту та IP
ssh -p 18826 root@ssh4.vast.ai

# Перевірте SSH ключ
ssh-add ~/.ssh/id_rsa
```

### Проблема: WebUI не запускається
```bash
# Перевірте логи
ssh -p 18826 root@ssh4.vast.ai 'tmux attach -t webui'

# Перевірте доступність порту
ssh -p 18826 root@ssh4.vast.ai 'netstat -tlnp | grep 7860'
```

### Проблема: Моделі не завантажуються
```bash
# Перевірте доступ до CivitAI
ssh -p 18826 root@ssh4.vast.ai 'curl -s https://civitai.com'

# Запустіть завантаження моделей вручну
ssh -p 18826 root@ssh4.vast.ai 'cd /workspace && python models_auto.py'
```

## 📊 Моніторинг ресурсів

### GPU використання
```bash
ssh -p 18826 root@ssh4.vast.ai 'nvidia-smi -l 1'
```

### Пам'ять та диск
```bash
ssh -p 18826 root@ssh4.vast.ai 'htop'
```

### Логи WebUI
```bash
ssh -p 18826 root@ssh4.vast.ai 'tail -f /workspace/stable-diffusion-webui/logs/webui.log'
```

## 🔄 Автоматизація

### Cron job для моніторингу
```bash
# Додайте в crontab
*/5 * * * * /path/to/monitor.sh >> /var/log/portrait-enhancer.log 2>&1
```

### Автоматичне завантаження результатів
```bash
# Скрипт для періодичного завантаження
while true; do
    make download
    sleep 300  # 5 хвилин
done
```

## 📞 Підтримка

При виникненні проблем:
1. Перевірте логи: `make logs`
2. Перевірте статус: `make status`
3. Перезапустіть: `make stop && make run`
4. Зверніться до документації або створіть issue
