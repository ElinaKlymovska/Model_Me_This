# Configuration Directory

Ця папка містить всі конфігураційні файли для Portrait Enhancement Pipeline.

## 📁 Файли

### `config.yaml`
**Основна конфігурація проекту**
- Загальні налаштування
- WebUI конфігурація
- ADetailer 2CN Plus налаштування
- Параметри обробки
- Налаштування детекції обличчя
- Параметри покращення
- ControlNet налаштування
- vast.ai розгортання
- Docker налаштування
- Логування

### `models.yaml`
**Конфігурація моделей**
- Stable Diffusion моделі
- ControlNet моделі
- ADetailer моделі
- Налаштування завантаження
- Шляхи до моделей

## 🔧 Використання

### Завантаження конфігурації
```python
import yaml
from pathlib import Path

# Завантаження основної конфігурації
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Завантаження конфігурації моделей
with open('config/models.yaml', 'r') as f:
    models_config = yaml.safe_load(f)
```

### Отримання значень
```python
# WebUI налаштування
webui_host = config['webui']['host']
webui_port = config['webui']['port']

# ADetailer налаштування
detectors = config['adetailer_2cn']['detectors']
confidence = config['adetailer_2cn']['confidence_threshold']

# Моделі
sd_models = models_config['stable_diffusion']['models']
cn_models = models_config['controlnet']['models']
```

## 📝 Налаштування

### Зміна параметрів
1. Відредагуйте відповідний .yaml файл
2. Перезапустіть програму
3. Зміни вступлять в силу

### Перемінні середовища
Деякі параметри можна перевизначити через змінні середовища:
```bash
export WEBUI_HOST="0.0.0.0"
export WEBUI_PORT="7860"
export DEBUG="true"
```

## 🔒 Безпека

- **Не комітьте** файли з API ключами
- Використовуйте `.env` файли для чутливих даних
- Перевіряйте права доступу до конфігураційних файлів

## 📚 Приклади

### Створення власної конфігурації
```yaml
# config/custom.yaml
custom:
  my_setting: "value"
  my_list: [1, 2, 3]
  my_dict:
    key: "value"
```

### Валідація конфігурації
```python
from pydantic import BaseModel, Field

class WebUIConfig(BaseModel):
    host: str = Field(default="127.0.0.1")
    port: int = Field(default=7860, ge=1, le=65535)
    timeout: int = Field(default=300, ge=1)
```
