# Scripts Directory

Ця папка містить всі bash скрипти для управління Portrait Enhancement Pipeline.

## 📁 Файли

### `bootstrap.sh`
**Основна установка та налаштування**
- Встановлює Stable Diffusion WebUI
- Налаштовує ControlNet та ADetailer розширення
- Завантажує необхідні моделі
- Запускає WebUI сервер

### `deploy_vast.sh`
**Розгортання на vast.ai**
- Створює необхідні директорії
- Копіює файли проекту
- Запускає bootstrap процес
- Налаштовує середовище

### `upload_images.sh`
**Завантаження зображень на vast.ai**
- Синхронізує input директорію
- Підтримує різні формати зображень
- Автоматичне створення структури папок

### `download_results.sh`
**Завантаження результатів з vast.ai**
- Скачує оброблені зображення
- Зберігає структуру папок
- Підтримує масове завантаження

### `monitor.sh`
**Моніторинг vast.ai інстансу**
- Перевіряє статус WebUI
- Показує використання ресурсів
- Відстежує процес обробки

## 🚀 Використання

### Всі скрипти
```bash
# З кореневої папки проекту
./scripts/bootstrap.sh
./scripts/deploy_vast.sh
./scripts/upload_images.sh
./scripts/download_results.sh
./scripts/monitor.sh
```

### Через Makefile
```bash
make deploy      # deploy_vast.sh
make upload      # upload_images.sh
make download    # download_results.sh
make monitor     # monitor.sh
```

## 🔧 Налаштування

### SSH доступ
Всі скрипти використовують SSH з'єднання з vast.ai:
- **Порт**: 18826
- **Користувач**: root
- **Хост**: ssh4.vast.ai

### Шляхи
Скрипти очікують наступну структуру:
- `portrait-enhancer/input/` - вхідні зображення
- `portrait-enhancer/output/` - результати обробки
- `portrait-enhancer/work/` - тимчасові файли

## 📝 Примітки

- Всі скрипти мають права на виконання (`chmod +x`)
- Переконайтеся, що SSH ключі налаштовані
- Скрипти автоматично створюють необхідні директорії
- Логи зберігаються в терміналі та можуть бути перенаправлені
