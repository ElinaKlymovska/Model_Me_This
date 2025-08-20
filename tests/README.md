# Tests Directory

Ця папка містить тести для Portrait Enhancement Pipeline.

## 📁 Файли

### `__init__.py`
**Ініціалізація пакету тестів**

### `test_config.py`
**Тести конфігураційних файлів**
- Валідація YAML файлів
- Перевірка структури конфігурації
- Тестування наявності обов'язкових секцій

### `test_basic.py`
**Базові тести функціональності**
- Перевірка структури проекту
- Тестування наявності файлів
- Перевірка імпорту модулів
- Тестування завантаження конфігурації

## 🚀 Запуск тестів

### Запуск всіх тестів
```bash
# З кореневої папки проекту
python -m pytest tests/

# Або через unittest
python -m unittest discover tests/
```

### Запуск конкретного тесту
```bash
# Тест конфігурації
python tests/test_config.py

# Тест базової функціональності
python tests/test_basic.py
```

### Запуск через Makefile
```bash
make test
```

## 📝 Створення нових тестів

### Структура тесту
```python
import unittest

class TestYourFeature(unittest.TestCase):
    """Test description"""
    
    def setUp(self):
        """Set up test fixtures"""
        pass
    
    def test_something(self):
        """Test specific functionality"""
        # Your test code here
        self.assertTrue(True)
    
    def tearDown(self):
        """Clean up after test"""
        pass
```

### Правила написання тестів
1. **Назва класу**: `TestFeatureName`
2. **Назва методу**: `test_specific_functionality`
3. **Документація**: кожен тест має docstring
4. **Assertions**: використовуйте assert методи unittest
5. **Cleanup**: використовуйте tearDown для очищення

## 🔧 Налаштування

### Залежності для тестування
```bash
pip install pytest pytest-cov
```

### Конфігурація pytest
Створіть `pytest.ini` в корені проекту:
```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v --tb=short
```

## 📊 Покриття тестами

### Перевірка покриття
```bash
# Запуск з покриттям
python -m pytest --cov=portrait-enhancer tests/

# Генерація HTML звіту
python -m pytest --cov=portrait-enhancer --cov-report=html tests/
```

### Цілі покриття
- **Мінімум**: 70% коду
- **Рекомендовано**: 80% коду
- **Ідеально**: 90%+ коду

## 🐛 Відладка тестів

### Детальний вивід
```bash
python -m pytest -v tests/
```

### Зупинка на першій помилці
```bash
python -m pytest -x tests/
```

### Запуск конкретного тесту з детальним виводом
```bash
python -m pytest -v -s tests/test_config.py::TestConfiguration::test_config_structure
```

## 📚 Приклади

### Тест API
```python
def test_api_response(self):
    """Test API endpoint response"""
    response = requests.get('http://localhost:7860/sdapi/v1/sd-models')
    self.assertEqual(response.status_code, 200)
    self.assertIsInstance(response.json(), list)
```

### Тест файлової системи
```python
def test_output_directory(self):
    """Test output directory creation"""
    output_dir = Path("portrait-enhancer/output")
    output_dir.mkdir(exist_ok=True)
    self.assertTrue(output_dir.exists())
    self.assertTrue(output_dir.is_dir())
```
