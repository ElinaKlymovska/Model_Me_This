.PHONY: help deploy upload download monitor build run clean

help: ## Показати довідку
	@echo "Portrait Enhancement Pipeline - Makefile"
	@echo ""
	@echo "Доступні команди:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

deploy: ## Розгорнути проект на vast.ai
	@echo "🚀 Розгортання проекту на vast.ai..."
	./deploy_vast.sh

upload: ## Завантажити зображення на vast.ai
	@echo "📤 Завантаження зображень..."
	./upload_images.sh

download: ## Завантажити результати з vast.ai
	@echo "📥 Завантаження результатів..."
	./download_results.sh

monitor: ## Моніторинг процесу на vast.ai
	@echo "📊 Моніторинг процесу..."
	./monitor.sh

build: ## Збудувати Docker образ
	@echo "🔨 Будування Docker образу..."
	docker-compose build

run: ## Запустити локально через Docker
	@echo "🏃 Запуск локально..."
	docker-compose up

run-detached: ## Запустити локально через Docker у фоновому режимі
	@echo "🏃 Запуск локально у фоновому режимі..."
	docker-compose up -d

stop: ## Зупинити Docker контейнери
	@echo "🛑 Зупинка контейнерів..."
	docker-compose down

clean: ## Очистити Docker ресурси
	@echo "🧹 Очищення Docker ресурсів..."
	docker-compose down --rmi all --volumes --remove-orphans

status: ## Перевірити статус Docker контейнерів
	@echo "📊 Статус контейнерів..."
	docker-compose ps

logs: ## Показати логи Docker контейнерів
	@echo "📝 Логи контейнерів..."
	docker-compose logs -f

ssh: ## Підключитися до vast.ai через SSH
	@echo "🔌 Підключення до vast.ai..."
	ssh -p 18826 root@ssh4.vast.ai -L 8080:localhost:8080

setup: ## Налаштувати проект (перший раз)
	@echo "⚙️ Налаштування проекту..."
	mkdir -p portrait-enhancer/{input,work,output}
	chmod +x *.sh
	@echo "✅ Проект налаштовано!"

test-connection: ## Перевірити з'єднання з vast.ai
	@echo "🔍 Перевірка з'єднання з vast.ai..."
	ssh -p 18826 -o ConnectTimeout=10 root@ssh4.vast.ai 'echo "✅ З'\''єднання успішне!"'

test-ad2cn: ## Тестувати ADetailer 2CN Plus інтеграцію
	@echo "🧪 Тестування ADetailer 2CN Plus..."
	./test_ad2cn.sh

enhanced-upload: ## Завантажити зображення та обробити з ADetailer 2CN Plus
	@echo "🎯 Завантаження зображень з розширеною обробкою..."
	./upload_images.sh
	@echo "🚀 Запуск розширеної обробки..."
	ssh -p 18826 root@ssh4.vast.ai 'cd /workspace && python portrait-enhancer/enhanced_batch.py --use-ad2cn'
