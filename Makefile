# Makefile for Portrait Enhancer Project
# Simplified version with essential commands only

.PHONY: help deploy upload download monitor build run clean test-connection

help: ## Show this help message
	@echo "🎨 Portrait Enhancer - Available Commands:"
	@echo ""
	@echo "🚀 Deployment:"
	@echo "  deploy          Deploy to vast.ai"
	@echo "  upload          Upload images to vast.ai"
	@echo "  download        Download results from vast.ai"
	@echo "  monitor         Monitor vast.ai instance"
	@echo "  download-models Download AI models from CivitAI"
	@echo ""
	@echo "🏗️  Local Development:"
	@echo "  build           Build Docker image"
	@echo "  run             Run locally with Docker"
	@echo "  clean           Clean up Docker resources"
	@echo ""
	@echo "🔗 Testing:"
	@echo "  test-connection Test SSH connection to vast.ai"
	@echo "  test            Run all tests"
	@echo ""
	@echo "📚 Help:"
	@echo "  help            Show this help message"

deploy: ## Deploy project to vast.ai
	@echo "🚀 Deploying to vast.ai..."
	./scripts/deploy_vast.sh

upload: ## Upload images to vast.ai
	@echo "📤 Uploading images to vast.ai..."
	./scripts/upload_images.sh

download: ## Download results from vast.ai
	@echo "📥 Downloading results from vast.ai..."
	./scripts/download_results.sh

monitor: ## Monitor vast.ai instance
	@echo "📊 Monitoring vast.ai instance..."
	./scripts/monitor.sh

download-models: ## Download AI models from CivitAI
	@echo "🤖 Downloading AI models from CivitAI..."
	cd scripts && python models_auto.py

build: ## Build Docker image
	@echo "🏗️  Building Docker image..."
	cd docker && docker-compose build

run: ## Run locally with Docker
	@echo "🏃 Running locally with Docker..."
	cd docker && docker-compose up -d

clean: ## Clean up Docker resources
	@echo "🧹 Cleaning up Docker resources..."
	cd docker && docker-compose down --volumes --remove-orphans
	docker system prune -f

test-connection: ## Test SSH connection to vast.ai
	@echo "🔗 Testing SSH connection to vast.ai..."
	ssh -p 18826 -o ConnectTimeout=10 root@ssh4.vast.ai 'echo "✅ З'\''єднання успішне!"'

test: ## Run all tests
	@echo "🧪 Running tests..."
	python -m pytest tests/ -v
