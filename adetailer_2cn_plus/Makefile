# ADetailer 2CN Plus Makefile
# Provides convenient commands for development and deployment
# Integrated with portrait-enhancer pipeline

.PHONY: help setup test fmt lint clean run-detect run-enhance install-dev install-prod

# Default target
help:
	@echo "ADetailer 2CN Plus - Available Commands:"
	@echo ""
	@echo "Setup & Installation:"
	@echo "  setup        - Install dependencies and setup environment"
	@echo "  install-dev  - Install development dependencies"
	@echo "  install-prod - Install production dependencies"
	@echo ""
	@echo "Development:"
	@echo "  test         - Run tests with pytest"
	@echo "  fmt          - Format code with black and ruff"
	@echo "  lint         - Run linting with ruff and mypy"
	@echo "  clean        - Clean build artifacts and cache"
	@echo ""
	@echo "Execution:"
	@echo "  run-detect   - Run face detection only"
	@echo "  run-enhance  - Run face detection + A-Pass + B-Pass enhancement"
	@echo "  test-run     - Quick test run"
	@echo ""
	@echo "Use 'make help' to see this message"

# Setup environment
setup: install-dev
	@echo "Setting up ADetailer 2CN Plus..."
	@mkdir -p models logs work output
	@echo "Created models/, logs/, work/, and output/ directories"
	@if [ ! -f .env ]; then \
		echo "Creating .env from env.example..."; \
		cp env.example .env; \
		echo "Please edit .env with your configuration"; \
	fi
	@echo "Setup complete!"

# Install development dependencies
install-dev:
	@echo "Installing development dependencies..."
	poetry install --with dev
	@echo "Development dependencies installed!"

# Install production dependencies only
install-prod:
	@echo "Installing production dependencies..."
	poetry install --only main
	@echo "Production dependencies installed!"

# Run tests
test:
	@echo "Running tests..."
	poetry run pytest tests/ -v --cov=ad2cn --cov-report=term-missing

# Format code
fmt:
	@echo "Formatting code with black..."
	poetry run black ad2cn/ scripts/ tests/
	@echo "Sorting imports with ruff..."
	poetry run ruff check --fix ad2cn/ scripts/ tests/
	@echo "Code formatting complete!"

# Run linting
lint:
	@echo "Running ruff linting..."
	poetry run ruff check ad2cn/ scripts/ tests/
	@echo "Running mypy type checking..."
	poetry run mypy ad2cn/ scripts/ tests/
	@echo "Linting complete!"

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	@echo "Cleanup complete!"

# Run face detection only
run-detect:
	@echo "Running face detection..."
	@if [ ! -f config.yaml ]; then \
		echo "Error: config.yaml not found. Please create it first."; \
		exit 1; \
	fi
	@echo "Note: This requires a valid config.yaml and input image"
	@echo "Usage: poetry run python scripts/detect.py -i image.jpg -o result.jpg -c config.yaml"

# Run face detection + enhancement
run-enhance:
	@echo "Running face detection + enhancement..."
	@if [ ! -f config.yaml ]; then \
		echo "Error: config.yaml not found. Please create it first."; \
		exit 1; \
	fi
	@echo "Note: This requires a valid config.yaml and input image"
	@echo "Usage: poetry run python scripts/enhance.py -i image.jpg -o output/ -c config.yaml --enhance"

# Quick test run (requires sample image)
test-run:
	@echo "Running quick test..."
	@if [ ! -f "tests/test_data/sample.jpg" ]; then \
		echo "Creating test data directory..."; \
		mkdir -p tests/test_data; \
		echo "Please add a sample.jpg to tests/test_data/ for testing"; \
		exit 1; \
	fi
	@echo "Testing detection only..."
	poetry run python scripts/detect.py -i tests/test_data/sample.jpg -o tests/test_data/result.jpg -c config.yaml
	@echo "Testing detection + enhancement..."
	poetry run python scripts/enhance.py -i tests/test_data/sample.jpg -o tests/test_data/enhanced/ -c config.yaml --enhance

# Test portrait-enhancer integration
test-portrait-enhancer:
	@echo "Testing portrait-enhancer integration..."
	@if [ ! -d "../portrait-enhancer" ]; then \
		echo "Warning: portrait-enhancer directory not found at ../portrait-enhancer"; \
		echo "Integration test will use fallback mode"; \
	fi
	@if [ ! -f "tests/test_data/sample.jpg" ]; then \
		echo "Error: Please add a sample.jpg to tests/test_data/ for testing"; \
		exit 1; \
	fi
	poetry run python scripts/enhance.py -i tests/test_data/sample.jpg -o tests/test_data/portrait_test/ -c config.yaml --enhance

# Build package
build:
	@echo "Building package..."
	poetry build

# Publish to PyPI (use with caution)
publish:
	@echo "Publishing to PyPI..."
	poetry publish

# Run pre-commit hooks
pre-commit:
	@echo "Running pre-commit hooks..."
	poetry run pre-commit run --all-files

# Install pre-commit hooks
install-hooks:
	@echo "Installing pre-commit hooks..."
	poetry run pre-commit install

# Docker commands
docker-build:
	@echo "Building Docker image..."
	docker build -t adetailer2cn-plus .

docker-run:
	@echo "Running Docker container..."
	docker run -it --gpus all -v $(PWD):/workspace adetailer2cn-plus

# Development server
dev-server:
	@echo "Starting development server..."
	poetry run python -m ad2cn.server

# Generate requirements.txt from poetry
requirements:
	@echo "Generating requirements.txt..."
	poetry export -f requirements.txt --output requirements.txt --without-hashes

# Check for security vulnerabilities
security-check:
	@echo "Checking for security vulnerabilities..."
	poetry run safety check

# Update dependencies
update-deps:
	@echo "Updating dependencies..."
	poetry update

# Show dependency tree
deps-tree:
	@echo "Dependency tree:"
	poetry show --tree

# Check portrait-enhancer integration
check-integration:
	@echo "Checking portrait-enhancer integration..."
	@if [ -d "../portrait-enhancer" ]; then \
		echo "✓ portrait-enhancer directory found"; \
		if [ -f "../portrait-enhancer/run_a_pass.py" ]; then \
			echo "✓ A-Pass script found"; \
		else \
			echo "⚠ A-Pass script not found"; \
		fi; \
		if [ -f "../portrait-enhancer/run_b_pass.py" ]; then \
			echo "✓ B-Pass script found"; \
		else \
			echo "⚠ B-Pass script not found"; \
		fi; \
		if [ -f "../portrait-enhancer/config.yaml" ]; then \
			echo "✓ portrait-enhancer config found"; \
		else \
			echo "⚠ portrait-enhancer config not found"; \
		fi; \
	else \
		echo "⚠ portrait-enhancer directory not found at ../portrait-enhancer"; \
		echo "Integration will use fallback mode"; \
	fi

# Setup portrait-enhancer integration
setup-integration:
	@echo "Setting up portrait-enhancer integration..."
	@if [ -d "../portrait-enhancer" ]; then \
		echo "✓ portrait-enhancer directory found"; \
		echo "Creating symlinks for easy access..."; \
		ln -sf ../portrait-enhancer/run_a_pass.py scripts/ || true; \
		ln -sf ../portrait-enhancer/run_b_pass.py scripts/ || true; \
		echo "Integration setup complete!"; \
	else \
		echo "⚠ portrait-enhancer directory not found"; \
		echo "Please clone portrait-enhancer to ../portrait-enhancer"; \
		echo "Or update the path in config.yaml"; \
	fi
