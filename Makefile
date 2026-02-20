.PHONY: install install-dev lint format typecheck test test-cov run clean help

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'

install: ## Install production dependencies
	pip install -e .

install-dev: ## Install development dependencies
	pip install -e ".[dev]"
	pre-commit install

lint: ## Run ruff linter checks
	ruff check app/ tests/

format: ## Auto-format code with ruff
	ruff check --fix app/ tests/
	ruff format app/ tests/

typecheck: ## Run mypy type checking
	mypy app/

test: ## Run tests
	pytest tests/

test-cov: ## Run tests with coverage report
	pytest tests/ --cov=app --cov-report=term-missing

run: ## Run the development server
	uvicorn app.main:app --reload --host 127.0.0.1 --port 8000

clean: ## Remove build artifacts and caches
	rm -rf .ruff_cache .mypy_cache .pytest_cache
	rm -rf __pycache__ app/__pycache__ tests/__pycache__
	rm -rf htmlcov .coverage
	rm -rf dist build *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
