# Makefile for Cortex Suite
# Version: 1.0.0
# Purpose: Convenient shortcuts for common development tasks

.PHONY: help install test coverage lint format clean docs

# Default target
.DEFAULT_GOAL := help

# Variables
PYTHON := python3.11
PIP := $(PYTHON) -m pip
PYTEST := pytest
BLACK := black
ISORT := isort
FLAKE8 := flake8
MYPY := mypy

help: ## Show this help message
	@echo "Cortex Suite Development Commands"
	@echo "=================================="
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Installation targets
install: ## Install production dependencies
	$(PIP) install -r requirements.txt
	$(PYTHON) -m spacy download en_core_web_sm
	$(PYTHON) -m nltk.downloader punkt stopwords wordnet

install-dev: install ## Install development dependencies
	$(PIP) install -r requirements-dev.txt
	pre-commit install

# Testing targets
test: ## Run all tests
	$(PYTEST)

test-unit: ## Run unit tests only
	$(PYTEST) -m unit

test-integration: ## Run integration tests only
	$(PYTEST) -m integration

test-fast: ## Run fast tests (skip slow and integration)
	$(PYTEST) -m "not slow and not integration"

test-watch: ## Run tests in watch mode (requires pytest-watch)
	ptw -- -m "not slow"

# Coverage targets
coverage: ## Run tests with coverage report
	$(PYTEST) --cov=cortex_engine --cov=pages --cov-report=html --cov-report=term-missing

coverage-report: ## Open HTML coverage report
	@$(PYTHON) -m webbrowser -t htmlcov/index.html || xdg-open htmlcov/index.html || open htmlcov/index.html

coverage-xml: ## Generate XML coverage report (for CI)
	$(PYTEST) --cov=cortex_engine --cov=pages --cov-report=xml

# Code quality targets
lint: ## Run all linters
	$(FLAKE8) cortex_engine pages
	$(MYPY) cortex_engine --ignore-missing-imports
	bandit -r cortex_engine -c pyproject.toml

format: ## Format code with black and isort
	$(BLACK) cortex_engine pages tests
	$(ISORT) cortex_engine pages tests

format-check: ## Check if code formatting is correct
	$(BLACK) --check cortex_engine pages tests
	$(ISORT) --check cortex_engine pages tests

# Security targets
security: ## Run security checks
	bandit -r cortex_engine -f json -o bandit-report.json
	safety check --json
	pip-audit

# Cleanup targets
clean: ## Remove generated files
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .coverage htmlcov coverage.json coverage.xml
	rm -rf build dist

clean-logs: ## Remove log files
	rm -rf logs/*.log

clean-all: clean clean-logs ## Remove all generated files including logs

# Version management targets
version-check: ## Check version consistency
	$(PYTHON) scripts/version_manager.py --check

version-sync: ## Sync version across all files
	$(PYTHON) scripts/version_manager.py --sync-all

version-changelog: ## Update changelog with current version
	$(PYTHON) scripts/version_manager.py --update-changelog

version-info: ## Show current version information
	$(PYTHON) scripts/version_manager.py --info

# Docker targets
docker-build: ## Build Docker image
	cd docker && docker build -t cortex-suite:latest .

docker-run: ## Run Docker container
	docker run -p 8501:8501 -p 8000:8000 cortex-suite:latest

docker-test: ## Run tests in Docker container
	docker run cortex-suite:latest pytest -m "not integration"

# Development server targets
run: ## Run Streamlit app locally
	streamlit run Cortex_Suite.py

run-api: ## Run API server locally
	uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Pre-commit targets
pre-commit: ## Run pre-commit hooks on all files
	pre-commit run --all-files

pre-commit-update: ## Update pre-commit hooks
	pre-commit autoupdate

# Documentation targets (if/when implemented)
docs: ## Build documentation
	@echo "Documentation build not yet implemented"

# Database targets
db-backup: ## Backup knowledge base
	@echo "Creating backup..."
	$(PYTHON) -c "from cortex_engine.backup_manager import BackupManager; BackupManager().create_backup()"

# Quick checks (runs before commit)
quick-check: format-check lint test-fast ## Run quick quality checks

# Full validation (runs in CI)
ci: clean install-dev lint coverage security ## Run full CI validation

# Initialize new development environment
init: install-dev pre-commit ## Initialize development environment
	@echo "Development environment initialized!"
	@echo "Run 'make test' to verify setup."

# Show project stats
stats: ## Show project statistics
	@echo "Project Statistics"
	@echo "=================="
	@echo "Python files:"
	@find cortex_engine pages -name "*.py" | wc -l
	@echo "Lines of code:"
	@find cortex_engine pages -name "*.py" -exec wc -l {} + | tail -1
	@echo "Test files:"
	@find tests -name "test_*.py" | wc -l
	@echo ""
	@echo "Git Status:"
	@git status --short

# Performance profiling
profile: ## Profile application performance
	$(PYTHON) -m cProfile -o profile.stats -m pytest -m "not slow"
	$(PYTHON) -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumulative').print_stats(20)"

# Memory profiling
memory-profile: ## Profile memory usage
	$(PYTHON) -m memory_profiler cortex_engine/ingest_cortex.py
