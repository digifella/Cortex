# Cortex Suite Testing Guide

**Version:** 1.0.0
**Last Updated:** 2025-12-23
**Purpose:** Comprehensive guide for testing the Cortex Suite codebase

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Test Structure](#test-structure)
4. [Running Tests](#running-tests)
5. [Writing Tests](#writing-tests)
6. [Coverage Goals](#coverage-goals)
7. [CI/CD Integration](#cicd-integration)
8. [Best Practices](#best-practices)

---

## Overview

The Cortex Suite uses **pytest** as its testing framework with comprehensive coverage reporting, mocking capabilities, and support for multiple test types (unit, integration, UI).

### Test Categories

| Type | Scope | Speed | Dependencies |
|------|-------|-------|--------------|
| **Unit** | Single function/class | Fast (<0.1s) | None - uses mocks |
| **Integration** | Multiple components | Medium (0.1-5s) | May need services |
| **UI** | Streamlit pages | Slow (>5s) | Streamlit testing |
| **Performance** | Benchmarking | Variable | Depends on test |

---

## Quick Start

### 1. Install Testing Dependencies

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Verify pytest installation
pytest --version
```

### 2. Run All Tests

```bash
# Run all tests with coverage
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/unit/test_path_utils.py

# Run specific test
pytest tests/unit/test_path_utils.py::TestConvertWindowsToWSLPath::test_windows_drive_path_conversion
```

### 3. View Coverage Report

```bash
# Generate HTML coverage report
pytest --cov=cortex_engine --cov-report=html

# Open in browser
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html  # Windows
```

---

## Test Structure

```
tests/
├── conftest.py                 # Shared fixtures and configuration
├── unit/                       # Unit tests - fast, isolated
│   ├── test_path_utils.py
│   ├── test_version_config.py
│   ├── test_model_checker.py
│   └── test_config.py
├── integration/                # Integration tests - slower
│   ├── test_ingestion_workflow.py
│   ├── test_search_workflow.py
│   └── test_backup_restore.py
├── ui/                         # UI/Streamlit tests
│   ├── test_knowledge_ingest.py
│   └── test_knowledge_search.py
└── fixtures/                   # Test data and fixtures
    ├── sample_documents/
    └── mock_databases/
```

---

## Running Tests

### By Category

```bash
# Run only unit tests (fast)
pytest -m unit

# Run only integration tests
pytest -m integration

# Run only UI tests
pytest -m ui

# Skip slow tests
pytest -m "not slow"
```

### By Service Requirement

```bash
# Run tests that require Ollama
pytest -m requires_ollama

# Run tests that require ChromaDB
pytest -m requires_chromadb

# Skip tests requiring external services
pytest -m "not requires_ollama and not requires_chromadb"
```

### With Coverage

```bash
# Basic coverage
pytest --cov=cortex_engine

# Coverage with missing lines
pytest --cov=cortex_engine --cov-report=term-missing

# Multiple output formats
pytest --cov=cortex_engine --cov-report=html --cov-report=json --cov-report=term
```

### Parallel Execution

```bash
# Run tests in parallel (requires pytest-xdist)
pytest -n auto

# Run with specific number of workers
pytest -n 4
```

### Debugging

```bash
# Stop on first failure
pytest -x

# Drop into debugger on failure
pytest --pdb

# Show local variables on failure
pytest --showlocals

# Verbose output with print statements
pytest -s -v
```

---

## Writing Tests

### Unit Test Template

```python
"""
Unit Tests for [Module Name]
Version: 1.0.0
Purpose: Test [specific functionality]
"""

import pytest
from cortex_engine.module import function_to_test


class TestFunctionName:
    """Test suite for specific function."""

    def test_basic_functionality(self):
        """Test basic use case."""
        result = function_to_test("input")
        assert result == "expected_output"

    def test_edge_case(self):
        """Test edge case handling."""
        result = function_to_test("")
        assert result is None

    @pytest.mark.parametrize("input,expected", [
        ("input1", "output1"),
        ("input2", "output2"),
        ("input3", "output3"),
    ])
    def test_multiple_inputs(self, input, expected):
        """Test multiple input scenarios."""
        assert function_to_test(input) == expected
```

### Integration Test Template

```python
"""
Integration Tests for [Workflow Name]
Version: 1.0.0
Purpose: Test end-to-end workflow
"""

import pytest


@pytest.mark.integration
class TestWorkflowName:
    """Integration test for complete workflow."""

    def test_complete_workflow(self, temp_db_path, mock_ollama_client):
        """Test the complete workflow from start to finish."""
        # Setup
        setup_test_data()

        # Execute workflow
        result = execute_workflow()

        # Verify
        assert result.success is True
        assert result.documents_processed == 10
```

### Using Fixtures

```python
def test_with_fixtures(temp_dir, sample_text_file, mock_chromadb_client):
    """Test using multiple fixtures."""
    # temp_dir: Temporary directory that auto-cleans
    # sample_text_file: Pre-created test file
    # mock_chromadb_client: Mocked ChromaDB client

    assert temp_dir.exists()
    assert sample_text_file.exists()

    # Use mocked ChromaDB
    collection = mock_chromadb_client.get_or_create_collection("test")
    collection.add(ids=["1"], documents=["test"])
```

### Mocking External Services

```python
from unittest.mock import Mock, patch

def test_with_ollama_mock(mock_ollama_client):
    """Test using mocked Ollama service."""
    # Mock is provided by conftest.py fixture
    from cortex_engine.utils.model_checker import model_checker

    result = model_checker.check_ollama_service()
    assert result[0] is True  # Service appears "running"

@patch('cortex_engine.utils.model_checker.ollama.Client')
def test_with_custom_mock(mock_client):
    """Test with custom mock behavior."""
    mock_client.return_value.list.return_value = {
        'models': [{'name': 'custom:model'}]
    }

    # Your test code here
```

---

## Coverage Goals

### Current Coverage Targets

| Phase | Timeframe | Target | Priority |
|-------|-----------|--------|----------|
| Phase 1 | Week 1-2 | 20% | Critical utilities |
| Phase 2 | Week 3-4 | 40% | Core workflows |
| Phase 3 | Month 2 | 60% | All business logic |
| Phase 4 | Month 3 | 75% | Production ready |

### Coverage by Component

**High Priority (75%+ coverage)**:
- `cortex_engine/utils/` - Shared utilities
- `cortex_engine/version_config.py` - Version management
- `cortex_engine/config.py` - Configuration
- `cortex_engine/model_checker.py` - Model validation

**Medium Priority (50%+ coverage)**:
- `cortex_engine/ingest_cortex.py` - Ingestion logic
- `cortex_engine/query_cortex.py` - Search logic
- `cortex_engine/backup_manager.py` - Backup/restore

**Lower Priority (30%+ coverage)**:
- `pages/*.py` - UI pages (harder to test, less critical)
- `cortex_engine/theme_visualizer.py` - Visualization

### Checking Coverage

```bash
# Generate coverage report
pytest --cov=cortex_engine --cov-report=term-missing

# Check if coverage meets threshold
pytest --cov=cortex_engine --cov-fail-under=30

# Detailed HTML report
pytest --cov=cortex_engine --cov-report=html
open htmlcov/index.html
```

---

## CI/CD Integration

### GitHub Actions Example

Create `.github/workflows/tests.yml`:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -r requirements-dev.txt

    - name: Run tests
      run: |
        pytest --cov=cortex_engine --cov-report=xml

    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

### Pre-commit Hooks

Create `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 24.10.0
    hooks:
      - id: black

  - repo: https://github.com/pycqa/isort
    rev: 5.13.2
    hooks:
      - id: isort

  - repo: local
    hooks:
      - id: pytest
        name: pytest
        entry: pytest -m "not slow"
        language: system
        pass_filenames: false
        always_run: true
```

Install pre-commit:
```bash
pip install pre-commit
pre-commit install
```

---

## Best Practices

### ✅ DO

1. **Write tests first** (TDD) when fixing bugs
2. **Use descriptive test names** that explain what is being tested
3. **Test one thing per test** - single assertion when possible
4. **Use fixtures** for common setup/teardown
5. **Mock external services** to keep tests fast and reliable
6. **Parametrize tests** for multiple similar test cases
7. **Add markers** (`@pytest.mark.unit`, `@pytest.mark.slow`) for organization
8. **Document test purpose** in docstrings
9. **Test edge cases** and error conditions
10. **Keep tests independent** - no test should depend on another

### ❌ DON'T

1. **Don't test implementation details** - test behavior, not internals
2. **Don't write flaky tests** - tests should be deterministic
3. **Don't skip tests** without good reason (use markers instead)
4. **Don't make tests depend on external services** unless marked `@pytest.mark.integration`
5. **Don't hardcode file paths** - use fixtures and temp directories
6. **Don't test third-party libraries** - trust they work
7. **Don't write tests that take >5s** without `@pytest.mark.slow`
8. **Don't commit failing tests** - fix or skip them
9. **Don't test UI layout** - focus on functionality
10. **Don't ignore coverage reports** - investigate uncovered code

### Test Naming Convention

```python
# Good naming - clear, descriptive
def test_convert_windows_path_to_wsl_format():
    """Test that Windows C: drive paths convert to /mnt/c."""
    pass

def test_empty_path_returns_empty_string():
    """Test that empty input returns empty output."""
    pass

# Bad naming - unclear
def test_path():
    pass

def test_case_1():
    pass
```

### Assertion Best Practices

```python
# Good - specific assertions
assert result == expected_value
assert len(items) == 3
assert "error" in message.lower()
assert result.success is True

# Bad - overly broad
assert result  # What does truthy mean?
assert items   # Just checking existence?

# Good - helpful failure messages
assert len(items) == 3, f"Expected 3 items, got {len(items)}"
```

---

## Troubleshooting

### Common Issues

**Issue: Tests fail with import errors**
```bash
# Solution: Ensure project root is in Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
pytest
```

**Issue: Tests fail to find fixtures**
```bash
# Solution: Ensure conftest.py is in the right location
# conftest.py should be in tests/ directory
```

**Issue: Coverage report doesn't update**
```bash
# Solution: Clear coverage data and regenerate
rm -rf .coverage htmlcov/
pytest --cov=cortex_engine --cov-report=html
```

**Issue: Tests run slowly**
```bash
# Solution: Run specific tests or use parallel execution
pytest -n auto -m "not slow"
```

---

## Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Coverage.py Documentation](https://coverage.readthedocs.io/)
- [Effective Python Testing](https://realpython.com/pytest-python-testing/)
- [Python Testing Best Practices](https://docs.python-guide.org/writing/tests/)

---

## Getting Help

1. Check this guide first
2. Review existing tests for examples
3. Check pytest documentation
4. Ask in team chat/discussion

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-12-23 | Initial testing infrastructure |

---

**Next Steps:**
1. Install development dependencies: `pip install -r requirements-dev.txt`
2. Run your first test: `pytest tests/unit/test_version_config.py -v`
3. Check coverage: `pytest --cov=cortex_engine`
4. Start writing tests for your code!
