# 🚀 Testing Quick Start Guide

**5-Minute Setup** to get your testing infrastructure running!

---

## Step 1: Install Dependencies (2 minutes)

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Verify installation
pytest --version
```

Expected output: `pytest 8.3.4` or similar

---

## Step 2: Run Your First Tests (1 minute)

```bash
# Run all tests
pytest

# Or use Make command (if make is installed)
make test
```

You should see output like:
```
========================= test session starts =========================
collected 45 items

tests/unit/test_path_utils.py ..................  [ 40%]
tests/unit/test_version_config.py .............  [ 70%]
tests/unit/test_model_checker.py .............   [100%]

========================= 45 passed in 2.5s ==========================
```

---

## Step 3: View Coverage Report (1 minute)

```bash
# Generate coverage report
pytest --cov=cortex_engine --cov-report=html

# Open in browser (choose your OS)
open htmlcov/index.html      # macOS
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html     # Windows

# Or use Make
make coverage coverage-report
```

---

## Step 4: Set Up Pre-Commit Hooks (1 minute)

```bash
# Install pre-commit hooks
pre-commit install

# Test them (optional)
pre-commit run --all-files
```

Now every commit will automatically:
- Format code with Black
- Sort imports with isort
- Check for security issues
- Run fast tests

---

## Common Commands Cheat Sheet

```bash
# Testing
make test              # Run all tests
make test-fast         # Run only fast tests
make test-unit         # Run only unit tests
make coverage          # Test with coverage report

# Code Quality
make format            # Auto-format code
make lint              # Check code quality
make security          # Security audit

# Quick Checks
make quick-check       # Format + lint + fast tests
make ci                # Full CI validation

# Utilities
make clean             # Clean generated files
make stats             # Project statistics
make help              # Show all commands
```

---

## Writing Your First Test

Create `tests/unit/test_my_module.py`:

```python
"""
Unit Tests for My Module
Version: 1.0.0
"""

import pytest
from cortex_engine.my_module import my_function


class TestMyFunction:
    """Test suite for my_function."""

    def test_basic_usage(self):
        """Test basic functionality."""
        result = my_function("input")
        assert result == "expected_output"

    def test_edge_case(self):
        """Test edge case."""
        result = my_function("")
        assert result is None
```

Run it:
```bash
pytest tests/unit/test_my_module.py -v
```

---

## Troubleshooting

**Tests fail with import errors?**
```bash
# Ensure you're in the project root
cd /home/longboardfella/cortex_suite

# Set Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

**Coverage not updating?**
```bash
make clean
pytest --cov=cortex_engine
```

**Pre-commit hooks failing?**
```bash
# Auto-fix formatting issues
make format

# Then try again
git commit
```

---

## Next Steps

1. ✅ Read [TESTING.md](TESTING.md) for comprehensive guide
2. ✅ Review existing tests in `tests/unit/`
3. ✅ Write tests for your code changes
4. ✅ Aim for 70%+ coverage on new code

---

## Success Metrics

By following this guide, you should have:
- ✅ 45+ passing tests
- ✅ ~20% initial code coverage
- ✅ Pre-commit hooks enabled
- ✅ Coverage reports generated
- ✅ Testing commands memorized

**You're ready to write quality, tested code!** 🎉
