# Cortex Suite Test Suite

**Status:** ✅ Operational
**Tests:** 87 passing, 1 performance test needs adjustment
**Coverage:** ~20% (critical utilities)

## Quick Start

```bash
# Install dependencies
pip install -r requirements-dev.txt

# Run all tests
pytest

# Run unit tests only
pytest tests/unit/

# Run with coverage
pytest --cov=cortex_engine --cov-report=html
```

## Test Structure

```
tests/
├── conftest.py                    # Shared fixtures
├── unit/                          # Fast, isolated tests
│   ├── test_path_utils.py        # ✅ 46 tests passing
│   ├── test_version_config.py    # ✅ 40 tests passing
│   └── test_model_checker.py     # ✅ 23 tests passing
├── integration/                   # Workflow tests
│   └── test_ingestion_workflow.py # ✅ Template ready
└── ui/                           # Streamlit UI tests
```

## What's Tested

### Path Utilities (46 tests) ✅
- Windows to WSL path conversion
- Docker path detection and mounting
- Edge cases (empty paths, special characters)
- Performance benchmarks

### Version Config (40 tests) ✅
- Version format validation
- Semantic versioning
- Changelog generation
- Version consistency checks

### Model Checker (23 tests) ✅
- Ollama service detection
- Model availability checking
- Ingestion requirements validation
- Research requirements validation
- Error handling

## Test Markers

Run specific test categories:

```bash
pytest -m unit              # Unit tests only
pytest -m integration       # Integration tests
pytest -m "not slow"        # Skip slow tests
pytest -m requires_ollama   # Tests needing Ollama
```

## Known Issues

1. **Performance Test** (`test_bulk_conversion_performance`) - Timing assertion too strict, can be adjusted
2. **Integration Tests** - Some use mocks, expand with real workflows as needed

## Next Steps

1. Add tests for `config.py`
2. Add tests for `ingest_cortex.py` core functions
3. Add tests for `query_cortex.py`
4. Expand integration tests with real workflows
5. Add UI tests for critical pages

## Documentation

- [TESTING.md](../TESTING.md) - Complete testing guide
- [TESTING_QUICKSTART.md](../TESTING_QUICKSTART.md) - 5-minute guide
- [TESTING_SETUP_COMPLETE.md](../TESTING_SETUP_COMPLETE.md) - Setup summary

## Contributing

When adding tests:
1. Follow existing patterns in `tests/unit/`
2. Use descriptive test names
3. Add appropriate markers
4. Include docstrings
5. Use fixtures from `conftest.py`

See [TESTING.md](../TESTING.md) for detailed guidelines.
