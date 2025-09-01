# Cortex Suite Deployment Guide

This guide covers deployment of the enhanced Cortex Suite with async processing, API integration, backup/restore, and comprehensive testing.

## Overview of New Features

### 🚀 High Priority Enhancements Implemented

1. **Async Processing Engine** - High-performance concurrent document ingestion and search
2. **RESTful API Layer** - Complete external integration with FastAPI
3. **Backup/Restore System** - Comprehensive data backup and recovery
4. **Test Suite** - Unit, integration, and API tests with pytest

## Architecture Changes

### New Components Added

```
cortex_suite/
├── cortex_engine/
│   ├── async_ingest.py      # Async document processing
│   ├── async_query.py       # Async search engine
│   └── backup_manager.py    # Backup/restore functionality
├── api/
│   ├── main.py             # FastAPI REST API
│   └── README.md           # API documentation
├── tests/                  # Comprehensive test suite
│   ├── conftest.py         # Test configuration
│   ├── test_async_ingest.py
│   ├── test_async_query.py
│   ├── test_backup_manager.py
│   └── test_api.py
├── examples/
│   └── api_usage_examples.py # API usage examples
├── pytest.ini             # Test configuration
└── run_tests.py           # Test runner script
```

## Deployment Steps

### 1. Environment Setup

```bash
# Update Python environment
source venv/bin/activate

# Install new dependencies
pip install -r requirements.txt

# Verify installations
python -c "import fastapi, pytest, aiofiles; print('✅ New dependencies installed')"
```

### 2. Database Migration

The new async system is backward compatible, but for optimal performance:

```bash
# Optional: Backup existing database
python -c "
from cortex_engine.backup_manager import create_knowledge_base_backup
import asyncio
asyncio.run(create_knowledge_base_backup('/mnt/f/ai_databases', 'pre_upgrade_backup'))
"

# Test database access
python -c "
from cortex_engine.async_query import search_knowledge_base_async
import asyncio
result = asyncio.run(search_knowledge_base_async('/mnt/f/ai_databases', 'test'))
print(f'✅ Database accessible: {result.status}')
"
```

### 3. API Server Deployment

#### Development Mode

```bash
# Start API server with auto-reload
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Test API health
curl http://localhost:8000/health
```

#### Production Mode

```bash
# Start with multiple workers
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4

# Or use gunicorn for production
pip install gunicorn
gunicorn api.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

#### Docker Deployment

```dockerfile
# Create Dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
```

```bash
# Build and run
docker build -t cortex-suite .
docker run -p 8000:8000 -v /mnt/f/ai_databases:/data cortex-suite
```

### 4. Testing Deployment

#### Run Test Suite

```bash
# Run all tests
python run_tests.py --type all --coverage

# Run specific test types
python run_tests.py --type unit
python run_tests.py --type integration
python run_tests.py --type api

# Run tests with coverage report
python run_tests.py --coverage
# Open htmlcov/index.html to view coverage report
```

#### Validate API Functionality

```bash
# Run API examples
python examples/api_usage_examples.py

# Test specific endpoints
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "machine learning", "search_type": "hybrid"}'
```

### 5. Streamlit Integration

The existing Streamlit app works unchanged, but you can now add async features:

```python
# In your Streamlit pages, you can now use:
import asyncio
from cortex_engine.async_ingest import ingest_documents_async
from cortex_engine.async_query import search_knowledge_base_async

# Example: Enhanced search with async
@st.cache_data
def async_search_wrapper(query, db_path):
    return asyncio.run(search_knowledge_base_async(db_path, query))

# Usage in Streamlit
if st.button("Enhanced Search"):
    with st.spinner("Searching..."):
        result = async_search_wrapper(query, db_path)
        st.success(f"Found {result.total_results} results in {result.processing_time:.2f}s")
```

## Configuration Updates

### Environment Variables

Add to your `.env` file:

```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# Async Processing
MAX_CONCURRENT_FILES=5
MAX_CONCURRENT_QUERIES=10
USE_PROCESS_POOL=true

# Backup Configuration  
BACKUP_RETENTION_DAYS=30
BACKUP_COMPRESSION=true
```

### New Configuration Options

The system now supports advanced configuration through `cortex_config.json`:

```json
{
  "ai_database_path": "/mnt/f/ai_databases",
  "knowledge_source_path": "/mnt/e/OneDrive - VentraIP Australia/Knowledge_Base",
  "async_processing": {
    "max_concurrent_files": 5,
    "max_concurrent_queries": 10,
    "use_process_pool": true
  },
  "api_settings": {
    "host": "0.0.0.0",
    "port": 8000,
    "workers": 4
  },
  "backup_settings": {
    "retention_days": 30,
    "compression": true,
    "include_images": true
  }
}
```

## Performance Monitoring

### New Monitoring Capabilities

1. **API Metrics**: Access via `/status` endpoint
2. **Processing Statistics**: Async operations provide detailed timing
3. **Test Coverage**: Comprehensive coverage reporting

```bash
# Monitor API performance
curl http://localhost:8000/status

# Check processing statistics
python -c "
from cortex_engine.async_ingest import AsyncIngestionEngine
import asyncio

async def get_stats():
    engine = AsyncIngestionEngine('/mnt/f/ai_databases')
    await engine.initialize()
    stats = await engine.get_ingestion_stats()
    print(f'Documents: {stats[\"total_documents\"]}')
    print(f'Database size: {stats[\"database_size\"]/1024/1024:.1f}MB')
    await engine.cleanup()

asyncio.run(get_stats())
"
```

## Troubleshooting

### Common Issues

#### 1. Import Errors

```bash
# Fix: Ensure all dependencies are installed
pip install -r requirements.txt

# Check specific imports
python -c "from cortex_engine.async_ingest import AsyncIngestionEngine; print('✅ Async imports work')"
```

#### 2. API Server Won't Start

```bash
# Check port availability
lsof -i :8000

# Try different port
uvicorn api.main:app --port 8001

# Check logs
uvicorn api.main:app --log-level debug
```

#### 3. Async Operations Fail

```bash
# Check event loop policy (on Windows/WSL)
python -c "
import asyncio
print(f'Event loop policy: {asyncio.get_event_loop_policy()}')
"

# Force specific policy if needed
export PYTHONASYNCIODEBUG=1
```

#### 4. Test Failures

```bash
# Run tests with verbose output
python run_tests.py --verbose

# Run specific failing test
python -m pytest tests/test_async_ingest.py::TestAsyncIngestionEngine::test_initialization -v

# Skip slow tests
python run_tests.py --fast
```

### Performance Tuning

#### 1. Async Processing

```python
# Optimize concurrency based on your system
config = AsyncIngestionConfig(
    max_concurrent_files=min(os.cpu_count(), 8),
    use_process_pool=True,  # Better for CPU-bound tasks
    max_workers=os.cpu_count()
)
```

#### 2. API Performance

```bash
# Production settings
uvicorn api.main:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers $(nproc) \
  --worker-class uvicorn.workers.UvicornWorker \
  --access-log \
  --log-level info
```

#### 3. Database Optimization

```python
# Enable ChromaDB optimizations
chroma_settings = chromadb.config.Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory=db_path,
    anonymized_telemetry=False
)
```

## Security Considerations

### API Security

1. **Authentication**: Implement proper token validation in `verify_token`
2. **Rate Limiting**: Configure appropriate rate limits per endpoint
3. **CORS**: Restrict origins in production
4. **Input Validation**: All inputs are validated via Pydantic models

```python
# Example: Enhanced authentication
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if not credentials:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    # Verify token against your auth system
    if not validate_token(credentials.credentials):
        raise HTTPException(status_code=401, detail="Invalid token")
    
    return True
```

### Data Protection

1. **Backup Encryption**: Backups include integrity checksums
2. **Path Validation**: All file paths are validated and sanitized
3. **Error Handling**: Sensitive information is not exposed in errors

## Maintenance

### Regular Tasks

1. **Backup Management**:
```bash
# Create scheduled backup
python -c "
from cortex_engine.backup_manager import create_knowledge_base_backup
import asyncio
asyncio.run(create_knowledge_base_backup('/mnt/f/ai_databases', compress=True))
"

# Cleanup old backups
python -c "
from cortex_engine.backup_manager import BackupManager
import asyncio

async def cleanup():
    manager = BackupManager('/mnt/f/ai_databases')
    deleted = await manager.cleanup_old_backups(keep_count=10)
    print(f'Cleaned up {deleted} old backups')

asyncio.run(cleanup())
"
```

2. **Performance Monitoring**:
```bash
# Weekly performance check
python examples/api_usage_examples.py 2>&1 | grep "Performance Test"
```

3. **Test Suite**:
```bash
# Run tests before deployments
python run_tests.py --type all --coverage
```

## Rollback Plan

If issues occur, you can rollback:

1. **Stop new services**:
```bash
pkill -f uvicorn
```

2. **Restore from backup** (if needed):
```python
from cortex_engine.backup_manager import restore_knowledge_base_backup
import asyncio

asyncio.run(restore_knowledge_base_backup(
    '/mnt/f/ai_databases', 
    'pre_upgrade_backup'
))
```

3. **Use original Streamlit-only mode**:
```bash
streamlit run Cortex_Suite.py
```

## Next Steps

With these enhancements deployed, you can:

1. **Integrate with external systems** via the REST API
2. **Scale processing** with async operations
3. **Ensure data safety** with automated backups
4. **Monitor performance** with comprehensive metrics
5. **Develop confidently** with full test coverage

The system maintains full backward compatibility while providing significant performance and capability improvements.