# Docker Fixes Documentation
## Complete Guide to Docker Compatibility Issues and Solutions

**Version**: 1.0.0  
**Date**: 2025-08-22  
**Context**: Cortex Suite Docker deployment fixes

---

## Overview

This document comprehensively documents all the fixes required to make Cortex Suite work properly in Docker containers, particularly focusing on document ingestion pipeline issues that occurred during August 2025.

## Issues Encountered and Fixes Applied

### 1. Ollama API Compatibility Issues

**Problem**: Docker containers with newer Ollama versions use `/api/chat` endpoint, while LlamaIndex was using deprecated `/api/generate` endpoint.

**Symptoms**:
```
httpx.HTTPStatusError: Client error '404 Not Found' for url 'http://localhost:11434/api/generate'
```

**Root Cause**: 
- LlamaIndex `llama-index-llms-ollama==0.1.3` uses deprecated API endpoints
- Modern Ollama versions (used in Docker) require `/api/chat` with different message format

**Solution**: Created smart Ollama LLM selector system:

1. **Smart Ollama LLM** (`cortex_engine/utils/smart_ollama_llm.py`):
   - Automatically detects environment (Docker vs non-Docker)
   - Tries original LlamaIndex Ollama first (works in most cases)
   - Falls back to modern wrapper if needed (Docker compatibility)

2. **Modern Ollama Wrapper** (`cortex_engine/utils/modern_ollama_llm.py`):
   - Uses correct `/api/chat` endpoint
   - Handles modern message format: `{"messages": [{"role": "user", "content": "text"}]}`
   - Implements all required LlamaIndex LLM interface methods

**Files Modified**:
- `cortex_engine/ingest_cortex.py` (v13.1.0 → v13.2.0)
- `cortex_engine/llm_service.py` (v1.1.0 → v1.2.0)
- `cortex_engine/task_engine.py` (v13.1.0 → v13.2.0)
- `cortex_engine/model_services/ollama_model_service.py`
- `cortex_engine/model_services/docker_model_service.py`

### 2. NLTK Data Dependencies

**Problem**: Docker containers missing required NLTK data for document processing.

**Symptoms**:
```
LookupError: Resource [93maveraged_perceptron_tagger[0m not found.
```

**Solution**: Added NLTK data downloads to Dockerfile:
```dockerfile
RUN python -c "import nltk; nltk.download('averaged_perceptron_tagger'); nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```

**Files Modified**:
- `docker/Dockerfile`

### 3. Missing Log Directories

**Problem**: Docker container missing required log directories causing permission errors.

**Symptoms**:
```
PermissionError: [Errno 13] Permission denied: '/home/cortex/app/logs/ingestion.log'
```

**Solution**: Added log directory creation to Dockerfile:
```dockerfile
RUN mkdir -p /home/cortex/data/ai_databases \
    /home/cortex/data/knowledge_base \
    /home/cortex/logs \
    /home/cortex/app/logs
```

**Files Modified**:
- `docker/Dockerfile`

### 4. Model Name Mismatches

**Problem**: Code trying to use model name `"mistral"` but Docker container only has `"mistral:7b-instruct-v0.3-q4_K_M"`.

**Symptoms**:
```
{"error":"model \"mistral\" not found, try pulling it first"}
```

**Solution**: Updated model references to use full tag names that match available models.

**Files Modified**:
- `cortex_engine/ingest_cortex.py`
- `cortex_engine/llm_service.py`
- `cortex_engine/task_engine.py`

### 5. Windows Path Handling Issues

**Problem**: Windows drive paths (E:\) not mounting correctly in Docker containers.

**Solution**: Enhanced path validation and mounting logic in startup scripts:

**Files Modified**:
- `docker/run-cortex.sh` - Added Windows drive detection and mounting
- `docker/run-cortex.bat` - Fixed Docker command construction for Windows
- `cortex_engine/utils/path_utils.py` - Enhanced Docker-aware path validation

### 6. LlamaIndex Abstract Method Implementation

**Problem**: Custom Ollama wrapper missing required abstract methods from LlamaIndex LLM base class.

**Symptoms**:
```
TypeError: Can't instantiate abstract class ModernOllama with abstract methods achat, acomplete, astream_chat, astream_complete
```

**Solution**: Implemented all required async methods in ModernOllama class:
- `acomplete()` - async completion
- `achat()` - async chat
- `astream_complete()` - async streaming (not implemented, raises NotImplementedError)
- `astream_chat()` - async streaming (not implemented, raises NotImplementedError)

**Files Modified**:
- `cortex_engine/utils/modern_ollama_llm.py`

---

## Architecture Overview

### Smart LLM Selection Strategy

```
User Request
     ↓
Smart Ollama LLM Selector
     ↓
Environment Detection
     ↓
┌─ Non-Docker ────────────────┐  ┌─ Docker ─────────────────────┐
│ Try: Original LlamaIndex    │  │ Try: Original LlamaIndex     │
│ ✅ Success → Use Original   │  │ ❌ 404 Error → Try Modern    │
│                             │  │ ✅ Success → Use Modern      │
└─────────────────────────────┘  └───────────────────────────────┘
```

### API Endpoint Mapping

| Environment | Endpoint | Message Format |
|-------------|----------|----------------|
| Non-Docker | `/api/generate` | `{"prompt": "text"}` |
| Docker | `/api/chat` | `{"messages": [{"role": "user", "content": "text"}]}` |

---

## Files Structure

### New Files Created:
```
cortex_engine/utils/
├── smart_ollama_llm.py      # Smart LLM selector
├── modern_ollama_llm.py     # Modern API wrapper
└── model_checker.py         # Model availability validation
```

### Modified Files:
```
cortex_engine/
├── ingest_cortex.py         # Uses smart LLM selector
├── llm_service.py           # Uses smart LLM selector  
├── task_engine.py           # Uses smart LLM selector
└── model_services/
    ├── ollama_model_service.py    # Updated endpoints
    └── docker_model_service.py    # Updated endpoints

docker/
├── Dockerfile               # Added NLTK data & log dirs
├── run-cortex.sh           # Enhanced Windows drive mounting
└── run-cortex.bat          # Fixed Docker command construction
```

---

## Testing Checklist

### Pre-Deployment Testing:
- [ ] Non-Docker environment: `python -c "from cortex_engine.utils.smart_ollama_llm import create_smart_ollama_llm; print('✅ Works')"` 
- [ ] Docker environment: Check ingestion logs for successful completion
- [ ] Model availability: `docker exec <container> ollama list`
- [ ] Path mounting: Verify external directories accessible in container
- [ ] Log creation: Verify `/home/cortex/app/logs/ingestion.log` created

### Post-Deployment Validation:
- [ ] Document ingestion completes without errors
- [ ] Staging file generated successfully  
- [ ] Metadata review populated with documents
- [ ] Proposal generation works (if tested)
- [ ] Search functionality works (if tested)

---

## Troubleshooting Guide

### Common Error Patterns:

1. **404 Not Found for /api/generate**:
   - Solution: Rebuild container with latest smart LLM selector

2. **Model not found errors**:
   - Check: `docker exec <container> ollama list`
   - Solution: Update model names to match available models

3. **NLTK data missing**:
   - Solution: Rebuild container with updated Dockerfile

4. **Permission denied on logs**:
   - Solution: Rebuild container with log directory creation

5. **Path mounting issues**:
   - Check: Windows drive letters properly detected
   - Solution: Use updated run scripts

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-08-22 | Initial Docker compatibility fixes |

---

## Future Considerations

1. **LlamaIndex Updates**: Monitor for newer versions that support modern Ollama API
2. **Model Management**: Consider implementing automatic model installation
3. **Environment Detection**: Enhance Docker detection for edge cases
4. **Performance Optimization**: Monitor overhead of smart LLM selection
5. **Error Handling**: Improve fallback strategies for LLM failures

---

## Dependencies

### Critical Version Requirements:
- `llama-index-llms-ollama==0.1.3` (contains /api/generate compatibility issue)
- Modern Ollama versions in Docker (require /api/chat endpoint)
- NLTK data packages for document processing
- Proper Docker volume mounting for persistence

### Environment Compatibility:
- ✅ Non-Docker: WSL2, Linux, macOS
- ✅ Docker: Linux containers with Ollama service
- ⚠️ Windows native: May require additional path handling

---

*This documentation should be updated whenever new Docker compatibility issues are discovered or additional fixes are implemented.*