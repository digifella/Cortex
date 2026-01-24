# Qwen3-VL Multimodal Embedding Upgrade Plan

**Created:** 2026-01-24
**Completed:** 2026-01-24
**Status:** ✅ COMPLETE
**Priority:** High - Required for multimodal embedding support

---

## ✅ Upgrade Completed

**Date:** 2026-01-24

### Packages Upgraded
| Package | Before | After |
|---------|--------|-------|
| transformers | 4.46.3 | **4.57.6** |
| sentence-transformers | 2.7.0 | **5.2.0** |
| chromadb | 0.5.23 | **1.4.1** |
| tokenizers | 0.20.3 | **0.22.2** |
| posthog | 6.1.1 | 5.4.0 |

### Verification Results
- ✅ Qwen3-VL architecture recognized (`model_type: qwen3_vl`)
- ✅ Qwen3VLConfig initialization working
- ✅ ChromaDB + LlamaIndex integration verified
- ✅ No dependency conflicts

### Files Updated
- `requirements.txt` - Updated with new versions
- `docker/requirements.txt` - Synced
- `cortex_engine/qwen3_vl_embedding_service.py` - Fixed `nonlocal` → `global` bug

---

## Executive Summary

The Cortex Suite currently has Qwen3-VL multimodal embedding code implemented but cannot use it because the installed `transformers` library (4.46.3) predates Qwen3-VL support. This document outlines the upgrade path to enable Qwen3-VL functionality.

---

## Current State

### Issues Encountered

1. **SyntaxError in qwen3_vl_embedding_service.py** (FIXED 2026-01-24)
   - Location: `cortex_engine/qwen3_vl_embedding_service.py:178`
   - Cause: Used `nonlocal` instead of `global` for module-level variables
   - Status: ✅ Fixed

2. **Transformers Architecture Not Recognized** (BLOCKING)
   - Error: `ValueError: The checkpoint you are trying to load has model type 'qwen3_vl' but Transformers does not recognize this architecture`
   - Cause: transformers 4.46.3 predates Qwen3-VL support (added in 4.57.0)
   - Status: ❌ Requires upgrade

### Current Package Versions

| Package | Installed | Required for Qwen3-VL |
|---------|-----------|----------------------|
| transformers | 4.46.3 | **>= 4.57.0** |
| sentence-transformers | 2.7.0 | >= 3.0.0 (recommended) |
| huggingface-hub | 0.36.0 | >= 0.20.0 ✅ |
| torch | 2.5.1 | >= 2.0.0 ✅ |
| tokenizers | 0.20.3 | Compatible ✅ |

---

## Upgrade Path

### Phase 1: Pre-Upgrade Preparation

#### 1.1 Backup Current Environment
```bash
# Create requirements backup
pip freeze > requirements_backup_20260124.txt

# Backup virtual environment (optional but recommended)
cp -r venv venv_backup_20260124
```

#### 1.2 Create Test Environment (Recommended)
```bash
# Create isolated test environment
python3.11 -m venv venv_qwen3vl_test
source venv_qwen3vl_test/bin/activate
pip install -r requirements.txt
```

### Phase 2: Core Upgrade

#### 2.1 Upgrade Transformers
```bash
# Upgrade transformers to latest stable with Qwen3-VL support
pip install "transformers>=4.57.0,<5.0.0"
```

**Expected Changes:**
- transformers: 4.46.3 → 4.57.6
- May auto-upgrade: tokenizers, huggingface-hub

#### 2.2 Upgrade Sentence-Transformers (Recommended)
```bash
# sentence-transformers 2.7.0 is very old (latest is 5.2.0)
# Upgrade to a compatible version
pip install "sentence-transformers>=3.0.0,<6.0.0"
```

**Why upgrade sentence-transformers:**
- Better compatibility with newer transformers
- Performance improvements
- Bug fixes for embedding models

#### 2.3 Combined Upgrade Command
```bash
pip install \
    "transformers>=4.57.0,<5.0.0" \
    "sentence-transformers>=3.0.0,<6.0.0" \
    "tokenizers>=0.20.0"
```

### Phase 3: Validation

#### 3.1 Verify Installation
```bash
# Check versions
python -c "import transformers; print(f'transformers: {transformers.__version__}')"
python -c "import sentence_transformers; print(f'sentence-transformers: {sentence_transformers.__version__}')"

# Test Qwen3-VL architecture recognition
python -c "from transformers import AutoConfig; AutoConfig.from_pretrained('Qwen/Qwen3-VL-Embedding-2B', trust_remote_code=True)"
```

#### 3.2 Test Embedding Service
```bash
# Test embedding service initialization
cd /home/longboardfella/cortex_suite
python -c "
from cortex_engine.qwen3_vl_embedding_service import _load_model, Qwen3VLConfig
config = Qwen3VLConfig.for_model_size('2B')  # Use smaller model for testing
model, processor, cfg = _load_model(config)
print(f'✅ Qwen3-VL model loaded: {cfg.model_name}')
"
```

#### 3.3 Test Full Ingest Pipeline
```bash
# Run a small test ingest
streamlit run Cortex_Suite.py
# Navigate to Knowledge Ingest and test with 1-2 documents
```

### Phase 4: Rollback Plan

If issues occur after upgrade:

```bash
# Option 1: Restore from backup
pip install -r requirements_backup_20260124.txt

# Option 2: Downgrade specific packages
pip install transformers==4.46.3 sentence-transformers==2.7.0

# Option 3: Restore entire venv
rm -rf venv
mv venv_backup_20260124 venv
```

---

## Compatibility Considerations

### Known Compatibility Matrix

| Component | Transformers 4.46.x | Transformers 4.57.x |
|-----------|--------------------|--------------------|
| LlamaIndex 0.14.x | ✅ Tested | ⚠️ Needs testing |
| sentence-transformers 2.7.0 | ✅ Works | ⚠️ May have issues |
| sentence-transformers 3.x+ | ❌ Untested | ✅ Recommended |
| torch 2.5.x | ✅ Works | ✅ Works |
| chromadb 0.5.x | ✅ Works | ✅ Works |
| spacy 3.7.x | ✅ Works | ✅ Works |

### Potential Breaking Changes

1. **sentence-transformers API changes (2.x → 3.x+)**
   - Some deprecated methods removed
   - Model loading syntax may differ
   - Check: `cortex_engine/embedding_service.py`

2. **transformers AutoModel changes**
   - Trust remote code handling
   - Flash Attention 2 API
   - Check: `cortex_engine/qwen3_vl_embedding_service.py`

3. **Tokenizer compatibility**
   - Fast tokenizer defaults may change
   - Check batch processing code

---

## Hardware Requirements for Qwen3-VL

### VRAM Requirements

| Model | VRAM (fp16) | VRAM (bf16) | Embedding Dim |
|-------|-------------|-------------|---------------|
| Qwen3-VL-Embedding-2B | ~5 GB | ~5 GB | 2048 |
| Qwen3-VL-Embedding-8B | ~16 GB | ~16 GB | 4096 |

### Current System (RTX 3090)
- VRAM: 24 GB
- Suitable for: Both 2B and 8B models
- Recommended: 8B for best quality (with Ollama models unloaded)

### Memory Management
When running Qwen3-VL alongside Ollama:
- Ollama models use VRAM (mistral ~10GB, llava ~5GB)
- May need to unload Ollama models during Qwen3-VL embedding
- Auto-selection in `smart_model_selector.py` handles this

---

## Implementation Checklist

### Pre-Upgrade
- [ ] Backup current requirements.txt
- [ ] Backup virtual environment (optional)
- [ ] Document current working state
- [ ] Create test environment

### Upgrade
- [ ] Upgrade transformers to >= 4.57.0
- [ ] Upgrade sentence-transformers to >= 3.0.0
- [ ] Verify no dependency conflicts

### Post-Upgrade Validation
- [ ] Verify package versions
- [ ] Test Qwen3-VL model loading
- [ ] Test text embedding
- [ ] Test image embedding (if applicable)
- [ ] Test full ingest pipeline
- [ ] Test search/query functionality
- [ ] Run existing unit tests

### Documentation
- [ ] Update requirements.txt with new versions
- [ ] Update CHANGELOG.md
- [ ] Update any version-specific documentation
- [ ] Sync docker/requirements.txt

---

## Quick Start Commands

### Conservative Upgrade (Recommended)
```bash
# Step 1: Backup
pip freeze > requirements_backup.txt

# Step 2: Upgrade core packages
pip install "transformers>=4.57.0" "sentence-transformers>=3.0.0"

# Step 3: Verify
python -c "import transformers; print(transformers.__version__)"

# Step 4: Test Qwen3-VL
python -c "from transformers import AutoConfig; AutoConfig.from_pretrained('Qwen/Qwen3-VL-Embedding-2B', trust_remote_code=True); print('✅ Qwen3-VL recognized')"
```

### Full Upgrade (All Latest)
```bash
pip install --upgrade \
    transformers \
    sentence-transformers \
    tokenizers \
    huggingface-hub \
    accelerate
```

---

## References

- [Qwen3-VL Transformers Documentation](https://huggingface.co/docs/transformers/en/model_doc/qwen3_vl)
- [Qwen3-VL-Embedding Model Card](https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B)
- [Qwen3-VL-Embedding GitHub](https://github.com/QwenLM/Qwen3-VL-Embedding)
- [Sentence Transformers Documentation](https://sbert.net/)
- [HuggingFace Hub Documentation](https://huggingface.co/docs/huggingface_hub)

---

## Appendix: Error Reference

### Error 1: SyntaxError (FIXED)
```
File "qwen3_vl_embedding_service.py", line 178
    nonlocal _processor, _embedding_model
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
SyntaxError: no binding for nonlocal '_processor' found
```
**Fix:** Changed `nonlocal` to `global` since variables are module-level globals.

### Error 2: Architecture Not Recognized (BLOCKING)
```
ValueError: The checkpoint you are trying to load has model type `qwen3_vl`
but Transformers does not recognize this architecture. This could be because
of an issue with the checkpoint, or because your version of Transformers is
out of date.
```
**Fix:** Upgrade transformers to >= 4.57.0
