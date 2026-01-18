# Qwen3-VL Multimodal Embedding & Reranking Integration

**Status:** Phase 3 Complete - Testing & Debugging
**Started:** 2026-01-17
**Last Updated:** 2026-01-17

## Overview

This document tracks the integration of Qwen3-VL multimodal embedding and reranking models into the Cortex Suite knowledge management system.

### What is Qwen3-VL?

Qwen3-VL-Embedding and Qwen3-VL-Reranker are multimodal models that map text, images, and videos into a **unified vector space**. This enables:

- **Cross-modal search**: Use text queries to find relevant images, charts, diagrams
- **Visual document search**: Embed PDFs/charts as images for semantic search (beyond OCR)
- **Two-stage retrieval**: Fast embedding recall (~85% precision) + precise reranking (~95%+ precision)

### Model Specifications

| Model | Size | VRAM | Dimensions | Notes |
|-------|------|------|------------|-------|
| Qwen3-VL-Embedding-2B | 2B params | ~5GB | 2048 | Efficient option |
| Qwen3-VL-Embedding-8B | 8B params | ~16GB | 4096 | High quality |
| Qwen3-VL-Reranker-2B | 2B params | ~5GB | - | Fast reranking |
| Qwen3-VL-Reranker-8B | 8B params | ~16GB | - | Precise reranking |

**Key Feature: Matryoshka Representation Learning (MRL)**
Allows truncating embedding vectors (e.g., 4096 → 1024) for storage efficiency while maintaining quality.

---

## Implementation Status

### Phase 1: Core Services ✅ COMPLETE

| Component | File | Status | Description |
|-----------|------|--------|-------------|
| Requirements | `requirements-qwen3-vl.txt` | ✅ Done | Dependencies for Qwen3-VL |
| Embedding Service | `cortex_engine/qwen3_vl_embedding_service.py` | ✅ Done | Full multimodal embedding (617 lines) |
| Reranker Service | `cortex_engine/qwen3_vl_reranker_service.py` | ✅ Done | Two-stage reranker (628 lines) |
| LlamaIndex Adapter | `cortex_engine/qwen3_vl_llamaindex_adapter.py` | ✅ Done | Drop-in LlamaIndex integration (509 lines) |
| Configuration | `cortex_engine/config.py` | ✅ Done | Added Qwen3-VL config options |
| Model Selector | `cortex_engine/utils/smart_model_selector.py` | ✅ Done | Auto-selects model based on VRAM |
| Embedding Validator | `cortex_engine/utils/embedding_validator.py` | ✅ Done | Added Qwen3-VL dimensions |

### Phase 2: Pipeline Integration ✅ COMPLETE

| Component | File | Status | Description |
|-----------|------|--------|-------------|
| Embedding Service | `cortex_engine/embedding_service.py` | ✅ Done | Switchable backend (BGE/NV-Embed ↔ Qwen3-VL) |
| Embedding Adapters | `cortex_engine/embedding_adapters.py` | ✅ Done | LlamaIndex multimodal support |
| Knowledge Search | `pages/3_Knowledge_Search.py` | ✅ Done | Optional neural reranking in direct search |
| Hybrid Search | `cortex_engine/graph_query.py` | ✅ Done | Three-stage retrieval with reranking |
| Knowledge Ingest | `pages/2_Knowledge_Ingest.py` | ⏳ Pending | UI controls for Qwen3-VL mode |
| Collection Manager | `pages/5_Collection_Management.py` | ⏳ Pending | Support mixed-modality collections |

### Phase 3: UI Integration 🚧 IN PROGRESS

| Component | File | Status | Description |
|-----------|------|--------|-------------|
| Ingest Sidebar | `pages/2_Knowledge_Ingest.py` | ✅ Done | Qwen3-VL status, model info, setup instructions |
| Search Sidebar | `pages/3_Knowledge_Search.py` | ✅ Done | Embedding info, reranker status display |
| Settings Page | `pages/Settings.py` | ⏳ Pending | Qwen3-VL enable/disable toggle |
| Main App | `Cortex_Suite.py` | ⏳ Pending | Global Qwen3-VL status indicator |

### Phase 4: Testing & Validation 🚧 PENDING

| Task | Status |
|------|--------|
| Unit tests for embedding service | ⏳ Pending |
| Unit tests for reranker service | ⏳ Pending |
| Integration tests with real documents | ⏳ Pending |
| Performance benchmarks | ⏳ Pending |
| Cross-modal search validation | ⏳ Pending |

---

## Installation Guide

### Prerequisites

- Python 3.11
- NVIDIA GPU with CUDA support (minimum 6GB VRAM for 2B models)
- Base Cortex Suite dependencies installed

### Step 1: Base Installation

```bash
cd /home/longboardfella/cortex_suite
source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Step 2: Install Qwen3-VL Dependencies

```bash
pip install -r requirements-qwen3-vl.txt
```

### Step 3: (Recommended) Install Flash Attention 2

For memory optimization on supported GPUs:

```bash
pip install flash-attn --no-build-isolation
```

### Step 4: Enable Qwen3-VL

Set environment variables or update `cortex_config.json`:

```bash
export QWEN3_VL_ENABLED=true
export QWEN3_VL_RERANKER_ENABLED=true
export QWEN3_VL_MODEL_SIZE=auto  # or "2B" or "8B"
```

---

## Configuration Reference

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `QWEN3_VL_ENABLED` | `false` | Enable Qwen3-VL embedding |
| `QWEN3_VL_MODEL_SIZE` | `auto` | Model size: "auto", "2B", "8B" |
| `QWEN3_VL_MRL_DIM` | (none) | MRL dimension reduction: 64, 128, 256, 512, 1024, 2048 |
| `QWEN3_VL_RERANKER_ENABLED` | `false` | Enable reranking |
| `QWEN3_VL_RERANKER_SIZE` | `auto` | Reranker size: "auto", "2B", "8B" |
| `QWEN3_VL_RERANKER_TOP_K` | `5` | Results after reranking |
| `QWEN3_VL_RERANKER_CANDIDATES` | `20` | Candidates before reranking |
| `QWEN3_VL_USE_FLASH_ATTENTION` | `true` | Use Flash Attention 2 |
| `QWEN3_VL_EMBED_BATCH_SIZE` | `8` | Embedding batch size |
| `QWEN3_VL_RERANK_BATCH_SIZE` | `4` | Reranking batch size |

### Auto-Selection by VRAM

The `auto` setting selects models based on available GPU memory:

| Available VRAM | Embedding Model | Reranker Model | Notes |
|----------------|-----------------|----------------|-------|
| ≥40GB | 8B | 8B | Premium (RTX 8000, A100) |
| ≥24GB | 8B | 2B | High (RTX 4090, A6000) |
| ≥16GB | 8B | 2B (on-demand) | Standard (RTX 3090) |
| ≥10GB | 2B | 2B | Efficient (RTX 3080) |
| ≥6GB | 2B | None | Minimal (RTX 3060) |

---

## Usage Examples

### Direct API Usage

```python
from cortex_engine.qwen3_vl_embedding_service import (
    embed_text, embed_image, embed_multimodal
)
from cortex_engine.qwen3_vl_reranker_service import rerank_results

# Text embedding
text_vec = embed_text("quarterly revenue chart")

# Image embedding (same vector space!)
img_vec = embed_image("/path/to/chart.png")

# Cross-modal similarity
similarity = text_vec @ img_vec.T

# Rerank search results
candidates = vector_search(query, top_k=20)
reranked = rerank_results(query, candidates, top_k=5)
```

### LlamaIndex Integration

```python
from cortex_engine.qwen3_vl_llamaindex_adapter import (
    Qwen3VLEmbedding, Qwen3VLReranker
)
from llama_index.core import VectorStoreIndex

# Create index with Qwen3-VL embeddings
embed_model = Qwen3VLEmbedding(model_size="auto")
index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)

# Query with reranking
reranker = Qwen3VLReranker(top_n=5)
query_engine = index.as_query_engine(
    node_postprocessors=[reranker],
    similarity_top_k=20  # Get more candidates for reranking
)
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Cortex Suite RAG Pipeline                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   Document   │───▶│   Qwen3-VL   │───▶│   ChromaDB   │       │
│  │   + Images   │    │  Embedding   │    │ Vector Store │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│                                                 │                │
│                                                 ▼                │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │    Query     │───▶│   Vector     │───▶│  Top-K       │       │
│  │  (text/img)  │    │   Search     │    │  Candidates  │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│                                                 │                │
│                                                 ▼                │
│                      ┌──────────────┐    ┌──────────────┐       │
│                      │   Qwen3-VL   │───▶│   Reranked   │       │
│                      │   Reranker   │    │   Results    │       │
│                      └──────────────┘    └──────────────┘       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Troubleshooting

### Common Issues & Solutions

#### 1. Embedding Model Not Loading (HuggingFace Offline Mode)

**Symptoms:**
- Search returns no results with vector search
- Logs show "Failed to load embedding model"
- `HF_HUB_OFFLINE` environment variable set to "1"

**Solution:**
```bash
# Temporarily disable offline mode to download models
HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python -c "
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('BAAI/bge-base-en-v1.5')
print('Model downloaded successfully')
"
```

#### 2. Torch/Torchvision Version Mismatch

**Symptoms:**
- `RuntimeError: CUDA error` or version compatibility warnings
- torch 2.9.x with torchvision 0.18.x causes issues

**Solution:**
```bash
pip install torchvision==0.20.1  # Match torch 2.9.1
```

#### 3. Numpy Array Comparison Error

**Symptoms:**
- "ValueError: truth value of an array with more than one element is ambiguous"
- Occurs in Database Embedding Inspector

**Solution:**
Fixed in code - use explicit length checks instead of boolean evaluation:
```python
# Wrong
if embeddings and embeddings[0]:

# Correct
has_embeddings = (embeddings is not None and len(embeddings) > 0 and len(embeddings[0]) > 0)
```

#### 4. ChromaDB Database Lock

**Symptoms:**
- "database is locked" errors
- Cannot access collections

**Solution:**
```bash
# Find and kill processes using the database
lsof +D /path/to/ai_databases/knowledge_hub_db/

# Remove stale lock files if necessary
rm /path/to/ai_databases/knowledge_hub_db/*.lock
```

#### 5. Qwen3-VL Reranker Missing Dependencies

**Symptoms:**
- "Could not import module 'AutoProcessor'"
- Reranker shows as disabled in UI

**Solution:**
```bash
pip install -r requirements-qwen3-vl.txt
```

### Database Compatibility

Use the **Database Embedding Inspector** in the Maintenance tab to check:
- Stored embedding dimensions
- Compatible models for your database
- Migration requirements for Qwen3-VL

| Stored Dimension | Original Model | Qwen3-VL Compatible? |
|------------------|----------------|----------------------|
| 768 | bge-base-en-v1.5 | Reranker only (no re-embed needed) |
| 1024 | bge-large-en-v1.5 | Reranker only (no re-embed needed) |
| 2048 | Qwen3-VL-2B | Full multimodal support |
| 4096 | Qwen3-VL-8B or NV-Embed-v2 | Check source model |

---

## Current Status (2026-01-18)

### What's Working
- **Traditional Vector Search**: Returns proper results with similarity scores (e.g., 0.358)
- **Hybrid Search**: Combines vector + GraphRAG, returns 13+ results with real scores
- **GraphRAG Enhanced Search**: Now returns real documents (via text-based fallback)
- **UI displays**: Qwen3-VL status in Ingest and Search sidebars
- **Database Embedding Inspector**: Utility in Maintenance tab
- **Search Debug Log**: Expandable panel showing search execution details

### Fixed Issues (2026-01-18 Session)
1. **EnhancedGraphManager missing methods** - Added `entity_index` property and `get_graph_stats()` method to `cortex_engine/graph_manager.py`
2. **GraphRAG returning only graph entities** - Added text-based fallback to `ChromaRetriever` class
3. **MockResult missing text attribute** - Graph-discovered documents now have proper text content
4. **Session state debug in threads** - Moved debug logging outside ThreadPoolExecutor

### Known Issues (To Investigate)
1. **ChromaRetriever vector search failing silently**:
   - Hybrid search uses `direct_chromadb_search` which works (returns real similarity scores)
   - GraphRAG uses `ChromaRetriever` which falls back to text search (returns 0.800 scores)
   - Debug print statements inside ChromaRetriever don't appear in terminal (even with flush=True)
   - Likely cause: Thread output being swallowed or ChromaDB client issue in thread context

2. **Numpy circular import**: Intermittently requires `pip install --force-reinstall numpy==1.26.4`

3. **Qwen3-VL not yet tested**: Dependencies need installation (`pip install -r requirements-qwen3-vl.txt`)

### Performance Observations
| Search Mode | Speed | Score Type | Results |
|-------------|-------|------------|---------|
| Traditional | ~0.6s | Real similarity (0.3-0.4) | 20 results |
| Hybrid | ~1.5s | Real similarity (0.3-0.4) | 13 results |
| GraphRAG Enhanced | ~4-15s | Text match (0.8) | 10 results |

### Next Steps
1. **Fix ChromaRetriever vector search** - Investigate why it fails while `direct_chromadb_search` works
2. **Install Qwen3-VL dependencies**: `pip install -r requirements-qwen3-vl.txt`
3. **Test reranker end-to-end** with real queries
4. **Enable Qwen3-VL embeddings** - Re-ingest documents with multimodal embeddings
5. **Test cross-modal search** with documents and images
6. **Benchmark performance** against current BGE embedding model

---

## References

- [Qwen3-VL-Embedding on HuggingFace](https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B)
- [Qwen3-VL-Reranker on HuggingFace](https://huggingface.co/Qwen/Qwen3-VL-Reranker-8B)
- [MMEB Benchmark Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
