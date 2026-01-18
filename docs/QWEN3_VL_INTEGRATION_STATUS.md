# Qwen3-VL Multimodal Embedding & Reranking Integration

**Status:** Phase 4 Complete - Production Ready
**Started:** 2026-01-17
**Last Updated:** 2026-01-19

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
| Graph Traversal | `cortex_engine/graph_query.py` | ✅ Done | Fixed undirected graph support |

### Phase 3: UI Integration ✅ COMPLETE

| Component | File | Status | Description |
|-----------|------|--------|-------------|
| Reranker Toggle | `pages/3_Knowledge_Search.py` | ✅ Done | Checkbox to enable/disable reranking |
| Result Count Slider | `pages/3_Knowledge_Search.py` | ✅ Done | Slider (5-50) for controlling output count |
| Background Preload | `pages/3_Knowledge_Search.py` | ✅ Done | Model preloads when page opens |
| Status Indicator | `pages/3_Knowledge_Search.py` | ✅ Done | Shows "Loading...", "Ready" status |
| Dynamic Timeout | `pages/3_Knowledge_Search.py` | ✅ Done | 300s timeout for first model load |

### Phase 4: Testing & Validation ✅ COMPLETE

| Test | Status | Results |
|------|--------|---------|
| Model Loading | ✅ Pass | 8B model loads in ~3-4 minutes |
| Traditional Vector Search | ✅ Pass | 50 candidates → 20 results after reranking |
| GraphRAG Enhanced Search | ✅ Pass | Graph context + reranking working |
| Hybrid Search | ✅ Pass | Combined results with graph enhancement |
| Performance Benchmarks | ✅ Pass | ~8s for 24 docs, ~43s for 50 docs |

---

## Performance Benchmarks (RTX 8000 - 48GB)

### Search Pipeline Performance

| Stage | Time | Notes |
|-------|------|-------|
| Vector Search (50 candidates) | ~0.3s | BGE embeddings on GPU |
| PageRank Calculation | ~0.7s | 6,324 documents |
| GraphRAG Enhancement | ~0.03s | Entity context |
| Neural Reranking (24 docs) | **7.6s** | Main cost |
| Neural Reranking (50 docs) | **43s** | Scales linearly |
| **Total (typical)** | **~8-10s** | With graph enhancement |

### Model Memory Usage

| Model | VRAM Usage |
|-------|------------|
| Embedding Model (BGE) | ~0.5GB |
| Reranker (Qwen3-VL-8B) | ~16GB |
| **Combined** | ~17GB |

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

### Step 4: Enable Qwen3-VL Reranker

**Option A: UI Toggle (Recommended)**
1. Start Cortex Suite: `streamlit run Cortex_Suite.py`
2. Navigate to Knowledge Search page
3. In sidebar under "Search Engine", check **"Neural Reranking"** checkbox
4. Model preloads automatically in background (~3-4 minutes)
5. Sidebar shows "🟢 Model ready" when loaded
6. Use slider to control result count (5-50)

**Option B: Environment Variables**
Set before starting Streamlit:

```bash
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
| `QWEN3_VL_RERANKER_TOP_K` | `20` | Results after reranking |
| `QWEN3_VL_RERANKER_CANDIDATES` | `50` | Candidates before reranking |
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

## Architecture

### Three-Stage Retrieval Pipeline

```
Query → [Stage 1: Vector Search] → 50 candidates
                    ↓
      [Stage 2: GraphRAG Enhancement] → Entity context, PageRank
                    ↓
      [Stage 3: Neural Reranking] → Precision scoring
                    ↓
              Top 20 results (configurable)
```

### Key Components

1. **Vector Search (ChromaDB)**
   - BGE embeddings (768 dimensions)
   - Fast approximate nearest neighbor search
   - Returns top 50 candidates

2. **GraphRAG Enhancement**
   - Entity extraction and relationship mapping
   - PageRank scoring for document importance
   - Multi-hop traversal for context

3. **Neural Reranking (Qwen3-VL)**
   - Binary classification: "Is document relevant to query?"
   - Uses `lm_head.weight[yes] - lm_head.weight[no]` for scoring
   - Sigmoid activation for 0-1 relevance score

---

## Troubleshooting

### Common Issues

**Model won't load:**
- Check GPU memory: `nvidia-smi`
- Try 2B model: `export QWEN3_VL_RERANKER_SIZE=2B`

**Slow first search:**
- Model loading takes 3-4 minutes first time
- Wait for sidebar to show "🟢 Model ready"
- Subsequent searches are fast (~8-10s)

**Graph traversal errors:**
- Fixed: `'Graph' object has no attribute 'predecessors'`
- Solution: Code now checks `is_directed()` before using `predecessors()`

**Out of memory:**
- Reduce batch size: `export QWEN3_VL_RERANK_BATCH_SIZE=2`
- Use smaller model: `export QWEN3_VL_RERANKER_SIZE=2B`

---

## Future Enhancements

### Potential Improvements

1. **Multimodal Embedding for Ingestion**
   - Use Qwen3-VL embeddings during document ingestion
   - Enable image/chart search within documents

2. **Batch Reranking Optimization**
   - Process multiple query-document pairs in parallel
   - Could reduce 50-doc reranking from 43s to ~15s

3. **Model Caching Across Sessions**
   - Keep model loaded between Streamlit reruns
   - Use `st.cache_resource` for persistence

4. **Cross-Modal Search**
   - Search images using text queries
   - Search text using image queries

---

## Changelog

### 2026-01-19
- ✅ Fixed undirected graph traversal (predecessors error)
- ✅ Verified Hybrid Search working correctly
- ✅ Performance benchmarks documented
- ✅ Phase 4 testing complete

### 2026-01-18
- ✅ Added UI toggle for reranker
- ✅ Added result count slider (5-50)
- ✅ Background preload when page opens
- ✅ Dynamic timeout (300s for model load)
- ✅ Increased default candidates (50) and results (20)

### 2026-01-17
- ✅ Fixed reranker scoring (official implementation)
- ✅ Flash Attention 2 fallback when not installed
- ✅ Initial integration complete
