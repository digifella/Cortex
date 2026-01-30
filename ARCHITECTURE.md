# Cortex Suite — Architecture Document

**Version:** 5.8.0
**Date:** 2026-01-30
**Purpose:** Comprehensive architectural reference for enterprise adaptation

---

## 1. Executive Summary

Cortex Suite is an AI-powered knowledge management and proposal generation platform built on Python/Streamlit. It ingests documents into a vector store (ChromaDB) and knowledge graph (NetworkX), provides multi-strategy search (vector, GraphRAG, hybrid), and uses those results to drive intelligent proposal generation with evidence-based responses.

The system runs locally in WSL2 or Docker, uses Ollama for LLM inference, and supports adaptive embedding models (BGE, NV-Embed, Qwen3-VL multimodal).

---

## 2. System Overview

```
+------------------------------------------------------------+
|                     Streamlit Frontend                      |
|  (Cortex_Suite.py + 18 page modules in pages/)             |
+-----------------------------+------------------------------+
                              |
              +---------------+---------------+
              |                               |
+-------------v-----------+   +--------------v--------------+
|    cortex_engine/        |   |         api/main.py          |
|  (Business Logic Layer)  |   |     (FastAPI REST API)       |
+-----------+----+---------+   +-----------------------------+
            |    |
    +-------v-+  +--------v--------+
    | ChromaDB |  | NetworkX Graph  |
    | (Vector) |  | (Knowledge)     |
    +----------+  +-----------------+
            |
    +-------v-------+
    |    Ollama      |
    | (LLM Service)  |
    +----------------+
```

### Key Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Frontend | Streamlit | Multi-page web UI |
| API | FastAPI | REST endpoints for external integration |
| Vector Store | ChromaDB | Document embeddings and similarity search |
| Knowledge Graph | NetworkX | Entity relationships, PageRank scoring |
| LLM Provider | Ollama | Local inference (Mistral, LLaVA, Qwen3-VL) |
| Embeddings | SentenceTransformers / Qwen3-VL | Document and query vectorisation |
| NER | spaCy | Named entity recognition |
| Document Parsing | Docling / PyMuPDF / python-docx / python-pptx | Multi-format document reading |
| Containerisation | Docker / Docker Compose | Deployment and distribution |

---

## 3. Directory Structure

```
cortex_suite/
├── Cortex_Suite.py              # Main entry point
├── cortex_config.json           # User configuration (persisted)
├── CHANGELOG.md                 # Release history
│
├── cortex_engine/               # Business logic (100+ modules)
│   ├── config.py                # Central configuration & model selection
│   ├── config_manager.py        # Persistent user settings (JSON)
│   ├── version_config.py        # Single source of truth for version
│   ├── embedding_service.py     # Embedding model management
│   ├── ingest_cortex.py         # Core document ingestion pipeline
│   ├── collection_manager.py    # Working collection CRUD
│   ├── graph_query.py           # GraphRAG search engine
│   ├── textifier.py             # Document-to-Markdown converter
│   ├── anonymizer.py            # PII redaction
│   ├── model_services/          # Hybrid model distribution
│   ├── proposals/               # Proposal field parsing & matching
│   ├── utils/                   # Path handling, logging, GPU, caching
│   └── ...
│
├── pages/                       # Streamlit page modules
│   ├── 1_AI_Assisted_Research.py
│   ├── 2_Knowledge_Ingest.py
│   ├── 3_Knowledge_Search.py
│   ├── 4_Collection_Management.py
│   ├── 5_Knowledge_Analytics.py
│   ├── 6_Maintenance.py
│   ├── 7_Document_Extract.py
│   ├── 8_Document_Summarizer.py
│   ├── 9_Knowledge_Synthesizer.py
│   ├── 10_Visual_Analysis.py
│   ├── 11_Metadata_Management.py
│   ├── 12_Document_Dialog.py
│   ├── Proposal_Workspace.py
│   ├── Proposal_Chunk_Review_V2.py
│   ├── Proposal_Intelligent_Completion.py
│   ├── Entity_Profile_Manager.py
│   └── components/              # Setup wizard, system terminal
│
├── api/
│   └── main.py                  # FastAPI REST API
│
├── scripts/                     # CLI utilities
│   ├── version_manager.py       # Version sync across 50+ files
│   ├── embedding_inspector.py   # Inspect embedding dimensions
│   ├── embedding_migrator.py    # Migrate between embedding models
│   └── graphrag_retroactive_extraction.py
│
└── docker/                      # Docker distribution
    ├── Dockerfile
    ├── docker-compose.yml
    ├── .env.example
    ├── run-cortex.bat / .sh
    └── (mirrored source files)
```

---

## 4. Core Workflows

### 4.1 Knowledge Workflow

```
  Documents (PDF, DOCX, PPTX, TXT, images)
       │
       v
  ┌─────────────────┐
  │ Knowledge Ingest │  ← Page 2
  │  (Batch/Single)  │
  └────────┬────────┘
           │
     ┌─────┴─────┐
     v           v
  ┌──────┐  ┌────────┐
  │Chunk │  │ Entity │
  │& Embed│  │Extract │
  └──┬───┘  └───┬────┘
     │          │
     v          v
  ChromaDB   NetworkX
  (vectors)  (graph)
     │          │
     └────┬─────┘
          v
  ┌────────────────┐
  │Knowledge Search│  ← Page 3
  │ Vector/Graph/  │
  │   Hybrid       │
  └───────┬────────┘
          v
  ┌──────────────────┐
  │   Collections    │  ← Page 4
  │ (Group Results)  │
  └──────────────────┘
```

### 4.2 Proposal Workflow

```
  ┌─────────────────────┐
  │ Entity Profile Mgr  │  Define company, capabilities, personnel
  └──────────┬──────────┘
             │
  ┌──────────v──────────┐
  │ Proposal Workspace  │  Upload tender, bind entity profile,
  │                     │  select evidence collection
  └──────────┬──────────┘
             │
  ┌──────────v──────────┐
  │  Chunk Review       │  Extract & review tender sections
  └──────────┬──────────┘
             │
  ┌──────────v──────────┐
  │ Intelligent         │  Two tiers:
  │ Completion          │  • Auto-complete (name, ABN, contacts)
  │                     │  • AI-generated (methodology, capability)
  └─────────────────────┘    with knowledge evidence injection
```

---

## 5. Document Ingestion Pipeline

### Processing Stages

| Stage | Component | Description |
|-------|-----------|-------------|
| 1. Read | Docling / LlamaIndex / fallback | Parse PDF, DOCX, PPTX, TXT into text |
| 2. Chunk | document_chunker.py | Split into overlapping chunks (section-aware optional) |
| 3. Extract | entity_extractor.py (spaCy) | NER for people, orgs, locations |
| 4. Relate | graph_extraction_worker.py | Build entity-entity relationships |
| 5. Embed | embedding_service.py | Generate vector embeddings per chunk |
| 6. Store | ChromaDB + NetworkX | Persist vectors and graph |
| 7. Classify | document_type_manager.py | Classify document type |
| 8. Collect | collection_manager.py | Add to default working collection |

### Document Reader Priority

1. **Docling** (IBM) — best quality, layout-aware, OCR support
2. **LlamaIndex readers** — good fallback for standard formats
3. **Plain text extraction** — last resort

### Metadata Stored Per Chunk

```json
{
  "doc_id": "unique-document-id",
  "chunk_id": "chunk_0",
  "file_name": "proposal.pdf",
  "doc_posix_path": "/path/to/proposal.pdf",
  "document_type": "Proposal/Quote",
  "proposal_outcome": "Won",
  "thematic_tags": "infrastructure, water, design",
  "summary": "Executive summary of the document...",
  "last_modified_date": "2026-01-15T10:30:00",
  "extracted_entities": "[{\"name\": \"Acme Corp\", \"type\": \"ORG\"}]"
}
```

---

## 6. Embedding System

### Adaptive Model Selection

The system auto-detects the best embedding model for available hardware:

| Model | Dimensions | Requirements | Use Case |
|-------|-----------|-------------|----------|
| Qwen3-VL 8B | 4096 | 16GB+ VRAM | Multimodal (text + image) |
| Qwen3-VL 2B | 2048 | 5GB+ VRAM | Multimodal, smaller GPU |
| NV-Embed-v2 | 4096 | 2-6GB VRAM | Text-only, high quality |
| BGE-base-en-v1.5 | 768 | CPU only | Universal fallback |

### Selection Logic

```
if GPU >= 16GB and qwen-vl-utils installed → Qwen3-VL 8B
elif GPU >= 5GB and qwen-vl-utils installed → Qwen3-VL 2B
elif NVIDIA GPU with 2-6GB                  → NV-Embed-v2
else                                        → BGE-base-en-v1.5
```

Override via environment: `CORTEX_EMBED_MODEL=BAAI/bge-base-en-v1.5`

### Architecture

```python
# embedding_service.py — Singleton with lazy loading
embed_query(text)       # Single query → vector (cached LRU 50)
embed_documents(texts)  # Batch embed with optimal GPU batch size
```

- Thread-safe model initialisation
- Query cache avoids re-embedding in hybrid search
- GPU memory monitoring with adaptive batch sizes

---

## 7. Search Architecture

### Three Search Modes

#### Traditional Vector Search
```
Query → embed_query() → ChromaDB.query(embedding, n=200)
     → threshold filter → diversity filter (5 chunks/doc)
     → post-search filters → optional reranking → results
```

#### GraphRAG Enhanced
```
Query → entity extraction → graph traversal (multi-hop)
     → PageRank scoring → merge with vector results
     → neural reranking → results
```

#### Hybrid Search
```
Query → [Vector Search] + [GraphRAG Search] in parallel
     → deduplicate → merge scores → strict mode filter
     → reranking → results
```

### Similarity Scoring

Distances from ChromaDB (L2) are converted to similarity:
```
similarity = 1.0 / (1.0 + l2_distance)
```

Model-aware thresholds:
- 2048D embeddings: threshold = 0.30
- 4096D embeddings: threshold = 0.40

### Text Fallback Recovery

When vector search returns candidates but all are below threshold, the system falls back to keyword matching:
- Scans up to 10,000 documents
- Strict mode: all query terms must be present
- Normal mode: at least 1 term must match
- Results scored by term frequency

### Neural Reranking (Optional)

Two-stage retrieval using Qwen3-VL reranker:
1. Fast recall: embedding search returns 50+ candidates
2. Precision: reranker scores and re-orders top results

---

## 8. Knowledge Graph (GraphRAG)

### Structure

- **Nodes**: Entities (PERSON, ORG, PROJECT, LOCATION, DOCUMENT)
- **Edges**: Relationships (authored, worked_on, client_of, collaborated_with, mentioned_in)
- **Storage**: NetworkX pickle at `<db_path>/knowledge_cortex.gpickle`

### Features

- **PageRank scoring** — importance-weighted entity ranking
- **Multi-hop traversal** — follow relationship chains (depth configurable)
- **Query expansion** — extract entities from query, find related entities
- **Persistent cache** — SQLite-backed LRU cache for repeated queries

### Graph Building (During Ingestion)

```
Document chunks → spaCy NER → entity nodes
                            → co-occurrence → relationship edges
                            → document-entity links
```

---

## 9. Database Architecture

### Storage Layout

```
<ai_database_path>/
├── knowledge_hub_db/          # ChromaDB persistent store
│   ├── chroma.sqlite3         # Metadata + collection info
│   └── [vector index files]   # HNSW index
├── knowledge_cortex.gpickle   # NetworkX knowledge graph
├── working_collections.json   # Collection definitions
└── workspaces/                # Proposal workspaces (YAML)
```

### ChromaDB Collection Schema

- **Collection name**: `knowledge_hub_collection`
- **Embedding function**: None (embeddings provided at insert time)
- **Distance metric**: L2 (Euclidean)

### Working Collections

JSON file mapping collection names to document ID sets:

```json
{
  "Deakin Projects": {
    "name": "Deakin Projects",
    "doc_ids": ["doc_abc123", "doc_def456"],
    "created_at": "2026-01-15T10:00:00",
    "modified_at": "2026-01-28T14:30:00",
    "embedding_model": "BAAI/bge-base-en-v1.5",
    "embedding_dim": 768
  }
}
```

---

## 10. Proposal Generation System

### Three-Stage Pipeline

#### Stage 1: Workspace Setup
- Upload tender document (PDF/DOCX)
- Bind entity profile (company details, personnel, capabilities)
- Select evidence collection from knowledge base
- Stored as YAML in `workspaces/`

#### Stage 2: Chunk Review
- Extract tender into sections (Company Details, Personnel, Methodology, etc.)
- Review and edit extracted chunks
- Navigate by section type

#### Stage 3: Intelligent Completion

**Tier 1 — Auto-Complete** (direct substitution):
- Company name, ABN, contact details, address
- Pulled directly from entity profile

**Tier 2 — AI-Generated** (evidence-driven):
1. Classify question type (capability, methodology, compliance, etc.)
2. Reformulate query based on question type
3. Search nominated collection for relevant evidence
4. Inject top evidence chunks into LLM prompt
5. Generate response with Ollama (Mistral)
6. Score confidence based on evidence quality
7. Flag for human review if low confidence

### Question Type Classification

| Type | Example | Search Strategy |
|------|---------|----------------|
| CAPABILITY | "Describe relevant experience" | Past project evidence |
| METHODOLOGY | "Outline your approach" | Process documentation |
| COMPLIANCE | "Provide certifications" | Policy documents |
| PERSONNEL | "Key team members" | Entity profiles + CVs |
| INNOVATION | "Novel approaches" | Technical docs + case studies |
| RISK | "Risk mitigation" | Risk frameworks |
| PRICING | "Fee schedule" | Financial templates |

---

## 11. API Layer

### FastAPI REST API (`api/main.py`)

Base URL: `http://localhost:8000`

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/search` | POST | Execute search query |
| `/search/batch` | POST | Batch search |
| `/ingest` | POST | Ingest documents |
| `/ingest/batch` | POST | Batch ingest |
| `/collections` | GET | List collections |
| `/collections` | POST | Create collection |
| `/collections/{name}` | GET | Get collection details |
| `/collections/{name}` | DELETE | Delete collection |
| `/backup` | POST | Create backup |
| `/restore` | POST | Restore from backup |

Authentication: Optional HTTPBearer token.

---

## 12. Page Modules Reference

### Knowledge Management

| Page | File | Purpose |
|------|------|---------|
| AI Research | `1_AI_Assisted_Research.py` | Multi-agent external research with synthesis |
| Knowledge Ingest | `2_Knowledge_Ingest.py` | Document ingestion with batch mode |
| Knowledge Search | `3_Knowledge_Search.py` | Vector/GraphRAG/hybrid search |
| Collection Mgmt | `4_Collection_Management.py` | CRUD for working collections |
| Analytics | `5_Knowledge_Analytics.py` | Usage patterns, knowledge gaps |
| Maintenance | `6_Maintenance.py` | DB cleanup, backup/restore |

### Document Tools

| Page | File | Purpose |
|------|------|---------|
| Document Extract | `7_Document_Extract.py` | Textifier (doc→Markdown) + Anonymizer |
| Summarizer | `8_Document_Summarizer.py` | Multi-level summarisation |
| Synthesizer | `9_Knowledge_Synthesizer.py` | Synthesise collection into structured output |
| Visual Analysis | `10_Visual_Analysis.py` | Theme network visualisation |
| Metadata Mgmt | `11_Metadata_Management.py` | Browse/edit document metadata and tags |
| Document Dialog | `12_Document_Dialog.py` | Interactive Q&A with documents |

### Proposal Generation

| Page | File | Purpose |
|------|------|---------|
| Workspace | `Proposal_Workspace.py` | Create proposal workspace |
| Chunk Review | `Proposal_Chunk_Review_V2.py` | Review extracted tender sections |
| Completion | `Proposal_Intelligent_Completion.py` | AI-assisted response generation |
| Entity Profiles | `Entity_Profile_Manager.py` | Company/personnel profiles |

---

## 13. Configuration Reference

### User Configuration (`cortex_config.json`)

```json
{
  "ai_database_path": "/path/to/ai_databases",
  "knowledge_source_path": "/path/to/documents",
  "db_path": "/path/to/ai_databases"
}
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CORTEX_EMBED_MODEL` | auto-detect | Force specific embedding model |
| `QWEN3_VL_ENABLED` | auto | Enable Qwen3-VL multimodal |
| `QWEN3_VL_MODEL_SIZE` | auto | 2B or 8B |
| `QWEN3_VL_RERANKER_ENABLED` | false | Neural reranking |
| `MODEL_DISTRIBUTION_STRATEGY` | hybrid_ollama_preferred | Model backend strategy |
| `DOCLING_VLM_ENABLED` | false | VLM for figure processing |
| `TABLE_AWARE_CHUNKING` | false | Section-aware chunking |
| `OLLAMA_HOST` | http://localhost:11434 | Ollama service URL |

### Model Requirements

| Model | Purpose | Install |
|-------|---------|---------|
| mistral-small3.2 | Proposals, synthesis | `ollama pull mistral-small3.2` |
| mistral:latest | General LLM tasks | `ollama pull mistral:latest` |
| qwen3-vl:8b | Vision, image description | `ollama pull qwen3-vl:8b` |
| en_core_web_sm | NER (spaCy) | `python -m spacy download en_core_web_sm` |

---

## 14. Docker Deployment

### Single Container (Simple)

```bash
docker build -t cortex-suite .
docker run -p 8501:8501 -v cortex_data:/data cortex-suite
```

### Multi-Container (Production)

```yaml
# docker-compose.yml services:
services:
  ollama:        # LLM inference (port 11434)
  chromadb:      # Vector database (port 8001)
  cortex-api:    # FastAPI backend (port 8000)
  cortex-ui:     # Streamlit frontend (port 8501)
  model-init:    # One-shot model download
```

### Storage Modes

- **Portable**: Docker volumes (data lost on container removal)
- **External**: Host directory mounts (persistent, production recommended)

### Path Handling

```python
if os.path.exists('/.dockerenv'):
    # Docker: use paths as-is (volumes handle mapping)
    path = configured_path
else:
    # WSL: convert Windows paths
    path = convert_windows_to_wsl_path(configured_path)
```

---

## 15. Cross-Cutting Concerns

### Caching Strategy

| Layer | Mechanism | TTL | Purpose |
|-------|-----------|-----|---------|
| Query embeddings | In-memory LRU (50) | Session | Avoid re-embedding same query |
| ChromaDB client | `@st.cache_resource` | 5 min | Reuse DB connections |
| GraphRAG results | SQLite persistent | Indefinite | Cache graph traversals |
| PageRank scores | Module-level dict | Session | Fast entity importance |
| Sidebar data | `@st.cache_data` | 2 min | Reduce repeated DB queries |

### Error Recovery

- **Ingestion**: Crash recovery with state persistence, resume from last chunk
- **Search**: Text fallback when vector search fails or filters everything
- **Embedding**: Graceful fallback from Qwen3-VL → NV-Embed → BGE
- **Services**: Health checks with auto-restart in Docker

### Thread Safety

- Model loading uses `threading.Lock` singletons
- Module-level locks for debug log buffers in search
- Session state isolation per Streamlit user

### Security Considerations

- PII anonymisation tool (Document Extract → Anonymizer)
- No credentials stored in code (environment variables)
- CORS configuration on API
- Optional API authentication (HTTPBearer)
- File permissions set on uploaded temp files (0o644)

---

## 16. Adapting for Enterprise

### What to Change

| Area | Current | Enterprise Adaptation |
|------|---------|----------------------|
| LLM Provider | Ollama (local) | Azure OpenAI, AWS Bedrock, or self-hosted vLLM |
| Vector Store | ChromaDB (local) | Pinecone, Weaviate, Qdrant, or Chroma Cloud |
| Auth | None / optional bearer | SSO (SAML/OIDC), RBAC |
| Storage | Local filesystem | S3, Azure Blob, NFS |
| Deployment | Docker Compose | Kubernetes, ECS, Cloud Run |
| Monitoring | Basic logging | Prometheus + Grafana, OpenTelemetry |
| Backup | Manual / script | Automated with retention policies |

### Integration Points

- **API layer** (`api/main.py`) — extend for enterprise integrations
- **embedding_service.py** — swap embedding backend
- **config.py** — add enterprise config providers (Vault, SSM)
- **collection_manager.py** — replace JSON with database-backed collections

### Scaling Considerations

- ChromaDB handles ~100K documents well; beyond that consider Qdrant/Weaviate
- Embedding generation is the bottleneck — GPU recommended for >10K docs
- Knowledge graph grows linearly; NetworkX handles ~1M nodes
- Ollama runs one model at a time; for concurrent users, use vLLM or TGI

---

## 17. Version Management

All versions are centralised in `cortex_engine/version_config.py`:

```bash
# Check version consistency
python scripts/version_manager.py --check

# Sync versions across 50+ files
python scripts/version_manager.py --sync-all

# Update changelog
python scripts/version_manager.py --update-changelog
```

---

*Document generated 2026-01-30. Reflects Cortex Suite v5.8.0 architecture.*
