AGENTS.md

Purpose: This file guides agentic coding assistants working on Cortex Suite.

Project Overview
- Streamlit-based local AI knowledge workbench for ingesting documents, extracting entities/relationships, searching, and generating proposals.
- Knowledge store: ChromaDB vectors + NetworkX graph; external DB path provided by user (must work on Windows/WSL/macOS/Linux and Docker).
- Key flows: Research → Ingest → Search → Collections → Proposal.

Single Source of Truth (Versioning)
- Edit only `cortex_engine/version_config.py` for version bumps and metadata.
- After changes, update CHANGELOG and sync Docker copies of version_config when distributing.

Non‑Negotiables
- Paths: Never hardcode. Always take the user’s DB path and normalize via `convert_windows_to_wsl_path` unless inside Docker (`/.dockerenv` → use as‑is).
- External storage: All artifacts (Chroma DB, graph, collections, staging JSON) live under the user DB path, not the repo.
- Embeddings: Use `cortex_engine/embedding_service.py` everywhere for consistent vectors. In search, call Chroma with `query_embeddings=[...]`.
- Collections: Use `WorkingCollectionManager` so `working_collections.json` is stored next to the external DB.
- Search stability: Do not import LlamaIndex on the search path. Hybrid search must union vector + graph results and cap to top_k.

Key Modules
- Ingest: `cortex_engine/ingest_cortex.py` (staging + finalization + graph build)
- Async ingest: `cortex_engine/async_ingest.py` (adds embeddings explicitly)
- Search UI: `pages/3_Knowledge_Search.py` (direct Chroma, GraphRAG wrapper)
- Graph: `cortex_engine/graphrag_integration.py`, `graph_query.py`
- Paths/utilities: `cortex_engine/utils/*`, `collection_manager.py`

Docker Distribution
- Keep Docker copies in `docker/` directory synchronized (especially `version_config.py`, `ingest_cortex.py`, `embedding_service.py`, wrapper for `3_Knowledge_Search.py`).
- In Docker, do not convert paths; rely on volume mapping.

Common Tasks
- Bump version: update `cortex_engine/version_config.py` and (if needed) `docker/cortex_engine/version_config.py`.
- Update changelog: add a new section with features/improvements/bug fixes.
- Sync Docker: copy updated engine/page files or use the provided wrapper for pages.

Commit & Push
- Group related changes; write clear commit messages (feature/improvement/bugfix). Include brief release notes for version bumps.

Do’s and Don’ts
- Do maintain one embedding model across ingest and search.
- Do add explicit embeddings on `collection.add` in any alternate ingestion path.
- Don’t write collections or DB files into the repo root.
- Don’t depend on LlamaIndex in the search path.

