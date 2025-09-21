Title: UX + Docker harmonization; ingest/finalize diagnostics; search fallback; maintenance clarity

Summary
- Unifies embeddings across ingest/finalize via adapter (fixes finalize crash).
- Adds staging diagnostics and Retry Finalization to host + Docker ingest pages.
- Adds Docker Knowledge Search fallback loader + minimal search implementation.
- Clarifies Clean Start (host + Docker) with pre/post verification of key files.
- Fixes indentation bug in Collection Management; adds diagnostics + sync helper.

Known Docker Issues (not blockers)
- Occasional .docx reader warnings about `file_path` arg; analysis still completes. Plan: normalize to `load_data(file=...)` everywhere.
- Rare staging UI edge case in container; diagnostics + Retry Finalization available.

Files of Interest
- Host ingest: pages/2_Knowledge_Ingest.py (diagnostics + retry).
- Docker ingest: docker/pages/2_Knowledge_Ingest.py (diagnostics + retry).
- Docker search loader + impl: docker/pages/3_Knowledge_Search.py, docker/pages/3_Knowledge_Search_impl.py.
- Embedding adapters: cortex_engine/embedding_adapters.py, docker/cortex_engine/embedding_adapters.py.
- Maintenance clarity: pages/7_Maintenance.py, docker/pages/7_Maintenance.py.
- Collections: pages/4_Collection_Management.py (diagnostics + sync), fix indentation bug.

Test Notes
- Host: Clean Start → small ingest (2–3 files) → watch auto-finalize; verify vector doc count + collections.
- Docker: Rebuild image; run 4‑doc ingest; if “no staged docs”, see diagnostics and use Retry Finalization; verify search fallback works.

