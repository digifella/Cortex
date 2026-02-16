# Repo Hygiene Pass (2026-02-16)

## Scope

Second-pass structured classification of current dirty workspace into:
1. Ready-to-commit bucket
2. Local-only bucket
3. In-progress bucket

## Validation Performed

Ran unit tests for newly added untracked test files:

```bash
venv/bin/pytest -q \
  tests/unit/test_api_security.py \
  tests/unit/test_batch_manager_cleanup.py \
  tests/unit/test_collection_manager.py \
  tests/unit/test_handoff_and_anonymizer_options.py \
  tests/unit/test_ingest_finalize.py \
  tests/unit/test_ingestion_recovery.py \
  tests/unit/test_persistent_cache_security.py
```

Result: `23 passed`.

## Buckets

## 1) Ready-to-Commit (Validated)

- `tests/unit/test_api_security.py`
- `tests/unit/test_batch_manager_cleanup.py`
- `tests/unit/test_collection_manager.py`
- `tests/unit/test_handoff_and_anonymizer_options.py`
- `tests/unit/test_ingest_finalize.py`
- `tests/unit/test_ingestion_recovery.py`
- `tests/unit/test_persistent_cache_security.py`
- `docs/IMPLEMENTATION_BLUEPRINT_MASTER.md`
- `docs/PROGRAM_TRANSPARENCY_PORTAL_TECHNICAL_BLUEPRINT_V2.md`

Rationale:
- Tests compile and pass.
- Blueprint docs are implementation-facing and align with current direction.

## 2) Local-Only (Excluded from Status Locally)

Added to `.git/info/exclude` (local-only, not committed):
- `PinPoint - Proposal.pdf`
- `Reframed prompt.pdf`
- `Taylor - Recruitment Proposal Feb 2026.pdf`
- `scimagojr 2024 (1).xlsx`
- `requirements_backup_20260124.txt`
- `journal_quality_rankings_scimagojr_2024.json`

Rationale:
- Large/binary/ad-hoc artifacts, not stable source files.

## 3) In-Progress (Leave Untouched for now)

Tracked modified files still present from broader active streams, e.g.:
- `api/main.py`
- `cortex_engine/anonymizer.py`
- `cortex_engine/config.py`
- `cortex_engine/docling_reader.py`
- `cortex_engine/utils/persistent_cache.py`
- `cortex_engine/utils/smart_model_selector.py`
- Docker mirrors of the above
- `pages/1_AI_Assisted_Research.py`
- `pages/components/_Maintenance_ResetRecovery.py`
- `requirements-docling.txt`
- `tests/unit/test_path_utils.py`

Rationale:
- These represent ongoing work that should be grouped by feature-specific commits.

## Recommended Next Hygiene Step

1. Commit the validated `Ready-to-Commit` bucket in one focused commit.
2. Then split the `In-Progress` tracked modifications into themed commits:
   - Docling/runtime deps
   - API/security/maintenance
   - UI/UX pages and docker mirrors
