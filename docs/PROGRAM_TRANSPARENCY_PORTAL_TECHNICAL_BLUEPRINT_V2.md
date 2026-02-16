# Program Transparency Portal - Technical Blueprint (v2.2)

**Author:** Cortex engineering review  
**Date:** 2026-02-16  
**Input Reviewed:** `/home/longboardfella/longboardfella_website/PROGRAM_TRANSPARENCY_PORTAL_PRD.md` (v1.1, 2026-02-15)

## 0.2 Implementation Sync (2026-02-16)

The following items are now implemented in Cortex and should be treated as current integration behavior for website work:

1. Queue worker supports job types:
   - `pdf_anonymise`
   - `pdf_textify`
   - `url_ingest`
2. Queue telemetry and operator controls are live:
   - Streamlit page: `pages/16_Queue_Monitor.py`
   - Local telemetry store: `cortex_engine/queue_monitor.py`
   - Operator controls: local cancel flag, local clear, and website cancel/fail update path.
3. Textifier execution model is unified across UI/worker/URL ingestor:
   - `DocumentTextifier.from_options(...)` is the shared options contract.
4. PDF processing defaults to Docling-first hybrid with fallbacks:
   - Docling pass first
   - image/LLM enrichment stage
   - fallback behavior with timeouts
5. URL ingest now forwards advanced textify options from job `input_data`:
   - `pdf_strategy`
   - `cleanup_provider`
   - `cleanup_model`
   - `docling_timeout_seconds`
   - `image_description_timeout_seconds`
   - `image_enrich_max_seconds`

## 1. Decision Summary

The PRD is strong and executable. The recommended pathway is:

1. Keep the PRD scope and phased rollout.
2. Tighten platform foundations early (tenant isolation, retrieval policy enforcement, auditability, async jobs).
3. Reuse Cortex + website assets, but introduce a clear service boundary so this can scale to multi-client safely.

This is a **go-forward blueprint** you can hand to the website-maintaining system for implementation alignment.

## 1.1 Alignment Status vs PRD v1.1

Alignment is now complete on architecture, phases, governance model, and model strategy.

One implementation clarification is now explicit:

1. PRD target queue stack is `Celery + Redis`.
2. Current working implementation in Cortex is the polling queue harness in `worker/`.
3. Build to a queue interface so MVP can run on current harness while preserving migration to Celery/Redis without API churn.

## 2. What to Keep As-Is from the PRD

1. Vault/Workbench dual-store model.
2. Curator approval gate before searchable publication.
3. Permission-aware RAG with citations and "no source, no answer."
4. Persona-based deck generation as a top differentiator.
5. Hybrid local/external model strategy with sanitised-only outbound data.
6. Phased delivery (MVP -> harden -> connectors).

## 3. Warranted Changes to the Design

## 3.1 Multi-tenant enforcement: move from "best effort" to "provable"

**Adjustment:** Keep shared PostgreSQL for MVP, but enforce tenancy at three layers from day one:

1. `tenant_id` + `project_id` scoping in application query builder.
2. PostgreSQL Row-Level Security (RLS) policies on all tenant tables.
3. Retrieval pre-filter tests that fail CI if cross-tenant results can surface.

**Why:** Prevents subtle cross-client leakage as soon as more than one tenant exists.

## 3.2 Retrieval architecture: policy filter before vector lookup

**Adjustment:** Introduce a retrieval gateway module:

1. Build a candidate set by `tenant_id`, `project_id`, `allowed_roles`, date range.
2. Only then run vector similarity over permitted chunk IDs.
3. Return citations tied to immutable chunk IDs + content hash.

**Why:** The PRD intent is correct; this makes it implementation-enforceable and testable.

## 3.3 Queue-first processing for all heavy tasks

**Adjustment:** Treat ingestion/anonymisation/embedding/structured extraction/deck generation as asynchronous jobs from day one, with states:

`queued -> claimed -> processing -> completed | failed | dead_letter`

**Why:** Your existing local worker harness already fits this pattern and reduces UI blocking and operational fragility.

**Implementation contract:** define a queue adapter interface (`enqueue`, `claim`, `heartbeat`, `complete`, `fail`, `retry`, `dead_letter`) with two backends:

1. `polling_api_adapter` (current harness, immediate use)
2. `celery_redis_adapter` (PRD target for multi-worker hosted operation)

## 3.4 Storage strategy: short-term Chroma, defined migration criteria

**Adjustment:** Start with Chroma only if needed for speed, but define migration trigger to `pgvector`:

1. More than 5 active tenants, or
2. Need strict SQL-governed retrieval joins, or
3. Operations burden from dual stores becomes material.

**Why:** Chroma is fast to ship; pgvector simplifies governance and reporting at scale.

## 3.5 De-identification policy: default irreversible in Workbench

**Adjustment:** Keep reversible mapping in Vault for admin-only incidents, but default Workbench outputs to irreversible pseudonyms unless explicitly approved.

**Why:** Reduces re-identification risk surface and strengthens trust posture.

## 3.6 LLM contract and model use

**Adjustment:** Separate model classes by task:

1. Embeddings: keep dedicated embedding models only.
2. Local generation (Qwen 30B/Ollama or LM Studio): routine Q&A and extraction on sanitised data.
3. External generation: deck narratives and deep synthesis on sanitised data only.

**Why:** 30B improves generation quality, not embedding quality. This preserves consistency and cost control.

## 3.7 Evaluation harness is mandatory, not optional

**Adjustment:** Add automated eval suite before production onboarding:

1. Citation validity rate.
2. Unsupported-claim rate.
3. PII leakage checks on outbound payloads.
4. Tenant isolation penetration tests.
5. Deck factual grounding checks against source chunks.

**Why:** This is the quickest way to keep quality and compliance stable as features grow.

## 4. Revised Build Pathway

## Phase 0 (Step 0) - Foundation Hardening (1-2 weeks)

1. Finalise Docling environment and fallback chain in Cortex.
2. Define canonical chunk schema:
   - `chunk_id`, `artefact_id`, `tenant_id`, `project_id`, `effective_date`, `allowed_roles`, `content_hash`.
3. Implement retrieval gateway contract (pre-filter then search).
4. Stand up queue job lifecycle + retry/dead-letter behavior.
5. Add baseline observability:
   - structured logs, request/job IDs, audit event emitter.

## Phase 1 - MVP Delivery (6-8 weeks)

1. Upload -> Vault -> anonymise -> curator approve -> Workbench publish.
2. Permission-aware chat with citation rendering and refusal policy.
3. Dashboard v1 (milestones, risks, spend).
4. Sponsor deck generation with citation appendix.
5. Tenant and role admin basics.

## Phase 2 - Hardening + Multi-client (4-6 weeks)

1. RLS validation suite and cross-tenant security tests.
2. Additional personas (Quality, Finance/CRO).
3. LLM-based structured extraction with human correction UI.
4. Alerting and trend deltas.
5. Operational dashboards (queue depth, failure rates, ingest latency).

## Phase 3 - Automation + Connectors (4-8 weeks)

1. Jira/Azure DevOps and SharePoint connectors.
2. Weekly auto-pack workflow with curator approval.
3. Change-detection summaries ("what changed since last report").
4. Optional GraphRAG enablement for cross-artefact relationship insights.

## 5. Integration Split: Website vs Cortex Responsibilities

## Website system (`longboardfella_website`)

1. Portal UX, auth/session shell, admin and curator interfaces.
2. Citation rendering components and evidence-first UX.
3. Queue API/admin screens and client-facing interaction layer.

## Cortex system (`cortex_suite`)

1. Parsing, anonymisation pipeline, chunking, embeddings, retrieval engine.
2. Deck narrative drafting + pptx assembly service.
3. Structured extraction services and model orchestration.

## Contract between systems

1. Versioned HTTP API with explicit request/response schemas.
2. Idempotency keys for ingest and deck jobs.
3. Every response carries `trace_id` and citation payload schema.

## 6. Job Queue Test Runbook (Current System Reminder)

This matches the queue worker currently present in `cortex_suite/worker`.

## 6.1 Configure worker

1. Copy config:
```bash
cp worker/config.env.example worker/config.env
```
2. Edit `worker/config.env` with:
   - `QUEUE_SERVER_URL` (your live queue worker API endpoint)
   - `QUEUE_SECRET_KEY`
   - optional `WORKER_ID`, `POLL_INTERVAL_SECONDS`, `HEARTBEAT_INTERVAL_SECONDS`

## 6.2 Install dependencies

```bash
pip install requests pymupdf
```

## 6.3 Run worker

```bash
venv/bin/python worker/worker.py
```

## 6.4 Functional test sequence

1. Create a test queue job from your website admin queue UI/API (`pdf_anonymise`).
2. Confirm worker logs show:
   - poll success
   - job claim
   - input download
   - heartbeat messages
   - completion upload
3. Confirm job status transitions correctly in admin UI:
   - `pending -> claimed -> completed` (or `failed`)
4. Verify output artifact is available and import path works.
5. Kill worker mid-job once to confirm claim expiry/retry behavior.

## 6.5 Status mapping to PRD lifecycle terms

Current harness and PRD terms should be treated as equivalent:

1. `pending` ~= `queued`
2. `claimed` ~= `claimed`
3. in-handler execution ~= `processing`
4. `completed` ~= `completed`
5. `failed` ~= `failed`
6. queued with retry exhausted ~= `dead_letter` (implement explicit dead-letter bucket when moving to adapter abstraction)

## 6.6 Minimum acceptance criteria

1. 3/3 test jobs complete without manual file intervention.
2. One forced failure correctly lands in `failed` with error details.
3. One restart test proves recovery from claimed-stale job.
4. Audit trail records poll/claim/complete/fail events.

## 7. Immediate Next Build Actions

1. Freeze API contract between website and Cortex services.
2. Implement retrieval gateway + isolation tests before adding new feature surface.
3. Ship one end-to-end golden path with a real client sample set (20-30 artefacts).
4. Run queue runbook and baseline metrics (ingest latency, failure rate, citation validity).

---

This v2 blueprint keeps your PRD direction, but makes the system safer to scale and easier to operate under real client load.
