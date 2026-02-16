# Cortex + Portal Master Implementation Blueprint

**Date:** 2026-02-15  
**Scope:** Program Transparency Portal delivery + existing Cortex enhancement stream  
**Priority Rule:** Portal-critical path first, preserve momentum on existing enhancements in parallel.

## Runtime Mode Decision (2026-02-16)

For immediate delivery, run in **single-tenant mode** with tenancy-ready structure:

1. Fixed runtime scope:
   - `tenant_id = "default"`
   - `project_id = "default"`
2. Keep tenant/project fields in schemas, API contracts, and retrieval filters from day one.
3. Keep permission checks abstracted behind callable policy functions.
4. Defer dynamic tenant admin UI and hard DB-level tenant enforcement to later phase.

## 1. Delivery Lanes

1. `Lane A (Critical Path)`: Website <-> Cortex handoff contract, queue reliability, retrieval policy enforcement.
2. `Lane B (Critical Path)`: Granular anonymisation and governance controls.
3. `Lane C (Value Path)`: Claim/citation mapper, deck quality, structured extraction quality.
4. `Lane D (Ops Path)`: Observability, test harnesses, release safety.

## 2. Prioritized Sequence

## Phase 0 - Platform Contract and Safety (Now)

1. Freeze handoff contract:
   - API contract version
   - `trace_id` propagation
   - idempotency key support
   - queue lifecycle state mapping
2. Implement and test queue handoff between website and Cortex worker path.
3. Ship granular anonymiser controls:
   - entity-type toggles
   - pronoun redaction toggle
   - company-name redaction controls (custom names)
4. Add targeted tests for handoff and anonymisation options.

## Phase 1 - Golden Path MVP

1. Upload -> anonymise -> approve -> index -> chat with citations.
2. Dashboard v1 (milestones/risks/spend).
3. Sponsor deck generation with evidence appendix.

## Phase 2 - Hardening

1. Tenant isolation tests and retrieval gateway verification.
2. Operational dashboards (queue depth/failure/latency/citation validity).
3. Expanded personas and structured extraction correction UI.

## Phase 3 - Connectors and Automation

1. Jira/Azure DevOps sync.
2. SharePoint/OneDrive sync.
3. Weekly curated deck pack automation.

## 3. Work Breakdown (Immediate)

## Work Item A1 - Handoff Contract (start now)

1. Add integration handshake endpoint in Cortex API:
   - returns contract version and supported capabilities
2. Add handoff validation endpoint:
   - validates `trace_id`, `idempotency_key`, job payload structure
3. Update worker to preserve trace/idempotency context in logs and completion metadata.
4. Add contract smoke-test script for local verification.

## Work Item B1 - Granular Anonymiser (start now)

1. Introduce anonymisation options model:
   - redact people/org/project/location
   - redact emails/phones/urls/headers
   - redact personal pronouns
   - redact custom company names
2. Wire options through:
   - `DocumentAnonymizer.anonymize_single_file`
   - queue handler input_data for `pdf_anonymise`
   - Document Extract UI controls
3. Add deterministic tests for:
   - pronoun redaction on/off
   - organization redaction toggles
   - custom company names redaction

## Work Item D1 - Validation Harness

1. Add focused pytest targets for new features.
2. Add queue runbook acceptance checks:
   - success path
   - forced failure
   - stale-claim recovery

## 4. Definition of Done (Current Sprint)

1. Website system can call Cortex handoff endpoints and validate payload contracts.
2. Worker processes queue jobs with preserved trace/idempotency metadata.
3. Anonymiser supports granular toggles and custom company-name masking.
4. UI exposes the new anonymiser controls.
5. Tests pass for new contract and anonymiser behaviors.

## 5. Parallel Track (Do Not Lose)

While critical path work is underway, retain backlog sequencing:

1. Claim/citation mapper enhancements.
2. Textifier polish and markdown quality refinements.
3. 30B local generation routing improvements (generation only, not embeddings).
4. Docling robustness and fallback validation.

## 6. Execution Start

Execution begins with:

1. Handoff contract code changes in `api/main.py` + worker trace/idempotency propagation.
2. Granular anonymiser options in `cortex_engine/anonymizer.py`, `worker/handlers/pdf_anonymise.py`, and `pages/7_Document_Extract.py`.
3. Targeted tests in `tests/unit/`.
