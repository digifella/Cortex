# Mailbox Entity Routing Spec

## Goal

Allow Cortex mailbox ingests to:

1. Route a submission to a different subscriber org when the email subject contains an explicit override such as `entity: Escient`.
2. Infer the document's primary organisation from a bare subject such as `Barwon Water`, especially for document-analysis uploads like org charts and strategic plans.

## Subject Conventions

### Explicit routing override

Accepted markers anywhere in the subject:

- `entity: Escient`
- `org: Escient`
- `organisation: Escient`
- `organization: Escient`

Examples:

- `entity: Escient | Barwon Water org chart`
- `Barwon Water strategic plan ; entity: Escient`

Behaviour:

- Cortex strips the routing directive from the cleaned subject before extraction and note-title generation.
- Cortex attempts to match the requested org to a known local org scope.
- If matched, Cortex uses that org as the effective `org_name` for extraction, signal ingest, and website note delivery.
- If not matched, Cortex falls back to the mailbox default org and marks the routing status as `unmatched_override`.

### Bare organisation subject hint

Examples:

- `Barwon Water`
- `Barwon Water org chart`
- `Barwon Water strategic plan`

Behaviour:

- For document attachments, Cortex derives a `subject_org_hint`.
- That hint is passed into extraction as `parsed_candidate_employer`.
- For document-analysis handoff, if no stronger organisation subject is found in the extracted document, Cortex promotes the subject org hint to the note `primary_entity`.

## Payload Additions Sent To Website

Structured note deliveries now include:

```json
"mailbox_routing": {
  "default_org_name": "Longboardfella",
  "requested_org_name": "Escient",
  "matched_org_name": "Escient",
  "effective_org_name": "Escient",
  "status": "matched_override",
  "subject_org_hint": "Barwon Water"
}
```

Notes:

- `effective_org_name` is the org Cortex actually used.
- `status` is one of:
  - `default`
  - `matched_override`
  - `unmatched_override`

## Website Expectations

### Immediate compatibility

The website can continue to trust the delivered top-level `org_name`. Cortex now sets it to the effective routed org when a local match exists.

### Recommended website enhancement

If `mailbox_routing.requested_org_name` is present but `status == "unmatched_override"`, the website should:

1. Attempt its own org lookup against subscriber organisations and aliases.
2. If it finds a match, use that org instead of sender-email fallback.
3. Persist both:
   - `requested_org_name`
   - `resolved_org_name`

### Recommended audit fields

For mailbox-origin notes, persist:

- `mailbox_default_org_name`
- `mailbox_requested_org_name`
- `mailbox_effective_org_name`
- `mailbox_routing_status`
- `mailbox_subject_org_hint`

## Current Cortex-side Behaviour

- Default mailbox org still comes from `INTEL_IMAP_ORG_NAME`.
- Explicit subject routing only succeeds when Cortex can locally match the requested org scope.
- Bare subjects improve primary-entity inference for document uploads even when routing remains on the default org.
