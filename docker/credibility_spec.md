# Credibility Classification Spec

Version: 1.1
Status: Active

## Canonical Fields
- `credibility_tier_value` (integer `0..5`)
- `credibility_tier_key` (`peer-reviewed|institutional|pre-print|editorial|commentary|unclassified`)
- `credibility_tier_label` (`Peer-Reviewed|Institutional|Pre-Print|Editorial|Commentary|Unclassified`)
- `credibility` (human-readable string)

## Source Integrity Fields
- `available_at` (canonical source URL if known)
- `availability_status` (`available|not_found|gone|client_error|server_error|unreachable|unknown`)
- `availability_http_code` (string HTTP code when available)
- `availability_checked_at` (date string `YYYY-MM-DD`)
- `availability_note` (human-readable source availability note)
- `source_integrity_flag` (`ok|deprecated_or_removed|unverified`)
- `source_integrity_risk` (`high|medium`) when scanner tags suspect items
- `source_integrity_last_scan_at` (date string `YYYY-MM-DD`) when scanner runs

## Canonical Tier Map
- `5 | peer-reviewed | Peer-Reviewed`
- `4 | institutional | Institutional`
- `3 | pre-print | Pre-Print`
- `2 | editorial | Editorial`
- `1 | commentary | Commentary`
- `0 | unclassified | Unclassified`

## Tier Examples
- Tier 5: NLM/PubMed, Nature, The Lancet, JAMA, BMJ
- Tier 4: WHO, UN/IPCC, OECD, World Bank, ABS, government departments, universities/institutes
- Tier 3: arXiv, SSRN, bioRxiv, ResearchGate
- Tier 2: Scientific American, The Conversation, HBR
- Tier 1: Blogs, newsletters, consulting reports, opinion
- Tier 0: Not yet assessed

## Human-Readable Field
- Format: `"{DocumentStage} {CredibilityTierLabel} Report"`
- `DocumentStage = Draft` when filename/content contains `draft`, otherwise `Final`
- Example: `Draft Institutional Report`
- Availability suffixes:
  - dead/removed links (`not_found|gone`): append `"(Source Link Unavailable)"`
  - uncertain links (`client_error|server_error|unreachable`): append `"(Source Link Unverified)"`

## Deterministic Policy
1. If `source_type == "AI Generated Report"` then force Tier 1 (`commentary`).
2. Else apply marker matching in priority order: `5 -> 4 -> 3 -> 2 -> 1`.
3. If no marker matched, keep valid model-provided tier if present; otherwise Tier 0.
4. If source availability is confirmed dead (`404`/`410`), reduce tier by 2 (floor at 0).
5. Do not tier-downgrade for temporary connectivity states (`client_error|server_error|unreachable`).
6. Always emit all canonical fields consistently.

## URL Availability Rules
- Preferred check flow: `HEAD`, then fallback to ranged `GET` (`bytes=0-0`) if required.
- `404`: classify as `not_found` and set note:
  - `"Previously available at: <url> but no longer available as at: <YYYY-MM-DD>."`
- `410`: classify as `gone` and set note:
  - `"Previously available at: <url> but has been removed (HTTP 410) as at: <YYYY-MM-DD>."`
- Missing/invalid URL: `availability_status = unknown`, `source_integrity_flag = unknown|unverified` depending on context.
- Flag mapping:
  - `not_found|gone` -> `source_integrity_flag = deprecated_or_removed`
  - `client_error|server_error|unreachable|unknown` -> `source_integrity_flag = unverified`
  - `available` -> `source_integrity_flag = ok`

## Search Inclusion Policy
- Search filter supports preserving valuable offline material:
  - `include_unavailable_sources = true` (default): include all results regardless of availability.
  - `include_unavailable_sources = false`: exclude `source_integrity_flag = deprecated_or_removed` or `availability_status in {not_found,gone}`.

## Batch Poisoning Mitigation (Maintenance)
- Source Integrity Scan and cleanup runs over selected collections.
- Supported modes:
  - scan only
  - metadata tagging
  - remove suspect docs from selected collections
  - tag + remove
- Weekly scheduled scan support stores:
  - `source_integrity/source_integrity_scan_<timestamp>.json`
  - `source_integrity/schedule_state.json`

## Marker Set
- Tier 5: `pubmed`, `nlm`, `nature`, `lancet`, `jama`, `bmj`, `peer-reviewed`, `peer reviewed`
- Tier 4: `who`, `un`, `ipcc`, `oecd`, `world bank`, `government`, `department`, `ministry`, `university`, `institute`, `centre`, `center`
- Tier 3: `arxiv`, `ssrn`, `biorxiv`, `researchgate`, `preprint`, `pre-print`
- Tier 2: `scientific american`, `the conversation`, `hbr`, `harvard business review`, `editorial`
- Tier 1: `blog`, `newsletter`, `opinion`, `consulting report`, `whitepaper`, `white paper`

## Legacy Alias Input Mapping
If present, map these aliases to canonical fields before normalization:
- `credibility_value` -> `credibility_tier_value`
- `credibility_source` -> `credibility_tier_key` (if value matches canonical keys)

## Parsing and Normalization
- If only one of value/key/label is present, infer the canonical triple.
- Invalid or missing values default to Tier 0.
- Normalize author names and metadata to ASCII-safe text for robust searching/parsing.
- Remove affiliation superscripts/index markers from author names.

## Keyword Rules (Document Preface)
- Target count: 5-12 keywords.
- Maximum 2 words per keyword.
- Remove error/noise tokens from failed image-processing paths.
