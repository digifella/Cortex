# Credibility Classification Spec

Version: 1.0
Status: Active

## Canonical Fields
- `credibility_tier_value` (integer `0..5`)
- `credibility_tier_key` (`peer-reviewed|institutional|pre-print|editorial|commentary|unclassified`)
- `credibility_tier_label` (`Peer-Reviewed|Institutional|Pre-Print|Editorial|Commentary|Unclassified`)
- `credibility` (human-readable string)

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

## Deterministic Policy
1. If `source_type == "AI Generated Report"` then force Tier 1 (`commentary`).
2. Else apply marker matching in priority order: `5 -> 4 -> 3 -> 2 -> 1`.
3. If no marker matched, keep valid model-provided tier if present; otherwise Tier 0.
4. Always emit all canonical fields consistently.

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
