# Cortex Network Explorer Spec

**Version:** 1.0  
**Date:** 2026-03-16  
**Status:** Implementation spec for Cortex + website handoff

## Purpose

Build an interactive stakeholder-network explorer that lets a website user browse:

- canonical people
- canonical organisations
- alumni groups
- lab-member introduction paths
- direct and indirect linkages
- recent intelligence-bearing relationships

This should be a **combined system**:

- **Cortex** owns graph truth, relationship scoring, subgraph generation, and intelligence enrichment
- **Website** owns rendering, interaction, auth-aware presentation, and operator workflow

Do **not** duplicate relationship logic in the website.  
Do **not** build the interactive UI inside Cortex.

## Core Decision

### Best ownership split

#### Cortex responsibilities

- maintain the canonical stakeholder graph
- ingest and compound profile sync, affiliations, alumni, LinkedIn connections, and signals
- generate a scoped graph payload for a given org / watch set / focal stakeholder
- calculate direct and indirect linkage strength
- explain why a link exists
- attach confidence, recency, corroboration, and actionability metadata
- optionally emit sidebar-ready intelligence summaries for the selected subgraph

#### Website responsibilities

- request a graph view from Cortex
- render nodes, edges, labels, filters, and drilldowns
- handle pan/zoom/search/select/expand
- show side panels for node detail, edge detail, and path explanations
- enforce user/org visibility rules in the UI
- cache and display completed graph snapshots

## Current Cortex Reality

Cortex already has the right substrate:

- synced profiles and affiliations in the stakeholder signal store
- alumni and LinkedIn connection metadata
- graph writeback from profile sync and signal ingest
- graph-backed relationship intelligence in `signal_digest`

Current graph node families already present or derivable:

- `person`
- `entity` / tracked organisation
- `organization`
- `alumni_group`
- `lab_member`
- `subscriber_org`
- `stakeholder_signal`
- `source`

Current graph edge families already present:

- `tracked_by`
- `works_at`
- `affiliated_with`
- `alumni_of`
- `linkedin_connection`
- `mentions_profile`
- `mentions_organization`
- `published_by`
- `co_mentioned`
- `org_alumni_context`

This means the missing piece is **graph view generation**, not graph existence.

## Target Outcome

The website should be able to request a graph snapshot like:

- “show Longboardfella’s watched network”
- “show the ego network for Karin Verspoor”
- “show all direct and indirect links around Healthdirect Australia”
- “show warm-introduction paths from our members to watched stakeholders”
- “show alumni clusters and cross-target overlaps”

and receive a structured JSON payload suitable for Cytoscape.js / Sigma.js / similar.

## Recommended Cortex Job Type

Add a new queue-backed Cortex job type:

- `stakeholder_graph_view`

This is preferable to trying to overload `signal_digest`, because the output contract is a graph payload, not markdown.

## Input Contract

Add `validate_stakeholder_graph_view_input()` in [handoff_contract.py](/home/longboardfella/cortex_suite/cortex_engine/handoff_contract.py).

### Required fields

```json
{
  "org_name": "Longboardfella",
  "view_mode": "watch_network"
}
```

### Supported fields

```json
{
  "org_name": "Longboardfella",
  "view_mode": "watch_network|ego|org_focus|warm_intro|alumni_cluster|cross_target",
  "profile_keys": ["..."],
  "focus_profile_key": "...",
  "focus_org_name": "Healthdirect Australia",
  "since_ts": "2026-03-01T00:00:00Z",
  "max_hops": 2,
  "max_nodes": 120,
  "max_edges": 240,
  "include_signals": true,
  "include_sources": false,
  "include_lab_members": true,
  "include_alumni": true,
  "include_unwatched_bridges": false,
  "edge_types": ["works_at", "alumni_of", "linkedin_connection", "co_mentioned"],
  "min_edge_weight": 0.2,
  "min_confidence": 5.0,
  "layout_hint": "force|concentric|cose|breadthfirst",
  "top_k_paths": 5
}
```

### Validation rules

- `org_name` required
- `view_mode` defaults to `watch_network`
- `max_hops` default `2`, cap at `4`
- `max_nodes` default `100`, cap at `250`
- `max_edges` default `200`, cap at `600`
- `focus_profile_key` required for `ego`
- `focus_org_name` required for `org_focus`
- unknown `edge_types` should fail validation
- missing alumni/signals should degrade gracefully

## Output Contract

The primary output artifact should be a JSON file, not markdown.

### Queue `output_data`

```json
{
  "graph_id": "graph_abc123",
  "org_name": "Longboardfella",
  "view_mode": "watch_network",
  "generated_at": "2026-03-16T10:00:00Z",
  "node_count": 84,
  "edge_count": 132,
  "focus_node_ids": ["stakeholder_person:..."],
  "output_filename": "graph_abc123.json",
  "has_paths": true,
  "has_signal_overlay": true
}
```

### JSON file shape

```json
{
  "graph_id": "graph_abc123",
  "org_name": "Longboardfella",
  "view_mode": "watch_network",
  "generated_at": "2026-03-16T10:00:00Z",
  "filters": {
    "since_ts": "2026-03-01T00:00:00Z",
    "max_hops": 2,
    "include_signals": true
  },
  "summary": {
    "watched_people": 18,
    "watched_orgs": 7,
    "alumni_groups": 5,
    "warm_intro_paths": 4,
    "cross_target_links": 11
  },
  "nodes": [],
  "edges": [],
  "paths": [],
  "insights": [],
  "legend": {
    "node_types": {},
    "edge_types": {}
  }
}
```

## Node Schema

Each node should contain enough metadata for immediate rendering and sidebar detail.

```json
{
  "id": "stakeholder_person:abcd1234",
  "type": "person",
  "label": "Karin Verspoor",
  "subtitle": "RMIT University",
  "org_name": "Longboardfella",
  "profile_key": "abcd1234",
  "watch_status": "watch",
  "is_focus": false,
  "is_watched": true,
  "importance_score": 8.6,
  "confidence_score": 9.2,
  "signal_count": 4,
  "recent_signal_count": 3,
  "tags": ["watched", "stakeholder", "high-confidence"],
  "meta": {
    "target_type": "person",
    "current_employer": "RMIT University",
    "current_role": "Executive Dean",
    "alumni": ["University of Melbourne"],
    "known_employers": ["RMIT University", "Uni X"]
  }
}
```

### Expected node types

- `person`
- `tracked_org`
- `organization`
- `alumni_group`
- `lab_member`
- `subscriber_org`
- `stakeholder_signal`
- `source`

### Node rendering guidance for website

- `person`: circle
- `tracked_org` / `organization`: rounded rectangle
- `alumni_group`: hexagon or pill
- `lab_member`: small highlighted circle
- `stakeholder_signal`: small transient dot, hidden by default unless signal overlay is enabled

## Edge Schema

Each edge must explain the relationship and provide a usable weight.

```json
{
  "id": "edge_001",
  "source": "stakeholder_person:abcd1234",
  "target": "stakeholder_organization:rmit",
  "type": "works_at",
  "label": "works at",
  "weight": 0.92,
  "confidence_score": 9.4,
  "recency_score": 8.7,
  "actionability_score": 7.1,
  "evidence_count": 4,
  "is_direct": true,
  "is_inferred": false,
  "path_role": "employment",
  "meta": {
    "role": "Executive Dean",
    "affiliation_type": "current",
    "source": "stakeholder_profile_sync"
  }
}
```

### Expected edge types

- `works_at`
- `affiliated_with`
- `alumni_of`
- `linkedin_connection`
- `tracked_by`
- `mentions_profile`
- `mentions_organization`
- `published_by`
- `co_mentioned`
- `org_alumni_context`
- derived display-only types:
  - `warm_intro_path`
  - `shared_alumni_bridge`
  - `shared_org_bridge`

## Path Schema

For indirect linkage browsing, Cortex should emit explicit path objects.

```json
{
  "id": "path_01",
  "type": "warm_intro",
  "source_node_id": "stakeholder_member:paul@example.com",
  "target_node_id": "stakeholder_person:abcd1234",
  "hop_count": 2,
  "strength": 0.78,
  "explanation": "1st-degree LinkedIn path via Paul Cooper",
  "node_ids": [
    "stakeholder_member:paul@example.com",
    "stakeholder_person:abcd1234"
  ],
  "edge_ids": ["edge_101"]
}
```

### Path types

- `warm_intro`
- `shared_alumni`
- `shared_org_context`
- `cross_target`
- `signal_bridge`

## Insight Schema

Cortex should also emit sidebar-ready intelligence insights derived from the subgraph.

```json
{
  "id": "insight_01",
  "type": "talent_flow",
  "severity": "medium",
  "score": 7.6,
  "title": "Healthdirect shows consulting-talent inflow",
  "summary": "Multiple watched profiles linked to consulting backgrounds now intersect Healthdirect-related signals.",
  "node_ids": ["...", "..."],
  "edge_ids": ["...", "..."],
  "source_signal_ids": ["sig_1", "sig_2"]
}
```

### Recommended insight types

- `warm_intro`
- `alumni_cluster`
- `cross_target_overlap`
- `talent_flow`
- `network_proximity`
- `signal_cluster`
- `recency_spike`

## Cortex-Side Implementation

### 1. Add new job validation

File:
- [handoff_contract.py](/home/longboardfella/cortex_suite/cortex_engine/handoff_contract.py)

Add:
- `validate_stakeholder_graph_view_input()`

### 2. Add new worker handler

File to add:
- `/home/longboardfella/cortex_suite/worker/handlers/stakeholder_graph_view.py`

Handler responsibilities:

- validate input
- call a new graph-view builder in `StakeholderSignalStore`
- write graph JSON under the external DB root, e.g.:
  - `stakeholder_intel/graph_views/{graph_id}.json`
- complete queue job with metadata and uploaded JSON file

### 3. Add graph snapshot builder to stakeholder store

File:
- [stakeholder_signal_store.py](/home/longboardfella/cortex_suite/cortex_engine/stakeholder_signal_store.py)

Add methods such as:

- `build_graph_view(...)`
- `_collect_focus_profiles(...)`
- `_collect_focus_orgs(...)`
- `_build_watch_network_subgraph(...)`
- `_score_graph_nodes(...)`
- `_score_graph_edges(...)`
- `_derive_graph_paths(...)`
- `_derive_graph_insights(...)`

### 4. Use current graph plus store metadata

The subgraph should be built from both:

- graph edges/nodes already persisted in `knowledge_cortex.gpickle`
- state metadata from stakeholder signal store

This matters because:

- graph knows connectivity
- signal store knows recency, confidence, actionability, and current reporting context

### 5. Derived scoring rules

Recommended formulas:

#### Node importance

Blend:

- watched status
- number of high-confidence signals
- centrality in the scoped subgraph
- whether the node sits on a warm-intro path
- whether the node is a bridge between clusters

#### Edge weight

Blend:

- relationship type base score
- confidence score
- recency score
- corroboration count
- whether the edge was derived from multiple signals

Example base weights:

- `linkedin_connection`: `0.95`
- `works_at`: `0.9`
- `alumni_of`: `0.75`
- `co_mentioned`: `0.55`
- `mentions_profile`: `0.5`
- `published_by`: `0.35`

### 6. View modes

#### `watch_network`

Default organisation-wide watched network:

- watched people and orgs
- their direct employment/alumni/member links
- major cross-target bridges

#### `ego`

Focused graph around one stakeholder:

- focal person/org
- 1-hop and 2-hop neighbors
- warm intro paths
- recent signals touching the focal node

#### `org_focus`

Focused graph around one organisation:

- org node
- watched people connected to it
- feeder orgs
- alumni bridges
- recent signal overlay

#### `warm_intro`

Member-to-target path explorer:

- lab members
- reachable watched targets
- shortest viable intro paths

#### `alumni_cluster`

Shared alumni/employer cluster map:

- alumni groups
- watched people
- subscriber org context

#### `cross_target`

Connections among watched targets:

- co-mentions
- shared alumni
- shared prior employers
- shared source clusters

## Website-Side Requirements

Pass this to the website agent.

### Website should queue `stakeholder_graph_view`

Recommended UI actions:

- `Open Network Explorer`
- `View Stakeholder Network`
- `Warm Intro Map`
- `Organisation Influence Map`

### Website should render returned JSON locally

Recommended library:

- `Cytoscape.js`

Why:

- strong graph interaction support
- filtering and styling are straightforward
- handles medium-sized graphs well

### Website should support

- pan / zoom / fit
- node search
- edge-type filters
- confidence threshold slider
- recency window filter
- direct / indirect toggle
- expand one more hop
- select node -> right-hand detail drawer
- select edge -> “why this link exists” drawer
- select path -> path explanation drawer

### Website should show two layers

#### Graph layer

- nodes and edges
- visual strength by weight / confidence

#### Intelligence layer

- insight pills / side panel
- recent signal badges
- trend highlights

### Website should keep signal nodes hidden by default

Reason:

- raw signal nodes can overwhelm the main graph
- better as an optional overlay or detail layer

## Security / Privacy Rules

### Cortex output must be org-scoped

- only include nodes reachable from the requested org/watch context
- do not dump the full graph

### Member identity handling

For `lab_member` nodes:

- website may display friendly name if safe
- if not available, display masked email or operator-approved label

### Sensitive pathing

Do not include:

- internal filesystem paths
- raw mailbox storage paths
- worker-local paths

## Performance Rules

### Cortex

- cap scoped graph size aggressively
- default max `100` nodes / `200` edges
- pre-trim signal overlays
- return focus-first subgraphs, not entire tenant graphs

### Website

- render initial graph from scoped payload only
- lazy-expand when user asks for more hops
- cache the latest completed graph snapshot per org/view mode

## Acceptance Criteria

A Longboardfella website user should be able to:

- open a network view for their organisation
- see watched people and organisations as canonical nodes
- see direct and indirect links clearly
- identify warm introduction paths from members to targets
- see alumni and cross-target clusters
- click an edge and understand why it exists
- click a node and see recent signal context
- filter by confidence and relationship type
- switch between watch-network, ego, org-focus, and warm-intro views

## Recommended File Targets

### Cortex

- [handoff_contract.py](/home/longboardfella/cortex_suite/cortex_engine/handoff_contract.py)
- [stakeholder_signal_store.py](/home/longboardfella/cortex_suite/cortex_engine/stakeholder_signal_store.py)
- `/home/longboardfella/cortex_suite/worker/handlers/stakeholder_graph_view.py`
- optional tests:
  - `/home/longboardfella/cortex_suite/tests/unit/test_stakeholder_signal_store.py`
  - `/home/longboardfella/cortex_suite/tests/unit/test_handoff_contract_validation.py`

### Website

Likely files:

- Market Radar API where queue jobs are created
- Market Radar UI where graph explorer is launched
- a dedicated JS graph component / modal

## Summary For Website Agent

Do not build relationship logic in the website.

Ask Cortex for a scoped graph snapshot via `stakeholder_graph_view`, then render it interactively. Cortex will provide the graph truth, edge explanations, indirect paths, and network intelligence metadata. The website should focus on visual exploration, filtering, drilldown, and user workflow.
