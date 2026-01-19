# Workspace Model Guide
**Version:** 2.0.0
**Date:** 2026-01-03
**Status:** Phase 1 Complete ✅

## Overview

The Workspace Model is a **per-tender working environment** that combines:
- **ChromaDB collection** for semantic search
- **JSON snapshots** of entity structured data
- **Progressive additions** (research, notes, custom narratives)
- **Hybrid data strategy** (structured + unstructured)

## Architecture

### What is a Workspace?

A workspace is created for EACH tender response. It's your **scratchpad** for that specific tender, containing:

1. **Tender document** (uploaded .docx file)
2. **Entity snapshot** (JSON copy of entity's structured data)
3. **Workspace collection** (ChromaDB collection with all relevant data)
4. **Field mappings** (Phase 2 - tender fields → data matches)
5. **User additions** (research notes, custom content, narratives)

### Directory Structure

```
/ai_databases/
├── workspaces/
│   ├── workspace_RFT12493_longboardfella_2026-01-03/
│   │   ├── metadata.json                 # Workspace metadata
│   │   ├── entity_snapshot.json          # Entity data snapshot
│   │   └── field_mappings.json           # Field matching results (Phase 2)
│   │
│   └── workspace_RFT99999_deakin_2026-01-05/
│       ├── metadata.json
│       ├── entity_snapshot.json
│       └── field_mappings.json
│
└── knowledge_hub_db/                      # ChromaDB collections
    ├── knowledge_hub_collection/          # Main KB
    ├── workspace_rft12493_longboardfella_2026_01_03/  # Workspace collection
    └── workspace_rft99999_deakin_2026_01_05/          # Another workspace

```

## Workflow

### Step 1: Create Workspace

**When:** User uploads tender document (RFT12493.docx)

**What Happens:**
1. Extract tender ID from filename (e.g., "RFT12493")
2. Create workspace with ID: `workspace_RFT12493_{entity}_2026-01-03`
3. Create ChromaDB collection: `workspace_rft12493_longboardfella_2026_01_03`
4. Save metadata.json

**Result:**
- Empty workspace created
- Status: `CREATED`

### Step 2: Link Entity

**When:** User selects entity profile (e.g., "longboardfella consulting")

**What Happens:**
1. Update workspace metadata with entity_id and entity_name
2. Link to entity's source documents

**Result:**
- Workspace linked to entity
- Ready for extraction

### Step 3: Extract & Populate

**When:** User clicks "Extract Structured Data"

**What Happens:**
1. Extract structured data from entity's source documents
2. **Save entity snapshot** to `entity_snapshot.json` (for structured queries)
3. **Populate workspace collection** with searchable text:
   - Organization profile
   - Insurance policies
   - Qualifications
   - Work experience
   - Projects
   - References
   - Capabilities
4. Update workspace status to `IN_PROGRESS`

**Result:**
- Workspace contains both:
  - JSON snapshot (for exact data lookups)
  - ChromaDB documents (for semantic search)
- Document count updated

### Step 4: Progressive Additions (Optional)

**User can add:**
- Research notes about the tender
- Competitor analysis
- Custom narratives
- Meeting notes
- Additional documents

**How:**
```python
workspace_manager.add_document_to_workspace(
    workspace_id="workspace_RFT12493_longboardfella_2026-01-03",
    content="Research note: DHA prioritizes Indigenous engagement",
    source_type=DocumentSource.USER_RESEARCH,
    metadata={"topic": "client priorities"}
)
```

### Step 5: Search Workspace

**Query workspace collection for relevant data:**

```python
results = workspace_manager.search_workspace(
    workspace_id="workspace_RFT12493_longboardfella_2026-01-03",
    query="insurance policy professional indemnity",
    top_k=5,
    source_type_filter=DocumentSource.ENTITY_DATA  # Optional filter
)
```

**Returns:** Top 5 relevant documents from workspace collection

## Data Models

### WorkspaceMetadata

```python
{
    "workspace_id": "workspace_RFT12493_longboardfella_2026-01-03",
    "workspace_name": "RFT12493 - longboardfella consulting",
    "tender_id": "RFT12493",
    "tender_filename": "RFT12493-Request-for-Tender.docx",
    "entity_id": "longboardfella_consulting",
    "entity_name": "longboardfella consulting pty ltd",
    "status": "IN_PROGRESS",
    "collection_name": "workspace_rft12493_longboardfella_2026_01_03",
    "document_count": 47,
    "field_count": 0,
    "matched_field_count": 0,
    "created_date": "2026-01-03T10:30:00",
    "last_modified": "2026-01-03T11:15:00"
}
```

### WorkspaceStatus

- **CREATED**: Workspace created, no data yet
- **IN_PROGRESS**: Extraction complete, user working
- **FIELD_MATCHING**: Phase 2 - matching tender fields
- **READY_TO_FILL**: All fields matched, ready to fill
- **COMPLETED**: Tender filled and exported
- **ARCHIVED**: Completed, no longer active

### DocumentSource

- **TENDER_DOCUMENT**: Chunks from uploaded tender
- **ENTITY_DATA**: Extracted structured data
- **USER_RESEARCH**: User-added research
- **USER_NOTES**: User's working notes
- **CUSTOM_NARRATIVE**: Drafted custom content
- **ADDITIONAL_UPLOAD**: Additional files uploaded

## API Reference

### WorkspaceManager

#### Create Workspace
```python
workspace = workspace_manager.create_workspace(
    tender_id="RFT12493",
    tender_filename="RFT12493-Request-for-Tender.docx",
    entity_id="longboardfella_consulting",
    entity_name="longboardfella consulting pty ltd"
)
```

#### List Workspaces
```python
workspaces = workspace_manager.list_workspaces(include_archived=False)
# Returns: List[WorkspaceMetadata]
```

#### Get Workspace
```python
workspace = workspace_manager.get_workspace("workspace_RFT12493_longboardfella_2026-01-03")
```

#### Add Document
```python
workspace_manager.add_document_to_workspace(
    workspace_id="workspace_RFT12493_longboardfella_2026-01-03",
    content="Document text content",
    source_type=DocumentSource.USER_RESEARCH,
    metadata={"topic": "competitor analysis"}
)
```

#### Search Workspace
```python
results = workspace_manager.search_workspace(
    workspace_id="workspace_RFT12493_longboardfella_2026-01-03",
    query="project health department",
    top_k=5
)
```

#### Archive Workspace
```python
workspace_manager.archive_workspace("workspace_RFT12493_longboardfella_2026-01-03")
```

#### Delete Workspace
```python
workspace_manager.delete_workspace("workspace_RFT12493_longboardfella_2026-01-03")
# Deletes collection + workspace directory
```

#### Cleanup Old Workspaces
```python
deleted_count = workspace_manager.cleanup_old_workspaces(age_days=180)
# Deletes archived workspaces older than 180 days
```

## Benefits

### 1. Progressive Workflow ✅

**Before (Entity-only):**
- Extract once from entity → 30 documents
- Fill tender
- No ability to add tender-specific context

**After (Workspace):**
- Extract from entity → 30 documents
- Add research notes → 5 documents
- Add custom narratives → 3 documents
- Add competitor analysis → 2 documents
- **Total: 40 documents** available for filling tender

### 2. Hybrid Data Strategy ✅

**Structured queries (JSON snapshot):**
```python
# Get exact ABN from snapshot
entity_data = workspace_manager.get_entity_snapshot(workspace_id)
abn = entity_data['organization']['abn']
```

**Semantic queries (Collection):**
```python
# Find relevant projects for this tender
results = workspace_manager.search_workspace(
    workspace_id,
    query="health department project case study"
)
```

### 3. Isolation & Organization ✅

- Each tender has its own collection
- No cross-contamination between tenders
- Easy to archive/delete old tenders
- Clear workspace lifecycle

### 4. Reusability ✅

- Entity data extracted once, reused across multiple tenders
- Workspace-specific additions stay isolated
- Next tender for same entity: instant entity snapshot + new workspace

## Use Cases

### Use Case 1: Quick Tender Response

**Scenario:** Respond to RFT12493 using existing "longboardfella" entity

**Steps:**
1. Upload tender → Create workspace
2. Select "longboardfella" entity → Link workspace
3. Extract data → Populate workspace (2-3 min from 30 docs)
4. Fill tender (Phase 2/3)

**Time:** ~15 minutes (vs 2-3 hours manual)

### Use Case 2: Tender with Research

**Scenario:** Respond to complex RFT requiring client research

**Steps:**
1. Create workspace for RFT99999
2. Extract from "longboardfella" entity
3. **Add research**: "Client prioritizes Indigenous engagement"
4. **Add competitor analysis**: Upload competitor's winning tender
5. **Add custom narrative**: "Why we're perfect for this client"
6. Search workspace for tender filling (includes ALL context)

**Benefit:** All context in one searchable workspace

### Use Case 3: Multiple Tenders for Same Entity

**Scenario:** 5 tenders in January for "longboardfella"

**Old Way:**
- Extract from entity 5 times (5 × 2-3 min = 10-15 min)
- Data duplicated across tenders

**New Way:**
- Extract from entity once → Used for all 5 workspaces
- Each workspace gets entity snapshot + tender-specific additions
- **Time saved:** 10 minutes

## Lifecycle Management

### Active Workspaces

**Status:** CREATED, IN_PROGRESS, FIELD_MATCHING, READY_TO_FILL

**Actions:**
- User actively working
- Can add documents
- Can search
- Can modify

### Completed Workspaces

**Status:** COMPLETED

**Actions:**
- Tender filled and exported
- Keep for reference
- Archive after 30 days

### Archived Workspaces

**Status:** ARCHIVED

**Actions:**
- Read-only
- Cleanup after 180 days

### Cleanup Strategy

```python
# Automatic cleanup (run monthly)
workspace_manager.cleanup_old_workspaces(age_days=180)
```

**Recommendation:**
- Archive completed workspaces after 30 days
- Delete archived workspaces after 180 days
- Keep workspace metadata even after collection deleted (for audit trail)

## Phase 2: Field Matching (Coming Soon)

Workspaces will support field mapping:

```python
workspace_manager.save_field_mappings(
    workspace_id,
    field_mappings=[
        FieldMapping(
            field_id="abn_field",
            field_location="Table 3, Row 5",
            field_description="ABN",
            matched_data="12 345 678 901",
            data_source="entity_snapshot:organization.abn",
            confidence=0.95,
            user_approved=True
        )
    ]
)
```

## Phase 3: Export (Coming Soon)

Workspaces will track export history:

```python
workspace.export_history = [
    {
        "export_date": "2026-01-03T15:30:00",
        "export_format": "docx",
        "export_filename": "RFT12493-Response-FINAL.docx",
        "fields_filled": 47
    }
]
```

## Troubleshooting

### Workspace Not Creating

**Error:** "Failed to create ChromaDB collection"

**Cause:** Collection name already exists

**Fix:**
- Check existing collections: `chroma_client.list_collections()`
- Delete old collection: `chroma_client.delete_collection(name)`
- Or use different tender ID

### Workspace Search Returns Empty

**Error:** No results from `search_workspace()`

**Cause:** Workspace not populated yet

**Fix:**
- Check `workspace.document_count` > 0
- Run extraction: "Extract Structured Data"
- Verify collection exists: `workspace.collection_name`

### Workspace Cleanup Failed

**Error:** "Failed to delete workspace"

**Cause:** Collection in use or permission issues

**Fix:**
- Close all ChromaDB clients
- Check file permissions on workspace directory
- Manually delete: `workspace_manager.delete_workspace(workspace_id)`

## Best Practices

1. **One workspace per tender** - Don't reuse workspaces
2. **Archive when done** - Move to ARCHIVED status after export
3. **Add progressively** - Add research/notes as you work
4. **Search before filling** - Use semantic search to find best data
5. **Clean up regularly** - Run cleanup monthly
6. **Descriptive names** - Use clear tender IDs in filenames

## Summary

The Workspace Model provides:

✅ **Per-tender isolation** - Each tender has its own working environment
✅ **Progressive workflow** - Add context as you work
✅ **Hybrid data** - Structured JSON + semantic search
✅ **Reusable entities** - Extract once, use many times
✅ **Clear lifecycle** - Created → In Progress → Completed → Archived
✅ **Manageable** - Easy cleanup and organization

**Status:** Phase 1 Complete (Workspace creation, extraction, population)
**Next:** Phase 2 (Field matching) → Phase 3 (Export)
