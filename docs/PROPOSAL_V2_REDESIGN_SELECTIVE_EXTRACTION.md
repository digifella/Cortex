# Proposal System v2.1 - Selective Entity-Based Extraction

## 🎯 Design Issue Identified

### Current Design (Wrong)
- ❌ Extracts from **entire KB** (58,185 documents!)
- ❌ No entity selection (longboardfella vs Deakin vs Escient)
- ❌ No document navigation/filtering
- ❌ One extraction blob for everything
- ❌ Slow and impractical

### User's Real Needs
- ✅ **Multiple entities**: longboardfella consulting, Deakin University, Escient Pty Ltd
- ✅ **Organized KB**: Semi-structured folders with good organization
- ✅ **Selective extraction**: Choose specific folders/documents
- ✅ **Entity-specific profiles**: Separate structured data per entity
- ✅ **Navigation**: Browse and select what to extract

---

## 🏗️ Redesigned Architecture

### Layer 1: Selective Entity-Based Extraction

#### New Workflow

**Step 1: Create Entity Profile**
```
Entity Name: longboardfella consulting pty ltd
Entity Type: My Company
Description: Primary trading entity
```

**Step 2: Browse & Select Source Documents**
```
KB Navigation (58,185 docs organized):
└── 📁 longboardfella_consulting/
    ├── 📁 company_registration/
    │   ├── ✅ ABN_Certificate.pdf          [SELECT]
    │   └── ✅ ACN_Certificate.pdf          [SELECT]
    ├── 📁 insurance/
    │   ├── ✅ PL_Policy_2024.pdf          [SELECT]
    │   └── ✅ PI_Policy_2024.pdf          [SELECT]
    ├── 📁 team_cvs/
    │   ├── ✅ JaneSmith_CV.pdf            [SELECT]
    │   └── ✅ JohnDoe_CV.pdf              [SELECT]
    └── 📁 projects/
        ├── ✅ ADHA_CaseStudy.pdf          [SELECT]
        └── ✅ Services_Australia.pdf       [SELECT]

Total selected: 8 documents (vs 58,185 total)
```

**Step 3: Extract Structured Data**
```
Extracting for: longboardfella consulting pty ltd
From: 8 selected documents

📋 Organization profile... ✓
🛡️ Insurance policies... ✓
🎓 Qualifications... ✓
🚀 Projects... ✓

Saved: structured_data/longboardfella_consulting.json
```

**Step 4: Manage Multiple Entities**
```
Structured Data Repository:
├── longboardfella_consulting.json    (Last extracted: 2026-01-03)
├── deakin_university.json            (Last extracted: 2025-12-15)
└── escient_pty_ltd.json              (Last extracted: 2025-12-10)
```

---

## 📊 New Data Structure

### Entity Profile Schema

```python
class EntityProfile(BaseModel):
    """An organizational entity for tender responses."""

    entity_id: str                          # Unique ID (e.g., "longboardfella_consulting")
    entity_name: str                        # Display name
    entity_type: EntityType                 # MY_COMPANY, CLIENT, PARTNER
    description: Optional[str]              # User description

    # Source document selection
    source_document_ids: List[str]          # Selected KB document IDs
    source_folders: List[str]               # Selected KB folders

    # Extraction metadata
    last_extracted: Optional[datetime]
    extraction_status: ExtractionStatus     # NEVER, EXTRACTING, COMPLETE, ERROR

    # Structured data reference
    structured_data_file: str               # Path to JSON file

class EntityType(str, Enum):
    MY_COMPANY = "my_company"               # Primary trading entities
    CLIENT = "client"                       # Client organizations
    PARTNER = "partner"                     # Partnership entities
    OTHER = "other"

class ExtractionStatus(str, Enum):
    NEVER = "never"                         # Never extracted
    EXTRACTING = "extracting"               # Currently extracting
    COMPLETE = "complete"                   # Extraction complete
    STALE = "stale"                         # Needs re-extraction (>30 days)
    ERROR = "error"                         # Extraction failed
```

### Enhanced Structured Knowledge

```python
class StructuredKnowledge(BaseModel):
    """Structured data for one entity."""

    # Entity reference
    entity_id: str
    entity_name: str

    # Core data (same as before)
    organization: Optional[OrganizationProfile]
    insurances: List[Insurance]
    team_qualifications: List[Qualification]
    team_work_experience: List[WorkExperience]
    projects: List[ProjectExperience]
    references: List[Reference]
    capabilities: List[Capability]

    # Extraction metadata
    extraction_date: datetime
    source_document_count: int              # How many docs analyzed
    source_document_ids: List[str]          # Which specific docs

    # Quality metrics
    extraction_confidence: Dict[str, float]  # Per-category confidence
    warnings: List[str]                      # Any extraction warnings
```

---

## 🎨 New UI Design

### Page: Entity Data Manager (New)

**Location:** `/pages/Proposal_Entity_Manager.py`

#### Section 1: Entity List
```
📊 Entity Data Manager
Version: v1.0.0

Manage organizational entities and their structured data for tender responses.

Your Entities:
┌─────────────────────────────────────────────────────────────────┐
│ 🏢 longboardfella consulting pty ltd                           │
│ Type: My Company                                                │
│ Status: ✅ Extracted (3 days ago)                               │
│ Data: 2 insurances, 5 qualifications, 8 projects, 3 references │
│ Sources: 8 documents                                            │
│ [View] [Re-Extract] [Edit Sources] [Delete]                    │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ 🎓 Deakin University                                            │
│ Type: Client                                                    │
│ Status: ⚠️ Stale (45 days ago)                                 │
│ Data: 1 insurance, 12 qualifications, 15 projects              │
│ Sources: 12 documents                                           │
│ [View] [Re-Extract] [Edit Sources] [Delete]                    │
└─────────────────────────────────────────────────────────────────┘

[+ Create New Entity]
```

#### Section 2: Create/Edit Entity

```
Create New Entity

Entity Name: [Escient Pty Ltd                    ]
Entity Type: [My Company ▼]
Description: [Technology consulting subsidiary    ]

📁 Select Source Documents
Browse your knowledge base and select documents containing data for this entity.

Search KB: [escient                              ] [🔍 Search]

Results (125 documents):
┌─────────────────────────────────────────────────────────────────┐
│ 📁 escient_pty_ltd/                             (45 documents)  │
│   ├── 📁 company_registration/                  (3 documents)   │
│   │   ☑️ ABN_Certificate_Escient.pdf                            │
│   │   ☑️ ACN_Certificate_Escient.pdf                            │
│   │   ☐ Business_Name_Registration.pdf                          │
│   ├── 📁 insurance/                             (2 documents)   │
│   │   ☑️ Combined_Policy_2024.pdf                               │
│   │   ☐ Workers_Comp_2024.pdf                                   │
│   ├── 📁 team/                                  (15 documents)  │
│   │   ☑️ TeamLead_CV.pdf                                        │
│   │   ☑️ Developer_CV.pdf                                       │
│   │   ... (13 more)                                             │
│   └── 📁 projects/                              (25 documents)  │
│       ☑️ Client_A_CaseStudy.pdf                                 │
│       ☑️ Client_B_Project.pdf                                   │
│       ... (23 more)                                             │
└─────────────────────────────────────────────────────────────────┘

Selected: 8 documents

[Cancel] [Save Entity] [Save & Extract Now]
```

#### Section 3: View Extracted Data

```
Entity: longboardfella consulting pty ltd
Last Extracted: 2026-01-01 14:30
Status: ✅ Complete
Sources: 8 documents

📊 Extraction Summary:
┌──────────────────────────────────────────────────────────────┐
│ Organization Profile          ✅ Complete                     │
│ Legal Name:   longboardfella consulting pty ltd              │
│ ABN:          12 345 678 901                                 │
│ ACN:          123 456 789                                    │
│ Address:      123 Surf St, Melbourne VIC 3000                │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│ Insurance Policies            2 policies                      │
│ • Professional Indemnity      $20M (Expires: 2025-06-30)     │
│ • Public Liability            $10M (Expires: 2025-06-30)     │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│ Team Qualifications           5 qualifications                │
│ • Jane Smith - PhD (Health Informatics) - 2018               │
│ • Jane Smith - CHIA - 2020                                   │
│ • John Doe - BSc (Computer Science) - 2015                   │
│ ... (2 more)                                                 │
└──────────────────────────────────────────────────────────────┘

[Download JSON] [Edit Data] [Re-Extract]

⚠️ Warnings (2):
• No references found in selected documents
• Some qualifications missing expiry dates
```

---

## 🔧 Implementation Changes

### New Files

**1. `/cortex_engine/entity_manager.py`**
```python
class EntityManager:
    """Manages organizational entities and their structured data."""

    def create_entity(self, name: str, type: EntityType, source_docs: List[str]) -> EntityProfile
    def list_entities(self) -> List[EntityProfile]
    def get_entity(self, entity_id: str) -> EntityProfile
    def update_entity(self, entity_id: str, updates: Dict) -> EntityProfile
    def delete_entity(self, entity_id: str) -> bool

    def extract_entity_data(self, entity_id: str, progress_callback) -> StructuredKnowledge
    def get_entity_data(self, entity_id: str) -> StructuredKnowledge
```

**2. `/cortex_engine/kb_navigator.py`**
```python
class KBNavigator:
    """Navigate and select documents from knowledge base."""

    def search_documents(self, query: str) -> List[KBDocument]
    def get_folder_structure(self) -> Dict  # Hierarchical folder view
    def get_document_metadata(self, doc_id: str) -> Dict
    def filter_by_folder(self, folder_path: str) -> List[KBDocument]
```

**3. `/pages/Proposal_Entity_Manager.py`**
- Entity list view
- Create/edit entity
- Browse & select KB documents
- View extracted data
- Re-extraction management

### Modified Files

**1. `/cortex_engine/tender_data_extractor.py`**

Add document filtering:
```python
async def extract_all_structured_data(
    self,
    entity_id: str,                          # NEW: Which entity
    document_ids: Optional[List[str]] = None,  # NEW: Filter to specific docs
    progress_callback=None
) -> StructuredKnowledge:
    """Extract structured data for specific entity from selected documents."""

    # Filter queries to only selected documents
    if document_ids:
        self.document_filter = document_ids

    # ... rest of extraction with filtering
```

Update `_query_vector_store` to filter by document IDs:
```python
async def _query_vector_store(self, query: str, top_k: int = 5) -> Dict[str, Any]:
    """Query vector store with optional document filtering."""

    # Add where clause to filter by document IDs if specified
    where = None
    if self.document_filter:
        where = {"doc_id": {"$in": self.document_filter}}

    results = self.vector_index.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where  # Filter to selected documents only
    )
```

**2. `/cortex_engine/tender_schema.py`**

Add entity metadata:
```python
class StructuredKnowledge(BaseModel):
    entity_id: str                          # NEW
    entity_name: str                        # NEW
    source_document_ids: List[str]          # NEW
    # ... rest of fields
```

---

## 🚀 User Workflow (Redesigned)

### First-Time Setup

**1. Create Entity Profiles**
```
Go to: Proposal Entity Manager

Create Entity: longboardfella consulting pty ltd
  - Browse KB → Select "longboardfella_consulting" folder
  - Auto-select all sub-documents (8 docs)
  - Save & Extract Now
  - Wait 2-3 minutes
  - ✅ Extraction complete!

Create Entity: Deakin University
  - Browse KB → Select "deakin_university" folder
  - Auto-select sub-documents (12 docs)
  - Save & Extract Now
  - ✅ Extraction complete!

Create Entity: Escient Pty Ltd
  - Browse KB → Select "escient_pty_ltd" folder
  - Auto-select sub-documents (6 docs)
  - Save & Extract Now
  - ✅ Extraction complete!
```

**2. Use in Tender Response**
```
Upload Tender: RFT12493.docx

Select Entity: [longboardfella consulting pty ltd ▼]
  ↓ Loads structured data for longboardfella

Auto-fill 45 fields with longboardfella data:
  ✅ ABN: 12 345 678 901
  ✅ Insurance: PI-2024-12345
  ✅ Team: Jane Smith, PhD
  ... (42 more)

Export completed tender
```

### Ongoing Use

**Add New Documents:**
```
Knowledge Ingest → Upload new insurance certificate

Go to: Proposal Entity Manager
Select: longboardfella consulting pty ltd
Edit Sources → Add new insurance doc
Re-Extract → 30 seconds
✅ Updated!
```

**Switch Entities:**
```
Tender for Deakin University project?

Select Entity: [Deakin University ▼]
Auto-fills Deakin's ABN, insurance, team, projects
```

---

## ✅ Benefits Over Current Design

| Current Design | Redesigned |
|---|---|
| ❌ Extracts 58,185 docs | ✅ Extracts 8-12 selected docs |
| ❌ 10-15 minutes | ✅ 2-3 minutes |
| ❌ One blob for everything | ✅ Separate per entity |
| ❌ No document selection | ✅ Browse & select |
| ❌ Can't switch entities | ✅ Select entity dropdown |
| ❌ No folder navigation | ✅ Folder tree view |
| ❌ Impractical for 58K docs | ✅ Practical for organized KB |

---

## 📝 Implementation Priority

### Phase 1A: Entity Management (Week 1)
1. Create `entity_manager.py` - CRUD for entities
2. Create `kb_navigator.py` - Document browsing
3. Update `tender_data_extractor.py` - Add document filtering
4. Create `Proposal_Entity_Manager.py` - Basic UI

### Phase 1B: Navigation UI (Week 1-2)
1. Folder tree view for KB navigation
2. Document search and selection
3. Multi-select checkboxes
4. Selected document count

### Phase 2: Field Matching (Week 2-3)
As planned, but with entity selection

### Phase 3: Auto-Fill UI (Week 3-4)
As planned, but with entity dropdown

---

**Status:** Design redesigned based on user feedback
**Next:** Implement Phase 1A (Entity Management + KB Navigation)
**Question for User:** Does this entity-based selective extraction approach match your needs?
