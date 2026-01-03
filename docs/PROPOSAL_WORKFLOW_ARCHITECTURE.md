# Proposal Workflow - Correct Architecture

## ✅ What Was Fixed

### ❌ Old (Incorrect) Architecture
- Extraction functionality in **Knowledge Search** page
- No clear separation between KB search and proposal workflow
- Confusing user experience

### ✅ New (Correct) Architecture
- **Knowledge Search** = Search existing knowledge base ONLY
- **Proposal Entity Manager** = Manage entity profiles (optional)
- **Proposal Workflow** = Complete tender response workflow with file pickers

---

## 🎯 New Proposal Workflow

### Page: `Proposal_Workflow.py`

**Purpose:** Complete end-to-end workflow for responding to tenders.

### 5-Step Workflow

```
📄 Step 1: Select Tender
    ↓ File picker for tender document (RFT/RFQ to fill out)

📁 Step 2: Select Sources
    ↓ File picker for source documents OR use entity profile

🔍 Step 3: Extract Data
    ↓ Extract structured data from selected sources

🎯 Step 4: Match Fields (Phase 2)
    ↓ Auto-match tender fields to extracted data

✅ Step 5: Fill & Export (Phase 3)
    ↓ Fill tender and export completed document
```

---

## 📋 Detailed Step Descriptions

### Step 1: Select Tender Document

**File Picker:**
- Upload tender/RFT/RFQ document (.docx)
- System loads and previews document
- Shows paragraph count, table count
- Preview first few paragraphs

**Example:**
```
Upload: RFT12493-Request-for-Tender-DHA-Health.docx
✅ Loaded: 777 paragraphs, 23 tables
Preview: "3. Technical Approach - Describe your methodology..."
```

### Step 2: Select Source Documents

**Two Options:**

**Option A: Upload Files (Ad-hoc)**
- File picker for source documents
- Select multiple files:
  - Company registration PDF
  - Insurance certificates
  - Team CVs
  - Project case studies
  - References
- Shows selected file list

**Option B: Use Entity Profile (Pre-configured)**
- Dropdown to select entity (longboardfella, Deakin, Escient)
- Uses entity's pre-selected KB documents
- Shows entity stats (folder count, document count, last extracted)

**Example Option A:**
```
Selected source files:
📄 ABN_Certificate.pdf
📄 Insurance_Policy_2024.pdf
📄 JaneSmith_CV.pdf
📄 ProjectCaseStudy_ADHA.pdf
Total: 4 files
```

**Example Option B:**
```
Selected entity: longboardfella consulting pty ltd
✅ 3 source folders
✅ 30 source documents
✅ Last extracted: 2026-01-02
```

### Step 3: Extract Structured Data

**Process:**
1. Initialize extractor with ChromaDB collection
2. Load knowledge graph
3. Run extraction from selected sources
4. Show progress (Organization → Insurances → Qualifications → ...)
5. Display extraction summary

**Output:**
```
📊 Extraction Summary:
✅ Organization: longboardfella consulting pty ltd
✅ Insurances: 2 policies
✅ Qualifications: 5 credentials
✅ Projects: 8 case studies
✅ References: 3 contacts
```

### Step 4: Match Fields (Phase 2 - Coming Soon)

**Planned Features:**
1. Parse tender document to find fillable fields
2. Classify field types (ABN, insurance policy, qualification, etc.)
3. Auto-match to extracted data
4. Show confidence scores (high/medium/low)
5. Review/approve interface

**Example:**
```
Tender Field → Matched Data → Confidence
ABN: [______] → 12 345 678 901 → High (95%)
Policy #: [___] → PI-2024-12345 → High (92%)
Team Lead: [__] → Dr. Jane Smith → Medium (75%) [Choose: Dr. Smith / John Doe]
```

### Step 5: Fill & Export (Phase 3 - Coming Soon)

**Planned Features:**
1. Fill tender document with matched data
2. Preserve original formatting
3. Allow manual edits
4. Export completed tender

---

## 🔄 Workflow Examples

### Example 1: Quick Response (Using Entity)

```
1. Select Tender: Upload RFT12493.docx
2. Select Sources: Choose entity "longboardfella consulting"
3. Extract Data: ✅ 30 docs processed, data extracted
4. Match Fields: (Phase 2)
5. Fill & Export: (Phase 3)

Time: ~5 minutes
```

### Example 2: Ad-hoc Response (Upload Files)

```
1. Select Tender: Upload NewTender.docx
2. Select Sources: Upload 5 PDFs (ABN, insurance, CVs, etc.)
3. Extract Data: ✅ 5 files processed, data extracted
4. Match Fields: (Phase 2)
5. Fill & Export: (Phase 3)

Time: ~10 minutes
```

---

## 📁 File Structure

```
/pages/
├── 3_Knowledge_Search.py         # Search KB ONLY (extraction removed)
├── Proposal_Entity_Manager.py    # Manage entity profiles (optional)
└── Proposal_Workflow.py          # NEW! Complete tender workflow

/cortex_engine/
├── entity_manager.py             # Entity CRUD operations
├── kb_navigator.py               # KB folder browsing
└── tender_data_extractor.py      # Data extraction engine

/docs/
├── PROPOSAL_WORKFLOW_ARCHITECTURE.md    # This file
└── ENTITY_MANAGER_QUICK_START.md        # Entity manager guide
```

---

## 🎯 User Journey

### Scenario: Respond to New Tender

**Step 1: Prepare (One-time)**
- Go to **Proposal Entity Manager**
- Create entity "longboardfella consulting"
- Select source folders from KB
- Extract structured data
- ✅ Entity ready for reuse

**Step 2: Respond to Tender**
- Go to **Proposal Workflow**
- Upload tender document (RFT12493.docx)
- Select entity "longboardfella consulting"
- Extract data (reuses entity's sources)
- Match fields (Phase 2)
- Fill and export (Phase 3)

**Total Time:** 10-15 minutes (vs 2-3 hours manual)

---

## ✅ Benefits of New Architecture

| Feature | Old | New |
|---------|-----|-----|
| **Separation** | ❌ Mixed with KB search | ✅ Separate workflow |
| **File Pickers** | ❌ No file selection | ✅ Tender + Sources |
| **Entity Support** | ❌ Not available | ✅ Pre-configured entities |
| **Ad-hoc Workflow** | ❌ Not supported | ✅ Upload files directly |
| **Clear Steps** | ❌ Unclear flow | ✅ 5-step progress |
| **Reusability** | ❌ Extract every time | ✅ Reuse entity data |

---

## 🚀 Current Status

### Phase 1: Extraction ✅ COMPLETE
- ✅ File picker for tender document
- ✅ File picker for source documents
- ✅ Entity selection option
- ✅ Structured data extraction
- ✅ Extraction summary display
- ✅ Removed from Knowledge Search

### Phase 2: Field Matching 🚧 IN PLANNING
- ⏳ Tender field parser
- ⏳ Field classifier
- ⏳ Data matcher
- ⏳ Review/approve UI

### Phase 3: Fill & Export 📋 PLANNED
- 📋 Document assembly
- 📋 Fill with matched data
- 📋 Export completed tender

---

## 📝 How to Use

**Access the new workflow:**
1. Start Streamlit: `streamlit run Cortex_Suite.py`
2. Go to sidebar → **"Proposal Workflow"**
3. Follow 5-step workflow

**For best results:**
1. First create entity profiles in **Proposal Entity Manager**
2. Then use those entities in **Proposal Workflow** for fast responses

---

**Status:** Phase 1 Complete (2026-01-03)
**Next:** Phase 2 - Field Matching & Auto-Fill
