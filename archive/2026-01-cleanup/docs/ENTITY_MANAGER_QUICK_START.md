# Entity Manager Quick Start Guide

## 🎉 What's Ready

You now have a complete **entity-based extraction system** that lets you:
- Create separate profiles for each organization (longboardfella, Deakin, Escient)
- Select specific KB folders/documents per entity
- Extract structured data from selected documents only (not entire 58K KB!)
- Manage multiple entities with separate data files

---

## 🚀 How to Use (5 Steps)

### Step 1: Navigate to Entity Manager

```bash
streamlit run Cortex_Suite.py
```

Go to the sidebar → **"Proposal Entity Manager"** page

### Step 2: Create Your First Entity

1. Click **"➕ Create New Entity"**
2. Fill in the form:
   - **Entity Name**: `longboardfella consulting pty ltd`
   - **Entity Type**: `My Company`
   - **Description**: `Primary trading entity`

### Step 3: Select Source Folders

1. In the **"Search folders"** box, type: `longboardfella`
2. You'll see matching folders like:
   ```
   📁 /Boilerplate/Longboardfella_Boilerplate (3 docs)
   📁 /Projects/longboardfella (25 docs)
   📁 /Insurance/longboardfella (2 docs)
   ```
3. Check the boxes next to folders you want to include
4. You'll see: "✅ Selected 3 folder(s)" and "📊 Total documents: 30"

### Step 4: Create & Extract

1. Click **"💾 Create Entity"**
2. Entity is created!
3. Click **"🔄 Re-Extract"** button to extract structured data
4. Wait 2-3 minutes (extracting from 30 docs, not 58K!)
5. See **"✅ Extraction complete!"**

### Step 5: View Results

- See extraction summary:
  - ✅ Organization Profile
  - 2 Insurance Policies
  - 5 Qualifications
  - 8 Projects
  - 3 References

- Data saved to: `{db_path}/structured_data/longboardfella_consulting.json`

---

## 📁 File Structure

```
{your_db_path}/
├── entities.json                           # Entity metadata
└── structured_data/                        # Structured data per entity
    ├── longboardfella_consulting.json      # Your entity
    ├── deakin_university.json              # Another entity
    └── escient_pty_ltd.json                # Another entity
```

---

## 🔄 Create More Entities

Repeat for other organizations:

**Entity 2: Deakin University**
```
Name: Deakin University
Type: Client
Search: "deakin"
Select: Deakin folders
Extract → Done!
```

**Entity 3: Escient Pty Ltd**
```
Name: Escient Pty Ltd
Type: Subsidiary
Search: "escient"
Select: Escient folders
Extract → Done!
```

---

## 🎯 Using Entities in Tender Responses

Once you have multiple entities with extracted data:

1. **Upload tender document** (in future Tender Auto-Fill page)
2. **Select entity dropdown**: Choose which organization
   - `[longboardfella consulting pty ltd ▼]`
3. **Auto-fill fields** from selected entity's data
4. **Switch entities** for different tenders

---

## ✅ What's New vs Old System

| Feature | Old System | New System |
|---------|-----------|------------|
| **Extraction Scope** | ❌ Entire 58K docs | ✅ Selected 8-30 docs per entity |
| **Extraction Time** | ❌ 10-15 minutes | ✅ 2-3 minutes |
| **Multiple Orgs** | ❌ One blob for all | ✅ Separate per entity |
| **Folder Selection** | ❌ No selection | ✅ Browse and select |
| **Entity Switching** | ❌ Not possible | ✅ Dropdown selection |

---

## 📊 Entity Manager Features

### Entity List View
- **Status Indicators**: ✅ Complete, ⚠️ Stale, ❌ Error, ⚪ Never extracted
- **Data Completeness**: See which categories have data
- **Actions**:
  - 👁️ View Data - See extracted structured data
  - 🔄 Re-Extract - Update extraction
  - ✏️ Edit Sources - Change selected folders
  - 🗑️ Delete - Remove entity

### KB Statistics
- Total documents in your KB
- Total folders detected
- Maximum folder depth

### Folder Navigation
- **Search by name**: Type folder name to find matches
- **See document counts**: Each folder shows how many docs
- **Multi-select**: Check multiple folders to include
- **Auto-calculate total**: See total docs before extracting

---

## 🔧 Troubleshooting

### Issue: No folders found when searching

**Solution:**
1. Check your folder name matches KB structure
2. Try broader search (e.g., just "long" instead of "longboardfella")
3. Check KB Statistics to see if documents are loaded

### Issue: Extraction takes longer than expected

**Possible causes:**
- Selected too many folders (>100 docs)
- Slow LLM model

**Solution:**
- Select fewer, more targeted folders
- Each entity should focus on 8-30 documents

### Issue: Extraction shows "No data found"

**Possible causes:**
- Selected folders don't contain relevant data
- Documents are images/scans (not searchable text)

**Solution:**
- Add folders with actual company docs (ABN certificates, insurance, CVs)
- Ensure documents are text-based PDFs or Word docs

---

## 📝 Next Steps

### After Creating Entities:

**Phase 1B (Current):**
- ✅ Create entities for your organizations
- ✅ Extract structured data per entity
- ✅ View and verify extraction results

**Phase 2 (Coming Soon):**
- Build Tender Field Classifier
- Build Tender Field Matcher
- Test with real tender (RFT12493)
- Match tender fields to entity data

**Phase 3 (Coming Soon):**
- Build Tender Auto-Fill UI
- Select entity dropdown
- Review/approve workflow
- Export completed tender

---

## 🆘 Need Help?

**Check entity status:**
- Green ✅ = Extraction complete and recent
- Yellow ⚠️ = Stale (>30 days old)
- Red ❌ = Error during extraction
- Gray ⚪ = Never extracted

**Re-extract if:**
- Added new documents to KB
- Updated insurance/qualifications
- Extraction is stale (>30 days)
- Previous extraction had errors

---

## 🎯 Example Workflow

**Scenario: Set up 3 entities**

```
1. Create "longboardfella consulting pty ltd"
   Search: "longboardfella"
   Select: Longboardfella_Boilerplate folder
   Extract: 2 minutes
   Result: ✅ 12 documents, complete data

2. Create "Deakin University"
   Search: "deakin"
   Select: Deakin folders
   Extract: 3 minutes
   Result: ✅ 25 documents, complete data

3. Create "Escient Pty Ltd"
   Search: "escient"
   Select: Escient folders
   Extract: 2 minutes
   Result: ✅ 8 documents, complete data

Total time: 10-15 minutes for 3 entities
Total data: 45 documents (not 58K!)
```

---

**Ready to test!** Go to **Proposal Entity Manager** and create your first entity!

**Files Created:**
- `/cortex_engine/entity_manager.py` - Entity CRUD
- `/cortex_engine/kb_navigator.py` - KB browsing
- `/pages/Proposal_Entity_Manager.py` - UI page
- `/cortex_engine/tender_data_extractor.py` - Updated with filtering
