# Testing Phase 2 After Reboot - Quick Start Guide

**Status:** Phase 1 & 2 complete, tested, and pushed to git ✅
**Commit:** `13e63c8` - "feat: Complete Phase 1 & 2 enhancements"
**Date:** 2026-01-13

---

## 🎯 What's Ready to Test

### Phase 2 Features (All Tests Passing: 7/7 ✅)

1. **Table-Aware Chunking**
   - Tables never split mid-table
   - Context added before/after tables
   - Better table retrieval and search

2. **Figure Entity Linking**
   - Figures linked to people/orgs in knowledge graph
   - Semantic search for figures improved
   - Entity extraction from VLM descriptions

---

## 🚀 Quick Testing Commands

### 1. Run Test Suites (Verify Everything Works)

```bash
# Change to project directory
cd /home/longboardfella/cortex_suite

# Activate virtual environment
source venv/bin/activate

# Run Phase 2 test suite
python scripts/test_phase2_enhancements.py

# Expected output: "🎉 ALL TESTS PASSED - Phase 2 is production-ready!"
```

### 2. Test with ONE Document (Safe - No Risk)

```bash
# Test with temporary database (won't touch your production DB)
python cortex_engine/ingest_cortex.py \
  --paths /path/to/one/test/document.pdf \
  --db-path /tmp/phase2_test

# Look for these log messages:
# "🚀 Applying Phase 2 enhancements to documents..."
# "📊 Applying table-aware chunking to N documents..."
# "✅ Table-aware chunking complete: X docs → Y chunks"
```

### 3. Test with Separate Database (Recommended)

```bash
# Create separate test database (keeps production untouched)
python cortex_engine/ingest_cortex.py \
  --paths /path/to/your/documents \
  --db-path /mnt/f/ai_databases_test

# This creates a completely separate database for testing
# Your production database at /mnt/f/ai_databases is NOT touched
```

---

## 📊 Database Options Explained

### ⭐ Recommended: Option 2 - Separate Test Database

**Why:** Safest way to evaluate Phase 2 without risk

```bash
# Step 1: Ingest to test database
python cortex_engine/ingest_cortex.py \
  --paths /path/to/docs \
  --db-path /mnt/f/ai_databases_test

# Step 2: Test search quality
# (Update Streamlit to point to test database temporarily)

# Step 3: Compare with production database
# - Search same queries in both databases
# - Evaluate table retrieval quality
# - Check if figure search is improved
```

**Pros:**
- ✅ Production database completely untouched
- ✅ Clean evaluation of Phase 2
- ✅ Can switch back instantly
- ✅ No risk to existing data

**Cons:**
- Need disk space for second database
- Need time to re-ingest documents

---

### Option 1: Quick Test with Existing Database

**Why:** Fastest way to see Phase 2 in action

```bash
# Just ingest new documents to existing database
python cortex_engine/ingest_cortex.py \
  --paths /path/to/new/test/docs \
  --db-path /mnt/f/ai_databases
```

**Result:**
- New documents get Phase 2 processing
- Old documents remain unchanged

**Pros:**
- ✅ Instant testing
- ✅ No data loss

**Cons:**
- ⚠️ Mixed chunking strategies (old vs new docs)
- ⚠️ Inconsistent search quality
- ⚠️ Not production-ready state

**Use this for:** Quick evaluation only

---

### Option 3: Fresh Production Database

**Why:** Best long-term solution for consistent quality

```bash
# Step 1: Backup existing database
cp -r /mnt/f/ai_databases /mnt/f/ai_databases_backup

# Step 2: Delete ChromaDB (knowledge graph can stay)
rm -rf /mnt/f/ai_databases/knowledge_hub_db

# Step 3: Re-ingest everything with Phase 2
python cortex_engine/ingest_cortex.py \
  --paths /path/to/all/documents \
  --db-path /mnt/f/ai_databases
```

**Pros:**
- ✅ All documents processed consistently
- ✅ Optimal search quality
- ✅ Production-ready state

**Cons:**
- ❌ Time-consuming (need to re-ingest everything)
- ⚠️ Backup recommended before starting

**Use this for:** Production deployment after testing

---

## 🔍 What to Look For

### During Ingestion

Watch for these log messages:

```
🚀 Applying Phase 2 enhancements to documents...
📎 Linking figures to knowledge graph entities...
✅ Figure entity linking complete
📊 Applying table-aware chunking to X documents...
✅ Table-aware chunking complete: X docs → Y chunks
```

### Success Indicators

1. **More chunks than documents** (tables split into separate chunks)
2. **Log shows "table chunks" vs "text chunks"**
3. **No errors or warnings about Phase 2 processing**

### If Phase 2 Isn't Running

Check:
1. Configuration flags enabled (should be by default)
2. Documents have table/figure metadata (requires Docling processing)
3. Phase 2 modules imported successfully (check startup logs)

---

## 📝 Testing Checklist

After reboot, do this in order:

- [ ] 1. Run `python scripts/test_phase2_enhancements.py`
  - Should see: "🎉 ALL TESTS PASSED"

- [ ] 2. Test with one document to `/tmp/phase2_test`
  - Verify Phase 2 log messages appear

- [ ] 3. If tests pass, create separate test database
  - Ingest sample documents to `/mnt/f/ai_databases_test`

- [ ] 4. Test search quality in Streamlit
  - Compare production vs test database
  - Look for improved table retrieval

- [ ] 5. Decide on production approach
  - Keep separate DB for now, OR
  - Plan full re-ingestion with Phase 2

---

## 🆘 Troubleshooting

### Tests Fail After Reboot?

```bash
# Reinstall dependencies if needed
pip install -r requirements.txt

# Reinstall spaCy model (for entity extraction)
python -m spacy download en_core_web_sm

# Re-run tests
python scripts/test_phase2_enhancements.py
```

### Phase 2 Not Processing Documents?

Check config:
```bash
# Verify Phase 2 is enabled
python -c "from cortex_engine.config import TABLE_AWARE_CHUNKING, FIGURE_ENTITY_LINKING; print(f'Table chunking: {TABLE_AWARE_CHUNKING}, Figure linking: {FIGURE_ENTITY_LINKING}')"

# Should output: "Table chunking: True, Figure linking: True"
```

### spaCy Model Missing?

```bash
python -m spacy download en_core_web_sm
```

---

## 📚 Documentation

All documentation is in the repo:

- **`PHASE2_STATUS.md`** - Comprehensive Phase 2 status and features
- **`TESTING_AFTER_REBOOT.md`** - This file (quick start guide)
- **`CLAUDE.md`** - Project guidelines (updated with version workflow)

Test scripts:
- **`scripts/test_phase1_enhancements.py`** - Phase 1 validation
- **`scripts/test_phase2_enhancements.py`** - Phase 2 validation

---

## 🎯 Recommended Next Steps

1. **Verify tests pass** after reboot
2. **Create test database** at `/mnt/f/ai_databases_test`
3. **Ingest sample documents** to test database
4. **Evaluate search quality** improvements
5. **Plan production deployment** (full re-ingestion vs incremental)

---

## 📊 Git Status

**Latest Commit:**
```
13e63c8 - feat: Complete Phase 1 & 2 enhancements - Table-aware chunking and figure entity linking
```

**Files Added:**
- `cortex_engine/table_chunking_enhancer.py` (450 lines)
- `cortex_engine/figure_entity_linker.py` (360 lines)
- `scripts/test_phase1_enhancements.py` (508 lines)
- `scripts/test_phase2_enhancements.py` (650 lines)
- `PHASE2_STATUS.md` (comprehensive docs)
- `TESTING_AFTER_REBOOT.md` (this guide)

**Files Modified:**
- `cortex_engine/docling_reader.py` (Docling 1.8.5 fix)
- `cortex_engine/ingest_cortex.py` (Phase 2 integration)
- `cortex_engine/query_cortex.py` (VLM workers increased)

**Status:** ✅ Committed and pushed to `origin/main`

---

**Quick Start:** Run `python scripts/test_phase2_enhancements.py` first! 🚀
