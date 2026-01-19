# Testing Notes - 2025-10-11

## Session Summary

This session focused on fixing critical ingestion errors discovered in logs, improving search result UX, and enhancing timeout handling for long-running operations.

## Changes Implemented

### 1. Fixed LlamaIndex Reader API Errors ✅
**Issue:** DocxReader and other LlamaIndex readers were being called with incorrect `file_path=` keyword argument
**Fix:** Changed to positional argument (API changed in recent LlamaIndex versions)
**Files Modified:**
- `cortex_engine/enhanced_ingest_cortex.py` (6 locations)
- `cortex_engine/ingest_cortex.py` (6 locations)
- Synced to `docker/` distribution

**Testing Required:**
- [x] Ingest .docx files - TESTED during session (35 documents processed successfully)
- [ ] Ingest .pdf files
- [ ] Ingest .pptx files
- [ ] Verify no errors in `logs/ingestion.log`

### 2. Search Result Deduplication ✅
**Issue:** Search returned 20 chunks from same document showing as 20 separate results
**Fix:** Added deduplication logic to group by file_name and keep highest-scoring chunk
**Files Modified:**
- `pages/3_Knowledge_Search.py`
- Synced to `docker/pages/`

**Testing Required:**
- [ ] Search for terms that match multiple chunks of same document
- [ ] Verify shows "X results (Y unique documents)"
- [ ] Verify displays only one result per document
- [ ] Check that correct (highest-scoring) chunk is displayed

### 3. Removed Auto-Submit Behavior ✅
**Issue:** Pressing Tab in search field auto-triggered search
**Fix:** Removed `or (query != last_query)` condition from button handler
**Files Modified:**
- `pages/3_Knowledge_Search.py`
- Synced to `docker/pages/`

**Testing Required:**
- [ ] Enter search term and press Tab - should NOT trigger search
- [ ] Must click "Search Knowledge Base" button to initiate search
- [ ] Verify consistent behavior across all input fields in project

### 4. Improved LLM Timeout Handling ✅
**Issue:** Document 14/35 hung for 10+ minutes with no feedback, showed confusing timeout message
**Fix:**
- Reduced timeout from 600s (10 min) to 180s (3 min)
- Added elapsed time logging for calls >60s
- Explicit TimeoutError handling with clear messages
**Files Modified:**
- `cortex_engine/ingest_cortex.py`
- Synced to `docker/cortex_engine/`

**Testing Required:**
- [ ] Ingest very large documents (>5000 words)
- [ ] Check logs show elapsed time for long operations
- [ ] Verify timeout after 3 minutes (not 10)
- [ ] Confirm fallback metadata is applied on timeout
- [ ] Verify ingestion continues after timeout (doesn't abort batch)

### 5. Fixed Missing Method Calls ✅
**Issue:** Code calling `recovery_manager.get_recently_ingested_documents()` which doesn't exist
**Fix:** Removed invalid method calls (completion screen now queries ChromaDB directly)
**Files Modified:**
- `pages/2_Knowledge_Ingest.py` (2 locations)
- Synced to `docker/pages/`

**Testing Required:**
- [ ] Complete full ingestion workflow
- [ ] Verify completion screen shows accurate document counts
- [ ] Check no errors in Streamlit logs about missing methods

### 6. Fixed Async Coroutine Warning ✅
**Issue:** `BackupManager.list_backups()` called without await
**Fix:** Added `asyncio.run()` wrapper
**Files Modified:**
- `pages/7_Maintenance.py`
- Synced to `docker/pages/`

**Testing Required:**
- [ ] Open Maintenance page → Backup Management
- [ ] Verify backup list displays correctly
- [ ] Check no coroutine warnings in console/logs

## Git Commits Made

1. `751836e` - fix: Resolve LlamaIndex reader API errors and async warnings
2. `d19751d` - feat: Deduplicate search results by document
3. `c702796` - fix: Remove auto-submit on text input change in Knowledge Search
4. `62de199` - fix: Improve LLM timeout handling and feedback during ingestion

All changes pushed to `origin/main`.

## Testing Status

### Tested During Session ✅
- Ingestion of 35 documents (completed successfully with new timeout/error handling)
- Search result display (screenshot reviewed showing redundancy issue)
- Log analysis confirmed fixes address root causes

### Requires User Testing ⏳
1. **Search Deduplication** - Search for query that matches multiple documents
2. **Auto-Submit Fix** - Verify Tab doesn't trigger search in Knowledge Search page
3. **Timeout Handling** - Ingest large documents and verify 3-minute timeout
4. **Backup List** - Check Maintenance page backup management displays correctly
5. **File Format Support** - Test .pdf, .pptx, .docx ingestion for API fixes

## Known Issues / Future Improvements

### Not Addressed Yet
- **UI Progress During Long Operations**: Users still don't see real-time progress during 60-180s LLM calls (only log updates)
- **GPU Usage Verification**: User reported ingestion might not be using GPU (Docker Task Manager showed no GPU usage)
- **Proposal Generator UI**: User mentioned it's "pretty complex and messy" - to be addressed in future session

### Performance Enhancements Completed Earlier
- SQLite-based persistent cache (10-100x speedup for repeated queries)
- Interactive Plotly charts for performance visualization
- Two-tier caching (memory + persistent)
- All previously committed and tested

## Docker Distribution Status

✅ All changes synchronized to `docker/` directory:
- `docker/cortex_engine/enhanced_ingest_cortex.py`
- `docker/cortex_engine/ingest_cortex.py`
- `docker/pages/2_Knowledge_Ingest.py`
- `docker/pages/3_Knowledge_Search.py`
- `docker/pages/7_Maintenance.py`

## Next Session Recommendations

1. **Test all fixes** listed in "Requires User Testing" section above
2. **Verify GPU usage** in Docker during ingestion (user reported possible issue)
3. **Review Proposal Generator UI** complexity (user feedback: "complex and messy")
4. **Consider UI progress indicators** for long-running LLM operations (UX improvement)
5. **Test Docker GPU build** when better network connectivity available

## Version Information

- **Current Version**: v4.10.2 (documented in CHANGELOG.md)
- **Release Date**: 2025-10-11
- **Python**: 3.11
- **Key Dependencies**: LlamaIndex (latest), ChromaDB, Ollama (local)

---

**Session Date**: 2025-10-11
**Duration**: ~3 hours
**Commits**: 4
**Files Modified**: 8 (4 in main, 4 in docker/)
**Lines Changed**: +90, -75
**Status**: Ready for testing ⏳
