# Chunk Review V2 - Testing Checklist

**Date**: 2026-01-07
**Version**: 2.0.0
**Status**: Ready for Testing

---

## Pre-Testing Setup

### Required Components
- [ ] Workspace with entity profile bound
- [ ] Document uploaded to workspace (preferably tender document with multiple sections)
- [ ] Document should have 20+ chunks for realistic testing
- [ ] Entity profile should have company data populated

### Test Environment
- [ ] Streamlit running: `streamlit run Cortex_Suite.py`
- [ ] LLM available: `ollama list` shows qwen2.5:72b-instruct-q4_K_M
- [ ] Database path accessible
- [ ] Git initialized in workspace directory

---

## Core Functionality Tests

### 1. Batch Analysis ✅

**Test**: Start batch analysis of all chunks

Steps:
1. Navigate to Proposal Chunk Review V2 page
2. Select workspace from dropdown
3. Click "🚀 Start Batch Analysis" button
4. Observe progress bar
5. Check live log output

**Expected**:
- [ ] Progress bar animates from 0% to 100%
- [ ] Status text updates: "Analyzing chunk X/Y (Z%)"
- [ ] Log container shows timestamped messages
- [ ] Each chunk logs: "[HH:MM:SS] Chunk X/Y: Found N mentions"
- [ ] Final summary: "Analysis complete in X.Xs"
- [ ] Total mentions found displayed
- [ ] Auto-transitions to review mode

**Pass Criteria**:
- All chunks analyzed without errors
- Progress updates in real-time
- Log messages appear chronologically
- Total mentions count is correct
- No crashes or freezes

---

### 2. Compact Navigation ✅

**Test**: Navigate between chunks efficiently

Steps:
1. After batch analysis completes
2. Test "← Prev" button (should be disabled on Chunk 1)
3. Test "Next →" button
4. Test jump dropdown - select different chunk
5. Navigate to last chunk
6. Test "Next →" button (should be disabled)

**Expected**:
- [ ] Prev button disabled when current_chunk_id == 1
- [ ] Next button disabled when current_chunk_id == total_chunks
- [ ] Jump dropdown shows all chunks with status indicators (✅/⏳)
- [ ] Selected chunk loads instantly
- [ ] Chunk title displays correctly
- [ ] Chunk line numbers shown (start-end)

**Pass Criteria**:
- Navigation buttons work correctly
- Boundary conditions handled (first/last chunk)
- Jump dropdown accurately reflects review status
- No lag when switching chunks

---

### 3. Mention Review Cards ✅

**Test**: Review and approve/reject mentions

Steps:
1. Navigate to chunk with pending mentions
2. Examine mention card layout
3. Click "Show Context" expander
4. Click "✅ Approve" on first mention
5. Click "❌ Reject" on second mention
6. Reload page and verify persistence

**Expected**:
- [ ] Mention cards display clearly
- [ ] @mention text shown in monospace font
- [ ] Field path displayed
- [ ] Location shown (e.g., "Line 1666")
- [ ] Context preview shows ±3 lines with >>> marker
- [ ] Approve removes mention from pending list
- [ ] Reject removes mention from pending list
- [ ] Changes persist after page reload

**Pass Criteria**:
- All mention details visible
- Context preview accurate
- Approve/reject actions work immediately
- UI updates without full page reload (st.rerun works)
- YAML file updated correctly

---

### 4. Inline Editing ✅

**Test**: Edit mention bindings

Steps:
1. Navigate to chunk with pending mention
2. Click "✏️ Edit Binding" button
3. Modify mention_text field
4. Modify field_path field
5. Add resolved_value text
6. Click "💾 Save Changes"
7. Verify changes saved

**Expected**:
- [ ] Edit dialog expands when clicked
- [ ] Current values pre-filled in form
- [ ] Can modify mention_text
- [ ] Can modify field_path
- [ ] Can add/edit resolved_value
- [ ] "Save Changes" commits updates
- [ ] Dialog closes after save
- [ ] Modified flag set to True in YAML

**Test Edge Cases**:
- [ ] Cancel button closes without saving
- [ ] Empty fields handled gracefully
- [ ] Invalid field paths accepted (user responsibility)
- [ ] Special characters in mention_text handled

**Pass Criteria**:
- All fields editable
- Changes persist to YAML
- No data loss on cancel
- UI responsive

---

### 5. Auto-Progression ✅

**Test**: Automatic chunk advancement

Steps:
1. Navigate to chunk with 3 pending mentions
2. Approve all 3 mentions
3. Observe behavior when last mention approved

**Expected**:
- [ ] Success message: "✅ Chunk complete! Advancing to next chunk..."
- [ ] 1-second delay for user feedback
- [ ] Auto-advances to next chunk
- [ ] Chunk status changes from ⏳ to ✅
- [ ] chunks_reviewed counter increments
- [ ] last_reviewed_chunk_id updated

**Test Last Chunk**:
1. Navigate to final chunk
2. Approve all mentions
3. Observe completion behavior

**Expected**:
- [ ] Balloons animation appears 🎉
- [ ] Message: "🎉 All chunks completed!"
- [ ] Export button enabled (if implemented)
- [ ] No auto-advance (already at last chunk)

**Pass Criteria**:
- Smooth progression between chunks
- User gets feedback before auto-advance
- Completion celebration on final chunk
- Progress tracking accurate

---

### 6. Multi-Session Support ✅

**Test**: Resume functionality

Steps:
1. Start reviewing workspace
2. Review 5 chunks (approve mentions)
3. Close browser/tab
4. Reopen page after 10 minutes
5. Select same workspace

**Expected**:
- [ ] last_reviewed_chunk_id persisted in metadata
- [ ] last_reviewed_at timestamp saved
- [ ] Can manually navigate to any chunk
- [ ] Review progress preserved (chunks_reviewed count)
- [ ] Previously approved mentions stay approved

**Ideal Enhancement** (not yet implemented):
- [ ] "Resume where you left off" button
- [ ] Display: "Last reviewed: Chunk 5, 10 minutes ago"

**Pass Criteria**:
- Session state persists across browser sessions
- No data loss when closing/reopening
- Can continue from any point

---

### 7. Real-Time Logging ✅

**Test**: Live log display during batch analysis

Steps:
1. Start batch analysis
2. Watch log container during processing
3. Verify log messages appear in real-time

**Expected**:
- [ ] Log container scrollable (height: 300px)
- [ ] Timestamps format: [HH:MM:SS]
- [ ] Messages appear as chunks analyzed (not all at end)
- [ ] Last 20 messages shown (prevents bloat)
- [ ] Final summary includes duration

**Example Log Output**:
```
[14:32:01] Starting batch analysis of 28 chunks...
[14:32:01] LLM Model: qwen2.5:72b-instruct-q4_K_M
[14:32:03] Chunk 1/28 (4%): Cover Letter
[14:32:05] Chunk 1: Found 2 mentions
[14:32:06] Chunk 2/28 (7%): Company Information
...
[14:32:58] Analysis complete in 57.3s
[14:32:58] Total mentions found: 47
```

**Pass Criteria**:
- Messages timestamped correctly
- Real-time updates (not batch)
- Scrollable container works
- Final summary accurate

---

### 8. Error Handling ✅

**Test**: Graceful failure recovery

**Test Scenarios**:

1. **LLM unavailable**:
   - Stop Ollama service
   - Start batch analysis
   - Expected: Error logged, chunk gets empty mention list, continues

2. **Invalid mention text**:
   - Edit mention with invalid syntax
   - Expected: System accepts (user responsibility) or shows warning

3. **Missing entity profile**:
   - Select workspace without entity bound
   - Expected: Clear error message, stops before analysis

4. **Corrupted chunk**:
   - Chunk with no content
   - Expected: Logs warning, returns empty mentions, continues

**Pass Criteria**:
- No crashes on errors
- Clear error messages to user
- Batch analysis continues despite individual chunk failures
- Failed chunks clearly marked in logs

---

### 9. Performance ✅

**Test**: System responsiveness

**Metrics to Track**:

| Action | Expected Time | Actual Time | Pass? |
|--------|---------------|-------------|-------|
| Batch analysis (28 chunks) | < 90 seconds | _____ | [ ] |
| Navigation (prev/next) | < 100ms | _____ | [ ] |
| Approve mention | < 200ms | _____ | [ ] |
| Edit save | < 300ms | _____ | [ ] |
| Jump to chunk | < 150ms | _____ | [ ] |
| Page load | < 2 seconds | _____ | [ ] |

**Memory Usage**:
- [ ] No memory leaks during long sessions
- [ ] Log limited to last 20 entries
- [ ] Session state cleaned appropriately

---

### 10. Data Persistence ✅

**Test**: YAML file integrity

Steps:
1. Perform various operations (approve, reject, edit)
2. Check workspace metadata.yaml
3. Check field_bindings.yaml

**Expected in metadata.yaml**:
```yaml
metadata:
  analysis_status: complete
  analysis_progress: 100
  analysis_completed_at: '2026-01-07T14:33:00'
  total_mentions_found: 47
  chunks_reviewed: 5
  last_reviewed_chunk_id: 5
  last_reviewed_at: '2026-01-07T15:00:00'
  approved_mentions: 12
  rejected_mentions: 3
```

**Expected in field_bindings.yaml**:
```yaml
- mention_text: '@companyname'
  field_path: company.legal_name
  chunk_id: 3
  location: 'Attachment 1 (Line 1666)'
  approved: true
  modified: false
  reviewed_at: '2026-01-07T14:45:00'
```

**Pass Criteria**:
- All fields populated correctly
- Timestamps in ISO format
- Counters accurate
- Boolean flags correct

---

## Visual Design Tests

### 11. CSS & Styling ✅

**Test**: Professional appearance

**Check**:
- [ ] Main header styled (#2D5F4F green)
- [ ] Progress bar with gradient fill
- [ ] Mention cards have hover effect
- [ ] @mention text in monospace font (#C85D3C)
- [ ] Status badges color-coded
- [ ] Smooth transitions (0.2-0.3s)
- [ ] Responsive layout on different screen sizes

**Browser Compatibility**:
- [ ] Chrome/Edge
- [ ] Firefox
- [ ] Safari (if available)

---

## Integration Tests

### 12. End-to-End Workflow ✅

**Complete User Journey**:

1. **Setup** (5 min):
   - [ ] Create new workspace
   - [ ] Upload tender document
   - [ ] Bind entity profile

2. **Analysis** (2 min):
   - [ ] Click "Start Batch Analysis"
   - [ ] Wait for completion
   - [ ] Verify mentions found

3. **Review** (10-15 min):
   - [ ] Navigate through all chunks
   - [ ] Approve valid mentions
   - [ ] Reject irrelevant mentions
   - [ ] Edit 2-3 incorrect bindings
   - [ ] Verify context previews accurate

4. **Completion** (1 min):
   - [ ] Reach last chunk
   - [ ] See balloons celebration
   - [ ] Verify all chunks marked ✅

5. **Verification** (2 min):
   - [ ] Check YAML files updated
   - [ ] Reload page - state persists
   - [ ] Progress metrics accurate

**Total Time**: ~20-25 minutes for realistic tender document

**Pass Criteria**:
- Complete workflow without crashes
- All features work as expected
- Data persists correctly
- Professional UX throughout

---

## Known Limitations (Acceptable)

1. **Keyboard Shortcuts**: Not implemented (nice-to-have for v3.0)
2. **Bulk Actions**: No "approve all" button (can add if requested)
3. **Field Path Validation**: No real-time validation against entity schema
4. **Undo/Redo**: Not implemented
5. **Export**: Placeholder only (implementation in next phase)

---

## Bug Tracking

### Issues Found During Testing

| # | Description | Severity | Status | Fix |
|---|-------------|----------|--------|-----|
| 1 | _______________ | ___ | ___ | ___ |
| 2 | _______________ | ___ | ___ | ___ |
| 3 | _______________ | ___ | ___ | ___ |

**Severity Levels**:
- **Critical**: Blocks core functionality
- **High**: Major feature broken
- **Medium**: Minor feature issue
- **Low**: Cosmetic or edge case

---

## Test Sign-Off

**Tester**: _______________
**Date**: _______________
**Version Tested**: 2.0.0

**Overall Assessment**:
- [ ] **PASS** - Production ready
- [ ] **PASS with minor issues** - Can ship with known limitations
- [ ] **FAIL** - Requires fixes before shipping

**Notes**:
```
_________________________________________________________
_________________________________________________________
_________________________________________________________
```

---

## Deployment Checklist

After testing passes:

- [ ] Update version number to 2.0.0 in all files
- [ ] Update documentation (CHUNK_REVIEW_V2_COMPLETE.md)
- [ ] Commit all changes to Git
- [ ] Create Git tag: `v2.0.0-chunk-review`
- [ ] Sync to Docker distribution (if applicable)
- [ ] Update CHANGELOG.md
- [ ] Notify users of new features
- [ ] Monitor for issues in production

---

*Ready for professional tender response workflows!*
