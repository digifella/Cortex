# Chunk Review V2: Professional Redesign

**Date**: 2026-01-07
**Status**: 🎯 Design Phase

## Problem Analysis

### Current UX Issues
1. **Manual Labor**: User must click "Analyze This Chunk" 28 times
2. **Verbose UI**: Long chunk grid showing all 28 chunks in sidebar
3. **No Batch Processing**: Analysis happens one-at-a-time, synchronously
4. **No Editing**: Can't edit field bindings (field_path, mention_text, etc.)
5. **Single Session**: Must complete review in one sitting
6. **Unprofessional**: Doesn't match workflow of busy consultant reviewing tenders

### User's Actual Need
> "I want something useful to a professional who would need to complete such a tender"

**Professional Workflow**:
1. Upload tender document
2. System processes everything in background
3. Get notified when ready
4. **Review in multiple sessions** at convenience
5. **Snappy navigation**: Jump around efficiently
6. **Edit bindings** directly when LLM got it wrong
7. Export when satisfied

---

## Redesigned Workflow

### Phase 1: Auto-Batch Analysis (Background)

```
User uploads document
  ↓
System automatically:
  • Chunks document (instant)
  • Starts batch analysis of ALL chunks (background)
  • Shows progress: "Analyzing chunk 8/28 (35% complete, ~4 min remaining)"
  ↓
User can navigate away, do other work
  ↓
Notification: "Analysis complete! 47 mentions found across 28 chunks"
```

**Implementation**:
- `analyze_all_chunks()` method in MarkupEngine
- Background processing with progress tracking
- Store progress in workspace metadata
- Streamlit auto-refresh to show live progress

---

### Phase 2: Professional Review Interface

**Compact Navigation**:
```
┌─────────────────────────────────────────────────────┐
│  Chunk 3 of 28: Tenderer's Details     [▓▓▓░░░] 35% │
│  ← Prev  |  Jump: [▼ Select chunk...]  |  Next →    │
└─────────────────────────────────────────────────────┘
```

**Mention Review Cards** (Clean, Scannable):
```
┌──────────────────────────────────────────────────────┐
│ @companyname                                    [✓][✗]│
│ ┌──────────────────────────────────────────────────┐ │
│ │ Field: company.legal_name            [Edit Path] │ │
│ │ Location: Attachment 1, Line 1666                │ │
│ │ Suggested Value: [Edit inline...]                │ │
│ └──────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────┘
```

**Quick Actions**:
- `A` key = Approve
- `R` key = Reject
- `E` key = Edit
- `→` key = Next mention
- `Ctrl+→` = Next chunk

**Session Resume**:
- "Last reviewed: Chunk 12, 3 hours ago"
- "Resume where you left off" button

---

### Phase 3: Inline Editing

**Editable Fields**:
1. **mention_text**: Change `@companyname` to `@company_legal_name`
2. **field_path**: Change `company.legal_name` to `company.trading_name`
3. **resolved_value**: Pre-fill or edit the actual value
4. **location**: Adjust if needed

**Edit Modal**:
```
┌─────────────────────────────────────┐
│  Edit Mention Binding               │
├─────────────────────────────────────┤
│  Mention Text:                      │
│  [@companyname____________]         │
│                                     │
│  Field Path:                        │
│  [company.legal_name______]         │
│                                     │
│  Resolved Value (optional):         │
│  [Longboardfella Consulting]        │
│                                     │
│  [ Cancel ]  [ Save Changes ]       │
└─────────────────────────────────────┘
```

---

## Technical Architecture

### Backend Changes

**1. Batch Analysis Engine**
```python
# cortex_engine/markup_engine.py

def analyze_all_chunks_batch(
    self,
    chunks: List[DocumentChunk],
    entity_id: str,
    progress_callback: Optional[Callable] = None
) -> Dict[int, List[MentionBinding]]:
    """
    Analyze all chunks in batch mode.

    Returns:
        Dict mapping chunk_id -> list of mentions
    """
    results = {}
    total = len(chunks)

    for idx, chunk in enumerate(chunks):
        mentions = self.analyze_chunk(chunk, entity_id)
        results[chunk.chunk_id] = mentions

        if progress_callback:
            progress_callback(idx + 1, total)

    return results
```

**2. Edit Mention Binding**
```python
# cortex_engine/workspace_manager.py

def edit_mention_binding(
    self,
    workspace_id: str,
    mention_id: str,  # Use unique ID (chunk_id + location + mention_text)
    new_mention_text: Optional[str] = None,
    new_field_path: Optional[str] = None,
    new_resolved_value: Optional[str] = None
) -> Workspace:
    """Allow editing of mention bindings."""
```

**3. Progress Tracking**
```python
# Add to WorkspaceMetadata
class WorkspaceMetadata(BaseModel):
    # Analysis progress
    analysis_status: str = "pending"  # pending, analyzing, complete, failed
    analysis_progress: int = 0  # 0-100%
    analysis_started_at: Optional[datetime] = None
    analysis_completed_at: Optional[datetime] = None
    total_mentions_found: int = 0

    # Review progress
    last_reviewed_chunk_id: Optional[int] = None
    last_reviewed_at: Optional[datetime] = None
```

---

### Frontend Changes

**1. Compact Chunk Navigator**
```python
# Replace verbose chunk grid with:
col1, col2, col3 = st.columns([1, 3, 1])

with col1:
    if st.button("← Prev"):
        # Go to previous unreviewed chunk

with col2:
    # Dropdown with status indicators
    chunk_options = [
        f"{'✅' if reviewed else '⏳'} Chunk {i}: {title}"
        for i, title, reviewed in chunks
    ]
    selected = st.selectbox("Jump to", chunk_options)

with col3:
    if st.button("Next →"):
        # Go to next unreviewed chunk
```

**2. Mention Cards (Not Expanders)**
```python
# Replace st.expander with clean cards
for mention in pending_mentions:
    st.markdown(f"""
    <div class="mention-card">
        <div class="mention-header">
            <span class="mention-text">{mention.mention_text}</span>
            <div class="actions">
                <button class="approve">✓</button>
                <button class="reject">✗</button>
                <button class="edit">Edit</button>
            </div>
        </div>
        <div class="mention-details">
            <span>Field: {mention.field_path}</span>
            <span>Location: {mention.location}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
```

**3. Edit Dialog**
```python
# Use st.dialog (Streamlit 1.31+) or modal
@st.dialog("Edit Mention")
def edit_mention_dialog(mention):
    new_text = st.text_input("Mention Text", value=mention.mention_text)
    new_path = st.text_input("Field Path", value=mention.field_path)
    new_value = st.text_area("Resolved Value", value=mention.resolved_value or "")

    if st.button("Save Changes"):
        # Update binding
        st.rerun()
```

---

## Implementation Plan

### Step 1: Batch Analysis Backend (30 min)
- [ ] Add `analyze_all_chunks_batch()` to MarkupEngine
- [ ] Add progress tracking to WorkspaceMetadata
- [ ] Add analysis status fields to workspace
- [ ] Test batch analysis with 28 chunks

### Step 2: Auto-Start Analysis (15 min)
- [ ] Trigger batch analysis on workspace load
- [ ] Show progress indicator during analysis
- [ ] Store results in workspace mentions list
- [ ] Update chunk progress automatically

### Step 3: Edit Capabilities (45 min)
- [ ] Add `edit_mention_binding()` to WorkspaceManager
- [ ] Create edit dialog UI component
- [ ] Wire up edit buttons
- [ ] Test editing field_path, mention_text, resolved_value

### Step 4: Compact Navigation (30 min)
- [ ] Replace verbose chunk grid with compact controls
- [ ] Add keyboard shortcuts (←/→ arrow keys)
- [ ] Add "Resume" functionality
- [ ] Improve visual design

### Step 5: Professional Polish (30 min)
- [ ] Clean CSS design (editorial aesthetic from chunk_navigator.html)
- [ ] Smooth animations
- [ ] Loading states
- [ ] Success notifications

**Total Estimated Time**: ~2.5 hours

---

## Success Criteria

✅ **Batch Processing**: All 28 chunks analyzed automatically in background
✅ **Fast Navigation**: Jump to any chunk, prev/next with keyboard
✅ **Inline Editing**: Edit mention_text, field_path, resolved_value
✅ **Multi-Session**: Can stop and resume review anytime
✅ **Professional UX**: Clean, efficient, consultant-friendly
✅ **YAML Editable**: All bindings editable through UI

---

## Open Questions

1. **Background vs Foreground**: Should analysis run truly in background (async) or show progress bar?
   - **Decision**: Show progress bar with "analyzing..." message, user can navigate away

2. **Edit Validation**: Should we validate field_path against entity schema?
   - **Decision**: Yes, show warning if path doesn't exist in entity profile

3. **Keyboard Shortcuts**: Implement now or later?
   - **Decision**: Start with buttons, add keyboard later if needed

4. **Resume Strategy**: Auto-resume or "Resume" button?
   - **Decision**: Show "Last reviewed: X" + "Resume" button

---

## Next Steps

**Immediate**: Get user approval on this design
**Then**: Implement Step 1 (Batch Analysis Backend)
**Goal**: Professional, efficient review experience for tender consultants
