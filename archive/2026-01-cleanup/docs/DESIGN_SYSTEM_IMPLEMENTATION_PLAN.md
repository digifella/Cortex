# Design System Implementation Plan
**Created:** 2026-01-01
**Status:** In Progress
**Goal:** Bring all 16 pages to 100% design system compliance

---

## 📊 Current State (Audit Results)

- **Total Pages:** 16
- **Fully Compliant:** 1 (6.2%)
- **Average Compliance:** 56.2%
- **Need Updates:** 15 pages

---

## 🎯 Implementation Strategy

### Phase 1: Critical Pages (Score < 50%) - 4 pages
Priority: **HIGHEST** - These pages have the most inconsistencies

1. **11_Knowledge_Synthesizer.py** (33.3%) - 100 lines
2. **1_AI_Assisted_Research.py** (33.3%) - 414 lines
3. **Proposal_Copilot.py** (33.3%) - 442 lines
4. **13_Metadata_Management.py** (41.7%) - 199 lines

**Estimated Time:** 2-3 hours total

### Phase 2: Moderate Pages (50-65%) - 6 pages
Priority: **HIGH** - Need significant updates

5. **4_Proposal_Step_1_Prep.py** (50.0%) - 281 lines
6. **5_Proposal_Step_2_Make.py** (50.0%) - 146 lines
7. **12_Visual_Analysis.py** (58.3%) - 577 lines
8. **3_Knowledge_Search.py** (58.3%) - 1735 lines ⚠️ **Large file**
9. **4_Collection_Management.py** (58.3%) - 1157 lines ⚠️ **Large file**
10. **6_Knowledge_Analytics.py** (58.3%) - 821 lines

**Estimated Time:** 3-4 hours total

### Phase 3: Good Pages (66-75%) - 4 pages
Priority: **MEDIUM** - Polish and minor updates

11. **8_Document_Anonymizer.py** (58.3%) - 403 lines
12. **9_Document_Summarizer.py** (58.3%) - 407 lines
13. **10_Idea_Generator.py** (66.7%) - 1001 lines
14. **7_Maintenance.py** (66.7%) - 2116 lines ⚠️ **Large file**

**Estimated Time:** 2-3 hours total

### Phase 4: Nearly Perfect (75-95%) - 1 page
Priority: **LOW** - Minor touch-ups

15. **2_Knowledge_Ingest.py** (75.0%) - 3843 lines ⚠️ **Largest file**

**Estimated Time:** 30-60 minutes

---

## ✅ Common Fixes Needed

### Template for All Pages

```python
# 1. Add at top after imports
from cortex_engine.ui_theme import apply_theme, section_header
from cortex_engine.ui_components import error_display, render_version_footer
from cortex_engine.utils.logging_utils import get_logger

logger = get_logger(__name__)

# 2. After st.set_page_config()
apply_theme()

# 3. Update page config to include icon
st.set_page_config(
    page_title="Page Name",
    page_icon="🎯",  # Choose appropriate emoji
    layout="wide"
)

# 4. Add navigation caption after title
st.caption("💡 Use the sidebar (←) to navigate between pages")
st.markdown("---")

# 5. Replace st.header() with section_header()
# OLD: st.header("Section Name")
# NEW: section_header("🔍", "Section Name", "Optional subtitle")

# 6. Replace error handling with error_display()
# OLD: st.error(f"Error: {e}")
# NEW: error_display(str(e), "Error Type", "Recovery suggestion")

# 7. Add version footer at bottom
render_version_footer()
```

---

## 📋 Update Checklist (Per Page)

For each page update:

- [ ] Read current implementation
- [ ] Add missing imports (theme, components, logger)
- [ ] Call `apply_theme()` after page config
- [ ] Add page icon to config
- [ ] Add navigation caption
- [ ] Add horizontal divider after header
- [ ] Replace headers with `section_header()`
- [ ] Replace error messages with `error_display()`
- [ ] Add `render_version_footer()` at bottom
- [ ] Add help text to inputs (if missing)
- [ ] Add logger for errors
- [ ] Test page functionality
- [ ] Run audit script to verify 100% compliance
- [ ] Commit changes with descriptive message
- [ ] Push to git

---

## 🎯 Success Criteria

**Per Page:**
- Design audit score: 100%
- All 12 checks passing
- No functionality regressions
- Improved UX consistency

**Overall Project:**
- All 16 pages at 100% compliance
- Consistent user experience across entire app
- Maintainable codebase with standardized patterns
- Updated documentation

---

## 📝 Commit Message Template

```
refactor: Update [Page Name] to design system v1.0.0

## Changes
- ✅ Added apply_theme() call
- ✅ Added page icon to config
- ✅ Added navigation caption
- ✅ Replaced headers with section_header()
- ✅ Replaced errors with error_display()
- ✅ Added version footer
- ✅ Added logging for errors
- ✅ Added help text to inputs

## Compliance
- Before: XX.X%
- After: 100.0%
- Status: ✅ All checks passing

🎨 Design system compliance: 100%

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
```

---

## 🚀 Execution Plan

**Today's Goal:** Complete Phase 1 (4 critical pages)

1. Start with smallest/simplest pages first
2. Update one page at a time
3. Test after each update
4. Commit after each successful update
5. Document any issues or patterns discovered
6. Create reusable code snippets for common updates

---

## 📊 Progress Tracking

### Phase 1: Critical Pages
- [ ] 11_Knowledge_Synthesizer.py (100 lines)
- [ ] 13_Metadata_Management.py (199 lines)
- [ ] 4_Proposal_Step_1_Prep.py (281 lines)
- [ ] 1_AI_Assisted_Research.py (414 lines)
- [ ] Proposal_Copilot.py (442 lines)

### Phase 2: Moderate Pages
- [ ] 5_Proposal_Step_2_Make.py (146 lines)
- [ ] 12_Visual_Analysis.py (577 lines)
- [ ] 6_Knowledge_Analytics.py (821 lines)
- [ ] 4_Collection_Management.py (1157 lines)
- [ ] 3_Knowledge_Search.py (1735 lines)

### Phase 3: Good Pages
- [ ] 8_Document_Anonymizer.py (403 lines)
- [ ] 9_Document_Summarizer.py (407 lines)
- [ ] 10_Idea_Generator.py (1001 lines)
- [ ] 7_Maintenance.py (2116 lines)

### Phase 4: Nearly Perfect
- [ ] 2_Knowledge_Ingest.py (3843 lines)

---

**Last Updated:** 2026-01-01
**Next Review:** After Phase 1 completion
