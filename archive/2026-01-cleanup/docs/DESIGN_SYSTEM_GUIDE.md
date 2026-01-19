# Cortex Suite Design System Guide
**Version:** 1.0.0
**Date:** 2026-01-01
**Purpose:** Consistent UI/UX across all pages for maintainability and user experience

---

## 🎨 Design Philosophy

**"Editorial Clarity with Professional Precision"**

The Cortex Suite uses a refined editorial design system inspired by high-end publications and research journals, conveying authority, intelligence, and professionalism.

### Core Principles
1. **Consistency**: Same patterns, same results across all pages
2. **Clarity**: Clear hierarchy, obvious actions, predictable behavior
3. **Efficiency**: Minimal clicks, fast access to functions
4. **Sophistication**: Professional aesthetics without being sterile
5. **Maintainability**: Centralized components, easy updates

---

## 📁 File Structure

```
cortex_suite/
├── cortex_engine/
│   ├── ui_theme.py              # Base theme (colors, typography, CSS)
│   ├── ui_components.py          # Reusable components (selectors, displays)
│   └── version_config.py         # Centralized version info
└── pages/
    ├── 1_Universal_Knowledge_Assistant.py  # ✓ Updated to standard
    ├── 2_Knowledge_Ingest.py               # → Needs update
    ├── 3_Knowledge_Search.py               # → Needs update
    └── ...                                  # → All need review
```

---

## 🏗️ Standard Page Structure

Every page should follow this structure:

```python
"""
Page Title - Brief Description
Version: X.Y.Z
Date: YYYY-MM-DD
Purpose: What this page does
"""

import streamlit as st
import sys
from pathlib import Path

# Set page config FIRST (before any other Streamlit commands)
st.set_page_config(
    page_title="Page Title",
    page_icon="🎯",  # Unique emoji for each page
    layout="wide"
)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import theme and components
from cortex_engine.ui_theme import apply_theme, section_header
from cortex_engine.ui_components import (
    llm_provider_selector,
    collection_selector,
    error_display,
    render_version_footer
)
from cortex_engine.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Apply theme IMMEDIATELY
apply_theme()

# ============================================
# PAGE HEADER
# ============================================

st.title("🎯 Page Title")
st.markdown("""
**Brief description** of what this page does and why it's useful.
Optional second line for additional context.
""")

st.caption("💡 Use the sidebar (←) to navigate between pages")
st.markdown("---")

# ============================================
# SIDEBAR CONFIGURATION
# ============================================

with st.sidebar:
    st.header("⚙️ Configuration")

    # Standard collection selector (if applicable)
    from cortex_engine.collection_manager import WorkingCollectionManager
    try:
        collection_manager = WorkingCollectionManager()
        selected_collection = collection_selector(
            collection_manager,
            key_prefix="page_name"
        )
    except Exception as e:
        error_display(str(e), "Collection Loading Error")
        selected_collection = None

    st.divider()

    # Standard LLM provider selector (if applicable)
    llm_provider, status_info = llm_provider_selector(
        task_type="research",  # or "ideation", "knowledge", "proposals"
        key_prefix="page_name"
    )

    st.divider()

    # Page-specific configuration
    # Add your custom settings here

# ============================================
# MAIN CONTENT
# ============================================

section_header("🔍", "Main Section", "Description of this section")

# Your page functionality here

# ============================================
# FOOTER
# ============================================

render_version_footer()
```

---

## 🎯 Component Usage Guide

### 1. **Theme Application**
**Always call first!**

```python
from cortex_engine.ui_theme import apply_theme
apply_theme()  # Call immediately after page config
```

### 2. **Section Headers**
**Use instead of plain st.header()**

```python
from cortex_engine.ui_theme import section_header

# With subtitle
section_header("🔍", "Search Results", "Found 10 documents")

# Without subtitle
section_header("📊", "Analytics")
```

### 3. **Status Messages**
**Standardized feedback**

```python
from cortex_engine.ui_components import error_display

# Simple error
error_display(
    "File not found",
    error_type="File Error",
    recovery_suggestion="Check the file path and try again"
)

# With details
error_display(
    "Database connection failed",
    error_type="Database Error",
    recovery_suggestion="Restart the database service",
    show_details=True
)
```

### 4. **Collection Selector**
**Standardized collection selection**

```python
from cortex_engine.ui_components import collection_selector
from cortex_engine.collection_manager import WorkingCollectionManager

collection_manager = WorkingCollectionManager()
selected = collection_selector(
    collection_manager,
    key_prefix="unique_page_id",  # Important for multiple selectors
    required=True  # False allows "None" option
)
```

### 5. **LLM Provider Selector**
**Standardized model selection**

```python
from cortex_engine.ui_components import llm_provider_selector

provider, status = llm_provider_selector(
    task_type="research",  # research, ideation, knowledge, proposals
    key_prefix="unique_page_id"
)

if status["status"] != "ready":
    st.error(f"LLM not ready: {status['message']}")
    st.stop()
```

### 6. **Version Footer**
**Always include at bottom**

```python
from cortex_engine.ui_components import render_version_footer

# At end of page
render_version_footer()  # Includes divider
# or
render_version_footer(show_divider=False)  # No divider
```

---

## 🎨 Typography Standards

### Headings
```python
# Page title (H1) - automatic from st.title()
st.title("🎯 Page Title")

# Major sections (H2) - use section_header()
section_header("📊", "Section Title", "Optional subtitle")

# Subsections (H3) - use st.subheader()
st.subheader("Subsection Title")

# Minor sections - use markdown
st.markdown("### Minor Section")
```

### Body Text
```python
# Regular text - use markdown for better control
st.markdown("""
Regular paragraph text with **bold** and *italic* emphasis.
Use markdown for consistent formatting.
""")

# Captions (small text)
st.caption("This is a caption or help text")

# Code blocks
st.code("""
code here
""", language="python")
```

---

## 🎭 Icon Standards

Use consistent emojis for page icons and section headers:

### Page Icons (in page_config)
- 🧠 **Universal Knowledge Assistant** - Knowledge work
- 📥 **Knowledge Ingest** - Input/upload
- 🔍 **Knowledge Search** - Search/query
- 📚 **Collection Management** - Organization
- 📝 **Proposal Tools** - Writing/creation
- 💡 **Idea Generator** - Innovation/creativity
- 📊 **Analytics** - Data/metrics
- ⚙️ **Settings/Maintenance** - Configuration
- 🛠️ **Tools/Utilities** - Helper functions

### Section Icons
- 🔍 Search
- 📊 Results/Analytics
- 📁 Files/Documents
- ⚙️ Configuration/Settings
- 💡 Tips/Information
- ✅ Success
- ❌ Error
- ⚠️ Warning
- 🔄 Processing
- 📥 Download/Export

---

## 🎨 Color Usage

The theme uses a professional editorial palette:

### Primary (Navy Blue) - Authority
- Buttons, links, key UI elements
- `COLORS['primary']['500']` = #374F77

### Secondary (Terracotta) - Warmth
- Accents, hover states
- `COLORS['secondary']['500']` = #B6704F

### Accent (Sage Green) - Success
- Success messages, confirmations
- `COLORS['accent']['500']` = #5B8A66

### Semantic Colors
```python
from cortex_engine.ui_theme import COLORS

# In custom HTML/CSS
st.markdown(f"""
<div style="color: {COLORS['primary']['700']};">
    Styled text
</div>
""", unsafe_allow_html=True)
```

---

## 📝 Form Patterns

### Simple Input Form
```python
# Use native Streamlit components
user_input = st.text_area(
    "Enter your query",
    placeholder="What would you like to explore?",
    height=120,
    help="Enter any question or topic"
)

# Action buttons
col1, col2 = st.columns([1, 1])
with col1:
    submit = st.button("🚀 Generate", type="primary", use_container_width=True)
with col2:
    clear = st.button("🗑️ Clear", use_container_width=True)
```

### Complex Form with Validation
```python
with st.form("my_form"):
    # Inputs
    name = st.text_input("Name", help="Enter your name")
    option = st.selectbox("Choose", ["A", "B", "C"])

    # Submit
    submitted = st.form_submit_button("Submit", use_container_width=True)

    if submitted:
        if not name:
            st.error("❌ Name is required")
        else:
            st.success("✅ Form submitted!")
```

---

## 📊 Data Display Patterns

### Metrics
```python
# Use columns for metrics
col1, col2, col3 = st.columns(3)
col1.metric("Documents", 1234, delta="+50")
col2.metric("Collections", 5)
col3.metric("Queries", 89, delta="+12")
```

### Results Cards
```python
# Standard result display
with st.container():
    st.subheader("Document Title")
    st.caption("Author: John Doe | Date: 2024-01-01")
    st.write("Document excerpt or content...")

    # Optional expandable content
    with st.expander("View full content"):
        st.write("Full content here...")

    st.divider()
```

### Tables
```python
import pandas as pd

# DataFrame display
df = pd.DataFrame(data)
st.dataframe(df, use_container_width=True)

# Or with more control
st.table(df)  # Static table
```

---

## ⚡ Loading States

### Simple Spinner
```python
with st.spinner("Processing..."):
    # Do work
    result = process_data()
```

### Status Updates
```python
with st.status("Processing query...", expanded=True) as status:
    status.write("Step 1: Loading data...")
    load_data()

    status.write("Step 2: Analyzing...")
    analyze()

    status.update(label="✅ Complete!", state="complete")
```

### Progress Bar
```python
progress_bar = st.progress(0)
for i in range(100):
    # Do work
    progress_bar.progress(i + 1, text=f"Processing... {i+1}%")
```

---

## 🚨 Error Handling Patterns

### Graceful Error Display
```python
try:
    result = risky_operation()
except FileNotFoundError as e:
    error_display(
        "File not found",
        error_type="File Error",
        recovery_suggestion="Check the file path and try again",
        show_details=True
    )
except Exception as e:
    error_display(
        str(e),
        error_type="Unexpected Error",
        recovery_suggestion="Please report this issue",
        show_details=True
    )
    logger.error(f"Error in operation: {e}", exc_info=True)
```

### Non-blocking Warnings
```python
if risky_condition:
    st.warning("⚠️ **Warning**: This operation may take a while")
    if st.button("Continue anyway"):
        proceed()
```

---

## 📥 Export Patterns

### Simple Download
```python
# Markdown export
st.download_button(
    "📥 Download Report",
    data=markdown_content,
    file_name="report.md",
    mime="text/markdown"
)
```

### Multiple Format Export
```python
from cortex_engine.ui_components import export_buttons

export_buttons(
    data={"content": result},
    filename_prefix="knowledge_report",
    export_types=["markdown", "json"]
)
```

---

## 🔄 State Management

### Session State Best Practices
```python
# Initialize at top of page
if 'query_history' not in st.session_state:
    st.session_state.query_history = []

# Use throughout page
st.session_state.query_history.append(new_query)

# Clear when needed
if st.button("Clear History"):
    st.session_state.query_history = []
    st.rerun()
```

---

## ✅ Page Checklist

Before committing a page update, verify:

- [ ] `apply_theme()` called immediately after page config
- [ ] Page icon set in `st.set_page_config()`
- [ ] Consistent header with title + description
- [ ] Navigation hint included (`st.caption`)
- [ ] Horizontal divider after header (`st.markdown("---")`)
- [ ] Sidebar uses standard configuration patterns
- [ ] Section headers use `section_header()` function
- [ ] Error messages use `error_display()` or standard st.error()
- [ ] Version footer included at bottom
- [ ] All user inputs have help text
- [ ] Loading states shown for slow operations
- [ ] Success/error feedback for all actions
- [ ] Export functionality where applicable
- [ ] Mobile-friendly (wide layout for desktop, responsive for mobile)
- [ ] Logging added for errors and important events

---

## 🎯 Next Steps

### For New Pages
1. Copy the standard page structure template
2. Customize for your use case
3. Follow the component usage guide
4. Test thoroughly
5. Run the checklist

### For Existing Pages
1. Read current page implementation
2. Identify inconsistencies with this guide
3. Update incrementally (header → body → footer)
4. Test after each change
5. Commit when verified

---

## 📚 Reference

- **Theme Details**: `cortex_engine/ui_theme.py`
- **Component Library**: `cortex_engine/ui_components.py`
- **Version Management**: `cortex_engine/version_config.py`
- **Example Page**: `pages/1_Universal_Knowledge_Assistant.py`

---

**Last Updated:** 2026-01-01
**Maintained By:** Cortex Suite Team
