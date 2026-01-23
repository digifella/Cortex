# Implementation Plan: Per-Question Source Selector

**Feature:** Multi-source evidence selection for Intelligent Completion questions
**Status:** Planning
**Date:** 2026-01-23
**Version Target:** 3.0.0

---

## Overview

Add a per-question source selector to the Intelligent Completion workflow, allowing users to choose where evidence comes from for each question. Inspired by NotebookLM's source input interface.

## Source Types

| Source Type | Description | Implementation |
|-------------|-------------|----------------|
| **Collection** | Existing knowledge collection | Use `EvidenceRetriever` with `collection_name` |
| **Upload File** | PDF/DOCX/TXT uploaded for this question | Use existing ingestion pipeline, save to collection |
| **Web Link** | URL to fetch content from | Use `requests` + `BeautifulSoup` for text extraction |
| **Paste Text** | Direct text input | Use as raw evidence context |

---

## Current State (v2.9.0)

### What We Have
- `EvidenceRetriever` - searches collections for relevant evidence
- `DocumentProcessor` - extracts text from PDF/DOCX/TXT files
- `WorkingCollectionManager` - manages knowledge collections
- `requests` + `beautifulsoup4` - available for web scraping
- Knowledge Ingest page - full ingestion pipeline

### Files Involved
- `pages/Proposal_Intelligent_Completion.py` - main UI (to modify)
- `cortex_engine/evidence_retriever.py` - evidence search (may extend)
- `cortex_engine/document_processor.py` - file text extraction (reuse)
- `cortex_engine/collection_manager.py` - collection management (reuse)

---

## UI Design

### Per-Question Source Selector

```
┌─────────────────────────────────────────────────────────────────┐
│ Question 1                                          PENDING     │
│ Describe your experience with similar projects...               │
├─────────────────────────────────────────────────────────────────┤
│ Source: ○ Collection  ○ Upload  ○ Web Link  ○ Paste Text       │
│                                                                 │
│ [When Collection selected]                                      │
│ Collection: [-- Entire Knowledge Base -- ▼]                     │
│                                                                 │
│ [When Upload selected]                                          │
│ [📁 Choose file] PDF, DOCX, TXT supported                       │
│ Save to: [Create new collection ▼] [Collection name: ____]      │
│                                                                 │
│ [When Web Link selected]                                        │
│ URL: [https://...                    ] [Fetch]                  │
│ ℹ️ Extracts visible text content only                           │
│                                                                 │
│ [When Paste Text selected]                                      │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ Paste your reference text here...                           │ │
│ └─────────────────────────────────────────────────────────────┘ │
│                                                                 │
│ [Edit]  [Balanced ▼]  [Generate]                                │
└─────────────────────────────────────────────────────────────────┘
```

### Model Capability Check

If PDF/DOCX processing requires unavailable model capabilities:
- Grey out "Upload" option
- Show tooltip: "Document processing requires [model]. Only text input available."

---

## Implementation Details

### 1. New Module: `cortex_engine/ic_source_manager.py`

```python
class ICSourceManager:
    """Manages per-question evidence sources for Intelligent Completion."""

    def __init__(self, db_path: Path, workspace_id: str):
        self.db_path = db_path
        self.workspace_id = workspace_id
        self.collection_manager = WorkingCollectionManager()
        self.document_processor = DocumentProcessor()

    def get_evidence_from_collection(
        self, question: str, collection_name: Optional[str], ...
    ) -> List[Evidence]:
        """Retrieve evidence from knowledge collection."""

    def process_uploaded_file(
        self, uploaded_file, save_to_collection: str
    ) -> Tuple[str, List[Evidence]]:
        """Process uploaded file, optionally save to collection, return evidence."""

    def fetch_web_content(self, url: str) -> Tuple[str, str]:
        """Fetch and extract text from URL. Returns (title, text)."""

    def create_evidence_from_text(
        self, text: str, source_name: str
    ) -> List[Evidence]:
        """Create Evidence objects from raw text input."""

    @staticmethod
    def check_upload_capability() -> Tuple[bool, str]:
        """Check if file upload/processing is available. Returns (available, reason)."""
```

### 2. Web Content Fetcher

```python
def fetch_web_content(url: str, timeout: int = 10) -> Tuple[str, str]:
    """
    Fetch visible text content from URL.

    Returns:
        Tuple of (page_title, extracted_text)

    Raises:
        ValueError: If URL invalid or fetch fails
    """
    import requests
    from bs4 import BeautifulSoup

    response = requests.get(url, timeout=timeout, headers={
        'User-Agent': 'Cortex-Suite/1.0 (Knowledge Assistant)'
    })
    response.raise_for_status()

    soup = BeautifulSoup(response.text, 'html.parser')

    # Remove script/style elements
    for element in soup(['script', 'style', 'nav', 'footer', 'header']):
        element.decompose()

    title = soup.title.string if soup.title else url
    text = soup.get_text(separator='\n', strip=True)

    return title, text
```

### 3. Session State Structure

```python
# Per-question source configuration
st.session_state.ic_question_sources = {
    "field_text_here": {
        "source_type": "collection" | "upload" | "weblink" | "paste",
        "collection_name": str | None,
        "uploaded_file_name": str | None,
        "uploaded_text": str | None,
        "web_url": str | None,
        "web_content": str | None,
        "pasted_text": str | None,
    }
}
```

### 4. Persistence Updates

Add to `ic_persistence_model.py`:
```python
class PersistedSourceConfig(BaseModel):
    source_type: str  # 'collection', 'upload', 'weblink', 'paste'
    collection_name: Optional[str] = None
    uploaded_file_name: Optional[str] = None
    web_url: Optional[str] = None
    pasted_text: Optional[str] = None  # Only for paste, others too large
```

Add to `ICCompletionState`:
```python
question_sources: Dict[str, PersistedSourceConfig] = Field(default_factory=dict)
```

---

## File Upload Flow

```
User uploads file
        ↓
Check file type (PDF/DOCX/TXT)
        ↓
Extract text using DocumentProcessor
        ↓
User chooses: [New Collection] or [Existing Collection] or [Workspace Only]
        ↓
If Collection: Call knowledge ingest pipeline (chunking, embedding)
If Workspace: Store extracted text in workspace temp storage
        ↓
Use extracted text as evidence context for generation
```

---

## Implementation Phases

### Phase 1: Core Infrastructure
1. Create `cortex_engine/ic_source_manager.py`
2. Add `fetch_web_content()` function
3. Add capability check for file processing
4. Update persistence models

### Phase 2: UI Components
1. Add source type radio buttons
2. Add conditional UI for each source type
3. Add collection dropdown for Collection source
4. Add file uploader for Upload source
5. Add URL input + fetch button for Web Link source
6. Add text area for Paste Text source

### Phase 3: Integration
1. Wire source selection to evidence retrieval
2. Handle file upload → collection ingestion flow
3. Handle web fetch → evidence creation
4. Handle paste text → evidence creation
5. Update Generate button to use selected source

### Phase 4: Polish
1. Add loading states for file processing and web fetching
2. Add error handling and user feedback
3. Grey out unavailable options with explanatory tooltips
4. Persist source selections per question
5. Sync to Docker distribution

---

## Dependencies

### Existing (No Changes Needed)
- `requests` - HTTP requests
- `beautifulsoup4` - HTML parsing
- `python-docx` - DOCX processing
- `PyMuPDF (fitz)` - PDF processing

### New Imports in IC Page
```python
from cortex_engine.ic_source_manager import ICSourceManager
from cortex_engine.document_processor import DocumentProcessor
```

---

## Risk Assessment

| Risk | Mitigation |
|------|------------|
| Large file uploads blocking UI | Use progress spinner, limit file size |
| Web fetch timeout/failure | Clear error message, suggest alternatives |
| PDF/DOCX extraction fails | Fallback error with "try TXT" suggestion |
| Model not available for processing | Grey out option, show capability message |

---

## Testing Checklist

- [ ] Collection source works (existing functionality)
- [ ] File upload extracts text correctly (PDF, DOCX, TXT)
- [ ] File can be saved to new collection
- [ ] File can be saved to existing collection
- [ ] Web URL fetches and extracts text
- [ ] Invalid URL shows clear error
- [ ] Paste text works as evidence
- [ ] Source selection persists across page reloads
- [ ] Greyed options show correct messages
- [ ] Generate uses selected source correctly

---

## Estimated Scope

- **New files:** 1 (`ic_source_manager.py`)
- **Modified files:** 3 (`Proposal_Intelligent_Completion.py`, `ic_persistence_model.py`, `workspace_manager.py`)
- **Lines of code:** ~400-500 new/modified
- **Complexity:** Medium-High

---

## Approval

- [ ] User approves plan
- [ ] Ready to implement

