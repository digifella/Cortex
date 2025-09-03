import re
from typing import Dict, List


class DocumentTypeManager:
    """Lightweight document type manager for Docker distribution.
    Provides in-memory categories, simple type suggestions, and basic editing.
    """

    def __init__(self):
        # Basic defaults; can be extended via UI during runtime (non-persistent)
        self._categories: Dict[str, Dict] = {
            "Business": {
                "description": "Client-facing and internal business docs",
                "types": [
                    "Proposal/Quote",
                    "Contract/SOW",
                    "Meeting Minutes",
                    "Email Correspondence",
                    "Financial Report",
                ],
            },
            "Technical": {
                "description": "Project and engineering documents",
                "types": [
                    "Project Plan",
                    "Technical Documentation",
                    "Final Report",
                    "Draft Report",
                    "Presentation",
                    "Research Paper",
                    "Image/Diagram",
                ],
            },
            "Other": {
                "description": "Uncategorized or miscellaneous",
                "types": ["Other"],
            },
        }

        # Compile simple patterns for suggestions
        self._patterns = [
            (re.compile(r"proposal|quote", re.I), "Proposal/Quote"),
            (re.compile(r"contract|sow", re.I), "Contract/SOW"),
            (re.compile(r"minutes|meeting", re.I), "Meeting Minutes"),
            (re.compile(r"invoice|financial|p&l|balance", re.I), "Financial Report"),
            (re.compile(r"plan|roadmap", re.I), "Project Plan"),
            (re.compile(r"spec|design|architecture|technical|doc", re.I), "Technical Documentation"),
            (re.compile(r"final", re.I), "Final Report"),
            (re.compile(r"draft", re.I), "Draft Report"),
            (re.compile(r"presentation|slides|deck|ppt|keynote", re.I), "Presentation"),
            (re.compile(r"research|paper|study", re.I), "Research Paper"),
            (re.compile(r"png|jpg|jpeg|gif|svg|tiff|bmp|image", re.I), "Image/Diagram"),
            (re.compile(r"email|msg|eml", re.I), "Email Correspondence"),
        ]

    def get_all_document_types(self) -> List[str]:
        types: List[str] = []
        for data in self._categories.values():
            types.extend(data.get("types", []))
        # Deduplicate while preserving order
        seen = set()
        uniq = []
        for t in types:
            if t not in seen:
                uniq.append(t)
                seen.add(t)
        return uniq

    def get_categories(self) -> Dict[str, Dict]:
        return self._categories

    def add_type_to_category(self, category: str, type_name: str) -> bool:
        if not type_name:
            return False
        cat = self._categories.get(category)
        if not cat:
            return False
        if type_name not in cat["types"]:
            cat["types"].append(type_name)
        return True

    def get_category_for_type(self, type_name: str) -> str:
        for name, data in self._categories.items():
            if type_name in data.get("types", []):
                return name
        return "Other"

    def suggest_document_type(self, filename: str) -> str:
        fn = filename or ""
        for pat, label in self._patterns:
            if pat.search(fn):
                return label
        return "Other"


_singleton: DocumentTypeManager = None


def get_document_type_manager() -> DocumentTypeManager:
    global _singleton
    if _singleton is None:
        _singleton = DocumentTypeManager()
    return _singleton

