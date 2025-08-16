# cortex_engine/document_type_manager.py
# V1.0 - Document Type Management System
# Date: 2025-07-26
# Purpose: Manage user-defined document type categories and mappings

import json
import os
from typing import Dict, List, Set
from pathlib import Path
from datetime import datetime

from .utils import get_project_root, get_logger
from .exceptions import ConfigurationError

logger = get_logger(__name__)

# Configuration
PROJECT_ROOT = get_project_root()
DOCUMENT_TYPES_FILE = str(PROJECT_ROOT / "document_types_config.json")

# Default document types and categories
DEFAULT_DOCUMENT_TYPES = {
    "categories": {
        "Project Documents": {
            "types": ["Project Plan", "Technical Documentation", "Final Report", "Draft Report"],
            "description": "Project-related deliverables and documentation"
        },
        "Business Documents": {
            "types": ["Proposal/Quote", "Contract/SOW", "Financial Report"],
            "description": "Business and financial documents"
        },
        "Research & Analysis": {
            "types": ["Case Study / Trophy", "Research Paper", "Analysis Report"],
            "description": "Research outputs and analytical documents"
        },
        "Communications": {
            "types": ["Email Correspondence", "Meeting Minutes", "Presentation"],
            "description": "Communication and meeting-related documents"
        },
        "Personal Records": {
            "types": ["CV", "Bio", "Personal Statement"],
            "description": "Personal and professional profiles"
        },
        "Meetings": {
            "types": ["Agenda", "Minutes", "Action Items"],
            "description": "Meeting-related documents"
        },
        "Other": {
            "types": ["Other", "Template", "Reference Material"],
            "description": "Miscellaneous document types"
        }
    },
    "type_mappings": {
        # Keywords that map to specific document types
        "bio": "CV",
        "biography": "CV", 
        "curriculum vitae": "CV",
        "resume": "CV",
        "agenda": "Agenda",
        "minutes": "Minutes",
        "meeting minutes": "Minutes",
        "action items": "Action Items",
        "proposal": "Proposal/Quote",
        "quote": "Proposal/Quote",
        "quotation": "Proposal/Quote",
        "contract": "Contract/SOW",
        "statement of work": "Contract/SOW",
        "sow": "Contract/SOW",
        "plan": "Project Plan",
        "technical": "Technical Documentation",
        "documentation": "Technical Documentation",
        "final report": "Final Report",
        "draft report": "Draft Report",
        "case study": "Case Study / Trophy",
        "trophy": "Case Study / Trophy",
        "presentation": "Presentation",
        "email": "Email Correspondence",
        "correspondence": "Email Correspondence",
        "financial": "Financial Report",
        "research": "Research Paper"
    },
    "created_at": datetime.now().isoformat(),
    "modified_at": datetime.now().isoformat()
}

class DocumentTypeManager:
    """Manages document type categories and mappings."""
    
    def __init__(self):
        self.config = self._load_config()
    
    def _load_config(self) -> Dict:
        """Load document types configuration from file."""
        if os.path.exists(DOCUMENT_TYPES_FILE):
            try:
                with open(DOCUMENT_TYPES_FILE, 'r') as f:
                    config = json.load(f)
                logger.debug(f"Loaded document types config from {DOCUMENT_TYPES_FILE}")
                return config
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load document types config, using defaults: {e}")
        
        # Create default config file
        self._save_config(DEFAULT_DOCUMENT_TYPES)
        return DEFAULT_DOCUMENT_TYPES.copy()
    
    def _save_config(self, config: Dict = None) -> bool:
        """Save document types configuration to file."""
        config_to_save = config or self.config
        config_to_save["modified_at"] = datetime.now().isoformat()
        
        try:
            with open(DOCUMENT_TYPES_FILE, 'w') as f:
                json.dump(config_to_save, f, indent=4)
            logger.debug(f"Saved document types config to {DOCUMENT_TYPES_FILE}")
            return True
        except IOError as e:
            logger.error(f"Failed to save document types config: {e}")
            return False
    
    def get_all_document_types(self) -> List[str]:
        """Get a flat list of all document types across all categories."""
        all_types = set()
        for category_data in self.config["categories"].values():
            all_types.update(category_data["types"])
        return sorted(list(all_types))
    
    def get_categories(self) -> Dict[str, Dict]:
        """Get all categories with their types and descriptions."""
        return self.config["categories"]
    
    def get_types_for_category(self, category_name: str) -> List[str]:
        """Get document types for a specific category."""
        category = self.config["categories"].get(category_name, {})
        return category.get("types", [])
    
    def add_category(self, category_name: str, description: str, types: List[str] = None) -> bool:
        """Add a new category."""
        if category_name in self.config["categories"]:
            return False  # Category already exists
        
        self.config["categories"][category_name] = {
            "types": types or [],
            "description": description
        }
        return self._save_config()
    
    def remove_category(self, category_name: str) -> bool:
        """Remove a category (but keep the document types available)."""
        if category_name not in self.config["categories"]:
            return False
        
        # Don't allow removing "Other" category
        if category_name == "Other":
            return False
        
        del self.config["categories"][category_name]
        return self._save_config()
    
    def add_type_to_category(self, category_name: str, document_type: str) -> bool:
        """Add a document type to a category."""
        if category_name not in self.config["categories"]:
            return False
        
        if document_type not in self.config["categories"][category_name]["types"]:
            self.config["categories"][category_name]["types"].append(document_type)
            self.config["categories"][category_name]["types"].sort()
            return self._save_config()
        
        return True  # Already exists
    
    def remove_type_from_category(self, category_name: str, document_type: str) -> bool:
        """Remove a document type from a category."""
        if category_name not in self.config["categories"]:
            return False
        
        if document_type in self.config["categories"][category_name]["types"]:
            self.config["categories"][category_name]["types"].remove(document_type)
            return self._save_config()
        
        return False
    
    def move_type_between_categories(self, document_type: str, from_category: str, to_category: str) -> bool:
        """Move a document type from one category to another."""
        if (from_category not in self.config["categories"] or 
            to_category not in self.config["categories"]):
            return False
        
        # Remove from old category
        if document_type in self.config["categories"][from_category]["types"]:
            self.config["categories"][from_category]["types"].remove(document_type)
        
        # Add to new category
        if document_type not in self.config["categories"][to_category]["types"]:
            self.config["categories"][to_category]["types"].append(document_type)
            self.config["categories"][to_category]["types"].sort()
        
        return self._save_config()
    
    def add_type_mapping(self, keyword: str, document_type: str) -> bool:
        """Add a keyword mapping to a document type."""
        self.config["type_mappings"][keyword.lower()] = document_type
        return self._save_config()
    
    def remove_type_mapping(self, keyword: str) -> bool:
        """Remove a keyword mapping."""
        keyword_lower = keyword.lower()
        if keyword_lower in self.config["type_mappings"]:
            del self.config["type_mappings"][keyword_lower]
            return self._save_config()
        return False
    
    def get_type_mappings(self) -> Dict[str, str]:
        """Get all keyword to document type mappings."""
        return self.config["type_mappings"]
    
    def suggest_document_type(self, filename: str, content_preview: str = "") -> str:
        """Suggest a document type based on filename and content."""
        filename_lower = filename.lower()
        content_lower = content_preview.lower()
        
        # Check filename and content against mappings
        for keyword, doc_type in self.config["type_mappings"].items():
            if keyword in filename_lower or keyword in content_lower:
                return doc_type
        
        # Default fallback
        return "Other"
    
    def validate_document_type(self, document_type: str) -> bool:
        """Check if a document type is valid (exists in any category)."""
        all_types = self.get_all_document_types()
        return document_type in all_types
    
    def get_category_for_type(self, document_type: str) -> str:
        """Find which category a document type belongs to."""
        for category_name, category_data in self.config["categories"].items():
            if document_type in category_data["types"]:
                return category_name
        return "Other"
    
    def export_config(self) -> str:
        """Export current configuration as JSON string."""
        return json.dumps(self.config, indent=4)
    
    def import_config(self, config_json: str) -> bool:
        """Import configuration from JSON string."""
        try:
            new_config = json.loads(config_json)
            
            # Validate basic structure
            required_keys = ["categories", "type_mappings"]
            if not all(key in new_config for key in required_keys):
                raise ValueError("Invalid config structure")
            
            self.config = new_config
            return self._save_config()
            
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to import config: {e}")
            return False
    
    def reset_to_defaults(self) -> bool:
        """Reset configuration to default values."""
        self.config = DEFAULT_DOCUMENT_TYPES.copy()
        return self._save_config()

# Global instance for easy access
_doc_type_manager_instance = None

def get_document_type_manager() -> DocumentTypeManager:
    """Get or create the global DocumentTypeManager instance."""
    global _doc_type_manager_instance
    
    if _doc_type_manager_instance is None:
        _doc_type_manager_instance = DocumentTypeManager()
    
    return _doc_type_manager_instance