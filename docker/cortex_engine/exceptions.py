# ## File: cortex_engine/exceptions.py
# Version: 1.0.0
# Date: 2025-07-23
# Purpose: Standardized exception hierarchy for the Cortex Suite.
#          Provides consistent error handling across all modules.


class CortexException(Exception):
    """Base exception for all Cortex Suite errors."""
    
    def __init__(self, message: str, details: str = None):
        self.message = message
        self.details = details
        super().__init__(message)
    
    def __str__(self):
        if self.details:
            return f"{self.message}: {self.details}"
        return self.message


class ConfigurationError(CortexException):
    """Raised when there's a configuration problem."""
    pass


class PathError(CortexException):
    """Raised when there's a path-related problem."""
    pass


class DocumentError(CortexException):
    """Base exception for document-related errors."""
    pass


class DocumentIngestionError(DocumentError):
    """Raised when document ingestion fails."""
    pass


class DocumentParsingError(DocumentError):
    """Raised when document parsing fails."""
    pass


class SearchError(CortexException):
    """Raised when search operations fail."""
    pass


class VectorStoreError(CortexException):
    """Raised when vector store operations fail."""
    pass


class GraphError(CortexException):
    """Raised when knowledge graph operations fail."""
    pass


class EntityExtractionError(CortexException):
    """Raised when entity extraction fails."""
    pass


class CollectionError(CortexException):
    """Raised when collection operations fail."""
    pass


class ProposalError(CortexException):
    """Base exception for proposal-related errors."""
    pass


class ProposalGenerationError(ProposalError):
    """Raised when proposal generation fails."""
    pass


class TemplateError(ProposalError):
    """Raised when template processing fails."""
    pass


class ModelError(CortexException):
    """Raised when AI model operations fail."""
    pass


class ValidationError(CortexException):
    """Raised when validation fails."""
    pass