# ## File: cortex_engine/utils/validation_utils.py
# Version: 1.0.0
# Date: 2025-08-08
# Purpose: Input validation utilities for security and data integrity

import os
import re
from pathlib import Path
from typing import List, Optional, Union, Dict, Any
from ..exceptions import ValidationError


class InputValidator:
    """Centralized input validation for security and data integrity"""
    
    # Safe filename patterns
    SAFE_FILENAME_PATTERN = re.compile(r'^[a-zA-Z0-9._\-\s()]+$')
    
    # Dangerous patterns to block
    DANGEROUS_PATTERNS = [
        r'\.\./',  # Path traversal
        r'\.\.\\',  # Windows path traversal
        r'<script',  # XSS attempts
        r'javascript:',  # JavaScript injection
        r'data:',  # Data URI schemes
        r'file://',  # File URI schemes
        r'\/\*.*\*\/',  # SQL comment patterns
        r'union\s+select',  # SQL injection
        r'drop\s+table',  # SQL injection
    ]
    
    @classmethod
    def validate_file_path(cls, file_path: Union[str, Path], must_exist: bool = True) -> Path:
        """
        Validate file path for security and existence
        
        Args:
            file_path: File path to validate
            must_exist: Whether the file must exist
            
        Returns:
            Validated Path object
            
        Raises:
            ValidationError: If path is invalid or unsafe
        """
        if not file_path:
            raise ValidationError("File path cannot be empty")
            
        # Convert to Path object
        path = Path(str(file_path)).resolve()
        
        # Check for path traversal attempts
        path_str = str(path)
        for pattern in cls.DANGEROUS_PATTERNS[:2]:  # Only path traversal patterns
            if re.search(pattern, path_str, re.IGNORECASE):
                raise ValidationError(f"Potentially dangerous path detected: {file_path}")
        
        # Check if path exists when required
        if must_exist and not path.exists():
            raise ValidationError(f"File does not exist: {file_path}")
            
        # Check if it's actually a file when it exists
        if path.exists() and not path.is_file():
            raise ValidationError(f"Path is not a file: {file_path}")
            
        return path
    
    @classmethod
    def validate_directory_path(cls, dir_path: Union[str, Path], must_exist: bool = True, create_if_missing: bool = False) -> Path:
        """
        Validate directory path for security and existence
        
        Args:
            dir_path: Directory path to validate
            must_exist: Whether the directory must exist
            create_if_missing: Create directory if it doesn't exist
            
        Returns:
            Validated Path object
            
        Raises:
            ValidationError: If path is invalid or unsafe
        """
        if not dir_path:
            raise ValidationError("Directory path cannot be empty")
            
        # Convert to Path object
        path = Path(str(dir_path)).resolve()
        
        # Check for path traversal attempts
        path_str = str(path)
        for pattern in cls.DANGEROUS_PATTERNS[:2]:  # Only path traversal patterns
            if re.search(pattern, path_str, re.IGNORECASE):
                raise ValidationError(f"Potentially dangerous path detected: {dir_path}")
        
        # Create directory if requested
        if create_if_missing and not path.exists():
            try:
                path.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                raise ValidationError(f"Cannot create directory {dir_path}: {e}")
        
        # Check if path exists when required
        if must_exist and not path.exists():
            raise ValidationError(f"Directory does not exist: {dir_path}")
            
        # Check if it's actually a directory when it exists
        if path.exists() and not path.is_dir():
            raise ValidationError(f"Path is not a directory: {dir_path}")
            
        return path
    
    @classmethod
    def validate_search_query(cls, query: str, max_length: int = 1000) -> str:
        """
        Validate search query for safety and length
        
        Args:
            query: Search query string
            max_length: Maximum allowed length
            
        Returns:
            Sanitized query string
            
        Raises:
            ValidationError: If query is invalid
        """
        if not query or not query.strip():
            raise ValidationError("Search query cannot be empty")
            
        query = query.strip()
        
        # Check length
        if len(query) > max_length:
            raise ValidationError(f"Query too long (max {max_length} characters)")
        
        # Check for dangerous patterns
        for pattern in cls.DANGEROUS_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE):
                raise ValidationError("Query contains potentially dangerous content")
        
        return query
    
    @classmethod
    def validate_filename(cls, filename: str, max_length: int = 255) -> str:
        """
        Validate filename for safety
        
        Args:
            filename: Filename to validate
            max_length: Maximum allowed length
            
        Returns:
            Validated filename
            
        Raises:
            ValidationError: If filename is invalid
        """
        if not filename or not filename.strip():
            raise ValidationError("Filename cannot be empty")
            
        filename = filename.strip()
        
        # Check length
        if len(filename) > max_length:
            raise ValidationError(f"Filename too long (max {max_length} characters)")
        
        # Check for dangerous characters
        if not cls.SAFE_FILENAME_PATTERN.match(filename):
            raise ValidationError("Filename contains invalid characters")
        
        # Check for reserved names
        reserved_names = ['CON', 'PRN', 'AUX', 'NUL'] + [f'COM{i}' for i in range(1, 10)] + [f'LPT{i}' for i in range(1, 10)]
        if filename.upper() in reserved_names:
            raise ValidationError(f"Filename '{filename}' is reserved")
            
        return filename
    
    @classmethod
    def validate_collection_name(cls, name: str) -> str:
        """
        Validate collection name
        
        Args:
            name: Collection name to validate
            
        Returns:
            Validated collection name
            
        Raises:
            ValidationError: If name is invalid
        """
        if not name or not name.strip():
            raise ValidationError("Collection name cannot be empty")
            
        name = name.strip()
        
        # Check length
        if len(name) > 100:
            raise ValidationError("Collection name too long (max 100 characters)")
        
        # Check for safe characters (allow more flexibility than filenames)
        if not re.match(r'^[a-zA-Z0-9._\-\s()]+$', name):
            raise ValidationError("Collection name contains invalid characters")
            
        return name
    
    @classmethod
    def validate_file_extensions(cls, file_paths: List[Union[str, Path]], allowed_extensions: Optional[List[str]] = None) -> List[Path]:
        """
        Validate file extensions against allowed list
        
        Args:
            file_paths: List of file paths to validate
            allowed_extensions: List of allowed extensions (with dots, e.g., ['.pdf', '.docx'])
            
        Returns:
            List of validated Path objects
            
        Raises:
            ValidationError: If any file has an invalid extension
        """
        if allowed_extensions is None:
            # Default allowed extensions for document ingestion
            allowed_extensions = [
                '.pdf', '.docx', '.txt', '.md', '.rtf',
                '.pptx', '.xlsx', '.csv',
                '.png', '.jpg', '.jpeg'
            ]
        
        allowed_extensions = [ext.lower() for ext in allowed_extensions]
        validated_paths = []
        
        for file_path in file_paths:
            path = Path(file_path)
            extension = path.suffix.lower()
            
            if extension not in allowed_extensions:
                raise ValidationError(f"File extension '{extension}' not allowed for file: {path.name}")
            
            validated_paths.append(path)
        
        return validated_paths
    
    @classmethod
    def sanitize_metadata(cls, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize metadata dictionary
        
        Args:
            metadata: Dictionary to sanitize
            
        Returns:
            Sanitized dictionary
        """
        sanitized = {}
        
        for key, value in metadata.items():
            # Sanitize key
            if isinstance(key, str):
                key = key.strip()
                # Check for dangerous patterns in keys
                safe_key = True
                for pattern in cls.DANGEROUS_PATTERNS:
                    if re.search(pattern, key, re.IGNORECASE):
                        safe_key = False
                        break
                
                if not safe_key:
                    continue  # Skip dangerous keys
            
            # Sanitize value
            if isinstance(value, str):
                value = value.strip()
                # Remove potentially dangerous content from string values
                for pattern in cls.DANGEROUS_PATTERNS:
                    value = re.sub(pattern, '', value, flags=re.IGNORECASE)
            elif isinstance(value, dict):
                value = cls.sanitize_metadata(value)
            elif isinstance(value, list):
                value = [cls.sanitize_metadata({0: item})[0] if isinstance(item, dict) 
                        else str(item).strip() if isinstance(item, str) 
                        else item for item in value]
            
            sanitized[key] = value
        
        return sanitized


def validate_api_input(func):
    """Decorator for API input validation"""
    def wrapper(*args, **kwargs):
        # Add any global API input validation here
        return func(*args, **kwargs)
    return wrapper