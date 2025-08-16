# ## File: cortex_engine/utils/file_utils.py
# Version: 1.0.0
# Date: 2025-07-23
# Purpose: File-related utility functions.
#          Consolidates file operations and hash generation.

import hashlib
from pathlib import Path
from typing import Union


def get_file_hash(filepath: Union[str, Path]) -> str:
    """
    Generates a SHA256 hash for a given file.

    Args:
        filepath: The path to the file.

    Returns:
        The hex digest of the file's SHA256 hash.
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        IOError: If the file cannot be read
    """
    path_obj = Path(filepath)
    
    if not path_obj.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    sha256_hash = hashlib.sha256()
    try:
        with open(path_obj, "rb") as f:
            # Read and update hash in chunks of 4K
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except IOError as e:
        raise IOError(f"Failed to read file {filepath}: {e}")