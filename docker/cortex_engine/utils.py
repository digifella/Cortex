# ## File: cortex_engine/utils.py
# Version: 1.0.0 (Initial Creation)
# Date: 2025-07-13
# Purpose: A module for common, low-dependency utility functions shared across the engine.
#          - FEATURE (v1.0.0): Created to house the `get_file_hash` function,
#            breaking a circular dependency between `ingest_cortex` and `collection_manager`.

import hashlib

def get_file_hash(filepath: str) -> str:
    """
    Generates a SHA256 hash for a given file.

    Args:
        filepath: The path to the file.

    Returns:
        The hex digest of the file's SHA256 hash.
    """
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        # Read and update hash in chunks of 4K
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()