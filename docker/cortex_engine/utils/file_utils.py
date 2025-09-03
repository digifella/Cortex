import hashlib
from pathlib import Path

def get_file_hash(path: str, chunk_size: int = 8192) -> str:
    p = Path(path)
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            h.update(chunk)
    return h.hexdigest()

