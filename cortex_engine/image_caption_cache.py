"""Corpus-wide, permanent cache mapping an image's content hash to its VLM
caption. Two different files that embed the same picture (a letterhead, a
logo, a repeated template graphic) hit the cache on the second and later
occurrence instead of paying for another VLM call.
"""
import hashlib
import os
import sqlite3
from typing import Optional

DEFAULT_DB_PATH = os.path.expanduser("~/.cortex_image_caption_cache.db")


class ImageCaptionCache:
    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        self._conn = sqlite3.connect(self.db_path)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS captions ("
            "  image_hash TEXT PRIMARY KEY,"
            "  caption TEXT NOT NULL,"
            "  created_at TEXT DEFAULT CURRENT_TIMESTAMP"
            ")"
        )
        self._conn.commit()

    @staticmethod
    def _hash(image_bytes: bytes) -> str:
        return hashlib.sha256(image_bytes).hexdigest()

    def get(self, image_bytes: bytes) -> Optional[str]:
        row = self._conn.execute(
            "SELECT caption FROM captions WHERE image_hash = ?",
            (self._hash(image_bytes),),
        ).fetchone()
        return row[0] if row else None

    def put(self, image_bytes: bytes, caption: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO captions (image_hash, caption) VALUES (?, ?)",
            (self._hash(image_bytes), caption),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()
