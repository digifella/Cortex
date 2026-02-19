"""
Persistent cache implementation using SQLite for query result caching.
Provides cross-session caching with automatic expiration and size management.
"""

import sqlite3
import json
import hashlib
import time
import threading
import re
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)
_SQL_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _validate_sql_identifier(identifier: str) -> str:
    """Allow only safe SQLite identifiers for table names."""
    if not _SQL_IDENTIFIER_RE.fullmatch(identifier):
        raise ValueError(f"Invalid SQL identifier: {identifier!r}")
    return identifier


class PersistentCache:
    """
    SQLite-based persistent cache for query results.

    Features:
    - Thread-safe operations
    - Automatic expiration (TTL)
    - LRU eviction when cache is full
    - JSON serialization for complex data
    - Cache statistics tracking
    """

    def __init__(
        self,
        cache_db_path: str,
        max_entries: int = 1000,
        default_ttl_hours: int = 24,
        table_name: str = "query_cache"
    ):
        """
        Initialize persistent cache.

        Args:
            cache_db_path: Path to SQLite database file
            max_entries: Maximum number of cache entries
            default_ttl_hours: Default time-to-live in hours
            table_name: Name of cache table
        """
        self.cache_db_path = Path(cache_db_path)
        self.max_entries = max_entries
        self.default_ttl = timedelta(hours=default_ttl_hours)
        self.table_name = _validate_sql_identifier(table_name)
        self._lock = threading.Lock()

        # Ensure parent directory exists
        self.cache_db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self._init_db()

        # Clean expired entries on startup
        self._cleanup_expired()

    def _init_db(self):
        """Initialize SQLite database and create table if not exists."""
        with sqlite3.connect(str(self.cache_db_path)) as conn:
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    cache_key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    expires_at REAL NOT NULL,
                    last_accessed REAL NOT NULL,
                    access_count INTEGER DEFAULT 0,
                    size_bytes INTEGER NOT NULL
                )
            """)

            # Create indexes for performance
            conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_expires_at
                ON {self.table_name}(expires_at)
            """)

            conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_last_accessed
                ON {self.table_name}(last_accessed)
            """)

            conn.commit()

    def _generate_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments."""
        key_data = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True)
        return hashlib.sha256(key_data.encode()).hexdigest()

    def get(self, cache_key: str) -> Optional[Any]:
        """
        Get value from cache.

        Args:
            cache_key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        with self._lock:
            try:
                with sqlite3.connect(str(self.cache_db_path)) as conn:
                    cursor = conn.execute(
                        f"""
                        SELECT value, expires_at
                        FROM {self.table_name}
                        WHERE cache_key = ?
                        """,
                        (cache_key,)
                    )
                    row = cursor.fetchone()

                    if not row:
                        return None

                    value_json, expires_at = row

                    # Check if expired
                    if time.time() > expires_at:
                        self._delete_key(cache_key)
                        return None

                    # Update access stats
                    conn.execute(
                        f"""
                        UPDATE {self.table_name}
                        SET last_accessed = ?, access_count = access_count + 1
                        WHERE cache_key = ?
                        """,
                        (time.time(), cache_key)
                    )
                    conn.commit()

                    # Deserialize and return
                    return json.loads(value_json)

            except Exception as e:
                logger.error(f"Cache get error: {e}")
                return None

    def put(
        self,
        cache_key: str,
        value: Any,
        ttl: Optional[timedelta] = None
    ):
        """
        Store value in cache.

        Args:
            cache_key: Cache key
            value: Value to cache (must be JSON-serializable)
            ttl: Time-to-live (uses default if not specified)
        """
        with self._lock:
            try:
                # Serialize value
                value_json = json.dumps(value)
                size_bytes = len(value_json.encode())

                # Calculate expiration
                ttl = ttl or self.default_ttl
                now = time.time()
                expires_at = now + ttl.total_seconds()

                with sqlite3.connect(str(self.cache_db_path)) as conn:
                    # Check if we need to evict entries
                    cursor = conn.execute(
                        f"SELECT COUNT(*) FROM {self.table_name}"
                    )
                    count = cursor.fetchone()[0]

                    if count >= self.max_entries:
                        # Evict least recently accessed entry
                        conn.execute(
                            f"""
                            DELETE FROM {self.table_name}
                            WHERE cache_key = (
                                SELECT cache_key FROM {self.table_name}
                                ORDER BY last_accessed ASC
                                LIMIT 1
                            )
                            """
                        )

                    # Insert or replace entry
                    conn.execute(
                        f"""
                        INSERT OR REPLACE INTO {self.table_name}
                        (cache_key, value, created_at, expires_at, last_accessed, access_count, size_bytes)
                        VALUES (?, ?, ?, ?, ?, 0, ?)
                        """,
                        (cache_key, value_json, now, expires_at, now, size_bytes)
                    )
                    conn.commit()

            except Exception as e:
                logger.error(f"Cache put error: {e}")

    def _delete_key(self, cache_key: str):
        """Delete specific cache key."""
        with sqlite3.connect(str(self.cache_db_path)) as conn:
            conn.execute(
                f"DELETE FROM {self.table_name} WHERE cache_key = ?",
                (cache_key,)
            )
            conn.commit()

    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            try:
                with sqlite3.connect(str(self.cache_db_path)) as conn:
                    conn.execute(f"DELETE FROM {self.table_name}")
                    conn.commit()
                    logger.info("Cache cleared")
            except Exception as e:
                logger.error(f"Cache clear error: {e}")

    def _cleanup_expired(self):
        """Remove expired entries."""
        with self._lock:
            try:
                with sqlite3.connect(str(self.cache_db_path)) as conn:
                    cursor = conn.execute(
                        f"DELETE FROM {self.table_name} WHERE expires_at < ?",
                        (time.time(),)
                    )
                    deleted = cursor.rowcount
                    conn.commit()
                    if deleted > 0:
                        logger.info(f"Cleaned up {deleted} expired cache entries")
            except Exception as e:
                logger.error(f"Cache cleanup error: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        with self._lock:
            try:
                with sqlite3.connect(str(self.cache_db_path)) as conn:
                    # Get basic counts
                    cursor = conn.execute(
                        f"""
                        SELECT
                            COUNT(*) as total_entries,
                            SUM(size_bytes) as total_size_bytes,
                            SUM(access_count) as total_accesses,
                            AVG(access_count) as avg_accesses
                        FROM {self.table_name}
                        """
                    )
                    row = cursor.fetchone()
                    total_entries, total_size, total_accesses, avg_accesses = row

                    # Get most accessed entries
                    cursor = conn.execute(
                        f"""
                        SELECT cache_key, access_count
                        FROM {self.table_name}
                        ORDER BY access_count DESC
                        LIMIT 5
                        """
                    )
                    top_entries = cursor.fetchall()

                    return {
                        'total_entries': total_entries or 0,
                        'max_entries': self.max_entries,
                        'utilization_pct': round((total_entries or 0) / self.max_entries * 100, 1),
                        'total_size_mb': round((total_size or 0) / (1024 * 1024), 2),
                        'total_accesses': total_accesses or 0,
                        'avg_accesses_per_entry': round(avg_accesses or 0, 1),
                        'top_entries': [
                            {'key': k[:16] + '...', 'accesses': a}
                            for k, a in top_entries
                        ]
                    }

            except Exception as e:
                logger.error(f"Cache stats error: {e}")
                return {
                    'total_entries': 0,
                    'max_entries': self.max_entries,
                    'utilization_pct': 0,
                    'error': str(e)
                }

    def warm_cache(self, key_value_pairs: List[tuple]):
        """
        Warm cache with pre-computed results.

        Args:
            key_value_pairs: List of (key, value) tuples to cache
        """
        for key, value in key_value_pairs:
            self.put(key, value)
        logger.info(f"Warmed cache with {len(key_value_pairs)} entries")


# Global cache instance
_global_cache: Optional[PersistentCache] = None
_cache_init_lock = threading.Lock()


def get_persistent_cache(
    cache_dir: Optional[str] = None,
    **kwargs
) -> PersistentCache:
    """
    Get or create global persistent cache instance.

    Args:
        cache_dir: Directory for cache database (uses default if None)
        **kwargs: Additional arguments for PersistentCache

    Returns:
        Global PersistentCache instance
    """
    global _global_cache

    if _global_cache is None:
        with _cache_init_lock:
            if _global_cache is None:
                # Use default cache path if not specified
                if cache_dir is None:
                    from cortex_engine.utils.path_utils import get_database_path
                    cache_dir = Path(get_database_path()) / "cache"

                cache_db = Path(cache_dir) / "persistent_cache.db"
                _global_cache = PersistentCache(str(cache_db), **kwargs)
                logger.info(f"Initialized persistent cache at {cache_db}")

    return _global_cache
