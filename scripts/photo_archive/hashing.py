# scripts/photo_archive/hashing.py
"""Stage 3 - hashing, restricted to size-collision groups.

A file whose size is globally unique cannot be a byte-duplicate and is never
read. Within a size group, a partial hash (head + tail) filters before any
full read. This is what makes a 5.4 TB run finish in hours rather than days.
"""
import hashlib

from .paths import win_long


def partial_hash(path: str, size: int, chunk: int = 65536) -> str:
    h = hashlib.sha256()
    h.update(str(size).encode())
    with open(win_long(path), "rb") as fh:
        h.update(fh.read(chunk))
        if size > chunk * 2:
            fh.seek(-chunk, 2)
            h.update(fh.read(chunk))
    return h.hexdigest()


def full_hash(path: str, bufsize: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(win_long(path), "rb") as fh:
        while True:
            block = fh.read(bufsize)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def hash_candidates(conn) -> dict[str, int]:
    stats = {"partial_hashed": 0, "full_hashed": 0, "errors": 0}

    dup_sizes = [r[0] for r in conn.execute(
        "SELECT size FROM files WHERE kind != 'sidecar' "
        "GROUP BY size HAVING COUNT(*) > 1")]

    for size in dup_sizes:
        rows = conn.execute(
            "SELECT id, path FROM files WHERE size = ? AND kind != 'sidecar' "
            "AND partial_hash IS NULL", (size,)).fetchall()
        for row in rows:
            try:
                ph = partial_hash(row["path"], size)
            except OSError:
                stats["errors"] += 1
                continue
            conn.execute("UPDATE files SET partial_hash = ? WHERE id = ?",
                         (ph, row["id"]))
            stats["partial_hashed"] += 1
        conn.commit()

    dup_parts = [r[0] for r in conn.execute(
        "SELECT partial_hash FROM files WHERE partial_hash IS NOT NULL "
        "GROUP BY partial_hash HAVING COUNT(*) > 1")]

    for ph in dup_parts:
        rows = conn.execute(
            "SELECT id, path FROM files WHERE partial_hash = ? AND sha256 IS NULL",
            (ph,)).fetchall()
        for row in rows:
            try:
                sha = full_hash(row["path"])
            except OSError:
                stats["errors"] += 1
                continue
            conn.execute("UPDATE files SET sha256 = ? WHERE id = ?",
                         (sha, row["id"]))
            stats["full_hashed"] += 1
        conn.commit()

    return stats
