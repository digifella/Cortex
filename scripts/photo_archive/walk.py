# scripts/photo_archive/walk.py
"""Stage 1 - walk and stat. Reads no file contents."""
import os

from .paths import win_long
from . import config, db


class ScopeUnreadable(Exception):
    """A configured scope root could not be read.

    Raised rather than recording zero files. A run that cannot see a folder
    must fail loudly: silently indexing a folder as empty is how the photo
    index lost 1990/1991 without an error.
    """


def walk_scope(conn, drive_root: str, scope_roots: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for top in scope_roots:
        top_path = os.path.join(drive_root, top)
        if not os.path.isdir(win_long(top_path)):
            raise ScopeUnreadable(f"scope root missing or unreadable: {top_path}")
        counts[top] = _walk_one(conn, top_path, top)
        conn.commit()
    return counts


def _walk_one(conn, top_path: str, top: str) -> int:
    n = 0
    pending_sidecars: list[tuple[int, str]] = []
    stems: dict[tuple[str, str], int] = {}

    for dirpath, dirnames, filenames in os.walk(win_long(top_path)):
        dirnames[:] = [
            d for d in dirnames
            if d not in config.HARD_EXCLUDE and d not in config.EXCLUDE_DIR_NAMES
        ]
        rel_dir = os.path.relpath(dirpath, win_long(top_path))
        rel_dir = "" if rel_dir == "." else rel_dir

        for fn in filenames:
            if config.is_skipped_file(fn):
                continue
            full = os.path.join(dirpath, fn)
            try:
                st = os.stat(full)
            except OSError:
                continue
            ext = os.path.splitext(fn)[1].lower()
            kind = config.kind_for(ext)
            fid = db.upsert_file(
                conn, path=full, top_folder=top, rel_dir=rel_dir,
                filename=fn, ext=ext, size=st.st_size,
                mtime=st.st_mtime, kind=kind,
            )
            stem = os.path.splitext(fn)[0]
            if kind == "sidecar":
                pending_sidecars.append((fid, os.path.join(rel_dir, stem)))
            else:
                stems.setdefault((rel_dir, stem), fid)
            n += 1

    for fid, relstem in pending_sidecars:
        rel_dir, stem = os.path.split(relstem)
        parent = stems.get((rel_dir, stem))
        if parent is not None:
            conn.execute("UPDATE files SET sidecar_of = ? WHERE id = ?", (parent, fid))
    return n
