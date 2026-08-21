import sqlite3
import pytest
from scripts.photo_archive import db


@pytest.fixture
def conn():
    c = db.connect(":memory:")
    db.init_schema(c)
    return c


def _row(**over):
    base = dict(path=r"P:\a\b.jpg", top_folder="a", rel_dir="", filename="b.jpg",
                ext=".jpg", size=100, mtime=1.0, kind="image")
    base.update(over)
    return base


def test_upsert_returns_id_and_is_idempotent(conn):
    first = db.upsert_file(conn, **_row())
    second = db.upsert_file(conn, **_row())
    assert first == second
    assert conn.execute("SELECT COUNT(*) FROM files").fetchone()[0] == 1


def test_upsert_updates_size_on_rescan(conn):
    db.upsert_file(conn, **_row())
    db.upsert_file(conn, **_row(size=222))
    assert conn.execute("SELECT size FROM files").fetchone()[0] == 222


def test_default_state_is_walked(conn):
    db.upsert_file(conn, **_row())
    assert conn.execute("SELECT state FROM files").fetchone()[0] == "walked"


def test_set_state_rejects_unknown_state(conn):
    fid = db.upsert_file(conn, **_row())
    with pytest.raises(ValueError):
        db.set_state(conn, fid, "banana")


def test_set_state_accepts_known_state(conn):
    fid = db.upsert_file(conn, **_row())
    db.set_state(conn, fid, "hashed")
    assert conn.execute("SELECT state FROM files").fetchone()[0] == "hashed"


def test_path_is_unique(conn):
    db.upsert_file(conn, **_row())
    assert conn.execute("SELECT COUNT(*) FROM files").fetchone()[0] == 1


def test_sidecar_links_to_parent(conn):
    parent = db.upsert_file(conn, **_row())
    db.upsert_file(conn, **_row(path=r"P:\a\b.xmp", filename="b.xmp",
                                ext=".xmp", kind="sidecar", sidecar_of=parent))
    got = conn.execute("SELECT sidecar_of FROM files WHERE ext='.xmp'").fetchone()[0]
    assert got == parent
