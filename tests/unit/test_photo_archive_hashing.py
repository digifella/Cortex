# tests/unit/test_photo_archive_hashing.py
import pytest
from scripts.photo_archive import db, hashing


@pytest.fixture
def conn():
    c = db.connect(":memory:")
    db.init_schema(c)
    return c


def _add(conn, path, size, content=None):
    if content is not None:
        with open(path, "wb") as fh:
            fh.write(content)
    return db.upsert_file(conn, path=str(path), top_folder="t", rel_dir="",
                          filename="f", ext=".jpg", size=size, mtime=1.0,
                          kind="image")


def test_partial_hash_differs_on_different_content(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.write_bytes(b"A" * 200000)
    b.write_bytes(b"B" * 200000)
    assert hashing.partial_hash(str(a), 200000) != hashing.partial_hash(str(b), 200000)


def test_partial_hash_stable_for_same_content(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.write_bytes(b"A" * 200000)
    b.write_bytes(b"A" * 200000)
    assert hashing.partial_hash(str(a), 200000) == hashing.partial_hash(str(b), 200000)


def test_partial_hash_catches_tail_difference(tmp_path):
    # Files identical for the first 64KB must still be distinguished.
    a, b = tmp_path / "a", tmp_path / "b"
    a.write_bytes(b"A" * 200000 + b"END1")
    b.write_bytes(b"A" * 200000 + b"END2")
    assert hashing.partial_hash(str(a), 200004) != hashing.partial_hash(str(b), 200004)


def test_full_hash_matches_hashlib(tmp_path):
    import hashlib
    p = tmp_path / "a"
    p.write_bytes(b"hello world")
    assert hashing.full_hash(str(p)) == hashlib.sha256(b"hello world").hexdigest()


def test_unique_size_is_never_hashed(conn, tmp_path):
    _add(conn, tmp_path / "solo.jpg", 999, b"S" * 999)
    stats = hashing.hash_candidates(conn)
    assert stats["partial_hashed"] == 0
    assert conn.execute(
        "SELECT partial_hash FROM files").fetchone()[0] is None


def test_colliding_sizes_get_partial_then_full(conn, tmp_path):
    _add(conn, tmp_path / "x.jpg", 100, b"X" * 100)
    _add(conn, tmp_path / "y.jpg", 100, b"X" * 100)
    stats = hashing.hash_candidates(conn)
    assert stats["partial_hashed"] == 2
    assert stats["full_hashed"] == 2
    shas = [r[0] for r in conn.execute("SELECT sha256 FROM files")]
    assert shas[0] == shas[1] is not None


def test_same_size_different_content_skips_full_hash(conn, tmp_path):
    _add(conn, tmp_path / "x.jpg", 100, b"X" * 100)
    _add(conn, tmp_path / "y.jpg", 100, b"Y" * 100)
    stats = hashing.hash_candidates(conn)
    assert stats["partial_hashed"] == 2
    assert stats["full_hashed"] == 0
