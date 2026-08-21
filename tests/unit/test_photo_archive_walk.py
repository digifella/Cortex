# tests/unit/test_photo_archive_walk.py
import os
import pytest
from scripts.photo_archive import db, walk


@pytest.fixture
def conn():
    c = db.connect(":memory:")
    db.init_schema(c)
    return c


@pytest.fixture
def tree(tmp_path):
    root = tmp_path / "drive"
    scope = root / "family_Randoms"
    (scope / "sub").mkdir(parents=True)
    (scope / "a.jpg").write_bytes(b"x" * 10)
    (scope / "a.xmp").write_text("<xmp/>")
    (scope / "b.tif_original").write_bytes(b"y" * 10)
    (scope / "sub" / "c.raf").write_bytes(b"z" * 10)
    (root / "New LR Catalog").mkdir()
    (root / "New LR Catalog" / "cat.lrcat").write_bytes(b"n")
    return root


def test_walk_records_images_and_sidecars(conn, tree):
    walk.walk_scope(conn, str(tree), ["family_Randoms"])
    names = {r["filename"] for r in conn.execute("SELECT filename FROM files")}
    assert names == {"a.jpg", "a.xmp", "c.raf"}


def test_walk_skips_original_files(conn, tree):
    walk.walk_scope(conn, str(tree), ["family_Randoms"])
    rows = conn.execute(
        "SELECT COUNT(*) FROM files WHERE filename LIKE '%_original'").fetchone()[0]
    assert rows == 0


def test_walk_never_enters_hard_excluded_root(conn, tree):
    walk.walk_scope(conn, str(tree), ["family_Randoms"])
    rows = conn.execute(
        "SELECT COUNT(*) FROM files WHERE ext = '.lrcat'").fetchone()[0]
    assert rows == 0


def test_walk_links_sidecar_to_parent(conn, tree):
    walk.walk_scope(conn, str(tree), ["family_Randoms"])
    side = conn.execute("SELECT sidecar_of FROM files WHERE ext='.xmp'").fetchone()[0]
    parent = conn.execute("SELECT id FROM files WHERE filename='a.jpg'").fetchone()[0]
    assert side == parent


def test_walk_records_rel_dir(conn, tree):
    walk.walk_scope(conn, str(tree), ["family_Randoms"])
    rel = conn.execute("SELECT rel_dir FROM files WHERE filename='c.raf'").fetchone()[0]
    assert rel == "sub"


def test_missing_scope_root_raises_not_silently_zero(conn, tree):
    with pytest.raises(walk.ScopeUnreadable):
        walk.walk_scope(conn, str(tree), ["No Such Folder"])


def test_walk_is_idempotent(conn, tree):
    walk.walk_scope(conn, str(tree), ["family_Randoms"])
    walk.walk_scope(conn, str(tree), ["family_Randoms"])
    assert conn.execute("SELECT COUNT(*) FROM files").fetchone()[0] == 3
