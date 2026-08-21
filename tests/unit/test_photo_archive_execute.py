# tests/unit/test_photo_archive_execute.py
import csv
import os
import pytest
from scripts.photo_archive import db, execute, journal


@pytest.fixture
def conn():
    c = db.connect(":memory:")
    db.init_schema(c)
    return c


def _plan(tmp_path, rows):
    p = tmp_path / "plan.csv"
    with p.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["group_id", "role", "path",
                                           "top_folder", "rel_dir", "size",
                                           "sha256"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return str(p)


def test_safe_move_suffixes_on_collision(tmp_path):
    a, b = tmp_path / "a.jpg", tmp_path / "b.jpg"
    a.write_bytes(b"1")
    b.write_bytes(b"2")
    result = execute.safe_move(str(a), str(b))
    assert result.endswith("b-2.jpg")
    assert not a.exists()


def test_safe_move_never_overwrites(tmp_path):
    a, b = tmp_path / "a.jpg", tmp_path / "b.jpg"
    a.write_bytes(b"1")
    b.write_bytes(b"2")
    execute.safe_move(str(a), str(b))
    assert b.read_bytes() == b"2"


def test_quarantine_dry_run_moves_nothing(conn, tmp_path):
    src = tmp_path / "family_Randoms" / "a.jpg"
    src.parent.mkdir(parents=True)
    src.write_bytes(b"x")
    plan = _plan(tmp_path, [{"group_id": 1, "role": "quarantine",
                             "path": str(src), "top_folder": "family_Randoms",
                             "rel_dir": "", "size": 1, "sha256": "s"}])
    j = journal.Journal(str(tmp_path / "j.csv"))
    stats = execute.quarantine(conn, plan, str(tmp_path), j, apply=False)
    assert stats["would_move"] == 1
    assert src.exists()


def test_quarantine_apply_moves_and_preserves_provenance(conn, tmp_path):
    src = tmp_path / "family_Randoms" / "sub" / "a.jpg"
    src.parent.mkdir(parents=True)
    src.write_bytes(b"x")
    plan = _plan(tmp_path, [{"group_id": 1, "role": "quarantine",
                             "path": str(src), "top_folder": "family_Randoms",
                             "rel_dir": "sub", "size": 1, "sha256": "s"}])
    j = journal.Journal(str(tmp_path / "j.csv"))
    stats = execute.quarantine(conn, plan, str(tmp_path), j, apply=True)
    assert stats["moved"] == 1
    assert not src.exists()
    assert (tmp_path / "_DUPES" / "family_Randoms" / "sub" / "a.jpg").exists()


def test_quarantine_ignores_keeper_rows(conn, tmp_path):
    keeper = tmp_path / "family_Randoms" / "k.jpg"
    keeper.parent.mkdir(parents=True)
    keeper.write_bytes(b"k")
    plan = _plan(tmp_path, [{"group_id": 1, "role": "keep",
                             "path": str(keeper), "top_folder": "family_Randoms",
                             "rel_dir": "", "size": 1, "sha256": "s"}])
    j = journal.Journal(str(tmp_path / "j.csv"))
    stats = execute.quarantine(conn, plan, str(tmp_path), j, apply=True)
    assert stats["moved"] == 0
    assert keeper.exists()


def test_quarantine_journals_every_move(conn, tmp_path):
    src = tmp_path / "family_Randoms" / "a.jpg"
    src.parent.mkdir(parents=True)
    src.write_bytes(b"x")
    plan = _plan(tmp_path, [{"group_id": 1, "role": "quarantine",
                             "path": str(src), "top_folder": "family_Randoms",
                             "rel_dir": "", "size": 1, "sha256": "s"}])
    jpath = tmp_path / "j.csv"
    execute.quarantine(conn, plan, str(tmp_path), journal.Journal(str(jpath)),
                       apply=True)
    rows = list(csv.DictReader(jpath.open()))
    assert len(rows) == 1 and rows[0]["operation"] == "move"


def test_quarantine_is_resumable(conn, tmp_path):
    src = tmp_path / "family_Randoms" / "a.jpg"
    src.parent.mkdir(parents=True)
    src.write_bytes(b"x")
    plan = _plan(tmp_path, [{"group_id": 1, "role": "quarantine",
                             "path": str(src), "top_folder": "family_Randoms",
                             "rel_dir": "", "size": 1, "sha256": "s"}])
    j = journal.Journal(str(tmp_path / "j.csv"))
    execute.quarantine(conn, plan, str(tmp_path), j, apply=True)
    stats = execute.quarantine(conn, plan, str(tmp_path), j, apply=True)
    assert stats["missing"] == 1 and stats["moved"] == 0
