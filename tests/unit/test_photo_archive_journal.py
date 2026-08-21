# tests/unit/test_photo_archive_journal.py
import csv
import pytest
from scripts.photo_archive import journal, hashing


def test_append_writes_header_once(tmp_path):
    j = journal.Journal(str(tmp_path / "j.csv"))
    j.append("quarantine", "move", "a", "b", 1, "s", "ok")
    j.append("quarantine", "move", "c", "d", 1, "s", "ok")
    lines = (tmp_path / "j.csv").read_text().strip().split("\n")
    assert lines[0].startswith("timestamp,")
    assert len(lines) == 3


def test_undo_dry_run_moves_nothing(tmp_path):
    src, dst = tmp_path / "src.jpg", tmp_path / "dst.jpg"
    dst.write_bytes(b"data")
    j = journal.Journal(str(tmp_path / "j.csv"))
    j.append("quarantine", "move", str(src), str(dst), 4,
             hashing.full_hash(str(dst)), "ok")
    stats = journal.undo(str(tmp_path / "j.csv"), apply=False)
    assert stats["would_restore"] == 1
    assert dst.exists() and not src.exists()


def test_undo_apply_restores_file(tmp_path):
    src, dst = tmp_path / "src.jpg", tmp_path / "dst.jpg"
    dst.write_bytes(b"data")
    j = journal.Journal(str(tmp_path / "j.csv"))
    j.append("quarantine", "move", str(src), str(dst), 4,
             hashing.full_hash(str(dst)), "ok")
    stats = journal.undo(str(tmp_path / "j.csv"), apply=True)
    assert stats["restored"] == 1
    assert src.exists() and not dst.exists()


def test_undo_refuses_when_hash_changed(tmp_path):
    src, dst = tmp_path / "src.jpg", tmp_path / "dst.jpg"
    dst.write_bytes(b"data")
    j = journal.Journal(str(tmp_path / "j.csv"))
    j.append("quarantine", "move", str(src), str(dst), 4, "deadbeef", "ok")
    stats = journal.undo(str(tmp_path / "j.csv"), apply=True)
    assert stats["refused"] == 1
    assert dst.exists() and not src.exists()


def test_undo_skips_missing_destination(tmp_path):
    j = journal.Journal(str(tmp_path / "j.csv"))
    j.append("quarantine", "move", str(tmp_path / "s"), str(tmp_path / "gone"),
             1, "x", "ok")
    stats = journal.undo(str(tmp_path / "j.csv"), apply=True)
    assert stats["missing"] == 1


def test_undo_replays_in_reverse_order(tmp_path):
    # A then B; undo must process B before A.
    a, b = tmp_path / "a", tmp_path / "b"
    b.write_bytes(b"bb")
    j = journal.Journal(str(tmp_path / "j.csv"))
    j.append("s", "move", str(a), str(b), 2, hashing.full_hash(str(b)), "ok")
    order = journal.undo(str(tmp_path / "j.csv"), apply=False)["order"]
    assert order == [str(b)]
