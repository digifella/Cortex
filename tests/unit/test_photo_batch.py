import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from scripts import photo_batch as pb


def test_description_is_bad_empty():
    assert pb.description_is_bad("") is True
    assert pb.description_is_bad(None) is True
    assert pb.description_is_bad("   ") is True


def test_description_is_bad_placeholder():
    assert pb.description_is_bad("[Image: description timed out]") is True


def test_description_is_bad_too_short():
    assert pb.description_is_bad("A dog.") is True            # < 40 chars
    assert pb.description_is_bad("A dog.", min_len=3) is False


def test_description_is_bad_refusal_prefix_even_when_long():
    text = "I must give a thorough and complete description of this scene before I continue"
    assert pb.description_is_bad(text) is True


def test_description_is_good():
    good = ("A wooden sailboat moored at a stone jetty under an overcast sky, "
            "with green hills rising behind the harbour.")
    assert pb.description_is_bad(good) is False


import os
import time


def test_file_key_changes_with_mtime(tmp_path):
    f = tmp_path / "a.jpg"
    f.write_bytes(b"x")
    k1 = pb.file_key(f)
    future = time.time() + 100
    os.utime(f, (future, future))
    assert pb.file_key(f) != k1


def test_checkpoint_roundtrip_and_is_done(tmp_path):
    a = tmp_path / "a.jpg"
    a.write_bytes(b"a")
    b = tmp_path / "b.jpg"
    b.write_bytes(b"b")
    cp = {pb.file_key(a): {"status": "tagged"}}
    pb.save_checkpoint(tmp_path, cp)
    loaded = pb.load_checkpoint(tmp_path)
    assert pb.is_done(a, loaded) is True
    assert pb.is_done(b, loaded) is False


def test_is_done_false_after_file_changes(tmp_path):
    a = tmp_path / "a.jpg"
    a.write_bytes(b"a")
    cp = {pb.file_key(a): {"status": "tagged"}}
    a.write_bytes(b"aa")  # size changes -> key changes
    assert pb.is_done(a, cp) is False


def test_load_checkpoint_missing_returns_empty(tmp_path):
    assert pb.load_checkpoint(tmp_path) == {}


def test_skipped_good_counts_as_done(tmp_path):
    a = tmp_path / "a.jpg"
    a.write_bytes(b"a")
    cp = {pb.file_key(a): {"status": "skipped-good"}}
    assert pb.is_done(a, cp) is True
