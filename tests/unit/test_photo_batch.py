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


def test_build_sync_config_dry_run_defaults():
    cfg = pb.build_sync_config("/raw", "/jpg", dry_run=True)
    assert cfg.dry_run is True
    assert cfg.keep_backups is True
    assert cfg.filter_keywords == ["nogps"]
    assert cfg.timestamp_tolerance_seconds == 0
    assert str(cfg.raw_root) == "/raw"
    assert str(cfg.jpg_dir) == "/jpg"


def test_build_sync_config_flags_passthrough():
    cfg = pb.build_sync_config(
        "/raw", "/jpg",
        dry_run=False, keep_backups=False,
        filter_keywords=["x", "y"], timestamp_tolerance=4,
    )
    assert cfg.dry_run is False
    assert cfg.keep_backups is False
    assert cfg.filter_keywords == ["x", "y"]
    assert cfg.timestamp_tolerance_seconds == 4


def test_scan_actions_matches_raw_by_stem(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    jpgdir = tmp_path / "jpg"
    jpgdir.mkdir()
    (raw / "2020-01-01 10-00-00-X-T1.NEF").write_bytes(b"raw")
    (jpgdir / "2020-01-01 10-00-00-X-T1.jpg").write_bytes(b"jpg")

    cfg = pb.build_sync_config(raw, jpgdir, dry_run=True)
    actions, orphaned = pb.scan_actions(cfg)

    assert len(actions) == 1
    assert actions[0].target_path.name == "2020-01-01 10-00-00-X-T1.xmp"
    assert orphaned == []
    # scanning is read-only — no sidecar is created
    assert not (raw / "2020-01-01 10-00-00-X-T1.xmp").exists()


def test_scan_actions_reports_orphan(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    jpgdir = tmp_path / "jpg"
    jpgdir.mkdir()
    (jpgdir / "loner.jpg").write_bytes(b"jpg")

    cfg = pb.build_sync_config(raw, jpgdir, dry_run=True)
    actions, orphaned = pb.scan_actions(cfg)

    assert actions == []
    assert [p.name for p in orphaned] == ["loner.jpg"]


def test_sync_photos_dry_run_writes_nothing(tmp_path, capsys):
    raw = tmp_path / "raw"
    raw.mkdir()
    jpgdir = tmp_path / "jpg"
    jpgdir.mkdir()
    (raw / "2020-01-01 10-00-00-X-T1.NEF").write_bytes(b"raw")
    (jpgdir / "2020-01-01 10-00-00-X-T1.jpg").write_bytes(b"jpg")

    summary = pb.sync_photos(jpgdir, raw, apply=False)

    assert summary["applied"] is False
    assert summary["actions"] == 1
    assert not (raw / "2020-01-01 10-00-00-X-T1.xmp").exists()
    out = capsys.readouterr().out
    assert "DRY RUN" in out
