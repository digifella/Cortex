import os
import pytest
from scripts.photo_archive.paths import win_long, strip_long
from scripts.photo_archive import config


def test_win_long_is_identity_off_windows(monkeypatch):
    monkeypatch.setattr(os, "name", "posix")
    assert win_long("/mnt/p/foo") == "/mnt/p/foo"


def test_win_long_prefixes_on_windows(monkeypatch):
    monkeypatch.setattr(os, "name", "nt")
    monkeypatch.setattr(os.path, "abspath", lambda p: p)
    assert win_long(r"P:\foo") == "\\\\?\\P:\\foo"


def test_win_long_is_idempotent(monkeypatch):
    monkeypatch.setattr(os, "name", "nt")
    monkeypatch.setattr(os.path, "abspath", lambda p: p)
    once = win_long(r"P:\foo")
    assert win_long(once) == once


def test_strip_long_removes_prefix():
    assert strip_long("\\\\?\\P:\\foo") == "P:\\foo"
    assert strip_long("P:\\foo") == "P:\\foo"


def test_original_files_are_skipped():
    assert config.is_skipped_file("x.tif_original") is True
    assert config.is_skipped_file("x.dng_original") is True
    assert config.is_skipped_file("x.tif") is False


def test_old_files_are_not_skipped():
    # .old goes through the normal hash pipeline; no special case.
    assert config.is_skipped_file("x.jpg.old") is False


def test_kind_classification():
    assert config.kind_for(".jpg") == "image"
    assert config.kind_for(".raf") == "raw"
    assert config.kind_for(".dng") == "raw"
    assert config.kind_for(".mov") == "video"
    assert config.kind_for(".xmp") == "sidecar"
    assert config.kind_for(".txt") == "other"


def test_scope_and_exclusions_are_disjoint():
    assert len(config.SCOPE_ROOTS) == 11
    assert not (set(config.SCOPE_ROOTS) & config.HARD_EXCLUDE)


def test_lr_catalog_folders_are_hard_excluded():
    assert "New LR Catalog" in config.HARD_EXCLUDE
    assert "LR Backups" in config.HARD_EXCLUDE
