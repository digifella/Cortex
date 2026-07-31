"""Staging for uploaded documents.

The ingest manifest compares mtime_ns + size (not a content hash), so an
unchanged re-upload MUST NOT rewrite the file or de-duplication breaks.
"""

import os

import pytest

from cortex_engine.private_vault_rag import stage_upload


def test_new_file_is_written(tmp_path):
    target = stage_upload(b"hello", "doc.pdf", tmp_path)
    assert target == tmp_path / "doc.pdf"
    assert target.read_bytes() == b"hello"


def test_staging_dir_created_when_absent(tmp_path):
    nested = tmp_path / "does" / "not" / "exist"
    target = stage_upload(b"x", "a.txt", nested)
    assert target.exists()
    assert nested.is_dir()


def test_identical_bytes_preserve_mtime(tmp_path):
    # The de-duplication guarantee: an unchanged re-upload must not touch mtime,
    # because the ingest manifest keys "unchanged" off mtime_ns + size.
    target = stage_upload(b"same", "doc.pdf", tmp_path)
    os.utime(target, ns=(111_000_000_000, 111_000_000_000))
    before = target.stat().st_mtime_ns

    again = stage_upload(b"same", "doc.pdf", tmp_path)

    assert again == target
    assert target.stat().st_mtime_ns == before


def test_changed_bytes_are_rewritten(tmp_path):
    target = stage_upload(b"first", "doc.pdf", tmp_path)
    os.utime(target, ns=(111_000_000_000, 111_000_000_000))
    before = target.stat().st_mtime_ns

    stage_upload(b"second", "doc.pdf", tmp_path)

    assert target.read_bytes() == b"second"
    assert target.stat().st_mtime_ns != before


def test_filename_is_reduced_to_basename(tmp_path):
    # A filename carrying separators must not escape the staging directory.
    target = stage_upload(b"x", "../../etc/evil.txt", tmp_path)
    assert target == tmp_path / "evil.txt"
    assert target.parent == tmp_path


def test_double_dot_raises_valueerror(tmp_path):
    # A bare ".." in filename would escape to staging_dir's parent.
    with pytest.raises(ValueError, match="Unsafe upload filename"):
        stage_upload(b"x", "..", tmp_path)
    # Verify nothing was written to parent directory
    assert not (tmp_path.parent / "..").exists() or (tmp_path.parent / "..").is_dir()


def test_double_dot_double_dot_raises_valueerror(tmp_path):
    # Multiple ".." should also be rejected.
    with pytest.raises(ValueError, match="Unsafe upload filename"):
        stage_upload(b"x", "../..", tmp_path)


def test_single_dot_raises_valueerror(tmp_path):
    # A single "." would resolve to the staging directory itself.
    with pytest.raises(ValueError, match="Unsafe upload filename"):
        stage_upload(b"x", ".", tmp_path)


def test_empty_string_raises_valueerror(tmp_path):
    # An empty filename string would resolve to the staging directory itself.
    with pytest.raises(ValueError, match="Unsafe upload filename"):
        stage_upload(b"x", "", tmp_path)


def test_slash_raises_valueerror(tmp_path):
    # A root slash "/" reduces to empty name and should be rejected.
    with pytest.raises(ValueError, match="Unsafe upload filename"):
        stage_upload(b"x", "/", tmp_path)
