from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from cortex_engine.llm_metadata_sync.models import (
    SidecarAction,
    SyncAction,
    SyncConfig,
    TargetType,
)
from cortex_engine.llm_metadata_sync import sync


def _cfg(tmp_path: Path, **kwargs) -> SyncConfig:
    defaults = dict(
        raw_root=tmp_path / "raw",
        jpg_dir=tmp_path / "jpg",
        dry_run=False,
    )
    defaults.update(kwargs)
    return SyncConfig(**defaults)


def _make_files(tmp_path: Path, names: list[str]) -> None:
    (tmp_path / "raw").mkdir(exist_ok=True)
    (tmp_path / "jpg").mkdir(exist_ok=True)
    for name in names:
        (tmp_path / "jpg" / f"{name}.jpg").touch()
        (tmp_path / "raw" / f"{name}.RAF").touch()


def _ok_result() -> MagicMock:
    r = MagicMock()
    r.ok = True
    r.filtered_stderr = ""
    return r


def _fail_result(msg: str = "permission denied") -> MagicMock:
    r = MagicMock()
    r.ok = False
    r.filtered_stderr = msg
    return r


def test_dry_run_does_not_call_exiftool(tmp_path):
    _make_files(tmp_path, ["shot"])
    cfg = _cfg(tmp_path, dry_run=True)

    with patch("cortex_engine.llm_metadata_sync.sync.read_jpg_metadata", return_value=(["kw1"], "desc")), \
         patch("cortex_engine.llm_metadata_sync.sync.read_existing_keywords", return_value=[]), \
         patch("cortex_engine.llm_metadata_sync.exiftool_runner.clear_keyword_lists") as mock_clear, \
         patch("cortex_engine.llm_metadata_sync.exiftool_runner.write_metadata") as mock_write:
        results = list(sync.run_sync(cfg))

    mock_clear.assert_not_called()
    mock_write.assert_not_called()
    assert len(results) == 1
    assert results[0].success is True
    assert results[0].keywords_written == 1


def test_one_file_failure_does_not_abort_others(tmp_path):
    _make_files(tmp_path, ["a", "b"])
    (tmp_path / "raw" / "a.xmp").touch()
    (tmp_path / "raw" / "b.xmp").touch()
    cfg = _cfg(tmp_path)

    with patch("cortex_engine.llm_metadata_sync.sync.read_jpg_metadata", return_value=(["kw"], "desc")), \
         patch("cortex_engine.llm_metadata_sync.sync.read_existing_keywords", return_value=[]), \
         patch("cortex_engine.llm_metadata_sync.exiftool_runner.clear_keyword_lists",
               side_effect=[_fail_result(), _ok_result()]), \
         patch("cortex_engine.llm_metadata_sync.exiftool_runner.write_metadata",
               return_value=_ok_result()):
        results = list(sync.run_sync(cfg))

    assert len(results) == 2
    assert any(not r.success for r in results)
    assert any(r.success for r in results)


def test_jpg_with_no_metadata_skips_exiftool(tmp_path):
    _make_files(tmp_path, ["empty"])
    cfg = _cfg(tmp_path)

    with patch("cortex_engine.llm_metadata_sync.sync.read_jpg_metadata", return_value=([], "")), \
         patch("cortex_engine.llm_metadata_sync.exiftool_runner.clear_keyword_lists") as mock_clear, \
         patch("cortex_engine.llm_metadata_sync.exiftool_runner.write_metadata") as mock_write:
        results = list(sync.run_sync(cfg))

    mock_clear.assert_not_called()
    mock_write.assert_not_called()
    assert results[0].success is True
    assert results[0].keywords_written == 0


def test_orphaned_jpgs_do_not_produce_results(tmp_path):
    (tmp_path / "raw").mkdir()
    (tmp_path / "jpg").mkdir()
    (tmp_path / "jpg" / "orphan.jpg").touch()
    cfg = _cfg(tmp_path)

    with patch("cortex_engine.llm_metadata_sync.sync.read_jpg_metadata", return_value=(["kw"], "desc")):
        results = list(sync.run_sync(cfg))

    assert results == []


def test_clear_called_before_write_for_existing_sidecar(tmp_path):
    _make_files(tmp_path, ["shot"])
    (tmp_path / "raw" / "shot.xmp").touch()
    cfg = _cfg(tmp_path)

    call_order = []

    def mock_clear(*a, **kw):
        call_order.append("clear")
        return _ok_result()

    def mock_write(*a, **kw):
        call_order.append("write")
        return _ok_result()

    with patch("cortex_engine.llm_metadata_sync.sync.read_jpg_metadata", return_value=(["kw"], "desc")), \
         patch("cortex_engine.llm_metadata_sync.sync.read_existing_keywords", return_value=[]), \
         patch("cortex_engine.llm_metadata_sync.exiftool_runner.clear_keyword_lists", side_effect=mock_clear), \
         patch("cortex_engine.llm_metadata_sync.exiftool_runner.write_metadata", side_effect=mock_write):
        list(sync.run_sync(cfg))

    assert call_order == ["clear", "write"]
