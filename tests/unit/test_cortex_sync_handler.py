from __future__ import annotations

from pathlib import Path
import zipfile

from worker.handlers.cortex_sync import (
    _collect_files_from_input_payload,
    _list_local_topic_dirs,
    _resolve_existing_path,
)


def test_resolve_existing_path_public_html_remap(tmp_path, monkeypatch):
    site_root = tmp_path / "site"
    target = site_root / "chatbot" / "knowledge" / "digital-health" / "doc.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("ok", encoding="utf-8")

    monkeypatch.setenv("CORTEX_SYNC_SITE_ROOT", str(site_root))
    # Submitted path resembles cPanel location and non-local username.
    submitted = "/home/longboar/public_html/chatbot/knowledge/digital-health/doc.md"
    resolved = _resolve_existing_path(submitted)
    assert Path(resolved) == target


def test_list_local_topic_dirs(tmp_path, monkeypatch):
    site_root = tmp_path / "site"
    knowledge = site_root / "chatbot" / "knowledge"
    (knowledge / "digital-health").mkdir(parents=True, exist_ok=True)
    (knowledge / "wellbeing").mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("CORTEX_SYNC_SITE_ROOT", str(site_root))
    topics = _list_local_topic_dirs()
    assert topics == ["digital-health", "wellbeing"]


def test_collect_files_from_uploaded_zip(tmp_path):
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    md_file = src_dir / "doc1.md"
    txt_file = src_dir / "doc2.txt"
    md_file.write_text("# Hello", encoding="utf-8")
    txt_file.write_text("hello", encoding="utf-8")
    zip_path = tmp_path / "payload.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(md_file, arcname="doc1.md")
        zf.write(txt_file, arcname="doc2.txt")

    files = _collect_files_from_input_payload(zip_path)
    names = sorted(Path(f).name for f in files)
    assert names == ["doc1.md", "doc2.txt"]
