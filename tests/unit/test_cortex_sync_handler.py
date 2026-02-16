from __future__ import annotations

from pathlib import Path

from worker.handlers.cortex_sync import _resolve_existing_path


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
