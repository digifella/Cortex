import pytest

from worker import lab_youtube_backlog as backlog
from worker import lab_youtube_direct as direct


BODY = """Your Lab job is complete — YouTube Summariser (#2587)
Completed: 2026-10-12T18:04:26 UTC

## FlyDubai story: commentary and sources
**Clip title:** A story
**URL:** https://youtu.be/NiIXMgIITO8?si=abc

### Summary
The video discusses an aviation incident and cites several sources.
"""


def test_backlog_stage_uses_existing_summary_without_publication(tmp_path, monkeypatch):
    monkeypatch.setattr(direct, "STAGING", tmp_path / "private")
    monkeypatch.setattr(direct, "STATE", tmp_path / "state.json")
    monkeypatch.setattr(backlog.direct, "classify_slug_safe", lambda *_: ("Transport", "aviation"))
    state = {}
    result = backlog.stage("gmail-123", BODY, "NiIXMgIITO8", state)
    assert result["status"] == "private_review"
    path = tmp_path / "private" / next((tmp_path / "private").iterdir()).name
    text = path.read_text()
    assert "wiki-ready: false" in text
    assert 'credibility_tier_value: 1' in text
    assert 'source_date: "2026-10-12"' in text
    assert state[result["id"]]["discord_notified"] is False
    assert backlog.stage("gmail-123", BODY, "NiIXMgIITO8", state)["status"] == "already_staged"


def test_backlog_rejects_wrong_video_id():
    with pytest.raises(ValueError, match="differs"):
        backlog.extract_completed_report(BODY, "l3u_FAv33G0")
