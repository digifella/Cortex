from pathlib import Path

from worker import lab_youtube_direct as direct
import pytest


def test_youtube_urls_dedupes_share_links_and_ignores_other_urls():
    body = (
        "See https://youtu.be/l3u_FAv33G0?si=abc and "
        "https://www.youtube.com/watch?v=l3u_FAv33G0, plus "
        "https://example.com/test and https://youtu.be/NiIXMgIITO8."
    )
    assert direct.youtube_urls(body) == [
        "https://www.youtube.com/watch?v=l3u_FAv33G0",
        "https://www.youtube.com/watch?v=NiIXMgIITO8",
    ]


def test_direct_lab_result_is_saved_mailed_and_idempotent(tmp_path, monkeypatch):
    monkeypatch.setattr(direct, "STATE", tmp_path / "state.json")
    monkeypatch.setattr(direct, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(direct, "STAGING", tmp_path / "private" / "Public_Review")
    report = "---\nsource_type: youtube_summary\n---\n\n## Example\n**URL:** https://youtu.be/l3u_FAv33G0\n### Summary\nThe speaker discusses AI.\n"
    output = tmp_path / "generated.md"
    output.write_text(report)
    calls = []

    def summarize(urls):
        calls.append((urls,))
        return {"output_data": {"success_count": 1, "error_count": 0,
                                "report_title": "Example"}, "output_file": output}

    monkeypatch.setattr(direct, "summarise_urls", summarize)

    class Graph:
        def send_reply(self, recipient, subject, body):
            calls.append((recipient, subject, body))

        def mark_read(self, graph_id):
            calls.append(("read", graph_id))

    item = {
        "id": "graph-123", "internetMessageId": "<abc@sender>",
        "from": {"emailAddress": {"address": "paul@project-143.com"}},
        "receivedDateTime": "2026-10-15T07:00:00Z",
        "body": {"contentType": "text", "content": "https://youtu.be/l3u_FAv33G0"},
    }
    state = {}
    assert direct.process_message(Graph(), item, state)
    drafts = list((tmp_path / "private" / "Public_Review").glob("*.md"))
    assert len(drafts) == 1
    assert "wiki-ready: false" in drafts[0].read_text()
    assert "review-status: pending" in drafts[0].read_text()
    assert len([call for call in calls if isinstance(call[0], list)]) == 1
    assert len([call for call in calls if call[0] == direct.OWNER]) == 1
    assert direct.process_message(Graph(), item, state)
    assert len([call for call in calls if call[0] == direct.OWNER]) == 1
    assert calls[-1] == ("read", "graph-123")


def test_direct_lab_rejects_non_owner_and_pre_cutover(tmp_path, monkeypatch):
    monkeypatch.setattr(direct, "STAGING", tmp_path / "private")

    class Graph:
        def send_reply(self, *_):
            raise AssertionError("email must not be sent")

        def mark_read(self, *_):
            raise AssertionError("message must not change")

    item = {
        "id": "graph-123", "from": {"emailAddress": {"address": "stranger@example.com"}},
        "receivedDateTime": "2026-10-15T07:00:00Z",
        "body": {"contentType": "text", "content": "https://youtu.be/l3u_FAv33G0"},
    }
    assert direct.process_message(Graph(), item, {}) is False
    item["from"]["emailAddress"]["address"] = "paul@project-143.com"
    item["receivedDateTime"] = "2026-10-15T06:00:00Z"
    assert direct.process_message(Graph(), item, {}) is False
    assert not (tmp_path / "private").exists()


def test_discord_review_notice_retries_without_resummarising(tmp_path, monkeypatch):
    from worker import lab_youtube_digest as digest
    from worker import lab_youtube_review as review

    monkeypatch.setattr(direct, "STATE", tmp_path / "state.json")
    staged = tmp_path / "private.md"
    staged.write_text("wiki-ready: false\n")
    state = {"abcdefgh1234": {"staged_path": str(staged), "review_status": "pending",
                               "discord_notified": False, "title": "Example", "urls": ["https://youtu.be/l3u_FAv33G0"]}}
    direct._save_state(state)
    monkeypatch.setattr(review, "pending", lambda: [{"id": "abcdefgh1234", "title": "Example",
                                                      "topic": "AI & Agents", "urls": ["https://youtu.be/l3u_FAv33G0"]}])
    calls = []

    def send_cards(items):
        calls.append(items)
        if len(calls) == 1:
            raise RuntimeError("relay down")

    monkeypatch.setattr(digest, "send_review_cards", send_cards)
    with pytest.raises(RuntimeError, match="relay down"):
        direct.send_pending_review_notice(state)
    assert state["abcdefgh1234"]["discord_notified"] is False
    assert direct.send_pending_review_notice(state)
    assert direct._load_state()["abcdefgh1234"]["discord_notified"] is True
    assert not direct.send_pending_review_notice(state)
    assert len(calls) == 2


def test_discord_intake_notice_lists_only_new_drafts(tmp_path, monkeypatch):
    from worker import lab_youtube_digest as digest
    from worker import lab_youtube_review as review

    monkeypatch.setattr(direct, "STATE", tmp_path / "state.json")
    fresh = tmp_path / "fresh.md"
    old = tmp_path / "old.md"
    fresh.write_text("wiki-ready: false\n")
    old.write_text("wiki-ready: false\n")
    state = {
        "12345678abcdef": {"staged_path": str(fresh), "discord_notified": False},
        "87654321abcdef": {"staged_path": str(old), "discord_notified": True},
    }
    monkeypatch.setattr(review, "pending", lambda: [
        {"id": "12345678abcdef", "title": "Fresh video", "topic": "AI & Agents", "urls": []},
        {"id": "87654321abcdef", "title": "Old video", "topic": "AI & Agents", "urls": []},
    ])
    messages = []
    monkeypatch.setattr(digest, "send_review_cards", messages.append)

    assert direct.send_pending_review_notice(state)
    assert len(messages) == 1
    assert [item["title"] for item in messages[0]] == ["Fresh video"]
