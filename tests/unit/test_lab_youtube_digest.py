import json

from worker import lab_youtube_digest as digest
from worker import lab_youtube_review as review

format_digest_pages = digest.format_digest_pages


def test_digest_paginates_every_review_without_youtube_previews():
    items = [
        {"id": f"{number:08x}abcdef", "title": f"Video {number} " + "A long descriptive title " * 4,
         "topic": "Biology & Life Sciences", "urls": ["https://youtu.be/example"]}
        for number in range(20)
    ]
    pages = format_digest_pages(items)
    assert len(pages) > 1
    assert all(len(page) <= 1700 for page in pages)
    combined = "\n".join(pages)
    assert all(f"`{number:08x}`" in combined for number in range(20))
    assert "youtu.be" not in combined
    assert "approve 12345678 as Biology & Life Sciences" in combined
    assert "(1/" in pages[0]


def test_review_cards_use_canonical_vault_topics_and_one_message_per_id(tmp_path, monkeypatch):
    for topic in ("AI & Agents", "Science & Physics", "Biology & Life Sciences"):
        (tmp_path / topic).mkdir()
    monkeypatch.setattr(digest, "TOPICS", tmp_path)
    monkeypatch.setattr(review, "TOPICS", tmp_path)
    calls = []

    def fake_run(command, **kwargs):
        payload = json.loads(kwargs["input"])
        calls.append(payload)
        class Result:
            returncode = 0
            stdout = json.dumps({"id": payload["id"], "status": "sent"})
        return Result()

    monkeypatch.setattr(digest.subprocess, "run", fake_run)
    items = [{"id": "12345678abcdef", "title": "Example", "topic": "Science, Physics & Research"}]
    assert digest.send_review_cards(items) == 1
    assert calls[0]["topics"] == ["AI & Agents", "Biology & Life Sciences", "Science & Physics"]
    assert calls[0]["topic"] == "Science & Physics"
    assert calls[0]["id"] == items[0]["id"]
