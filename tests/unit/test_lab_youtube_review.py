from pathlib import Path
import importlib.util

from worker import lab_youtube_direct as direct
from worker import lab_youtube_review as review


def _setup(tmp_path, monkeypatch):
    staging = tmp_path / "private" / "lab-youtube"
    staging.mkdir(parents=True)
    draft = staging / "draft.md"
    draft.write_text("---\nwiki-ready: false\nreview-status: pending\ndomain: undecided\n"
                     "title: Example\n---\n\n## Example\nSummary.\n")
    topics = tmp_path / "public" / "Topics"
    (topics / "Entertainment & Games").mkdir(parents=True)
    (topics / "AI & Agents").mkdir()
    monkeypatch.setattr(direct, "STATE", tmp_path / "state.json")
    monkeypatch.setattr(direct, "STAGING", staging)
    monkeypatch.setattr(review, "TOPICS", topics)
    monkeypatch.setattr(review, "PUBLIC", tmp_path / "public" / "lab-notes")
    monkeypatch.setattr(review, "DECLINED", tmp_path / "private" / "declined")
    key = "0123456789abcdef0123"
    direct._save_state({key: {"title": "Example", "staged_path": str(draft),
                              "suggested_topic": "AI & Agents", "urls": ["https://youtu.be/l3u_FAv33G0"]}})
    return key, draft


def test_redirect_then_approve_records_selected_topic(tmp_path, monkeypatch):
    key, draft = _setup(tmp_path, monkeypatch)
    changed = review.decide(key[:8], "redirect", "Entertainment & Games")
    assert changed["topic"] == "Entertainment & Games"
    assert draft.exists()
    assert "wiki-ready: false" in draft.read_text()
    approved = review.decide(key[:8], "approve")
    public = Path(approved["path"])
    assert public.exists() and not draft.exists()
    text = public.read_text()
    assert "wiki-ready: true" in text
    assert 'domain: "entertainment-games"' in text
    organiser_source = Path.home() / "nemoclaw-vault-organiser.py"
    spec = importlib.util.spec_from_file_location("vault_organiser_under_test", organiser_source)
    organiser = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(organiser)
    _prepared, folder = organiser.prepare_note(text, "lab-notes/draft.md")
    assert folder == "Entertainment & Games"
    assert not review.pending()


def test_decline_keeps_private_archive(tmp_path, monkeypatch):
    key, draft = _setup(tmp_path, monkeypatch)
    result = review.decide(key[:8], "decline")
    assert result["status"] == "declined"
    assert not draft.exists()
    assert Path(result["path"]).exists()
    assert not review.PUBLIC.exists()
