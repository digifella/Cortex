import json

from worker.handlers import youtube_summarise as yt


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeTextResponse:
    def __init__(self, text: str):
        self._text = text.encode("utf-8")

    def read(self, *args, **kwargs):
        return self._text

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_fetch_youtube_metadata_uses_oembed(monkeypatch):
    def fake_urlopen(url, timeout=0):
        assert "youtube.com/oembed" in url
        assert "format=json" in url
        return _FakeResponse({
            "title": "ACCC Chair on Competition Reform",
            "author_name": "ACCC",
            "author_url": "https://www.youtube.com/@accc",
            "provider_name": "YouTube",
        })

    monkeypatch.setattr(yt.urllib.request, "urlopen", fake_urlopen)
    meta = yt._fetch_youtube_metadata("https://youtu.be/qpEZoK2JmOg")

    assert meta["video_title"] == "ACCC Chair on Competition Reform"
    assert meta["author"] == "ACCC"
    assert meta["author_url"] == "https://www.youtube.com/@accc"
    assert meta["provider"] == "YouTube"


def test_youtube_helpers_parse_duration_and_canonical_url():
    assert yt._canonical_youtube_url("https://youtu.be/qpEZoK2JmOg?si=tracking") == "https://www.youtube.com/watch?v=qpEZoK2JmOg"
    assert yt._canonical_youtube_url("https://www.youtube.com/watch?v=qpEZoK2JmOg&si=tracking") == "https://www.youtube.com/watch?v=qpEZoK2JmOg"
    assert yt._parse_iso8601_duration("PT1H3M12S") == 3792
    assert yt._parse_iso8601_duration("PT24M59S") == 1499


def test_fetch_duration_from_watch_page(monkeypatch):
    def fake_urlopen(req, timeout=0):
        assert "youtube.com/watch" in req.full_url
        return _FakeTextResponse('"lengthSeconds":"3805"')

    monkeypatch.setattr(yt.urllib.request, "urlopen", fake_urlopen)

    assert yt._fetch_youtube_duration_seconds_from_watch_page("qpEZoK2JmOg") == 3805


def test_long_youtube_uses_lowres_gemini_rest(monkeypatch):
    calls = []

    class FakeGenAI:
        class GenerativeModel:
            def __init__(self, model_id):
                self.model_id = model_id

            def generate_content(self, *args, **kwargs):
                raise AssertionError("low-res long videos should use the REST path")

    monkeypatch.setattr(yt, "_gemini_client", lambda: FakeGenAI)
    monkeypatch.setattr(yt, "_fetch_youtube_duration_seconds", lambda url: 63 * 60)

    def fake_rest(model_id, prompt, youtube_url, *, media_resolution="", timeout=0):
        calls.append({
            "model_id": model_id,
            "youtube_url": youtube_url,
            "media_resolution": media_resolution,
            "timeout": timeout,
        })
        return "Long video summary"

    monkeypatch.setattr(yt, "_generate_gemini_rest", fake_rest)

    result = yt._summarise_gemini(
        "https://youtu.be/qpEZoK2JmOg?si=tracking",
        "gemini-flash",
        ["summary"],
    )

    assert result["summary"] == "Long video summary"
    assert calls == [{
        "model_id": "gemini-2.5-flash",
        "youtube_url": "https://www.youtube.com/watch?v=qpEZoK2JmOg",
        "media_resolution": "MEDIA_RESOLUTION_LOW",
        "timeout": 180,
    }]


def test_timeout_retries_lowres_when_duration_lookup_misses(monkeypatch):
    calls = []

    class FakeGenAI:
        class GenerativeModel:
            def __init__(self, model_id):
                self.model_id = model_id

            def generate_content(self, *args, **kwargs):
                raise RuntimeError("504 The request timed out. Please try again.")

    monkeypatch.setattr(yt, "_gemini_client", lambda: FakeGenAI)
    monkeypatch.setattr(yt, "_fetch_youtube_duration_seconds", lambda url: None)

    def fake_rest(model_id, prompt, youtube_url, *, media_resolution="", timeout=0):
        calls.append({
            "model_id": model_id,
            "youtube_url": youtube_url,
            "media_resolution": media_resolution,
            "timeout": timeout,
        })
        return "Recovered via low-res retry"

    monkeypatch.setattr(yt, "_generate_gemini_rest", fake_rest)

    result = yt._summarise_gemini(
        "https://youtu.be/WajgNhbbeHM?si=tracking",
        "gemini-flash",
        ["summary"],
    )

    assert result["summary"] == "Recovered via low-res retry"
    assert calls == [{
        "model_id": "gemini-2.5-flash",
        "youtube_url": "https://www.youtube.com/watch?v=WajgNhbbeHM",
        "media_resolution": "MEDIA_RESOLUTION_LOW",
        "timeout": 180,
    }]


def test_build_report_includes_report_title_and_clip_metadata():
    report = yt._build_report(
        results=[{
            "url": "https://youtu.be/qpEZoK2JmOg",
            "report_title": "Competition Reform Priorities",
            "video_title": "Chair Discusses Competition Reform",
            "author": "ACCC",
            "sections": {"summary": "A concise summary."},
        }],
        output_modes=["summary"],
        api_choice="gemini-flash",
    )

    assert "title: Competition Reform Priorities" in report
    assert "# Competition Reform Priorities" in report
    assert "**Clip title:** Chair Discusses Competition Reform" in report
    assert "**Author / channel:** ACCC" in report


def test_remove_sponsor_paragraphs_drops_sponsor_copy():
    text = (
        "The video explains the latest changes to community care delivery.\n\n"
        "The video introduces Zocdoc as a sponsor for easily finding and booking doctor appointments.\n\n"
        "It closes by outlining practical implementation risks for providers."
    )

    cleaned = yt._remove_sponsor_paragraphs(text)

    assert "Zocdoc" not in cleaned
    assert "community care delivery" in cleaned
    assert "implementation risks" in cleaned


def test_chunk_transcript_entries_splits_requested_windows():
    chunks = yt._chunk_transcript_entries(
        [
            {"start": 0, "duration": 15, "text": "Intro"},
            {"start": 1800, "duration": 20, "text": "Middle"},
            {"start": 3700, "duration": 20, "text": "Second hour"},
        ],
        chunk_duration_seconds=3600,
        start_time_seconds=0,
        end_time_seconds=7200,
    )

    assert len(chunks) == 2
    assert chunks[0]["start_seconds"] == 0
    assert chunks[0]["end_seconds"] == 3600
    assert [item["text"] for item in chunks[0]["entries"]] == ["Intro", "Middle"]
    assert chunks[1]["start_seconds"] == 3600
    assert chunks[1]["end_seconds"] == 7200
    assert [item["text"] for item in chunks[1]["entries"]] == ["Second hour"]


def test_handle_chunked_transcript_mode_returns_chunk_metadata(monkeypatch):
    monkeypatch.setattr(yt, "_fetch_youtube_metadata", lambda url: {
        "video_title": "Long Interview",
        "author": "Channel",
        "author_url": "https://www.youtube.com/@channel",
        "provider": "YouTube",
    })
    monkeypatch.setattr(yt, "_fetch_youtube_extra_context", lambda url: {})
    monkeypatch.setattr(yt, "_extract_transcript_entries", lambda url: [
        {"start": 0, "duration": 30, "text": "Opening remarks"},
        {"start": 3500, "duration": 30, "text": "Closing first hour"},
        {"start": 3700, "duration": 30, "text": "Second hour starts"},
    ])
    monkeypatch.setattr(yt, "_summarise_gemini_transcript", lambda transcript, context_label, model_name, output_modes, language="": {
        "summary": f"Summary for {context_label}",
        "timestamps": "Chunk timestamps",
    })
    monkeypatch.setattr(yt, "_generate_report_title", lambda url, api_choice, metadata, sections, index, language="": "Long Interview Summary")

    result = yt.handle(
        None,
        {
            "urls": ["https://youtu.be/qpEZoK2JmOg"],
            "api_choice": "gemini-flash",
            "output_modes": ["summary", "timestamps"],
            "youtube_options": {
                "chunk_duration_seconds": 3600,
                "start_time_seconds": 0,
                "end_time_seconds": 7200,
            },
        },
        {"id": 321},
    )

    output_data = result["output_data"]
    assert output_data["chunked"] is True
    assert output_data["chunk_count"] == 2
    assert output_data["videos"][0]["chunk_count"] == 2
    rendered = result["output_file"].read_text(encoding="utf-8")
    assert "### Chunk 1 (00:00 - 01:00:00)" in rendered
    assert "### Chunk 2 (01:00:00 - 02:00:00)" in rendered
    assert "Processed as:** 2 time chunk(s)" in rendered


def test_is_youtube_url_rejects_non_youtube_and_truncated():
    assert yt._is_youtube_url("https://youtu.be/qpEZoK2JmOg") is True
    assert yt._is_youtube_url("https://www.youtube.com/watch?v=qpEZoK2JmOg") is True
    # the reported failures: a website URL and a truncated 10-char id
    assert yt._is_youtube_url("http://www.longboardfella.com.au") is False
    assert yt._is_youtube_url("https://www.youtube.com/watch?v=ije_fF8SWc") is False  # 10 chars


def test_non_youtube_url_skipped_not_sent_to_gemini(monkeypatch):
    monkeypatch.setattr(yt, "_fetch_youtube_metadata", lambda url: {"video_title": "", "author": "", "author_url": "", "provider": "YouTube"})
    monkeypatch.setattr(yt, "_fetch_youtube_extra_context", lambda url: {})

    def _must_not_call(*a, **k):
        raise AssertionError("non-YouTube URL must never reach Gemini")
    monkeypatch.setattr(yt, "_summarise_gemini", _must_not_call)
    monkeypatch.setattr(yt, "_generate_report_title", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no title gen for skipped URL")))

    result = yt.handle(None, {"urls": ["http://www.longboardfella.com.au"], "api_choice": "gemini-flash"}, {"id": 2087})
    od = result["output_data"]
    assert od["error_count"] == 1
    assert od["videos_processed"] == 0
    assert "youtube" in od["errors"][0]["error"].lower()


def test_mixed_batch_skips_invalid_processes_valid(monkeypatch):
    monkeypatch.setattr(yt, "_fetch_youtube_metadata", lambda url: {"video_title": "V", "author": "A", "author_url": "", "provider": "YouTube"})
    monkeypatch.setattr(yt, "_fetch_youtube_extra_context", lambda url: {})
    seen = []
    def fake_summary(url, model_name, output_modes, language=""):
        seen.append(url)
        return {"summary": "Real summary."}
    monkeypatch.setattr(yt, "_summarise_gemini", fake_summary)
    monkeypatch.setattr(yt, "_generate_report_title", lambda *a, **k: "Title")

    result = yt.handle(None, {"urls": ["http://www.longboardfella.com.au", "https://youtu.be/qpEZoK2JmOg"], "api_choice": "gemini-flash"}, {"id": 1})
    od = result["output_data"]
    assert od["videos_processed"] == 1
    assert od["error_count"] == 1
    assert seen == ["https://youtu.be/qpEZoK2JmOg"]  # only the valid URL hit Gemini


def test_no_vault_note_when_all_content_failed(monkeypatch):
    monkeypatch.setattr(yt, "_fetch_youtube_metadata", lambda url: {"video_title": "", "author": "", "author_url": "", "provider": "YouTube"})
    monkeypatch.setattr(yt, "_fetch_youtube_extra_context", lambda url: {})
    monkeypatch.setattr(yt, "_summarise_gemini", lambda *a, **k: {"summary": "[Error generating summary: 400 Request contains an invalid argument.]"})
    monkeypatch.setattr(yt, "_generate_report_title", lambda *a, **k: "Should Not Publish")
    wrote = []
    monkeypatch.setattr(yt, "_write_vault_lab_note", lambda content, title, job: wrote.append(title))

    # a valid URL but the summary errored -> no successful content -> no vault note
    yt.handle(None, {"urls": ["https://youtu.be/qpEZoK2JmOg"], "api_choice": "gemini-flash", "source_system": "email"}, {"id": 9})
    assert wrote == []


def test_vault_note_written_on_success(monkeypatch):
    monkeypatch.setattr(yt, "_fetch_youtube_metadata", lambda url: {"video_title": "", "author": "", "author_url": "", "provider": "YouTube"})
    monkeypatch.setattr(yt, "_fetch_youtube_extra_context", lambda url: {})
    monkeypatch.setattr(yt, "_summarise_gemini", lambda *a, **k: {"summary": "A real summary."})
    monkeypatch.setattr(yt, "_generate_report_title", lambda *a, **k: "Good Title")
    wrote = []
    monkeypatch.setattr(yt, "_write_vault_lab_note", lambda content, title, job: wrote.append(title))

    yt.handle(None, {"urls": ["https://youtu.be/qpEZoK2JmOg"], "api_choice": "gemini-flash", "source_system": "email"}, {"id": 10})
    assert wrote == ["Good Title"]


def test_handle_returns_rich_output_data(monkeypatch):
    monkeypatch.setattr(yt, "_fetch_youtube_metadata", lambda url: {
        "video_title": "Silverchain CEO on Community Care",
        "author": "Silverchain Group",
        "author_url": "https://www.youtube.com/@silverchain",
        "provider": "YouTube",
    })
    monkeypatch.setattr(yt, "_summarise_gemini", lambda url, model_name, output_modes, language="": {
        "summary": (
            "This is the summary.\n\n"
            "The video introduces Zocdoc as a sponsor for easily finding and booking doctor appointments."
        ),
        "timestamps": "- [00:00] Intro",
    })
    monkeypatch.setattr(yt, "_generate_report_title", lambda url, api_choice, metadata, sections, index, language="": "Community Care Strategy Update")

    result = yt.handle(
        None,
        {
            "urls": ["https://youtu.be/qpEZoK2JmOg"],
            "api_choice": "gemini-flash",
            "output_modes": ["summary", "timestamps"],
        },
        {"id": 123},
    )

    output_data = result["output_data"]
    assert output_data["provider"] == "Google"
    assert output_data["model"] == "Gemini 2.5 Flash"
    assert output_data["video_count"] == 1
    assert output_data["videos_processed"] == 1
    assert output_data["report_title"] == "Community Care Strategy Update"
    assert output_data["videos"][0]["clip_title"] == "Silverchain CEO on Community Care"
    assert output_data["videos"][0]["author"] == "Silverchain Group"
    assert result["output_file"].exists()
    rendered = result["output_file"].read_text(encoding="utf-8")
    assert "Zocdoc" not in rendered
