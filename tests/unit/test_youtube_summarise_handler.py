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
    meta = yt._fetch_youtube_metadata("https://youtu.be/example")

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
            "url": "https://youtu.be/example",
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
            "urls": ["https://youtu.be/example"],
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
