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


def test_handle_returns_rich_output_data(monkeypatch):
    monkeypatch.setattr(yt, "_fetch_youtube_metadata", lambda url: {
        "video_title": "Silverchain CEO on Community Care",
        "author": "Silverchain Group",
        "author_url": "https://www.youtube.com/@silverchain",
        "provider": "YouTube",
    })
    monkeypatch.setattr(yt, "_summarise_gemini", lambda url, model_name, output_modes: {
        "summary": "This is the summary.",
        "timestamps": "- [00:00] Intro",
    })
    monkeypatch.setattr(yt, "_generate_report_title", lambda url, api_choice, metadata, sections, index: "Community Care Strategy Update")

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
