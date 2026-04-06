from __future__ import annotations

from pathlib import Path

from cortex_engine.url_ingestor import URLIngestor


class _FakeResponse:
    def __init__(self, *, url: str, status_code: int, headers: dict[str, str] | None = None, text: str = ""):
        self.url = url
        self.status_code = status_code
        self.headers = headers or {}
        self.text = text


class _FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.headers = {}

    def get(self, url, timeout=None, allow_redirects=True, stream=False, headers=None):
        assert self.responses, f"Unexpected GET for {url}"
        return self.responses.pop(0)


def test_url_ingestor_captures_html_markdown_on_403_when_enabled(tmp_path):
    ingestor = URLIngestor(tmp_path)
    ingestor.session = _FakeSession(
        [
            _FakeResponse(
                url="https://example.org/paywalled",
                status_code=403,
                headers={"Content-Type": "text/html"},
                text="<html><head><title>Paywalled article</title></head><body><main>Abstract preview only.</main></body></html>",
            )
        ]
    )

    results = ingestor.process_urls(
        ["https://doi.org/10.1000/example"],
        capture_web_md_on_no_pdf=True,
    )

    assert len(results) == 1
    result = results[0]
    assert result.status == "web_markdown"
    assert result.reason == "web_page_captured_http_error"
    assert result.web_captured is True
    assert result.converted_to_md is True
    assert result.md_path
    assert Path(result.md_path).exists()


def test_url_ingestor_captures_minimal_markdown_for_thin_redirect_pages(tmp_path):
    ingestor = URLIngestor(tmp_path)
    ingestor.session = _FakeSession(
        [
            _FakeResponse(
                url="https://example.org/redirect",
                status_code=200,
                headers={"Content-Type": "text/html"},
                text="<html><head><title>Redirecting</title></head><body></body></html>",
            )
        ]
    )

    results = ingestor.process_urls(
        ["https://doi.org/10.1000/redirect"],
        capture_web_md_on_no_pdf=True,
    )

    assert len(results) == 1
    result = results[0]
    assert result.status == "web_markdown"
    assert result.web_captured is True
    assert result.md_path
    content = Path(result.md_path).read_text(encoding="utf-8")
    assert "Redirecting" in content
    assert "Captured web landing page from" in content


def test_url_ingestor_uses_unique_markdown_filenames_for_same_page_title(tmp_path):
    ingestor = URLIngestor(tmp_path)
    ingestor.session = _FakeSession(
        [
            _FakeResponse(
                url="https://example.org/a",
                status_code=403,
                headers={"Content-Type": "text/html"},
                text="<html><head><title>Just a moment...</title></head><body></body></html>",
            ),
            _FakeResponse(
                url="https://example.org/b",
                status_code=403,
                headers={"Content-Type": "text/html"},
                text="<html><head><title>Just a moment...</title></head><body></body></html>",
            ),
        ]
    )

    results = ingestor.process_urls(
        ["https://doi.org/10.1000/a", "https://doi.org/10.1000/b"],
        capture_web_md_on_no_pdf=True,
    )

    assert len(results) == 2
    assert results[0].status == "web_markdown"
    assert results[1].status == "web_markdown"
    assert results[0].md_path
    assert results[1].md_path
    assert results[0].md_path != results[1].md_path


def test_url_ingestor_uses_browser_fallback_to_find_pdf_on_block_page(tmp_path):
    ingestor = URLIngestor(tmp_path)
    ingestor.session = _FakeSession(
        [
            _FakeResponse(
                url="https://example.org/challenge",
                status_code=403,
                headers={"Content-Type": "text/html"},
                text="<html><head><title>Just a moment...</title></head><body>Checking if the site connection is secure</body></html>",
            )
        ]
    )
    ingestor._fetch_html_with_browser = lambda url: '<html><head><title>Article</title><meta name="citation_pdf_url" content="/content/example.pdf"></head><body>Article</body></html>'  # type: ignore[method-assign]
    ingestor._download_pdf = lambda pdf_url, title_hint: (True, "200", str(tmp_path / "pdfs" / "example.pdf"), "")  # type: ignore[method-assign]

    results = ingestor.process_urls(
        ["https://doi.org/10.1000/challenge"],
        capture_web_md_on_no_pdf=True,
    )

    assert len(results) == 1
    result = results[0]
    assert result.status == "downloaded"
    assert result.open_access_pdf_found is True
    assert result.pdf_url == "https://example.org/content/example.pdf"


def test_url_ingestor_uses_browser_fallback_to_find_pdf_on_thin_landing_page(tmp_path):
    ingestor = URLIngestor(tmp_path)
    ingestor.session = _FakeSession(
        [
            _FakeResponse(
                url="https://example.org/landing",
                status_code=200,
                headers={"Content-Type": "text/html"},
                text="<html><head><title>Redirecting</title></head><body></body></html>",
            )
        ]
    )
    ingestor._fetch_html_with_browser = lambda url: '<html><head><title>Article</title></head><body><a href="/downloads/paper.pdf">Download PDF</a></body></html>'  # type: ignore[method-assign]
    ingestor._download_pdf = lambda pdf_url, title_hint: (True, "200", str(tmp_path / "pdfs" / "paper.pdf"), "")  # type: ignore[method-assign]

    results = ingestor.process_urls(
        ["https://doi.org/10.1000/landing"],
        capture_web_md_on_no_pdf=True,
    )

    assert len(results) == 1
    result = results[0]
    assert result.status == "downloaded"
    assert result.open_access_pdf_found is True
    assert result.pdf_url == "https://example.org/downloads/paper.pdf"
