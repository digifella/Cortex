from cortex_engine.url_ingestor import URLIngestResult
from worker.handlers import url_ingest


def _result(url, title):
    return URLIngestResult(input_url=url, page_title=title, status="web_markdown")


def test_single_url_report_has_no_generic_header_or_stats():
    report = url_ingest._build_markdown_report(
        [_result("https://example.com/a", "Page A")], {"total_urls": 1}
    )

    assert "## Page A" in report
    assert "**URL:** https://example.com/a" in report
    assert "URL Ingest Summary\nGenerated:" not in report
    assert "Generated:" not in report
    assert "Total URLs processed" not in report
    # frontmatter is closed once and the page heading follows it directly
    assert report.split("\n---\n", 1)[1].lstrip().startswith("## Page A")


def test_multi_url_report_keeps_overview():
    report = url_ingest._build_markdown_report(
        [_result("https://example.com/a", "Page A"), _result("https://example.com/b", "Page B")],
        {"total_urls": 2},
    )

    assert "# URL Ingest Summary" in report
    assert "**Total URLs processed:** 2" in report
    assert "## Page A" in report and "## Page B" in report
