from pathlib import Path

import pytest

from cortex_engine.pre_ingest_organizer import PreIngestScanCancelled, run_pre_ingest_organizer_scan


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_pre_ingest_manifest_generation_and_policy_tags(tmp_path):
    source_root = tmp_path / "knowledge_source"
    db_root = tmp_path / "db_root"

    _write(source_root / "reports" / "strategy_report_v1.txt", "draft internal report")
    _write(source_root / "reports" / "strategy_report_v2.txt", "final research report")
    _write(source_root / "admin" / "invoice_2026.txt", "invoice amount due")
    _write(source_root / "clientA" / "client_confidential_brief.txt", "strictly confidential client brief")
    _write(source_root / "market" / "deloitte_industry_scan.txt", "industry report by Deloitte")
    _write(source_root / "do_not_ingest" / "private_note.txt", "internal note")

    result = run_pre_ingest_organizer_scan(
        source_dirs=[source_root.as_posix()],
        db_path=db_root.as_posix(),
    )

    manifest_path = Path(result["manifest_path"])
    assert manifest_path.exists()

    import json

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = payload["records"]
    by_name = {r["file_name"]: r for r in records}

    assert len(records) == 6

    assert by_name["strategy_report_v2.txt"]["is_canonical_version"] is True
    assert by_name["strategy_report_v1.txt"]["ingest_policy_class"] == "exclude"

    assert by_name["invoice_2026.txt"]["ingest_policy_class"] == "exclude"
    assert by_name["client_confidential_brief.txt"]["ingest_policy_class"] == "review_required"
    assert by_name["client_confidential_brief.txt"]["source_ownership"] == "client_owned"

    assert by_name["deloitte_industry_scan.txt"]["source_ownership"] == "external_ip"
    assert by_name["deloitte_industry_scan.txt"]["ingest_policy_class"] == "review_required"
    assert by_name["deloitte_industry_scan.txt"]["external_ip_owner"] == "Deloitte"

    assert by_name["private_note.txt"]["ingest_policy_class"] == "do_not_ingest"

    summary = payload["summary"]["policy_counts"]
    assert summary["exclude"] >= 2
    assert summary["review_required"] >= 2
    assert summary["do_not_ingest"] == 1


def test_pre_ingest_custom_policy_markers(tmp_path):
    source_root = tmp_path / "source"
    db_root = tmp_path / "db"

    _write(source_root / "acme" / "engagement_note.txt", "contains proprietary details")

    result = run_pre_ingest_organizer_scan(
        source_dirs=[source_root.as_posix()],
        db_path=db_root.as_posix(),
        config={
            "client_markers": ["acme"],
            "external_ip_owners": [],
            "restricted_path_markers": [],
        },
    )

    manifest_path = Path(result["manifest_path"])
    import json

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = payload["records"][0]

    assert "acme" in record["client_related"]
    assert record["source_ownership"] == "client_owned"
    assert record["ingest_policy_class"] == "review_required"
    assert record["sensitivity_level"] == "confidential"


def test_pre_ingest_business_planning_markers_classified_internal(tmp_path):
    source_root = tmp_path / "source"
    db_root = tmp_path / "db"

    _write(source_root / "sales" / "q1_sales_plan.txt", "weekly planning and pipeline report")
    _write(source_root / "finance" / "quarterly_budget_report.txt", "quarterly report and budgets")

    result = run_pre_ingest_organizer_scan(
        source_dirs=[source_root.as_posix()],
        db_path=db_root.as_posix(),
    )

    import json

    payload = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    by_name = {r["file_name"]: r for r in payload["records"]}

    sales_doc = by_name["q1_sales_plan.txt"]
    budget_doc = by_name["quarterly_budget_report.txt"]

    assert sales_doc["sensitivity_level"] == "internal"
    assert sales_doc["sensitivity_reason"] == "business_planning_marker"
    assert budget_doc["sensitivity_level"] == "internal"
    assert budget_doc["sensitivity_reason"] == "business_planning_marker"


def test_pre_ingest_progress_callback_emits_events(tmp_path):
    source_root = tmp_path / "source"
    db_root = tmp_path / "db"

    _write(source_root / "alpha" / "doc1.txt", "first report")
    _write(source_root / "alpha" / "doc2.md", "second report")

    events = []

    def _on_progress(event, data):
        events.append((event, data))

    result = run_pre_ingest_organizer_scan(
        source_dirs=[source_root.as_posix()],
        db_path=db_root.as_posix(),
        progress_callback=_on_progress,
    )

    assert Path(result["manifest_path"]).exists()
    event_names = [name for name, _ in events]
    assert "scan_started" in event_names
    assert "scan_complete" in event_names
    assert "analyze_progress" in event_names
    assert "manifest_written" in event_names


def test_pre_ingest_control_callback_can_stop_scan(tmp_path):
    source_root = tmp_path / "source"
    db_root = tmp_path / "db"

    _write(source_root / "alpha" / "doc1.txt", "one")
    _write(source_root / "alpha" / "doc2.txt", "two")

    calls = {"n": 0}

    def _control():
        calls["n"] += 1
        if calls["n"] >= 1:
            raise PreIngestScanCancelled("stopped in test")

    with pytest.raises(PreIngestScanCancelled):
        run_pre_ingest_organizer_scan(
            source_dirs=[source_root.as_posix()],
            db_path=db_root.as_posix(),
            control_callback=_control,
        )
