"""Unit tests for the private vault ingest wrapper."""

import pytest

from cortex_engine.vault_ingest import (
    IngestSummary,
    parse_ingest_summary,
    should_index,
)

DONE = "[private-ingest] done changed={c} skipped={s} failures={f} dry_run={d}"


def test_parse_extracts_all_fields():
    summary = parse_ingest_summary(DONE.format(c=7, s=2, f=1, d="False"))
    assert summary == IngestSummary(changed=7, skipped=2, failures=1, dry_run=False)


def test_parse_finds_summary_among_other_output():
    output = "\n".join([
        "[private-ingest] converting a.pdf",
        "[private-ingest] ERROR: b.pdf: boom",
        DONE.format(c=1, s=0, f=1, d="False"),
    ])
    summary = parse_ingest_summary(output)
    assert summary.changed == 1
    assert summary.failures == 1


def test_parse_reads_dry_run_true():
    summary = parse_ingest_summary(DONE.format(c=3, s=0, f=0, d="True"))
    assert summary.dry_run is True


def test_parse_returns_none_when_no_summary_line():
    assert parse_ingest_summary("killed before finishing") is None


def test_index_runs_on_happy_path():
    assert should_index(IngestSummary(changed=5, skipped=0, failures=0, dry_run=False)) is True


def test_index_runs_on_partial_failure():
    # Exit code 2, but 97 files converted -- halting would strand them unindexed.
    assert should_index(IngestSummary(changed=97, skipped=0, failures=3, dry_run=False)) is True


def test_index_skipped_when_nothing_changed():
    assert should_index(IngestSummary(changed=0, skipped=12, failures=0, dry_run=False)) is False


def test_index_skipped_on_dry_run():
    # dry-run still increments `changed`, so dry_run must be checked explicitly.
    assert should_index(IngestSummary(changed=4, skipped=0, failures=0, dry_run=True)) is False


def test_index_skipped_when_summary_missing():
    assert should_index(None) is False
