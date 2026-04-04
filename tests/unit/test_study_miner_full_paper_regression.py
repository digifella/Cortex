from __future__ import annotations

import csv
from pathlib import Path


_FIXTURE_DIR = Path("/home/longboardfella/cortex_suite/tests/fixtures/study_miner_full_paper")


def _load_rows(name: str) -> list[dict[str, str]]:
    path = _FIXTURE_DIR / name
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _groups(rows: list[dict[str, str]]) -> set[str]:
    return {(row.get("table_group") or "").strip() for row in rows if (row.get("table_group") or "").strip()}


def _reference_numbers(rows: list[dict[str, str]]) -> set[str]:
    return {(row.get("reference_number") or "").strip() for row in rows if (row.get("reference_number") or "").strip()}


def test_full_paper_table_2_fixture_stays_hrqol_focused():
    rows = _load_rows("table_2.csv")

    assert len(rows) == 13
    assert len(_reference_numbers(rows)) == 13

    groups = _groups(rows)
    assert {
        "Global / JULIET",
        "Global / ORCHARRD",
        "Global / SADAL",
        "US / NCT00667615",
        "NR / TRANSFORM",
        "US / TRANSCE",
    } <= groups
    assert all("CUA" not in group and "CEA" not in group and "TTO" not in group for group in groups)

    transcend_rows = [row for row in rows if (row.get("table_group") or "").strip() == "US / TRANSCE"]
    assert {row["reference_number"] for row in transcend_rows} == {"17", "35", "36", "37"}


def test_full_paper_table_3_fixture_stays_separate_from_economic_rows():
    rows = _load_rows("table_3.csv")

    assert len(rows) == 7
    assert len(_reference_numbers(rows)) == 7

    groups = _groups(rows)
    assert {
        "Global / CheckMate 436",
        "Global / SOLAR",
        "Global / ZUMA-1",
        "US / TRANSCEND NHL 001",
        "NR / Orlova 2022 [46]",
    } <= groups
    assert all("CUA" not in group and "CEA" not in group and "TTO" not in group for group in groups)


def test_full_paper_table_4_fixture_stays_economic_focused():
    rows = _load_rows("table_4.csv")

    assert len(rows) == 18
    assert len(_reference_numbers(rows)) == 18

    groups = _groups(rows)
    assert {
        "China / CUA",
        "Japan / CUA",
        "Singapore / CUA",
        "Spain / CUA",
        "UK / CUA",
        "US / CUA",
        "US / CEA",
    } <= groups
    assert not any(
        marker in group
        for group in groups
        for marker in ("JULIET", "SADAL", "TRANSFORM", "CheckMate", "SOLAR", "ZUMA-1")
    )


def test_full_paper_table_5_fixture_is_reference_free_low_value_table():
    rows = _load_rows("table_5.csv")

    assert len(rows) == 4
    assert _reference_numbers(rows) == set()
    assert _groups(rows) == {"Juliet study", "TRANSCEND NHL 001"}
