from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_script_module():
    path = Path("/home/longboardfella/cortex_suite/scripts/refresh_study_miner_full_paper_fixtures.py")
    spec = importlib.util.spec_from_file_location("refresh_study_miner_fixtures", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_find_latest_table_exports_picks_latest_file_per_table(temp_dir):
    module = _load_script_module()

    filenames = [
        "2026-04-04T16-00_review_table_2_study_miner_export.csv",
        "2026-04-04T17-00_review_table_2_study_miner_export.csv",
        "2026-04-04T16-01_review_table_3_study_miner_export.csv",
        "2026-04-04T17-01_review_table_3_study_miner_export.csv",
        "2026-04-04T16-02_review_table_4_study_miner_export.csv",
        "2026-04-04T17-02_review_table_4_study_miner_export.csv",
        "2026-04-04T16-03_review_table_5_study_miner_export.csv",
        "2026-04-04T17-03_review_table_5_study_miner_export.csv",
    ]
    for name in filenames:
        (temp_dir / name).write_text("header\n", encoding="utf-8")

    exports = module.find_latest_table_exports(temp_dir)

    assert exports[2].name == "2026-04-04T17-00_review_table_2_study_miner_export.csv"
    assert exports[3].name == "2026-04-04T17-01_review_table_3_study_miner_export.csv"
    assert exports[4].name == "2026-04-04T17-02_review_table_4_study_miner_export.csv"
    assert exports[5].name == "2026-04-04T17-03_review_table_5_study_miner_export.csv"


def test_copy_exports_to_fixture_dir_writes_stable_fixture_names(temp_dir):
    module = _load_script_module()

    exports = {}
    for table_number in (2, 3, 4, 5):
        source = temp_dir / f"source_table_{table_number}.csv"
        source.write_text(f"table,{table_number}\n", encoding="utf-8")
        exports[table_number] = source

    fixture_dir = temp_dir / "fixtures"
    copied = module.copy_exports_to_fixture_dir(exports, fixture_dir)

    assert [destination.name for _, destination in copied] == [
        "table_2.csv",
        "table_3.csv",
        "table_4.csv",
        "table_5.csv",
    ]
    assert (fixture_dir / "table_2.csv").read_text(encoding="utf-8") == "table,2\n"


def test_find_latest_table_exports_requires_all_expected_tables(temp_dir):
    module = _load_script_module()

    (temp_dir / "2026-04-04T17-00_review_table_2_study_miner_export.csv").write_text("header\n", encoding="utf-8")
    (temp_dir / "2026-04-04T17-01_review_table_3_study_miner_export.csv").write_text("header\n", encoding="utf-8")
    (temp_dir / "2026-04-04T17-02_review_table_4_study_miner_export.csv").write_text("header\n", encoding="utf-8")

    try:
        module.find_latest_table_exports(temp_dir)
    except FileNotFoundError as exc:
        assert "table 5" in str(exc)
    else:
        raise AssertionError("Expected missing table export to raise FileNotFoundError")
