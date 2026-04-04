# Study Miner Full-Paper Fixtures

These CSVs are the pinned regression baseline for the full systematic-review PDF
used during Study Miner table-rescue tuning.

Refresh them intentionally with:

```bash
python scripts/refresh_study_miner_full_paper_fixtures.py --source-dir /home/longboardfella/cortex_suite
```

That command selects the latest matching exports for tables `2`, `3`, `4`, and
`5`, then copies them into this directory as:

- `table_2.csv`
- `table_3.csv`
- `table_4.csv`
- `table_5.csv`

After refreshing the fixtures, rerun:

```bash
pytest tests/unit/test_study_miner_full_paper_regression.py
```
