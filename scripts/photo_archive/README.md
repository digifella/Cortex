# P: drive photo archive pipeline

Runs under Windows Python: `C:\Python311\python.exe -m scripts.photo_archive.cli`

Every writing stage is dry-run unless `--apply` is passed. Nothing is ever
deleted.

## Rehearsal order — do not skip

1. `SMS_Photos` (80 files) — verify by hand.
2. `Google Drive Photos` (1,058 files) — verify by hand.
3. `family_Randoms` (118,975 files) — first at-scale run.
4. Everything else.

Restrict scope for a rehearsal by editing `SCOPE_ROOTS` in `config.py`.

## Full sequence

```bat
set PY=C:\Python311\python.exe
%PY% -m scripts.photo_archive.cli walk          --db C:\pindex.db
%PY% -m scripts.photo_archive.cli exif          --db C:\pindex.db
%PY% -m scripts.photo_archive.cli hash          --db C:\pindex.db
%PY% -m scripts.photo_archive.cli plan-dupes    --db C:\pindex.db --plan dupes.csv
REM  review dupes.csv before the next line
%PY% -m scripts.photo_archive.cli quarantine    --db C:\pindex.db --plan dupes.csv --apply
%PY% -m scripts.photo_archive.cli plan-organise --db C:\pindex.db --plan org.csv
REM  review org.csv before the next line
%PY% -m scripts.photo_archive.cli organise      --db C:\pindex.db --plan org.csv --apply
```

## Undo

```bat
%PY% -m scripts.photo_archive.cli undo --db C:\pindex.db --journal photo_archive_journal.csv
%PY% -m scripts.photo_archive.cli undo --db C:\pindex.db --journal photo_archive_journal.csv --apply
```

Undo refuses any file whose hash changed since it was moved.
