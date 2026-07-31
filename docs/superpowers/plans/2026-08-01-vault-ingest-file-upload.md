# Vault Ingest Single-Document Upload Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a second input mode to the Vault GraphRAG ingest panel — a browser file dialog for ingesting a single document — alongside the existing folder-path mode.

**Architecture:** Streamlit's uploader gives bytes and a filename but no path, so uploads are staged to a stable directory with a content-aware write that preserves `mtime` when the bytes are unchanged (this is what keeps the ingest manifest's de-duplication working). The run is then scoped to that one file using the ingest script's existing `--file-list` flag, threaded as one optional parameter through the existing command chain.

**Tech Stack:** Python 3.11, Streamlit, pytest, hashlib. No new dependencies.

## Global Constraints

- Python 3.11 via the repo venv (`~/cortex_suite/venv`). Run tests with `venv/bin/python -m pytest`.
- Scope is the **private** vault only: `/mnt/c/Users/paul/OneDrive - VentraIP Australia/Vault_OneDrive`.
- The external scripts `~/nemoclaw-private-knowledge-ingest.py` and `~/nemoclaw-vault-indexer.py` are **called, never modified** — the cron queue runner depends on their current behaviour.
- **The folder-path mode must remain behaviourally unchanged.** Any command built without a file list must be byte-identical to what is built today.
- Staging directory: `~/.nemoclaw/vault-ingest-uploads/`. File list: `.filelist.txt` inside it (single stable name, overwritten per run — `start_vault_ingest`'s `flock` already prevents concurrent starts).
- Uploader allowed types must mirror the ingest script's `SUPPORTED_EXTS`: `.pdf`, `.docx`, `.pptx`, `.txt`.
- No test may spawn a real non-dry-run ingest, write to the real vault, write to the real manifest, or touch `vault-rag-db`.
- Branch: `feat/vault-ingest-file-upload` (already created; spec committed as `fc28363`).

### Reference: the external contract this depends on

`~/nemoclaw-private-knowledge-ingest.py` accepts `--file-list <path>`, read at lines 371-379 as: one path per line, blank lines and lines starting with `#` skipped, each line `expanduser().resolve()`d.

**Critical:** even on the explicit-file path the script calls `source_path.relative_to(source_root)` (lines 134, 155, 233, 318). `--source-root` must therefore remain an ancestor of every listed file. With uploads, `--source-root` is the staging directory.

`is_unchanged` (line ~238) compares `mtime_ns`, `size`, and `output_path` — **not** a content hash. That is why the staging write must preserve mtime on unchanged bytes.

---

### Task 1: Content-aware upload staging

The de-duplication guarantee lives here. Pure logic, no Streamlit.

**Files:**
- Modify: `cortex_engine/private_vault_rag.py` (append near the existing `VAULT_INGEST_STATE` / `VAULT_INGEST_LOG_DIR` constants and the vault-ingest functions)
- Test: `tests/unit/test_vault_upload_staging.py`

**Interfaces:**
- Consumes: existing module constant `HOME = Path.home()` (defined near line 27 — do not redefine).
- Produces: `VAULT_INGEST_UPLOAD_DIR: Path`, `stage_upload(data: bytes, filename: str, staging_dir: Path | None = None) -> Path`.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_vault_upload_staging.py`:

```python
"""Staging for uploaded documents.

The ingest manifest compares mtime_ns + size (not a content hash), so an
unchanged re-upload MUST NOT rewrite the file or de-duplication breaks.
"""

import os

from cortex_engine.private_vault_rag import stage_upload


def test_new_file_is_written(tmp_path):
    target = stage_upload(b"hello", "doc.pdf", tmp_path)
    assert target == tmp_path / "doc.pdf"
    assert target.read_bytes() == b"hello"


def test_staging_dir_created_when_absent(tmp_path):
    nested = tmp_path / "does" / "not" / "exist"
    target = stage_upload(b"x", "a.txt", nested)
    assert target.exists()
    assert nested.is_dir()


def test_identical_bytes_preserve_mtime(tmp_path):
    # The de-duplication guarantee: an unchanged re-upload must not touch mtime,
    # because the ingest manifest keys "unchanged" off mtime_ns + size.
    target = stage_upload(b"same", "doc.pdf", tmp_path)
    os.utime(target, ns=(111_000_000_000, 111_000_000_000))
    before = target.stat().st_mtime_ns

    again = stage_upload(b"same", "doc.pdf", tmp_path)

    assert again == target
    assert target.stat().st_mtime_ns == before


def test_changed_bytes_are_rewritten(tmp_path):
    target = stage_upload(b"first", "doc.pdf", tmp_path)
    os.utime(target, ns=(111_000_000_000, 111_000_000_000))
    before = target.stat().st_mtime_ns

    stage_upload(b"second", "doc.pdf", tmp_path)

    assert target.read_bytes() == b"second"
    assert target.stat().st_mtime_ns != before


def test_filename_is_reduced_to_basename(tmp_path):
    # A filename carrying separators must not escape the staging directory.
    target = stage_upload(b"x", "../../etc/evil.txt", tmp_path)
    assert target == tmp_path / "evil.txt"
    assert target.parent == tmp_path
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd ~/cortex_suite && venv/bin/python -m pytest tests/unit/test_vault_upload_staging.py -v`
Expected: FAIL — `ImportError: cannot import name 'stage_upload'`

- [ ] **Step 3: Write the implementation**

Add `import hashlib` to the imports of `cortex_engine/private_vault_rag.py` (alphabetically, between `glob` and `json`). Then add alongside the other vault-ingest constants and functions:

```python
VAULT_INGEST_UPLOAD_DIR = HOME / ".nemoclaw" / "vault-ingest-uploads"


def stage_upload(data: bytes, filename: str, staging_dir: Path | None = None) -> Path:
    """Write an uploaded document to the staging dir, preserving mtime when unchanged.

    Streamlit's uploader supplies bytes and a name but no path, so uploads must be
    written somewhere the ingest script can read them. The manifest keys on the
    absolute source path and treats a file as unchanged by comparing mtime_ns and
    size -- so rewriting identical bytes would defeat de-duplication and re-ingest
    the same document. Skip the write when the staged copy already matches.
    """
    staging_dir = Path(staging_dir) if staging_dir else VAULT_INGEST_UPLOAD_DIR
    staging_dir.mkdir(parents=True, exist_ok=True)
    target = staging_dir / Path(filename).name

    if target.exists():
        try:
            if hashlib.sha256(target.read_bytes()).hexdigest() == hashlib.sha256(data).hexdigest():
                return target
        except OSError:
            pass

    target.write_bytes(data)
    return target
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd ~/cortex_suite && venv/bin/python -m pytest tests/unit/test_vault_upload_staging.py -v`
Expected: PASS, 5 passed

- [ ] **Step 5: Confirm nothing existing broke**

Run: `cd ~/cortex_suite && venv/bin/python -m pytest tests/unit/test_vault_ingest.py tests/unit/test_vault_ingest_state.py -q`
Expected: 36 passed

- [ ] **Step 6: Commit**

```bash
cd ~/cortex_suite
git add cortex_engine/private_vault_rag.py tests/unit/test_vault_upload_staging.py
git commit -m "feat: content-aware staging for uploaded documents

The ingest manifest compares mtime_ns + size rather than a content hash, so an
unchanged re-upload must not rewrite the staged file or de-duplication breaks.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 2: Thread `--file-list` through the command chain

**Files:**
- Modify: `cortex_engine/vault_ingest.py` (`build_ingest_command`, `run_ingest_then_index`, `main`)
- Modify: `cortex_engine/private_vault_rag.py` (`start_vault_ingest`)
- Test: `tests/unit/test_vault_ingest.py`, `tests/unit/test_vault_ingest_state.py`

**Interfaces:**
- Consumes: `VAULT_INGEST_UPLOAD_DIR` from Task 1 (used only by tests here).
- Produces:
  - `build_ingest_command(..., manifest_path: Path | None, file_list: Path | None = None) -> list[str]` — `file_list` added as the LAST keyword-only parameter so existing keyword calls are unaffected.
  - `run_ingest_then_index(..., manifest_path=None, file_list: Path | None = None, runner=None) -> int`
  - `start_vault_ingest(..., dry_run: bool = False, file_list: Path | None = None, state_path: Path | None = None) -> dict`

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_vault_ingest.py`:

```python
def test_ingest_command_includes_file_list_when_given(tmp_path):
    listing = tmp_path / ".filelist.txt"
    command = build_ingest_command(
        tmp_path, "b", None,
        pdf_strategy="hybrid", use_vision=False, limit=0,
        dry_run=False, manifest_path=None, file_list=listing,
    )
    assert command[command.index("--file-list") + 1] == str(listing)


def test_ingest_command_omits_file_list_by_default(tmp_path):
    # Regression guard: the folder-path mode must build exactly the command it
    # built before --file-list existed.
    command = build_ingest_command(
        tmp_path, "b", None,
        pdf_strategy="hybrid", use_vision=False, limit=0,
        dry_run=False, manifest_path=None,
    )
    assert "--file-list" not in command


def test_run_passes_file_list_to_phase_one(tmp_path):
    listing = tmp_path / ".filelist.txt"
    runner = FakeRunner(DONE.format(c=1, s=0, f=0, d="False"))
    run_ingest_then_index(tmp_path, "b", tmp_path / "d", file_list=listing, runner=runner)
    assert "--file-list" in runner.commands[0]
    assert str(listing) in runner.commands[0]
```

Append to `tests/unit/test_vault_ingest_state.py`:

```python
def test_start_vault_ingest_forwards_file_list(tmp_path, monkeypatch):
    # start_vault_ingest logs the command it spawned as the log's first line,
    # so that line is the contract we can assert against without a real ingest.
    from cortex_engine import private_vault_rag as pvr

    monkeypatch.setattr(pvr, "VAULT_INGEST_LOG_DIR", tmp_path / "logs")
    source = tmp_path / "staging"
    source.mkdir()
    (source / "doc.txt").write_text("hello", encoding="utf-8")
    listing = source / ".filelist.txt"
    listing.write_text(f"{source / 'doc.txt'}\n", encoding="utf-8")

    started = pvr.start_vault_ingest(
        source, "upload-doc", tmp_path / "dest",
        dry_run=True, file_list=listing, state_path=tmp_path / "state.json",
    )

    first_line = Path(started["log_path"]).read_text(encoding="utf-8").splitlines()[0]
    assert "--file-list" in first_line
    assert str(listing) in first_line
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd ~/cortex_suite && venv/bin/python -m pytest tests/unit/test_vault_ingest.py tests/unit/test_vault_ingest_state.py -v`
Expected: FAIL — `TypeError: build_ingest_command() got an unexpected keyword argument 'file_list'`

- [ ] **Step 3: Add `file_list` to `build_ingest_command`**

In `cortex_engine/vault_ingest.py`, add the parameter as the last keyword-only argument and append the flag:

```python
def build_ingest_command(
    source_root: Path,
    branch_name: str,
    dest_root: Path | None,
    *,
    pdf_strategy: str,
    use_vision: bool,
    limit: int,
    dry_run: bool,
    manifest_path: Path | None,
    file_list: Path | None = None,
) -> list[str]:
```

and, immediately after the existing `if manifest_path:` block:

```python
    if file_list:
        command += ["--file-list", str(file_list)]
```

Leave every other line of the function untouched, so a call without `file_list` produces the identical command.

- [ ] **Step 4: Thread it through `run_ingest_then_index` and `main`**

In `run_ingest_then_index`, add `file_list: Path | None = None` immediately before the `runner` parameter, and pass it into the `build_ingest_command(...)` call:

```python
    ingest = run(build_ingest_command(
        source_root, branch_name, dest_root,
        pdf_strategy=pdf_strategy, use_vision=use_vision,
        limit=limit, dry_run=dry_run, manifest_path=manifest_path,
        file_list=file_list,
    ))
```

In `main()`, add the argument and pass it through:

```python
    parser.add_argument("--file-list", default="")
```

and in the `return run_ingest_then_index(...)` call:

```python
        file_list=Path(args.file_list) if args.file_list else None,
```

- [ ] **Step 5: Thread it through `start_vault_ingest`**

In `cortex_engine/private_vault_rag.py`, add `file_list: Path | None = None` to the keyword-only parameters of `start_vault_ingest` (immediately after `dry_run`), and append the flag to the spawned command, immediately after the existing `if dest_root:` block:

```python
            if file_list:
                command += ["--file-list", str(file_list)]
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `cd ~/cortex_suite && venv/bin/python -m pytest tests/unit/test_vault_ingest.py tests/unit/test_vault_ingest_state.py tests/unit/test_vault_upload_staging.py -v`
Expected: PASS, 45 passed (36 existing + 5 from Task 1 + 4 new)

- [ ] **Step 7: Verify the CLI exposes the flag**

Run: `cd ~/cortex_suite && venv/bin/python -m cortex_engine.vault_ingest --help`
Expected: usage text now lists `--file-list`

- [ ] **Step 8: Commit**

```bash
cd ~/cortex_suite
git add cortex_engine/vault_ingest.py cortex_engine/private_vault_rag.py tests/unit/test_vault_ingest.py tests/unit/test_vault_ingest_state.py
git commit -m "feat: thread --file-list through the ingest command chain

Scopes a run to specific files. Omitted by default, so the folder-path mode
builds exactly the command it built before.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 3: Upload mode in the ingest panel

**Files:**
- Modify: `pages/18_Private_Vault_GraphRAG.py` (`_ingest_panel`, lines 62-120, and the imports at lines 10-29)

**Interfaces:**
- Consumes: `stage_upload`, `VAULT_INGEST_UPLOAD_DIR` (Task 1); `start_vault_ingest(..., file_list=...)` (Task 2); existing `PRIVATE_VAULT`, `vault_ingest_status`, `convert_windows_to_wsl_path`.
- Produces: nothing consumed by later tasks.

This task has no pytest unit test — Streamlit page functions are not unit-testable without a harness this repo lacks, and the logic they call is covered by Tasks 1-2. Verify with `streamlit.testing.v1.AppTest`, as the previous panel work did.

- [ ] **Step 1: Extend the imports**

Add `stage_upload` and `VAULT_INGEST_UPLOAD_DIR` to the existing `from cortex_engine.private_vault_rag import (...)` block, keeping it alphabetised.

- [ ] **Step 2: Replace the input section of `_ingest_panel`**

Replace everything from `source_raw = st.text_input(` down to and including the `dry_run = col6.checkbox(...)` line with:

```python
        mode = st.radio(
            "Source", ["Folder", "Single document"],
            horizontal=True, disabled=running, key="vault_ingest_mode",
        )

        source_path = ""
        uploaded = None

        if mode == "Folder":
            source_raw = st.text_input(
                "Source folder",
                placeholder=r"C:\Users\paul\Documents\Manuals  or  /mnt/c/...",
                disabled=running,
            )
            source_path = convert_windows_to_wsl_path(source_raw.strip()) if source_raw.strip() else ""
            if source_path and st.session_state.get("_vault_ingest_src") != source_path:
                st.session_state["_vault_ingest_src"] = source_path
                derived = Path(source_path).name.lower().replace(" ", "-")
                st.session_state["vault_ingest_branch"] = derived
                st.session_state["vault_ingest_dest"] = f"30 Resources/Imported Knowledge/{derived}"
        else:
            uploaded = st.file_uploader(
                "Document", type=["pdf", "docx", "pptx", "txt"], disabled=running,
            )
            if uploaded is not None and st.session_state.get("_vault_ingest_src") != uploaded.name:
                st.session_state["_vault_ingest_src"] = uploaded.name
                derived = Path(uploaded.name).stem.lower().replace(" ", "-")
                st.session_state["vault_ingest_branch"] = derived
                st.session_state["vault_ingest_dest"] = f"30 Resources/Imported Knowledge/{derived}"
            st.caption(
                f"Uploads are staged in `{VAULT_INGEST_UPLOAD_DIR}` and kept, so re-uploading "
                "the same document is skipped rather than duplicated. Clear that folder manually "
                "if it grows. The vault note will cite the staged path, not the original location."
            )

        col1, col2 = st.columns(2)
        branch = col1.text_input("Branch name", key="vault_ingest_branch", disabled=running)
        dest = col2.text_input("Destination (relative to vault)", key="vault_ingest_dest", disabled=running)

        limit = 0
        cols = st.columns(4 if mode == "Folder" else 3)
        strategy = cols[0].selectbox("PDF strategy", ["hybrid", "docling", "pymupdf"], disabled=running)
        nxt = 1
        if mode == "Folder":
            limit = cols[1].number_input("File limit (0 = all)", min_value=0, value=0, step=10, disabled=running)
            nxt = 2
        use_vision = cols[nxt].checkbox("Describe images", value=False, disabled=running)
        dry_run = cols[nxt + 1].checkbox("Dry run", value=False, disabled=running)
```

- [ ] **Step 3: Replace the validation and button block**

Replace everything from `if source_path and not Path(source_path).is_dir():` down to **and including** the `st.rerun()` that follows the `except` clause (currently line 123) with the block below. Stop there — the `if status["state"] != "idle":` block that follows must be left exactly as it is, and there must be exactly one `st.rerun()` in the result:

```python
        ready = False
        if mode == "Folder":
            if source_path and not Path(source_path).is_dir():
                st.error(f"Not a directory: `{source_path}`")
            ready = bool(source_path) and Path(source_path).is_dir()
        else:
            if uploaded is not None and uploaded.size == 0:
                st.error("That file is empty.")
            ready = uploaded is not None and uploaded.size > 0

        dest_root = None
        dest_error = ""
        if dest.strip():
            candidate = (PRIVATE_VAULT / dest.strip()).resolve()
            if candidate.is_relative_to(PRIVATE_VAULT.resolve()):
                dest_root = candidate
            else:
                dest_error = "Destination must stay inside the vault."
        if dest_error:
            st.error(dest_error)

        if st.button(
            "Ingest", type="primary",
            disabled=running or not ready or not branch or bool(dest_error),
        ):
            try:
                file_list = None
                if mode == "Folder":
                    run_source = Path(source_path)
                else:
                    staged = stage_upload(uploaded.getvalue(), uploaded.name)
                    file_list = VAULT_INGEST_UPLOAD_DIR / ".filelist.txt"
                    file_list.write_text(f"{staged}\n", encoding="utf-8")
                    run_source = VAULT_INGEST_UPLOAD_DIR

                started = start_vault_ingest(
                    run_source, branch.strip(), dest_root,
                    pdf_strategy=strategy, use_vision=use_vision,
                    limit=int(limit), dry_run=dry_run, file_list=file_list,
                )
                st.success(f"Started (pid {started['pid']}). Log: `{started['log_path']}`")
            except (ValueError, OSError) as exc:
                st.error(str(exc))
            st.rerun()
```

Note `OSError` is now caught alongside `ValueError` because staging writes to disk. Staging happens on click rather than on upload, so a staging failure surfaces at the moment of action and nothing is spawned.

- [ ] **Step 4: Verify syntax and imports**

Run: `cd ~/cortex_suite && venv/bin/python -c "import ast; ast.parse(open('pages/18_Private_Vault_GraphRAG.py').read()); print('syntax ok')"`
Expected: `syntax ok`

Run: `cd ~/cortex_suite && venv/bin/python -c "from cortex_engine.private_vault_rag import stage_upload, VAULT_INGEST_UPLOAD_DIR, start_vault_ingest; print('imports ok')"`
Expected: `imports ok`

- [ ] **Step 5: Verify with AppTest**

Write a throwaway script (do not commit it) using `streamlit.testing.v1.AppTest` against `pages/18_Private_Vault_GraphRAG.py` and confirm:

- the page runs with zero exceptions
- a "Source" radio exists with options `Folder` and `Single document`
- in `Folder` mode a "Source folder" text input and a "File limit (0 = all)" number input are present
- switching the radio to `Single document` removes the "Source folder" input and the "File limit (0 = all)" input, and the run still raises no exception
- the Ingest button is disabled in `Single document` mode with no file uploaded

Paste the script and its full output into your report. If AppTest cannot drive `st.file_uploader` (it has limited support for upload widgets), say so plainly rather than claiming coverage you did not achieve — and report which of the checks above you could and could not perform.

- [ ] **Step 6: Commit**

```bash
cd ~/cortex_suite
git add pages/18_Private_Vault_GraphRAG.py
git commit -m "feat: single-document upload mode in the ingest panel

Adds a Folder / Single document radio. Upload mode stages the file, scopes the
run with --file-list, and hides the meaningless file-limit control.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 4: Version 6.5.0 and Docker sync

**Files:**
- Modify: `cortex_engine/version_config.py`
- Modify: `pages/18_Private_Vault_GraphRAG.py` (header comment lines 2-3)
- Modify: `CHANGELOG.md` (via the version manager)
- Modify: `docker/cortex_engine/`, `docker/pages/` (copies)

- [ ] **Step 1: Bump the version**

In `cortex_engine/version_config.py` set `CORTEX_VERSION = "6.5.0"` and update `VERSION_METADATA`. **Keep the existing `version` and `breaking_changes` keys** — `tests/unit/test_version_config.py` asserts both are present and that `VERSION_METADATA["version"] == CORTEX_VERSION`. Set:

```python
    "release_date": "2026-08-01",
    "release_name": "Single-Document Vault Upload",
    "description": "Upload one document straight into the private vault from the ingest panel",
    "new_features": [
        "Folder / Single document mode toggle on the Vault GraphRAG ingest panel",
        "Browser file dialog for ingesting one PDF, DOCX, PPTX or TXT",
    ],
    "improvements": [
        "Uploaded documents stage to a stable path with content-aware writes, so re-uploading the same file is skipped rather than duplicated",
    ],
    "bug_fixes": [],
```

- [ ] **Step 2: Update the page header**

In `pages/18_Private_Vault_GraphRAG.py` set lines 2-3 to:

```python
# Version: v6.5.0
# Date: 2026-08-01
```

- [ ] **Step 3: Run the version sync**

```bash
cd ~/cortex_suite
venv/bin/python scripts/version_manager.py --sync-all
venv/bin/python scripts/version_manager.py --update-changelog
venv/bin/python scripts/version_manager.py --check
```

`--check` reports 14 pre-existing mismatches from gaps in the script's own regex coverage. That is a known, pre-existing condition — do NOT try to fix it. Confirm the count has not grown.

- [ ] **Step 4: Sync the Docker distribution**

```bash
cd ~/cortex_suite
cp cortex_engine/version_config.py cortex_engine/vault_ingest.py cortex_engine/private_vault_rag.py docker/cortex_engine/
cp pages/18_Private_Vault_GraphRAG.py docker/pages/
```

Then verify each is byte-identical:

```bash
for f in cortex_engine/version_config.py cortex_engine/vault_ingest.py cortex_engine/private_vault_rag.py pages/18_Private_Vault_GraphRAG.py; do
  diff -q "$f" "docker/$f" && echo "IN SYNC $f"
done
```

- [ ] **Step 5: Run the test suite**

Run: `cd ~/cortex_suite && venv/bin/python -m pytest tests/unit -q`

The 45 vault-ingest tests must pass. There are **~10-12 pre-existing failures** in `test_document_extract_study_miner`, `test_intel_extract_handler`, `test_maintenance_runner`, `test_model_checker`, `test_path_utils`, and `test_url_ingestor`. These predate this work and some are flaky (they query the local Ollama server, and `test_model_checker` failures in particular vary between runs). Do NOT fix them. Report the count and the list; if a failure appears OUTSIDE those six files, stop and report BLOCKED.

- [ ] **Step 6: Commit**

⚠️ The working tree contains pre-existing unrelated changes belonging to the user that must NOT be committed: deleted `*.pdf` files and their `:Zone.Identifier` companions, a modified `.claude/settings.local.json`, and untracked `RLB Maintenance plan 2012.txt*` / `SciAm_06_2025.pdf:Zone.Identifier`. **Never `git add -A` or `git add .`.**

Stage tracked modifications with those paths excluded, then verify before committing:

```bash
cd ~/cortex_suite
git add -u -- ':!*.pdf' ':!*Zone.Identifier' ':!.claude'
git status --short          # confirm none of the unrelated paths are staged
git commit -m "release: Version 6.5.0 - Single-Document Vault Upload

Adds a file-upload mode to the private vault ingest panel.

- Version consistency verified
- Docker distribution updated

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Notes for the implementer

- **Do not modify** `~/nemoclaw-private-knowledge-ingest.py` or `~/nemoclaw-vault-indexer.py`.
- The mtime-preservation behaviour in Task 1 is the whole point of the feature's de-duplication. If a test there ever seems awkward, do not relax it — it is guarding a real behaviour.
- `st.file_uploader` returns `None` until a file is chosen; every read of `uploaded` must be guarded.
- `uploaded.getvalue()` returns the full bytes; `uploaded.name` is the original filename with no path.
