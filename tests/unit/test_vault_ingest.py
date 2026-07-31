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


import subprocess
import sys
from pathlib import Path

from cortex_engine.vault_ingest import (
    CORTEX_PYTHON,
    INDEXER_SCRIPT,
    INGEST_SCRIPT,
    VAULT_RAG_PYTHON,
    build_index_command,
    build_ingest_command,
    run_ingest_then_index,
)


class FakeRunner:
    """Records commands and returns canned stdout per phase."""

    def __init__(self, ingest_stdout: str, ingest_rc: int = 0, index_rc: int = 0):
        self.ingest_stdout = ingest_stdout
        self.ingest_rc = ingest_rc
        self.index_rc = index_rc
        self.commands: list[list[str]] = []

    def __call__(self, command: list[str]) -> subprocess.CompletedProcess:
        self.commands.append(command)
        if str(INGEST_SCRIPT) in command:
            return subprocess.CompletedProcess(
                command, self.ingest_rc, self.ingest_stdout, ""
            )
        return subprocess.CompletedProcess(command, self.index_rc, "indexed", "")

    @property
    def ran_index(self) -> bool:
        return any(str(INDEXER_SCRIPT) in cmd for cmd in self.commands)


def test_ingest_command_includes_required_flags(tmp_path):
    command = build_ingest_command(
        tmp_path, "manuals-hifi", tmp_path / "dest",
        pdf_strategy="hybrid", use_vision=False, limit=0,
        dry_run=False, manifest_path=tmp_path / "m.json",
    )
    assert str(CORTEX_PYTHON) in command
    assert "--source-root" in command
    assert "manuals-hifi" in command
    assert str(tmp_path / "dest") in command
    assert str(tmp_path / "m.json") in command
    # Falsy options must not appear at all.
    assert "--dry-run" not in command
    assert "--use-vision" not in command
    assert "--limit" not in command


def test_ingest_command_includes_optional_flags_when_set(tmp_path):
    command = build_ingest_command(
        tmp_path, "b", None,
        pdf_strategy="docling", use_vision=True, limit=25,
        dry_run=True, manifest_path=None,
    )
    assert "--use-vision" in command
    assert "--dry-run" in command
    assert command[command.index("--limit") + 1] == "25"
    assert command[command.index("--pdf-strategy") + 1] == "docling"
    assert "--dest-root" not in command


def test_index_command_targets_private_vault_only():
    command = build_index_command()
    assert str(VAULT_RAG_PYTHON) in command
    assert "--private-only" in command
    assert "--public-only" not in command


def test_phases_run_in_order_on_happy_path(tmp_path):
    runner = FakeRunner(DONE.format(c=5, s=0, f=0, d="False"))
    rc = run_ingest_then_index(tmp_path, "b", tmp_path / "d", runner=runner)
    assert rc == 0
    assert runner.ran_index is True
    assert str(INGEST_SCRIPT) in runner.commands[0]
    assert str(INDEXER_SCRIPT) in runner.commands[1]


def test_partial_failure_still_indexes_and_returns_nonzero(tmp_path):
    runner = FakeRunner(DONE.format(c=97, s=0, f=3, d="False"), ingest_rc=2)
    rc = run_ingest_then_index(tmp_path, "b", tmp_path / "d", runner=runner)
    assert runner.ran_index is True
    assert rc == 2


def test_zero_changed_skips_index(tmp_path):
    runner = FakeRunner(DONE.format(c=0, s=9, f=0, d="False"))
    rc = run_ingest_then_index(tmp_path, "b", tmp_path / "d", runner=runner)
    assert runner.ran_index is False
    assert rc == 0


def test_dry_run_skips_index(tmp_path):
    runner = FakeRunner(DONE.format(c=4, s=0, f=0, d="True"))
    run_ingest_then_index(tmp_path, "b", tmp_path / "d", dry_run=True, runner=runner)
    assert runner.ran_index is False


def test_crash_without_summary_halts_chain(tmp_path):
    runner = FakeRunner("segfault", ingest_rc=-9)
    rc = run_ingest_then_index(tmp_path, "b", tmp_path / "d", runner=runner)
    assert runner.ran_index is False
    assert rc != 0


def test_index_failure_surfaces_in_return_code(tmp_path):
    runner = FakeRunner(DONE.format(c=5, s=0, f=0, d="False"), index_rc=1)
    rc = run_ingest_then_index(tmp_path, "b", tmp_path / "d", runner=runner)
    assert runner.ran_index is True
    assert rc == 1


def test_unexpected_exception_still_prints_done_marker(tmp_path, capsys):
    """A crash must resolve to `failed`, not leave the UI wedged on `running`.

    The UI falls back to a pid-liveness check when there is no done marker, so an
    uncaught exception here (missing interpreter, missing script) would strand the
    panel. Catch it, log it, and emit a non-zero done marker.
    """
    def boom(command):
        raise FileNotFoundError("no such interpreter")

    rc = run_ingest_then_index(tmp_path, "b", tmp_path / "d", runner=boom)
    out = capsys.readouterr().out
    assert rc != 0
    assert "[vault-ingest] phase=done rc=1" in out
    assert "FileNotFoundError" in out   # traceback goes to the log


def test_default_runner_streams_and_accumulates_output(capsys):
    from cortex_engine.vault_ingest import _default_runner

    result = _default_runner(
        [sys.executable, "-u", "-c", "import sys; print('one'); print('two', file=sys.stderr)"]
    )
    assert result.returncode == 0
    # Same lines both echoed live and returned for parse_ingest_summary.
    assert "one" in capsys.readouterr().out
    assert "one" in result.stdout
    assert "two" in result.stdout   # stderr merged into stdout


def test_skip_index_uses_real_returncode_not_synthesized(tmp_path):
    # Fixture deliberately violates ingest script contract: exited 1 but reported failures=2
    runner = FakeRunner(DONE.format(c=0, s=0, f=2, d="False"), ingest_rc=1)
    rc = run_ingest_then_index(tmp_path, "b", tmp_path / "d", runner=runner)
    assert runner.ran_index is False
    assert rc == 1


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
