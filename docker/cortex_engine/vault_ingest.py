# ## File: cortex_engine/vault_ingest.py
# Version: v6.4.0
# Date: 2026-07-31
# Purpose: Two-phase private vault ingest -- textify documents, then index them.

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

# Emitted as the final line of nemoclaw-private-knowledge-ingest.py.
SUMMARY_RE = re.compile(
    r"done changed=(\d+) skipped=(\d+) failures=(\d+) dry_run=(\w+)"
)


@dataclass(frozen=True)
class IngestSummary:
    changed: int
    skipped: int
    failures: int
    dry_run: bool


def parse_ingest_summary(output: str) -> IngestSummary | None:
    """Pull the ingest script's summary line out of its stdout, or None."""
    match = SUMMARY_RE.search(output or "")
    if not match:
        return None
    return IngestSummary(
        changed=int(match.group(1)),
        skipped=int(match.group(2)),
        failures=int(match.group(3)),
        dry_run=match.group(4).lower() == "true",
    )


def should_index(summary: IngestSummary | None) -> bool:
    """Decide whether phase 2 should run.

    A missing summary means the ingest crashed or was killed: state is unknown,
    so we do not index a partial branch as if it were complete. Partial failures
    (exit code 2) still index -- `changed` may be 97 of 100.
    """
    if summary is None:
        return False
    if summary.dry_run:
        return False
    return summary.changed > 0


HOME = Path.home()
INGEST_SCRIPT = HOME / "nemoclaw-private-knowledge-ingest.py"
INDEXER_SCRIPT = HOME / "nemoclaw-vault-indexer.py"
CORTEX_PYTHON = HOME / "cortex_suite" / "venv" / "bin" / "python"
VAULT_RAG_PYTHON = HOME / "venvs" / "vault-rag" / "bin" / "python3"


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
    command = [
        str(CORTEX_PYTHON), "-u", str(INGEST_SCRIPT),
        "--source-root", str(source_root),
        "--branch-name", branch_name,
        "--pdf-strategy", pdf_strategy,
    ]
    if dest_root:
        command += ["--dest-root", str(dest_root)]
    if manifest_path:
        command += ["--manifest-path", str(manifest_path)]
    if file_list:
        command += ["--file-list", str(file_list)]
    if limit:
        command += ["--limit", str(limit)]
    if use_vision:
        command.append("--use-vision")
    if dry_run:
        command.append("--dry-run")
    return command


def build_index_command() -> list[str]:
    # Phase 2 runs under the vault-rag interpreter: different chromadb pin.
    return [str(VAULT_RAG_PYTHON), "-u", str(INDEXER_SCRIPT), "--private-only"]


def _default_runner(command: list[str]) -> subprocess.CompletedProcess:
    """Run a phase, echoing each line as it arrives, and return the full output.

    Buffering the phase (capture_output=True) hides the ingest script's per-file
    `[private-ingest] i/N -> file` progress until the phase ends -- hours, for a
    large folder -- and loses it entirely if the run is cancelled mid-phase. So we
    stream: print each line immediately, and accumulate it for parse_ingest_summary.
    stderr is merged into stdout so the interleaving in the log matches reality.
    """
    env = {**os.environ, "HF_HOME": "/mnt/f/hf-home", "TOKENIZERS_PARALLELISM": "false"}
    lines: list[str] = []
    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    with proc.stdout:
        for line in proc.stdout:
            print(line, end="", flush=True)
            lines.append(line)
    return subprocess.CompletedProcess(command, proc.wait(), "".join(lines), "")


def run_ingest_then_index(
    source_root: Path,
    branch_name: str,
    dest_root: Path | None = None,
    *,
    pdf_strategy: str = "hybrid",
    use_vision: bool = False,
    limit: int = 0,
    dry_run: bool = False,
    manifest_path: Path | None = None,
    file_list: Path | None = None,
    runner: Callable[[list[str]], subprocess.CompletedProcess] | None = None,
) -> int:
    """Textify a source branch into the private vault, then index it.

    Any unexpected exception is caught and still reported as a done marker with a
    non-zero rc. The UI resolves a run with no done marker by checking whether the
    detached pid is alive, so an uncaught crash here (missing interpreter, missing
    script) would otherwise leave the panel wedged on "running".
    """
    try:
        run = runner or _default_runner

        print(f"[vault-ingest] phase=textify branch={branch_name}", flush=True)
        # The runner echoes phase output line by line as it arrives -- do not re-print.
        ingest = run(build_ingest_command(
            source_root, branch_name, dest_root,
            pdf_strategy=pdf_strategy, use_vision=use_vision,
            limit=limit, dry_run=dry_run, manifest_path=manifest_path,
            file_list=file_list,
        ))

        summary = parse_ingest_summary(ingest.stdout)
        if not should_index(summary):
            reason = "ingest produced no parseable summary" if summary is None else (
                "dry run" if summary.dry_run else "nothing changed"
            )
            print(f"[vault-ingest] phase=skip-index reason={reason}", flush=True)
            rc = ingest.returncode
            print(f"[vault-ingest] phase=done rc={rc}", flush=True)
            return rc

        print("[vault-ingest] phase=index", flush=True)
        index = run(build_index_command())

        # Index failure dominates: the branch is on disk but not searchable.
        rc = index.returncode or (2 if summary.failures else 0)
        print(f"[vault-ingest] phase=done rc={rc}", flush=True)
        return rc
    except Exception:
        traceback.print_exc(file=sys.stdout)
        print("[vault-ingest] phase=done rc=1", flush=True)
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Textify a branch into the private vault, then index it")
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--branch-name", required=True)
    parser.add_argument("--dest-root", default="")
    parser.add_argument("--manifest-path", default="")
    parser.add_argument("--pdf-strategy", default="hybrid", choices=["hybrid", "docling", "pymupdf"])
    parser.add_argument("--use-vision", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--file-list", default="")
    args = parser.parse_args()
    return run_ingest_then_index(
        Path(args.source_root),
        args.branch_name,
        Path(args.dest_root) if args.dest_root else None,
        pdf_strategy=args.pdf_strategy,
        use_vision=args.use_vision,
        limit=args.limit,
        dry_run=args.dry_run,
        manifest_path=Path(args.manifest_path) if args.manifest_path else None,
        file_list=Path(args.file_list) if args.file_list else None,
    )


if __name__ == "__main__":
    sys.exit(main())
