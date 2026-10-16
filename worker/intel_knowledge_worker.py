"""Read intel@ into private review until public quality checks are restored.

This deliberately never writes AI-Vault or invokes the wiki publisher. It is
safe to run while the old Shoebox pipeline is unavailable: new mail is retained
without replaying pre-cutover messages or treating mail text as an instruction.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import signal
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

from worker.notes_mailbox_worker import (
    NotesGraphClient,
    NotesMailboxConfig,
    NotesMailboxProcessor,
    _normalize_graph_message,
)

CUTOVER = "2026-10-15T06:15:00Z"
MAILBOX = "intel@longboardfella.com.au"
PRIVATE_VAULT = "/mnt/c/Users/paul/OneDrive - VentraIP Australia/Vault_OneDrive"
PRIVATE_MARKER = re.compile(r"(?i)\b(?:private|sensitive|confidential|client[- ]confidential)\b")


def classify_intel_route(subject: str, body: str) -> dict[str, str]:
    if PRIVATE_MARKER.search(subject or "") or PRIVATE_MARKER.search(body or ""):
        return {"route": "private_vault", "reason": "sensitivity_marker"}
    return {"route": "public_review", "reason": "awaiting_source_and_sensitivity_checks"}


def load_config() -> NotesMailboxConfig:
    path = Path.home() / ".config/shoebox/graph.json"
    if path.stat().st_mode & 0o077:
        raise RuntimeError("intel@ Graph config must be mode 0600")
    graph = json.loads(path.read_text(encoding="utf-8"))
    if graph.get("mailbox", "").lower() != MAILBOX:
        raise RuntimeError("intel@ Graph config points at another mailbox")
    root = Path.home() / "vault-rag-db/intel-mailbox"
    return NotesMailboxConfig(
        source_system="intel_mailbox", poll_interval=600,
        transport_mode="graph", mailbox_identity=MAILBOX,
        public_outbox_dir=str(root / "public"),
        private_outbox_dir=str(root / "private"),
        public_vault_dir="/mnt/c/Users/paul/Documents/AI-Vault/Inbox",
        private_vault_dir=f"{PRIVATE_VAULT}/Intel_Inbox",
        public_review_vault_dir=f"{PRIVATE_VAULT}/Public_Review/intel",
        write_vault_markdown=True,
        graph_tenant_id=graph["tenant_id"], graph_client_id=graph["client_id"],
        graph_client_secret=graph["client_secret"], graph_mailbox=MAILBOX,
        graph_page_size=25, graph_timeout=30,
    )


def poll_once(client: NotesGraphClient, processor: NotesMailboxProcessor) -> tuple[int, int]:
    processed = skipped = 0
    # Read all post-cutover mail regardless of its read flag; local IDs provide
    # dedupe and prevent a user marking a new message read from hiding it.
    last = str(processor.state.get("last_received_at") or "")
    since = CUTOVER
    if last:
        overlap = datetime.fromisoformat(last.replace("Z", "+00:00")) - timedelta(days=2)
        since = max(CUTOVER, overlap.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    for item in client.list_unread_messages(since=since, unread_only=False):
        message = _normalize_graph_message(item)
        graph_id = message["graph_message_id"]
        if processor.has_processed_graph_id(graph_id):
            skipped += 1
            continue
        if message["has_attachments"]:
            message["attachments"] = client.list_attachments(graph_id)
            if not message["attachments"]:
                raise RuntimeError("Graph flagged attachments but returned none")
        result = processor.process_message(message)
        processor.record_processed_graph_id(graph_id, message["received_at"])
        processed += 1
        logging.info("intel@ staged route=%s vault_path=%s", result["route"], result["vault_path"])
    return processed, skipped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config()
    client = NotesGraphClient(config)
    processor = NotesMailboxProcessor(config, classifier=classify_intel_route)
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    while True:
        processed, skipped = poll_once(client, processor)
        logging.info("intel@ private staging poll processed=%s skipped=%s", processed, skipped)
        if args.once or stop.wait(config.poll_interval):
            return


if __name__ == "__main__":
    main()
