from __future__ import annotations

import logging
import signal
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cortex_engine.intel_mailbox import IntelMailboxConfig, IntelMailboxPoller


def load_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _read_mailbox_config() -> tuple[IntelMailboxConfig, int, str]:
    env_path = ROOT / "worker" / "config.env"
    file_vars = load_env_file(env_path)
    for key, value in file_vars.items():
        if key not in os.environ:
            os.environ[key] = value

    def get(name: str, default: str = "") -> str:
        return str(os.environ.get(name, file_vars.get(name, default)) or "").strip()

    cfg = IntelMailboxConfig.from_env(file_vars)
    poll_interval = max(15, int(get("INTEL_IMAP_POLL_INTERVAL", "60") or "60"))
    log_level = get("LOG_LEVEL", "INFO").upper()
    return cfg, poll_interval, log_level


import os


def main() -> int:
    cfg, poll_interval, log_level = _read_mailbox_config()
    logging.basicConfig(level=getattr(logging, log_level, logging.INFO), format="%(asctime)s %(levelname)s %(message)s")
    logging.info(
        "Intel mailbox worker started: host=%s folder=%s org=%s poll=%ss callback=%s",
        cfg.host,
        cfg.folder,
        cfg.org_name,
        poll_interval,
        "configured" if cfg.callback_url else "outbox",
    )

    poller = IntelMailboxPoller(cfg)
    stop_event = threading.Event()

    def _stop(*_args):
        stop_event.set()

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    while not stop_event.is_set():
        try:
            summary = poller.poll_once()
            for result in summary.get("results", []) or []:
                delivery = result.get("delivery") or {}
                delivery_parts = [str(delivery.get("status") or "")]
                if delivery.get("http_status"):
                    delivery_parts.append(f"http={delivery['http_status']}")
                response = delivery.get("response") if isinstance(delivery.get("response"), dict) else {}
                if response:
                    if "imported_suggested" in response:
                        delivery_parts.append(f"suggested={response.get('imported_suggested', 0)}")
                    if "imported_updates" in response:
                        delivery_parts.append(f"updates={response.get('imported_updates', 0)}")
                    if response.get("skipped"):
                        delivery_parts.append(f"skipped={response.get('skipped')}")
                logging.info(
                    "Intel mailbox extracted trace_id=%s signal_id=%s entities=%s update_suggestions=%s warnings=%s delivery=%s",
                    result.get("trace_id", ""),
                    result.get("signal_id", ""),
                    ", ".join(result.get("entity_names") or []) or "none",
                    result.get("update_suggestion_count", 0),
                    result.get("warning_count", 0),
                    " ".join(part for part in delivery_parts if part),
                )
                for warning in result.get("warnings") or []:
                    logging.warning("Intel mailbox extraction warning trace_id=%s: %s", result.get("trace_id", ""), warning)
            processed = int(summary.get("processed", 0) or 0)
            skipped = int(summary.get("skipped", 0) or 0)
            failures = int(summary.get("failures", 0) or 0)
            if processed or failures:
                logging.info(
                    "Intel mailbox poll complete: processed=%s skipped=%s failures=%s",
                    processed,
                    skipped,
                    failures,
                )
            elif skipped:
                logging.debug(
                    "Intel mailbox poll skipped=%s with no new processed messages",
                    skipped,
                )
        except Exception:
            logging.exception("Intel mailbox poll failed")
        if stop_event.wait(poll_interval):
            break

    logging.info("Intel mailbox worker stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
