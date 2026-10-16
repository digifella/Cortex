"""Send a daily Hermes/Discord list of pending YouTube review decisions."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from worker.lab_youtube_review import DOMAIN_SLUGS, TOPICS, normalize_topic, pending

SP4 = "paul@100.71.168.3"
KEY = Path.home() / ".ssh/fastfella_recovery_ed25519"
CARD_HANDLER = "/home/paul/.hermes/hermes-agent/plugins/platforms/discord/lab_review_cards.py"


def public_topics() -> list[str]:
    """Use the actual Obsidian public-vault folders, not old website labels."""
    if not TOPICS.is_dir():
        raise RuntimeError("Public Obsidian Topics folder is unavailable")
    topics = sorted(folder.name for folder in TOPICS.iterdir()
                    if folder.is_dir() and folder.name.casefold() in DOMAIN_SLUGS)
    if not 1 <= len(topics) <= 25:
        raise RuntimeError(f"Expected 1–25 public-vault topics, found {len(topics)}")
    return topics


def send_review_cards(items: list[dict]) -> int:
    """One Discord topic-picker card per draft; SP4 deduplicates by review ID."""
    if not items:
        return 0
    topics = public_topics()
    for item in items:
        try:
            suggested, _slug = normalize_topic(str(item["topic"]))
        except ValueError:
            suggested = "Undecided"
        payload = {"id": item["id"], "title": item["title"],
                   "topic": suggested, "topics": topics}
        result = subprocess.run(
            ["ssh", "-i", str(KEY), "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
             SP4, "python3", CARD_HANDLER, "send"],
            input=json.dumps(payload, ensure_ascii=False), text=True,
            capture_output=True, timeout=45, check=False,
        )
        if result.returncode:
            raise RuntimeError(f"SP4 review-card send failed for {item['id'][:8]} (rc={result.returncode})")
        reply = json.loads(result.stdout)
        if reply.get("id") != item["id"] or reply.get("status") not in {"sent", "already-sent"}:
            raise RuntimeError(f"SP4 review-card response invalid for {item['id'][:8]}")
    return len(items)


def format_digest_pages(items: list[dict], *, new: bool = False) -> list[str]:
    """Keep every review ID visible and each Discord message safely below its limit."""
    if not items:
        return ["📺 Lab YouTube review: no drafts awaiting a publication decision."]
    heading = "New Lab YouTube drafts" if new else "Lab YouTube review"
    footer = ("Reply with IDs: `approve 12345678`; "
              "`approve 12345678 as Biology & Life Sciences`; or `decline 12345678`.")
    entries = []
    for item in items:
        title = " ".join(str(item.get("title") or "Untitled").split())
        entries.append(f"• `{str(item['id'])[:8]}` {title}\n  Topic: {item['topic']}")
    groups: list[list[str]] = []
    current: list[str] = []
    # Reserve room for the heading, page number, and reply instructions.
    for entry in entries:
        if current and len("\n".join(current)) + len(entry) + len(footer) + 110 > 1700:
            groups.append(current)
            current = []
        current.append(entry)
    if current:
        groups.append(current)
    total = len(groups)
    return [f"📺 {heading} — {len(items)} pending ({page}/{total})\n"
            + "\n".join(group) + "\n" + footer
            for page, group in enumerate(groups, 1)]


def send(message: str) -> None:
    result = subprocess.run(
        ["ssh", "-i", str(KEY), "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
         SP4, "~/hermes-alert-from-sp8.sh"],
        input=message, text=True, capture_output=True, timeout=45, check=False,
    )
    if result.returncode != 0 or "OK" not in result.stdout.split():
        raise RuntimeError(f"Hermes Discord relay failed (rc={result.returncode})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--send", action="store_true")
    parser.add_argument("--send-text", action="store_true", help="Use the text-list fallback")
    args = parser.parse_args()
    items = pending()
    messages = format_digest_pages(items)
    if args.send and not args.send_text:
        if items:
            count = send_review_cards(items)
            print(f"Sent or confirmed {count} per-video Lab review card(s) via Hermes Discord")
        else:
            send(messages[0])
            print("Sent empty Lab review status via Hermes Discord")
    elif args.send_text:
        for message in messages:
            send(message)
        print(f"Sent {len(messages)} Lab YouTube review digest page(s) via Hermes Discord")
    else:
        print("\n\n".join(messages))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
