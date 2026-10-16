"""Inspect Hermes review requests; apply only when --apply is explicitly given."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
from pathlib import Path

from worker import lab_youtube_direct as direct
from worker import lab_youtube_review as review

SP4 = "paul@100.71.168.3"
KEY = Path.home() / ".ssh/fastfella_recovery_ed25519"
CARD_HANDLER = "/home/paul/.hermes/hermes-agent/plugins/platforms/discord/lab_review_cards.py"
FILENAME = re.compile(r"\d{8}T\d{6}Z-[0-9a-f]{32}\.json")
READ_REMOTE = (
    "import json,pathlib; p=pathlib.Path.home()/'.hermes/lab-youtube-review-requests'; "
    "print(json.dumps([{'file':f.name,'request':json.loads(f.read_text())} "
    "for f in sorted(p.glob('*.json'))[:50]] if p.is_dir() else []))"
)
ACK_REMOTE = (
    "import pathlib,sys; p=pathlib.Path.home()/'.hermes/lab-youtube-review-requests'/sys.argv[1]; "
    "p.rename(p.with_suffix('.done'))"
)


def _ssh(code: str, filename: str = "") -> str:
    remote = ["python3", "-c", code]
    if filename:
        remote.append(filename)
    command = ["ssh", "-i", str(KEY), "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
               SP4, shlex.join(remote)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=30, check=False)
    if result.returncode:
        raise RuntimeError(f"SP4 review request read failed (rc={result.returncode})")
    return result.stdout


def _card_status(review_id: str, status: str) -> None:
    command = ["ssh", "-i", str(KEY), "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
               SP4, "python3", CARD_HANDLER, "status", review_id, status]
    result = subprocess.run(command, capture_output=True, text=True, timeout=35, check=False)
    if result.returncode:
        raise RuntimeError(f"SP4 review card update failed (rc={result.returncode})")


def collect(*, apply: bool = False) -> int:
    requests = json.loads(_ssh(READ_REMOTE))
    applied = 0
    for row in requests:
        filename = str(row.get("file", ""))
        request = row.get("request") or {}
        if not FILENAME.fullmatch(filename) or not isinstance(request, dict):
            continue
        action = request.get("action")
        review_id = str(request.get("id", ""))
        topic = str(request.get("topic", ""))
        if action not in {"approve", "decline", "redirect"} or not re.fullmatch(r"[0-9a-f]{8,20}", review_id):
            continue
        if len(topic) > 100 or any(char in topic for char in "\r\n\x00"):
            continue
        if not apply:
            print(f"[lab-review] pending {filename}: {action} {review_id[:8]} topic={topic}")
            continue
        try:
            state = direct._load_state()
            _key, entry = review._resolve(state, review_id)
            if entry.get("review_status") not in {"approved", "declined"}:
                result = review.decide(review_id, action, topic)
                print(f"[lab-review] {result['status']} {result['id'][:8]} topic={result.get('topic', '')}")
                applied += 1
            elif entry.get("review_status") != action:
                raise ValueError(f"conflicting prior decision: {entry['review_status']}")
            _ssh(ACK_REMOTE, filename)
            if action in {"approve", "decline"}:
                try:
                    _card_status(_key, "approved" if action == "approve" else "declined")
                except Exception as exc:
                    print(f"[lab-review] card display needs retry for {_key[:8]}: {exc}")
        except Exception as exc:
            print(f"[lab-review] held {filename}: {type(exc).__name__}: {exc}")
    return applied


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Apply validated queued decisions")
    args = parser.parse_args()
    print(f"Applied {collect(apply=args.apply)} review decision(s)" if args.apply else
          f"Inspected {collect()} review queue (no changes)")
