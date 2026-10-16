"""Review staged Lab YouTube summaries before public-vault publication."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path.home()))
from worker import lab_youtube_direct as direct

PUBLIC = Path("/mnt/c/Users/paul/Documents/AI-Vault/lab-notes")
DECLINED = direct.STAGING.parent / "lab-youtube-declined"
TOPICS = Path("/mnt/c/Users/paul/Documents/AI-Vault/Topics")
DOMAIN_SLUGS = {
    "ai & agents": "ai-agents", "biology & life sciences": "biology-life-sciences",
    "built environment & architecture": "built-environment-architecture",
    "business & strategy": "business-strategy", "cosmology & space": "cosmology-space",
    "creative pursuits": "creative-pursuits", "earth systems, geology & climate": "earth-systems-geology-climate",
    "entertainment & games": "entertainment-games", "food & nutrition": "food-nutrition",
    "health & wellbeing": "health-wellbeing", "history & anthropology": "history-anthropology",
    "maths & cryptography": "maths-cryptography", "philosophy & religion": "philosophy-religion",
    "science & physics": "science-physics", "security & infrastructure": "security-infrastructure",
    "society, politics & institutions": "society-politics-institutions",
    "tools & platforms": "tools-platforms", "travels & journeys": "travels-journeys",
    "ux & design": "ux-design", "undecided": "undecided",
}
TOPIC_ALIASES = {
    "tools, platforms & infrastructure": "tools & platforms",
    "science, physics & research": "science & physics",
    "maths, logic & crypto": "maths & cryptography",
    "philosophy, ethics & religion": "philosophy & religion",
    "society, politics & conflict": "society, politics & institutions",
}


def normalize_topic(value: str) -> tuple[str, str]:
    wanted = TOPIC_ALIASES.get(value.strip().casefold(), value.strip().casefold())
    if not TOPICS.is_dir():
        raise ValueError("Public Topics folder is unavailable")
    for folder in TOPICS.iterdir():
        label = folder.name.casefold()
        slug = DOMAIN_SLUGS.get(label)
        if folder.is_dir() and slug and wanted in {label, slug}:
            return folder.name, slug
    raise ValueError("Unknown topic. Use one of the top-level AI-Vault Topics folders.")


def _field(text: str, key: str, value: str) -> str:
    if not text.startswith("---\n"):
        raise ValueError("Draft is missing YAML frontmatter")
    rendered = value if key == "wiki-ready" and value in {"true", "false"} else json.dumps(value, ensure_ascii=False)
    line = f"{key}: {rendered}"
    pattern = rf"(?m)^{re.escape(key)}:.*$"
    if re.search(pattern, text):
        return re.sub(pattern, lambda _: line, text, count=1)
    return text.replace("---\n", f"---\n{line}\n", 1)


def _resolve(state: dict, review_id: str) -> tuple[str, dict]:
    matches = [(key, value) for key, value in state.items() if key.startswith(review_id.lower())]
    if len(matches) != 1 or len(review_id) < 8:
        raise ValueError("Review ID must uniquely match at least eight characters")
    return matches[0]


def decide(review_id: str, action: str, topic: str = "") -> dict:
    state = direct._load_state()
    key, entry = _resolve(state, review_id)
    if entry.get("review_status") in {"approved", "declined"}:
        raise ValueError(f"Already {entry['review_status']}")
    staged = Path(entry.get("staged_path", ""))
    if not staged.is_file() or staged.parent.resolve() != direct.STAGING.resolve():
        raise ValueError("Staged draft is missing or outside the review folder")
    text = staged.read_text(encoding="utf-8")
    selected = topic or entry.get("selected_topic") or entry.get("suggested_topic") or "Undecided"
    label, slug = normalize_topic(selected)

    if action == "redirect":
        text = _field(text, "selected-topic", label)
        text = _field(text, "domain", slug)
        staged.write_text(text, encoding="utf-8")
        entry["selected_topic"] = label
        direct._save_state(state)
        return {"id": key, "status": "pending", "topic": label, "path": str(staged)}

    if action == "decline":
        DECLINED.mkdir(parents=True, exist_ok=True)
        target = DECLINED / staged.name
        if target.exists():
            raise ValueError("Declined archive file already exists")
        os.replace(staged, target)
        entry.update({"review_status": "declined", "reviewed_at": datetime.now(timezone.utc).isoformat(),
                      "archive_path": str(target)})
        direct._save_state(state)
        return {"id": key, "status": "declined", "path": str(target)}

    if action != "approve":
        raise ValueError(f"Unknown action {action}")
    text = _field(text, "wiki-ready", "true")
    text = _field(text, "review-status", "approved")
    text = _field(text, "selected-topic", label)
    text = _field(text, "domain", slug)
    PUBLIC.mkdir(parents=True, exist_ok=True)
    target = PUBLIC / staged.name
    if target.exists():
        raise ValueError("Public note with this name already exists; review collision manually")
    fd, name = tempfile.mkstemp(prefix=".lab-approved-", dir=PUBLIC)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            file.write(text)
            file.flush()
            os.fsync(file.fileno())
        os.replace(name, target)
    finally:
        Path(name).unlink(missing_ok=True)
    if target.read_text(encoding="utf-8") != text:
        raise RuntimeError("Public note read-back failed; draft retained")
    staged.unlink()
    entry.update({"review_status": "approved", "reviewed_at": datetime.now(timezone.utc).isoformat(),
                  "selected_topic": label, "public_path": str(target)})
    direct._save_state(state)
    return {"id": key, "status": "approved", "topic": label, "path": str(target)}


def pending() -> list[dict]:
    state = direct._load_state()
    return [{"id": key, "title": entry.get("title", ""),
             "topic": entry.get("selected_topic") or entry.get("suggested_topic") or "Undecided",
             "urls": entry.get("urls", [])}
            for key, entry in state.items()
            if entry.get("staged_path") and entry.get("review_status", "pending") == "pending"
            and Path(entry["staged_path"]).is_file()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("list")
    for action in ("approve", "decline", "redirect"):
        cmd = sub.add_parser(action)
        cmd.add_argument("id")
        cmd.add_argument("--topic", default="")
    args = parser.parse_args()
    result = pending() if args.action == "list" else decide(args.id, args.action, args.topic)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
