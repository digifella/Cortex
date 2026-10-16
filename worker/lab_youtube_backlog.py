"""Stage old, already-completed Lab summaries from forwarded Gmail for review.

Reads Gmail with the existing paul@project-143.com readonly OAuth grant. It
does not call Gemini, change mail, approve a draft, or publish to AI-Vault.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path.home()))
import nemoclaw_gmail_live as gmail
from worker import lab_youtube_direct as direct

REPORT = Path.home() / "recovery-2026-10-14/project-gmail-lab-backlog-audit.json"
HEADING = re.compile(r"(?m)^##\s+(.+)$")
MAIN_URL = re.compile(r"(?im)^\s*(?:\*\*)?URL:(?:\*\*)?\s*(https?://\S+)")
COMPLETED = re.compile(r"(?im)^Completed:\s*(\d{4}-\d{2}-\d{2})")


def extract_completed_report(body: str, expected_id: str) -> tuple[str, str, str, str]:
    if "Your Lab job is complete" not in body or "YouTube Summariser" not in body:
        raise ValueError("Not a forwarded Lab YouTube completion")
    heading = HEADING.search(body)
    if not heading:
        raise ValueError("Completed email has no Markdown report heading")
    report = body[heading.start():].strip()
    if "[body truncated]" in report:
        raise ValueError("Completed report exceeds safe Gmail read limit")
    urls = MAIN_URL.findall(report)
    if not urls:
        raise ValueError("Completed report has no primary video URL")
    canonical = direct.youtube_urls(urls[0])
    if len(canonical) != 1 or direct.youtube_summarise._youtube_video_id(canonical[0]) != expected_id:
        raise ValueError("Completed report URL differs from the audited video")
    date_match = COMPLETED.search(body)
    return heading.group(1).strip(), canonical[0], report, date_match.group(1) if date_match else ""


def stage(message_id: str, body: str, expected_id: str, state: dict) -> dict:
    title, url, report, completed_date = extract_completed_report(body, expected_id)
    for key, entry in state.items():
        if url in entry.get("urls", []) and entry.get("staged_path"):
            return {"status": "already_staged", "id": key}
    key = hashlib.sha256(("historical-lab:" + message_id).encode()).hexdigest()[:20]
    if key in state:
        raise ValueError("Backlog state key collision")
    topic, _group = direct.classify_slug_safe(title, report[:4000])
    name = f"recovered-{expected_id}-{direct.youtube_summarise._safe_filename(title)}.md"
    path = direct.STAGING / name
    front = {
        "title": title, "source_type": "lab-youtube-completion-recovered",
        "source_url": url, "source_date": completed_date,
        "source_date_source": "lab_completion_fallback" if completed_date else "unknown",
        "credibility_tier_value": 1, "credibility_tier_key": "commentary",
        "credibility_tier_label": "Commentary", "credibility": "Video summary; claims unverified",
        "review-status": "pending", "lab-review-id": key,
        "suggested-topic": topic, "domain": direct.slugify(topic),
    }
    content = ("---\n" + "\n".join(f"{k}: {json.dumps(v, ensure_ascii=False)}"
                                      for k, v in front.items()) +
               "\nwiki-ready: false\n---\n\n" + report + "\n")
    direct.STAGING.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise FileExistsError("A different backlog review draft has this video ID and title")
    else:
        path.write_text(content, encoding="utf-8")
    state[key] = {"staged_path": str(path), "urls": [url], "title": title,
                  "suggested_topic": topic, "review_status": "pending",
                  "discord_notified": False, "historical_gmail_message_id": message_id,
                  "source_date": completed_date}
    direct._save_state(state)
    return {"status": "private_review", "id": key, "title": title}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Stage audited gaps in private review")
    args = parser.parse_args()
    audit = json.loads(REPORT.read_text(encoding="utf-8"))
    if audit.get("mailbox") != "paul@project-143.com" or not audit.get("read_only"):
        raise ValueError("Expected the project Gmail read-only audit")
    candidates = [row for row in audit.get("rows", [])
                  if row.get("status") == "candidate_missing_from_public_wiki"
                  and len(row.get("video_ids") or []) == 1]
    if not args.apply:
        print(json.dumps({"candidate_count": len(candidates), "action": "none"}))
        return
    service = gmail._gmail_service()
    gmail._verify_account(service)
    state = direct._load_state()
    results = []
    for row in candidates:
        raw = service.users().messages().get(
            userId="me", id=row["gmail_message_id"], format="full").execute()
        msg = gmail._message_result(raw, 100000)
        results.append(stage(msg["id"], msg["body"], row["video_ids"][0], state))
    if any(result["status"] == "private_review" for result in results):
        direct.send_pending_review_notice(state)
    print(json.dumps({"staged": sum(r["status"] == "private_review" for r in results),
                      "already_staged": sum(r["status"] == "already_staged" for r in results)}))


if __name__ == "__main__":
    main()
