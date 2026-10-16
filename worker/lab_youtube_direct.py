"""Direct lab@ YouTube intake: Graph mail -> Gemini -> private review -> owner mail.

Run once from cron. No website queue or legacy Gmail mailbox is involved.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests
sys.path.insert(0, str(Path.home()))
from nemoclaw_taxonomy import classify_slug_safe, slugify

from worker.lab_mailbox_worker import LabGraphClient, LabMailboxConfig, load_env_file, normalize_graph_message
from worker.handlers import youtube_summarise

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "worker" / "config.env"
STATE = Path.home() / ".local/share/cortex-lab-youtube/state.json"
RESULTS = Path.home() / ".local/share/cortex-lab-youtube/results"
STAGING = Path("/mnt/c/Users/paul/OneDrive - VentraIP Australia/Vault_OneDrive/Public_Review/lab-youtube")
OWNER = "paul@project-143.com"
SENDERS = {"paul@project-143.com"}
CUTOVER = datetime.fromisoformat("2026-10-15T17:15:00+11:00")
MODEL = "gemini-3.5-flash-lite"
URL_RE = re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)


def youtube_urls(text: str) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    for match in URL_RE.finditer(text):
        url = match.group().rstrip(".,;:!?)]}")
        if not youtube_summarise._is_youtube_url(url):
            continue
        video_id = youtube_summarise._youtube_video_id(url)
        if video_id not in seen:
            seen.add(video_id)
            found.append(youtube_summarise._canonical_youtube_url(url))
    return found


def _load_state() -> dict:
    try:
        data = json.loads(STATE.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (FileNotFoundError, ValueError):
        return {}


def _save_state(state: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".lab-youtube-", dir=STATE.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump(state, file, ensure_ascii=False, indent=2)
            file.flush()
            os.fsync(file.fileno())
        os.chmod(name, 0o600)
        os.replace(name, STATE)
    finally:
        Path(name).unlink(missing_ok=True)


def _safe_message_key(message_id: str) -> str:
    return hashlib.sha256(message_id.encode("utf-8")).hexdigest()[:20]


def summarise_urls(urls: list[str]) -> dict:
    """Use the supported Gemini SDK's native YouTube URL input, free-tier Flash-Lite."""
    from google import genai
    from google.genai import types

    key = os.environ.get("GOOGLE_API_KEY", "")
    if not key:
        raise RuntimeError("GOOGLE_API_KEY is unavailable")
    results = []
    with genai.Client(api_key=key, http_options=types.HttpOptions(timeout=120000)) as client:
        for url in urls:
            metadata = youtube_summarise._fetch_youtube_metadata(url)
            extra_context = youtube_summarise._fetch_youtube_extra_context(url)
            prompt = (
                youtube_summarise.MODE_PROMPTS["summary"]
                + "\nAttribute claims to the video or its speaker. Do not present them as independently verified facts."
            )
            interaction = client.interactions.create(
                model=MODEL,
                input=[{"type": "text", "text": prompt}, {"type": "video", "uri": url}],
            )
            summary = (interaction.output_text or "").strip()
            if not summary:
                raise RuntimeError(f"Gemini returned an empty summary for {url}")
            results.append({
                "url": url,
                "report_title": metadata.get("video_title") or f"YouTube video {youtube_summarise._youtube_video_id(url)}",
                "video_title": metadata.get("video_title", ""),
                "author": metadata.get("author", ""),
                "author_url": metadata.get("author_url", ""),
                "sections": {"summary": summary},
                "extra_context": extra_context,
            })
    report = youtube_summarise._build_report(results, ["summary"], MODEL)
    fd, name = tempfile.mkstemp(prefix="lab-youtube-", suffix=".md")
    with os.fdopen(fd, "w", encoding="utf-8") as file:
        file.write(report)
    return {"output_data": {"success_count": len(results), "error_count": 0,
                            "report_title": results[0]["report_title"] if len(results) == 1
                            else f"YouTube Summary Report - {len(results)} Videos"},
            "output_file": Path(name)}


def list_recent_unread(graph: LabGraphClient) -> list[dict]:
    """Page the current intake window so older unrelated unread mail cannot block it."""
    url = f"https://graph.microsoft.com/v1.0/users/{graph.config.graph_mailbox}/mailFolders/inbox/messages"
    params = {
        "$filter": f"isRead eq false and receivedDateTime ge {CUTOVER.astimezone(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}",
        "$top": "50",
        "$select": "id,subject,from,toRecipients,receivedDateTime,body,internetMessageId",
    }
    messages: list[dict] = []
    while url:
        response = requests.get(url, headers=graph._headers(), params=params,
                                timeout=graph.config.graph_timeout)
        response.raise_for_status()
        data = response.json()
        messages.extend(data.get("value") or [])
        url = data.get("@odata.nextLink", "")
        if url and urlparse(url).netloc != "graph.microsoft.com":
            raise RuntimeError("Unexpected Graph pagination host")
        params = None
    return messages


def _write_staged_note(report: str, title: str, urls: list[str], key: str) -> tuple[Path, str]:
    STAGING.mkdir(parents=True, exist_ok=True)
    video_id = youtube_summarise._youtube_video_id(urls[0])
    name = f"{datetime.now().date()}-{video_id}-{youtube_summarise._safe_filename(title)}.md"
    path = STAGING / name
    topic, _group = classify_slug_safe(title, report[:4000])
    if not report.startswith("---\n"):
        raise RuntimeError("Gemini report has no frontmatter")
    staged = report.replace("---\n", "---\n" +
        f"wiki-ready: false\nreview-status: pending\nlab-review-id: {key}\n"
        f"suggested-topic: {json.dumps(topic, ensure_ascii=False)}\n"
        f"domain: {slugify(topic)}\n", 1).rstrip() + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != staged:
            raise RuntimeError(f"Staged note filename collision: {path}")
        return path, topic
    tmp = path.with_name(f".{path.name}.{key}.tmp")
    with open(tmp, "x", encoding="utf-8") as file:
        file.write(staged)
        file.flush()
        os.fsync(file.fileno())
    os.replace(tmp, path)
    return path, topic


def process_message(graph: LabGraphClient, item: dict, state: dict) -> bool:
    message = normalize_graph_message(item)
    graph_id = message["graph_message_id"]
    if not graph_id:
        return False
    sender = message["from_email"].casefold()
    if sender not in SENDERS:
        print(f"[lab-youtube] skipped non-owner sender for message {graph_id[:8]}")
        return False
    try:
        received = datetime.fromisoformat(message["received_at"].replace("Z", "+00:00"))
    except ValueError:
        return False
    if received < CUTOVER:
        return False
    urls = youtube_urls(message["text_body"] + "\n" + message["subject"])
    if not urls:
        return False
    if len(urls) > 5:
        print(f"[lab-youtube] skipped message {graph_id[:8]} with more than five YouTube URLs")
        return False

    key = _safe_message_key(graph_id)
    entry = state.setdefault(key, {})
    if not entry.get("result_path"):
        result = summarise_urls(urls)
        output = result["output_data"]
        if output.get("error_count") or output.get("success_count") != len(urls):
            raise RuntimeError(f"YouTube summary incomplete for message {graph_id[:8]}")
        generated = Path(result["output_file"])
        report = generated.read_text(encoding="utf-8")
        generated.unlink(missing_ok=True)
        RESULTS.mkdir(parents=True, exist_ok=True)
        result_path = RESULTS / f"{key}.md"
        result_path.write_text(report, encoding="utf-8")
        os.chmod(result_path, 0o600)
        entry.update({"result_path": str(result_path), "urls": urls,
                      "title": output["report_title"]})
        _save_state(state)

    if not entry.get("staged_path"):
        report = Path(entry["result_path"]).read_text(encoding="utf-8")
        staged_path, topic = _write_staged_note(report, entry["title"], entry["urls"], key)
        entry["staged_path"] = str(staged_path)
        entry["suggested_topic"] = topic
        _save_state(state)

    if not entry.get("mail_sent"):
        report = Path(entry["result_path"]).read_text(encoding="utf-8")
        graph.send_reply(OWNER, f"Lab YouTube draft for review — {entry['title']}",
                         f"Suggested topic: {entry['suggested_topic']}\n"
                         f"Review ID: {key}\nStatus: pending publication decision\n\n" + report)
        entry["mail_sent"] = True
        _save_state(state)
    graph.mark_read(graph_id)
    entry["completed_at"] = datetime.now(timezone.utc).isoformat()
    entry.setdefault("discord_notified", False)
    _save_state(state)
    print(f"[lab-youtube] staged message {graph_id[:8]}: {entry['staged_path']}")
    return True


def send_pending_review_notice(state: dict) -> bool:
    """Notify Hermes after intake; retry on the next poll if delivery fails."""
    outstanding = [key for key, entry in state.items()
                   if entry.get("staged_path") and entry.get("review_status", "pending") == "pending"
                   and entry.get("discord_notified") is False]
    if not outstanding:
        return False
    from worker.lab_youtube_digest import send_review_cards
    from worker.lab_youtube_review import pending

    items = [item for item in pending() if item["id"] in outstanding]
    send_review_cards(items)
    for key in outstanding:
        state[key]["discord_notified"] = True
    _save_state(state)
    return True


def main() -> int:
    env = load_env_file(CONFIG)
    key = env.get("GOOGLE_API_KEY", "").strip()
    if not key:
        print("[lab-youtube] GOOGLE_API_KEY missing; no mail changed")
        return 2
    os.environ["GOOGLE_API_KEY"] = key
    config = LabMailboxConfig.from_env(env)
    if not all((config.graph_tenant_id, config.graph_client_id,
                config.graph_client_secret, config.graph_mailbox)):
        print("[lab-youtube] Lab Graph config incomplete; no mail changed")
        return 2
    graph = LabGraphClient(config)
    state = _load_state()
    count = 0
    failures = 0
    for item in list_recent_unread(graph):
        try:
            count += bool(process_message(graph, item, state))
        except Exception as exc:
            failures += 1
            print(f"[lab-youtube] processing failed: {type(exc).__name__}: {exc}", file=sys.stderr)
    try:
        if send_pending_review_notice(state):
            print("[lab-youtube] sent Hermes review notice")
    except Exception as exc:
        failures += 1
        print(f"[lab-youtube] Hermes review notice failed; will retry: {type(exc).__name__}: {exc}",
              file=sys.stderr)
    print(f"[lab-youtube] completed {count} message(s); failures={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
