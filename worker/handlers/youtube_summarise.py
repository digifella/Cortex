"""
YouTube Summarise Handler
Summarises one or more YouTube videos using Gemini (native URL support) or Claude
(transcript extraction via youtube-transcript-api).

Deploy this file to the Cortex suite worker handlers directory.

Dependencies (add to cortex_suite requirements):
    google-generativeai
    youtube-transcript-api
    anthropic  (already present)

Config (config.env):
    GOOGLE_API_KEY=...           # For Gemini API
    ANTHROPIC_API_KEY=...        # For Claude API (already present)

input_data schema:
    urls          list[str]   YouTube URLs to summarise
    api_choice    str         gemini-flash | gemini-pro | claude-haiku | claude-sonnet
    output_modes  list[str]   summary | timestamps | meeting_notes | action_items | transcript
    push_to_kb    bool        Whether to also push output to KB (via QUEUE_SERVER_URL)
    kb_category   str         KB category for push_to_kb (optional)
    source_system str         Origin system (lab / admin)
"""

import os
import json
import logging
import tempfile
import textwrap
import re
from datetime import date
from pathlib import Path

logger = logging.getLogger(__name__)

# ── API clients (lazy import) ──

def _gemini_client():
    import google.generativeai as genai
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY not set in config.env")
    genai.configure(api_key=api_key)
    return genai


def _anthropic_client():
    import anthropic
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set in config.env")
    return anthropic.Anthropic(api_key=api_key)


# ── Mode prompts ──

MODE_PROMPTS = {
    "summary": (
        "Provide a clear, well-structured summary of the video content. "
        "Include the main topic, key points discussed, and any conclusions or takeaways. "
        "Aim for 3-5 paragraphs."
    ),
    "timestamps": (
        "List the key moments and topics in this video with timestamps. "
        "Format as:\n- [HH:MM:SS or MM:SS] — Brief description of what happens\n"
        "Cover the major sections, topic shifts, and highlighted moments."
    ),
    "meeting_notes": (
        "Format the content as structured meeting notes:\n"
        "## Participants / Speakers\n(list if identifiable)\n"
        "## Key Discussion Points\n(bullet points)\n"
        "## Decisions Made\n(if any)\n"
        "## Action Items\n(if any)\n"
        "## Next Steps\n(if mentioned)"
    ),
    "action_items": (
        "Extract all action items, tasks, recommendations, or calls-to-action mentioned in this video. "
        "Format as a numbered checklist. If none are explicitly stated, infer the key take-actions "
        "a viewer would want to act on."
    ),
    "transcript": (
        "Provide a clean, readable transcript of the video content. "
        "Use speaker labels where identifiable (e.g. 'Host:', 'Guest:'). "
        "Preserve the natural flow of conversation."
    ),
}


# ── Gemini path ──

def _summarise_gemini(url: str, model_name: str, output_modes: list[str]) -> dict:
    """Use Gemini's native YouTube URL understanding."""
    genai = _gemini_client()

    model_id = "gemini-1.5-pro" if model_name == "gemini-pro" else "gemini-1.5-flash"
    model = genai.GenerativeModel(model_id)

    sections = {}
    for mode in output_modes:
        prompt = MODE_PROMPTS.get(mode, f"Provide {mode} for this video.")
        full_prompt = (
            f"{prompt}\n\nVideo: {url}"
        )
        try:
            response = model.generate_content([
                {"text": full_prompt},
                {"file_data": {"mime_type": "video/youtube", "file_uri": url}},
            ])
            sections[mode] = response.text.strip()
        except Exception as e:
            logger.warning(f"Gemini mode '{mode}' failed for {url}: {e}")
            sections[mode] = f"[Error generating {mode}: {e}]"

    return sections


# ── Transcript extraction (for Claude path) ──

def _extract_transcript(url: str) -> str:
    """Extract transcript text using youtube-transcript-api."""
    from youtube_transcript_api import YouTubeTranscriptApi
    from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound

    # Extract video ID
    video_id = None
    if "youtu.be/" in url:
        video_id = url.split("youtu.be/")[1].split("?")[0].split("/")[0]
    elif "v=" in url:
        video_id = url.split("v=")[1].split("&")[0]

    if not video_id:
        raise ValueError(f"Cannot extract video ID from: {url}")

    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
    except (TranscriptsDisabled, NoTranscriptFound) as e:
        # Try auto-generated
        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en", "en-US", "en-GB"])
        except Exception:
            raise RuntimeError(f"No transcript available for video {video_id}: {e}")

    # Format with timestamps
    lines = []
    for entry in transcript_list:
        secs = int(entry["start"])
        mm, ss = divmod(secs, 60)
        hh, mm = divmod(mm, 60)
        ts = f"{hh:02d}:{mm:02d}:{ss:02d}" if hh else f"{mm:02d}:{ss:02d}"
        lines.append(f"[{ts}] {entry['text']}")

    return "\n".join(lines)


# ── Claude path ──

def _summarise_claude(transcript: str, url: str, model_name: str, output_modes: list[str]) -> dict:
    """Summarise using Claude with an extracted transcript."""
    client = _anthropic_client()

    model_id = (
        "claude-sonnet-4-6" if model_name == "claude-sonnet"
        else "claude-haiku-4-5-20251001"
    )

    # Truncate long transcripts (100k chars ≈ ~75k tokens, well within context)
    transcript_excerpt = transcript[:90_000]
    if len(transcript) > 90_000:
        transcript_excerpt += "\n\n[Transcript truncated due to length]"

    sections = {}
    for mode in output_modes:
        prompt = MODE_PROMPTS.get(mode, f"Provide {mode} for this video.")
        user_message = (
            f"Here is the transcript of a YouTube video ({url}):\n\n"
            f"---\n{transcript_excerpt}\n---\n\n"
            f"Task: {prompt}"
        )
        try:
            response = client.messages.create(
                model=model_id,
                max_tokens=4096,
                messages=[{"role": "user", "content": user_message}],
            )
            sections[mode] = response.content[0].text.strip()
        except Exception as e:
            logger.warning(f"Claude mode '{mode}' failed for {url}: {e}")
            sections[mode] = f"[Error generating {mode}: {e}]"

    return sections


# ── Report builder ──

MODE_LABELS = {
    "summary":       "Summary",
    "timestamps":    "Key Timestamps",
    "meeting_notes": "Meeting Notes",
    "action_items":  "Action Items",
    "transcript":    "Transcript",
}


def _build_report(results: list[dict], output_modes: list[str], api_choice: str) -> str:
    today = date.today().isoformat()
    mode_labels = ", ".join(MODE_LABELS.get(m, m) for m in output_modes)
    api_label = {
        "gemini-flash": "Gemini 1.5 Flash",
        "gemini-pro": "Gemini 1.5 Pro",
        "claude-haiku": "Claude Haiku",
        "claude-sonnet": "Claude Sonnet",
    }.get(api_choice, api_choice)

    lines = [
        "---",
        "title: YouTube Summary Report",
        f"date: {today}",
        "source_type: youtube_summary",
        f"api: {api_label}",
        f"modes: {mode_labels}",
        "---",
        "",
        "# YouTube Summary Report",
        f"Generated: {today} · API: {api_label} · Modes: {mode_labels}",
        "",
    ]

    for i, result in enumerate(results, 1):
        lines.append(f"---\n")
        url = result.get("url", "")
        lines.append(f"## Video {i}")
        lines.append(f"**URL:** {url}")
        lines.append("")

        sections = result.get("sections", {})
        for mode in output_modes:
            label = MODE_LABELS.get(mode, mode)
            content = sections.get(mode, "[Not generated]")
            lines.append(f"### {label}")
            lines.append(content)
            lines.append("")

    return "\n".join(lines)


# ── Push to KB ──

def _push_to_kb(content: str, kb_category: str, job: dict) -> None:
    """POST the report as a knowledge document to the website API."""
    import urllib.request
    import urllib.parse

    server_url = os.environ.get("QUEUE_SERVER_URL", "").rstrip("/")
    secret_key = os.environ.get("QUEUE_SECRET_KEY", "")
    if not server_url or not secret_key:
        logger.warning("push_to_kb: QUEUE_SERVER_URL or QUEUE_SECRET_KEY not set — skipping")
        return

    kb_api = f"{server_url}/admin/knowledge_api.php"
    today = date.today().isoformat()
    filename = f"youtube_summary_{today}_{job.get('id', 'job')}.md"

    data = urllib.parse.urlencode({
        "action": "upload_text",
        "filename": filename,
        "content": content,
        "category": kb_category or "General",
        "_secret": secret_key,
    }).encode()

    try:
        req = urllib.request.Request(kb_api, data=data, method="POST")
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read().decode()
            logger.info(f"push_to_kb response: {body[:200]}")
    except Exception as e:
        logger.warning(f"push_to_kb failed: {e}")


# ── Main handler ──

def handle(input_path, input_data: dict, job: dict):
    """
    Entry point called by the queue worker.

    Returns:
        {"output_data": dict, "output_file": Path | None}
    """
    urls         = input_data.get("urls", [])
    api_choice   = input_data.get("api_choice", "gemini-flash")
    output_modes = input_data.get("output_modes", ["summary"])
    push_to_kb   = input_data.get("push_to_kb", False)
    kb_category  = input_data.get("kb_category", "")

    if not urls:
        raise ValueError("No YouTube URLs provided in input_data")

    use_gemini = api_choice.startswith("gemini")
    results = []
    errors  = []

    for url in urls:
        logger.info(f"Processing: {url} via {api_choice}")
        try:
            if use_gemini:
                sections = _summarise_gemini(url, api_choice, output_modes)
            else:
                transcript = _extract_transcript(url)
                sections   = _summarise_claude(transcript, url, api_choice, output_modes)
            results.append({"url": url, "sections": sections})
        except Exception as e:
            logger.error(f"Failed to process {url}: {e}")
            errors.append({"url": url, "error": str(e)})
            results.append({"url": url, "sections": {m: f"[Error: {e}]" for m in output_modes}})

    # Build output markdown
    report_md = _build_report(results, output_modes, api_choice)

    # Write to temp file
    suffix = f"_yt_summary_{date.today().isoformat()}.md"
    with tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False, encoding="utf-8") as f:
        f.write(report_md)
        output_path = Path(f.name)

    output_data = {
        "url_count":    len(urls),
        "success_count": len(urls) - len(errors),
        "error_count":  len(errors),
        "api_used":     api_choice,
        "modes":        output_modes,
        "errors":       errors,
    }

    if push_to_kb:
        _push_to_kb(report_md, kb_category, job)

    return {"output_data": output_data, "output_file": output_path}
