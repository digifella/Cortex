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
    language      str         Output language (optional, e.g. "Danish", "French"). Defaults to English.
"""

import json
import logging
import os
import tempfile
import urllib.parse
import urllib.request
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

MODEL_DETAILS = {
    "gemini-flash": {"provider": "Google", "label": "Gemini 2.5 Flash"},
    "gemini-pro": {"provider": "Google", "label": "Gemini 2.5 Pro"},
    "claude-haiku": {"provider": "Anthropic", "label": "Claude Haiku"},
    "claude-sonnet": {"provider": "Anthropic", "label": "Claude Sonnet"},
}


def _model_details(api_choice: str) -> dict:
    details = MODEL_DETAILS.get(api_choice)
    if details:
        return details
    return {"provider": "", "label": api_choice}


def _fetch_youtube_metadata(url: str) -> dict:
    """Best-effort public metadata lookup via YouTube oEmbed."""
    oembed_url = "https://www.youtube.com/oembed?" + urllib.parse.urlencode(
        {"url": url, "format": "json"}
    )
    try:
        with urllib.request.urlopen(oembed_url, timeout=15) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        return {
            "video_title": (payload.get("title") or "").strip(),
            "author": (payload.get("author_name") or "").strip(),
            "author_url": (payload.get("author_url") or "").strip(),
            "provider": (payload.get("provider_name") or "YouTube").strip(),
        }
    except Exception as exc:
        logger.info("YouTube metadata lookup failed for %s: %s", url, exc)
        return {
            "video_title": "",
            "author": "",
            "author_url": "",
            "provider": "YouTube",
        }


def _title_context(url: str, metadata: dict, sections: dict) -> str:
    parts = []
    if metadata.get("video_title"):
        parts.append(f"Video title: {metadata['video_title']}")
    if metadata.get("author"):
        parts.append(f"Channel/author: {metadata['author']}")
    parts.append(f"URL: {url}")
    for key in ("summary", "meeting_notes", "action_items", "timestamps", "transcript"):
        value = (sections.get(key) or "").strip()
        if value:
            parts.append(f"{key.capitalize()} excerpt:\n{value[:1600]}")
            break
    return "\n\n".join(parts)


def _fallback_report_title(metadata: dict, index: int) -> str:
    return (
        metadata.get("video_title")
        or metadata.get("author")
        or f"YouTube Summary Report {index}"
    )


def _generate_report_title_gemini(url: str, model_name: str, metadata: dict, sections: dict, language: str = "") -> str:
    genai = _gemini_client()
    model_id = "gemini-2.5-pro" if model_name == "gemini-pro" else "gemini-2.5-flash"
    model = genai.GenerativeModel(model_id)
    lang_note = f" Write the title in {language}." if language else ""
    prompt = (
        "Create a concise, professional title for a written summary report of this YouTube clip. "
        "Use the actual subject matter, not generic wording. Max 12 words. "
        f"Return title text only. No quotes. No markdown.{lang_note}\n\n"
        + _title_context(url, metadata, sections)
    )
    response = model.generate_content(prompt)
    return (response.text or "").strip()


def _generate_report_title_claude(url: str, model_name: str, metadata: dict, sections: dict, language: str = "") -> str:
    client = _anthropic_client()
    model_id = (
        "claude-sonnet-4-6" if model_name == "claude-sonnet"
        else "claude-haiku-4-5-20251001"
    )
    lang_note = f" Write the title in {language}." if language else ""
    response = client.messages.create(
        model=model_id,
        max_tokens=64,
        messages=[{
            "role": "user",
            "content": (
                "Create a concise, professional title for a written summary report of this YouTube clip. "
                "Use the actual subject matter, not generic wording. Max 12 words. "
                f"Return title text only. No quotes. No markdown.{lang_note}\n\n"
                + _title_context(url, metadata, sections)
            ),
        }],
    )
    return response.content[0].text.strip()


def _generate_report_title(url: str, api_choice: str, metadata: dict, sections: dict, index: int, language: str = "") -> str:
    try:
        if api_choice.startswith("gemini"):
            title = _generate_report_title_gemini(url, api_choice, metadata, sections, language)
        else:
            title = _generate_report_title_claude(url, api_choice, metadata, sections, language)
        cleaned = " ".join((title or "").split()).strip().strip("#").strip()
        return cleaned or _fallback_report_title(metadata, index)
    except Exception as exc:
        logger.info("AI report title generation failed for %s: %s", url, exc)
        return _fallback_report_title(metadata, index)


# ── Gemini path ──

def _summarise_gemini(url: str, model_name: str, output_modes: list[str], language: str = "") -> dict:
    """Use Gemini's native YouTube URL understanding."""
    genai = _gemini_client()

    model_id = "gemini-2.5-pro" if model_name == "gemini-pro" else "gemini-2.5-flash"
    model = genai.GenerativeModel(model_id)

    lang_instruction = f"\n\nIMPORTANT: Write your entire response in {language}." if language else ""

    sections = {}
    for mode in output_modes:
        prompt = MODE_PROMPTS.get(mode, f"Provide {mode} for this video.")
        full_prompt = (
            f"{prompt}{lang_instruction}\n\nVideo: {url}"
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

def _summarise_claude(transcript: str, url: str, model_name: str, output_modes: list[str], language: str = "") -> dict:
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

    lang_instruction = f"\n\nIMPORTANT: Write your entire response in {language}." if language else ""

    sections = {}
    for mode in output_modes:
        prompt = MODE_PROMPTS.get(mode, f"Provide {mode} for this video.")
        user_message = (
            f"Here is the transcript of a YouTube video ({url}):\n\n"
            f"---\n{transcript_excerpt}\n---\n\n"
            f"Task: {prompt}{lang_instruction}"
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


def _build_report(results: list[dict], output_modes: list[str], api_choice: str, language: str = "") -> str:
    today = date.today().isoformat()
    mode_labels = ", ".join(MODE_LABELS.get(m, m) for m in output_modes)
    model_info = _model_details(api_choice)
    api_label = model_info["label"]
    if len(results) == 1:
        report_title = results[0].get("report_title") or results[0].get("video_title") or "YouTube Summary Report"
    else:
        report_title = f"YouTube Summary Report - {len(results)} Videos"

    lines = [
        "---",
        f"title: {report_title}",
        f"date: {today}",
        "source_type: youtube_summary",
        f"provider: {model_info['provider']}",
        f"api: {api_label}",
        f"modes: {mode_labels}",
    ]
    if language:
        lines.append(f"language: {language}")
    lines += [
        "---",
        "",
        f"# {report_title}",
        f"Generated: {today} · API: {api_label} · Modes: {mode_labels}",
        "",
    ]

    for i, result in enumerate(results, 1):
        lines.append(f"---\n")
        url = result.get("url", "")
        report_title = result.get("report_title") or result.get("video_title") or f"Video {i}"
        video_title = result.get("video_title", "")
        author = result.get("author", "")
        lines.append(f"## {report_title}")
        if video_title:
            lines.append(f"**Clip title:** {video_title}")
        if author:
            lines.append(f"**Author / channel:** {author}")
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
    language     = input_data.get("language", "")

    if not urls:
        raise ValueError("No YouTube URLs provided in input_data")

    use_gemini = api_choice.startswith("gemini")
    results = []
    errors  = []

    for url in urls:
        logger.info(f"Processing: {url} via {api_choice}")
        metadata = _fetch_youtube_metadata(url)
        try:
            if use_gemini:
                sections = _summarise_gemini(url, api_choice, output_modes, language)
            else:
                transcript = _extract_transcript(url)
                sections   = _summarise_claude(transcript, url, api_choice, output_modes, language)
            report_title = _generate_report_title(url, api_choice, metadata, sections, len(results) + 1, language)
            results.append({
                "url": url,
                "sections": sections,
                "video_title": metadata.get("video_title", ""),
                "author": metadata.get("author", ""),
                "author_url": metadata.get("author_url", ""),
                "report_title": report_title,
            })
        except Exception as e:
            logger.error(f"Failed to process {url}: {e}")
            errors.append({"url": url, "error": str(e)})
            results.append({
                "url": url,
                "sections": {m: f"[Error: {e}]" for m in output_modes},
                "video_title": metadata.get("video_title", ""),
                "author": metadata.get("author", ""),
                "author_url": metadata.get("author_url", ""),
                "report_title": _fallback_report_title(metadata, len(results) + 1),
            })

    # Build output markdown
    report_md = _build_report(results, output_modes, api_choice, language)

    # Write to temp file
    suffix = f"_yt_summary_{date.today().isoformat()}.md"
    with tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False, encoding="utf-8") as f:
        f.write(report_md)
        output_path = Path(f.name)

    model_info = _model_details(api_choice)
    output_data = {
        "url_count": len(urls),
        "video_count": len(urls),
        "videos_processed": len(urls) - len(errors),
        "success_count": len(urls) - len(errors),
        "error_count": len(errors),
        "api_used": api_choice,
        "provider": model_info["provider"],
        "model": model_info["label"],
        "modes": output_modes,
        "language": language or "English",
        "report_title": results[0].get("report_title", "YouTube Summary Report") if len(results) == 1 else f"YouTube Summary Report - {len(results)} Videos",
        "videos": [
            {
                "url": item.get("url", ""),
                "clip_title": item.get("video_title", ""),
                "author": item.get("author", ""),
                "report_title": item.get("report_title", ""),
            }
            for item in results
        ],
        "errors": errors,
    }

    if push_to_kb:
        _push_to_kb(report_md, kb_category, job)

    return {"output_data": output_data, "output_file": output_path}
