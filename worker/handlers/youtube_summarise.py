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
    youtube_options dict      Optional time slicing:
                              - start_time_seconds
                              - end_time_seconds
                              - chunk_duration_seconds
                              - chunk_overlap_seconds
"""

import json
import logging
import os
import re
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from html import unescape
from datetime import date
from pathlib import Path

from cortex_engine.handoff_contract import validate_youtube_summarise_input

logger = logging.getLogger(__name__)
VAULT_LAB_NOTES_DIR = Path(os.environ.get("NEMOCLAW_LAB_NOTES_DIR", "/mnt/c/Users/paul/Documents/AI-Vault/lab-notes"))
VAULT_NOTE_FILENAME_MAX = int(os.environ.get("NEMOCLAW_NOTE_FILENAME_MAX", "56"))

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

LONG_VIDEO_LOWRES_SECONDS = int(os.environ.get("YOUTUBE_LOWRES_THRESHOLD_SECONDS", str(25 * 60)))
GEMINI_LOWRES_VALUE = "MEDIA_RESOLUTION_LOW"
TRANSCRIPT_CHAR_LIMIT = 90_000


def _is_gemini_timeout_error(exc: Exception) -> bool:
    text = str(exc or "").lower()
    return (
        "504" in text
        or "timed out" in text
        or "timeout" in text
    )


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


def _youtube_video_id(url: str) -> str:
    parsed = urllib.parse.urlparse(str(url or "").strip())
    host = parsed.netloc.lower().removeprefix("www.")

    if host == "youtu.be":
        return parsed.path.strip("/").split("/", 1)[0]
    if host in {"youtube.com", "m.youtube.com", "music.youtube.com"}:
        if parsed.path == "/watch":
            return urllib.parse.parse_qs(parsed.query).get("v", [""])[0]
        if parsed.path.startswith("/shorts/"):
            return parsed.path.split("/shorts/", 1)[1].split("/", 1)[0]
        if parsed.path.startswith("/embed/"):
            return parsed.path.split("/embed/", 1)[1].split("/", 1)[0]
    return ""


def _is_youtube_url(url: str) -> bool:
    """True only for real YouTube video URLs. A valid YouTube video id is exactly
    11 characters; this rejects non-YouTube URLs (id "") and truncated/malformed
    ids (e.g. 10 chars) before they are ever sent to Gemini as a video file_uri."""
    return len(_youtube_video_id(url)) == 11


def _parse_iso8601_duration(value: str) -> int | None:
    """Parse YouTube ISO-8601 durations such as PT1H3M12S into seconds."""
    match = re.fullmatch(
        r"P(?:\d+Y)?(?:\d+M)?(?:\d+W)?(?:\d+D)?"
        r"(?:T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?)?",
        str(value or "").strip(),
    )
    if not match:
        return None
    hours, minutes, seconds = (int(part or 0) for part in match.groups())
    return hours * 3600 + minutes * 60 + seconds


def _fetch_youtube_duration_seconds(url: str) -> int | None:
    """Best-effort duration lookup via YouTube Data API when a key is configured."""
    video_id = _youtube_video_id(url)
    api_key = os.environ.get("YOUTUBE_API_KEY") or os.environ.get("GOOGLE_API_KEY", "")
    if not video_id:
        return None

    if api_key:
        api_url = "https://www.googleapis.com/youtube/v3/videos?" + urllib.parse.urlencode(
            {"id": video_id, "part": "contentDetails", "key": api_key}
        )
        try:
            with urllib.request.urlopen(api_url, timeout=15) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            items = payload.get("items") or []
            if items:
                parsed = _parse_iso8601_duration(items[0].get("contentDetails", {}).get("duration", ""))
                if parsed is not None:
                    return parsed
        except Exception as exc:
            logger.info("YouTube Data API duration lookup failed for %s: %s", url, exc)

    return _fetch_youtube_duration_seconds_from_watch_page(video_id)


def _youtube_api_key() -> str:
    return os.environ.get("YOUTUBE_API_KEY") or os.environ.get("GOOGLE_API_KEY", "")


def _youtube_api_get(path: str, params: dict, timeout: int = 15) -> dict:
    api_key = _youtube_api_key()
    if not api_key:
        return {}
    params = dict(params)
    params["key"] = api_key
    api_url = f"https://www.googleapis.com/youtube/v3/{path}?" + urllib.parse.urlencode(params)
    try:
        with urllib.request.urlopen(api_url, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        logger.info("YouTube Data API lookup failed for %s: %s", path, exc)
        return {}


def _strip_html(value: str) -> str:
    text = re.sub(r"<br\s*/?>", "\n", str(value or ""), flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    return unescape(text).strip()


def _extract_urls(text: str) -> list[str]:
    urls = []
    seen = set()
    for match in re.finditer(r"https?://[^\s<>)\]]+", text or ""):
        url = match.group(0).rstrip(".,;:")
        if url not in seen:
            seen.add(url)
            urls.append(url)
    return urls


def _extract_js_object(text: str, marker: str) -> dict:
    start = text.find(marker)
    if start < 0:
        return {}
    start = text.find("{", start)
    if start < 0:
        return {}

    depth = 0
    in_string = False
    escape = False
    quote = ""
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == quote:
                in_string = False
            continue
        if char in {'"', "'"}:
            in_string = True
            quote = char
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:index + 1])
                except Exception as exc:
                    logger.info("YouTube watch page JSON parse failed: %s", exc)
                    return {}
    return {}


def _fetch_youtube_watch_page_context(video_id: str) -> dict:
    watch_url = f"https://www.youtube.com/watch?v={video_id}"
    req = urllib.request.Request(watch_url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            text = resp.read(4_000_000).decode("utf-8", errors="ignore")
    except Exception as exc:
        logger.info("YouTube watch page context lookup failed for %s: %s", watch_url, exc)
        return {"description": "", "tags": []}

    player = _extract_js_object(text, "ytInitialPlayerResponse")
    video_details = player.get("videoDetails") or {}
    description = (video_details.get("shortDescription") or "").strip()
    tags = [str(tag).strip() for tag in (video_details.get("keywords") or []) if str(tag).strip()]

    if not description:
        microformat = player.get("microformat", {}).get("playerMicroformatRenderer", {})
        description_obj = microformat.get("description") or {}
        description = (
            description_obj.get("simpleText")
            or "".join(run.get("text", "") for run in description_obj.get("runs", []) or [])
        ).strip()

    return {"description": description, "tags": tags}


def _fetch_youtube_extra_context(url: str) -> dict:
    """Fetch description, tags, links, playlist links, and top comments when API access exists."""
    video_id = _youtube_video_id(url)
    if not video_id:
        return {"description": "", "tags": [], "urls": [], "playlist_urls": [], "comments": []}

    video_payload = _youtube_api_get("videos", {"id": video_id, "part": "snippet"}) if _youtube_api_key() else {}
    snippet = ((video_payload.get("items") or [{}])[0].get("snippet") or {}) if video_payload else {}
    description = (snippet.get("description") or "").strip()
    tags = [str(tag).strip() for tag in (snippet.get("tags") or []) if str(tag).strip()]
    if not description and not tags:
        watch_context = _fetch_youtube_watch_page_context(video_id)
        description = watch_context.get("description", "")
        tags = watch_context.get("tags", [])
    urls = _extract_urls(description)
    playlist_urls = [
        item for item in urls
        if "youtube.com" in urllib.parse.urlparse(item).netloc.lower()
        and urllib.parse.parse_qs(urllib.parse.urlparse(item).query).get("list")
    ]

    comments = []
    comments_payload = (
        _youtube_api_get(
            "commentThreads",
            {
                "videoId": video_id,
                "part": "snippet",
                "maxResults": 3,
                "order": "relevance",
                "textFormat": "html",
            },
        )
        if _youtube_api_key()
        else {}
    )
    for item in comments_payload.get("items") or []:
        top_comment = (
            item.get("snippet", {})
            .get("topLevelComment", {})
            .get("snippet", {})
        )
        text = _strip_html(top_comment.get("textDisplay", ""))
        if not text:
            continue
        comments.append({
            "author": (top_comment.get("authorDisplayName") or "").strip(),
            "text": text,
            "like_count": top_comment.get("likeCount"),
            "published_at": (top_comment.get("publishedAt") or "").strip(),
        })

    return {
        "description": description,
        "tags": tags,
        "urls": urls,
        "playlist_urls": playlist_urls,
        "comments": comments,
    }


def _fetch_youtube_duration_seconds_from_watch_page(video_id: str) -> int | None:
    """Extract public lengthSeconds from the watch page when YouTube Data API is unavailable."""
    watch_url = f"https://www.youtube.com/watch?v={video_id}"
    req = urllib.request.Request(watch_url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            text = resp.read(2_000_000).decode("utf-8", errors="ignore")
    except Exception as exc:
        logger.info("YouTube watch page duration lookup failed for %s: %s", watch_url, exc)
        return None

    match = re.search(r'"lengthSeconds"\s*:\s*"(\d+)"', text)
    if match:
        return int(match.group(1))
    match = re.search(r'"approxDurationMs"\s*:\s*"(\d+)"', text)
    if match:
        return max(1, round(int(match.group(1)) / 1000))
    return None


def _canonical_youtube_url(url: str) -> str:
    """Return a clean watch URL for Gemini native YouTube input."""
    video_id = _youtube_video_id(url)
    if not video_id:
        return str(url or "").strip()
    return f"https://www.youtube.com/watch?v={video_id}"


def _format_seconds_label(total_seconds: int) -> str:
    total_seconds = max(0, int(total_seconds))
    mm, ss = divmod(total_seconds, 60)
    hh, mm = divmod(mm, 60)
    return f"{hh:02d}:{mm:02d}:{ss:02d}" if hh else f"{mm:02d}:{ss:02d}"


def _transcript_text_excerpt(value: str, limit: int = TRANSCRIPT_CHAR_LIMIT) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return text[:limit] + "\n\n[Transcript truncated due to length]"


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


SPONSOR_PARAGRAPH_PATTERNS = [
    re.compile(r"\bsponsor(?:ed|ship)?\b", re.IGNORECASE),
    re.compile(r"\bthis video (?:is|was) sponsored by\b", re.IGNORECASE),
    re.compile(r"\b(?:today'?s|this) sponsor\b", re.IGNORECASE),
    re.compile(r"\bbrought to you by\b", re.IGNORECASE),
    re.compile(r"\bpartner(?:ed)? with\b", re.IGNORECASE),
    re.compile(r"\bpaid promotion\b", re.IGNORECASE),
    re.compile(r"\bpromo code\b", re.IGNORECASE),
    re.compile(r"\baffiliate link\b", re.IGNORECASE),
    re.compile(r"\bdiscount code\b", re.IGNORECASE),
    re.compile(r"\buse code\b", re.IGNORECASE),
]


def _looks_like_sponsor_paragraph(paragraph: str) -> bool:
    text = " ".join((paragraph or "").split())
    if not text:
        return False
    return any(pattern.search(text) for pattern in SPONSOR_PARAGRAPH_PATTERNS)


def _remove_sponsor_paragraphs(text: str) -> str:
    """Drop sponsor/ad-read paragraphs from model-written notes and summaries."""
    if not text or not text.strip():
        return text

    blocks = re.split(r"\n\s*\n", text.strip())
    kept_blocks = [block for block in blocks if not _looks_like_sponsor_paragraph(block)]
    cleaned = "\n\n".join(kept_blocks).strip()
    return cleaned or text.strip()


def _sanitize_sections(sections: dict) -> dict:
    cleaned = {}
    for mode, content in (sections or {}).items():
        if mode in {"summary", "meeting_notes", "action_items"}:
            cleaned[mode] = _remove_sponsor_paragraphs(content or "")
        else:
            cleaned[mode] = content
    return cleaned


# ── Gemini path ──

def _gemini_response_text(payload: dict) -> str:
    parts = []
    for candidate in payload.get("candidates") or []:
        for part in candidate.get("content", {}).get("parts", []) or []:
            text = part.get("text", "")
            if text:
                parts.append(text)
    return "\n".join(parts).strip()


def _generate_gemini_rest(
    model_id: str,
    prompt: str,
    youtube_url: str,
    *,
    media_resolution: str = "",
    timeout: int = 120,
) -> str:
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY not set in config.env")

    body = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {"file_data": {"file_uri": youtube_url}},
            ],
        }],
    }
    if media_resolution:
        body["generation_config"] = {"media_resolution": media_resolution}

    endpoint = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model_id}:generateContent?key={urllib.parse.quote(api_key)}"
    )
    req = urllib.request.Request(
        endpoint,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Gemini REST error {exc.code}: {detail[:500]}") from exc

    text = _gemini_response_text(payload)
    if not text:
        raise RuntimeError(f"Gemini REST returned no text: {json.dumps(payload)[:500]}")
    return text


def _summarise_gemini(url: str, model_name: str, output_modes: list[str], language: str = "") -> dict:
    """Use Gemini's native YouTube URL understanding."""
    genai = _gemini_client()

    model_id = "gemini-2.5-pro" if model_name == "gemini-pro" else "gemini-2.5-flash"
    model = genai.GenerativeModel(model_id)
    gemini_url = _canonical_youtube_url(url)
    duration_seconds = _fetch_youtube_duration_seconds(gemini_url)
    use_lowres = bool(duration_seconds and duration_seconds >= LONG_VIDEO_LOWRES_SECONDS)
    if use_lowres:
        logger.info(
            "Using Gemini low media resolution for long YouTube video duration=%ss url=%s",
            duration_seconds,
            gemini_url,
        )

    lang_instruction = f"\n\nIMPORTANT: Write your entire response in {language}." if language else ""

    sections = {}
    for mode in output_modes:
        prompt = MODE_PROMPTS.get(mode, f"Provide {mode} for this video.")
        full_prompt = (
            f"{prompt}{lang_instruction}\n\nVideo: {gemini_url}"
        )
        native_error = None
        try:
            if use_lowres:
                sections[mode] = _generate_gemini_rest(
                    model_id,
                    full_prompt,
                    gemini_url,
                    media_resolution=GEMINI_LOWRES_VALUE,
                    timeout=180,
                )
            else:
                response = model.generate_content([
                    {"text": full_prompt},
                    {"file_data": {"mime_type": "video/youtube", "file_uri": gemini_url}},
                ], request_options={"timeout": 90})
                sections[mode] = response.text.strip()
        except Exception as e:
            native_error = e
            logger.warning(f"Gemini native YouTube mode '{mode}' failed for {gemini_url}: {e}")
            if not use_lowres and _is_gemini_timeout_error(e):
                try:
                    logger.info(
                        "Retrying Gemini low media resolution after timeout for url=%s mode=%s",
                        gemini_url,
                        mode,
                    )
                    sections[mode] = _generate_gemini_rest(
                        model_id,
                        full_prompt,
                        gemini_url,
                        media_resolution=GEMINI_LOWRES_VALUE,
                        timeout=180,
                    )
                    logger.info(
                        "Gemini low media resolution retry succeeded for mode '%s' url=%s",
                        mode,
                        gemini_url,
                    )
                    continue
                except Exception as lowres_error:
                    logger.warning(
                        "Gemini low media resolution retry failed for %s: %s",
                        gemini_url,
                        lowres_error,
                    )
            try:
                transcript = _extract_transcript(gemini_url)
                transcript_excerpt = transcript[:90_000]
                if len(transcript) > 90_000:
                    transcript_excerpt += "\n\n[Transcript truncated due to length]"
                fallback_prompt = (
                    f"Here is the transcript of a YouTube video ({gemini_url}):\n\n"
                    f"---\n{transcript_excerpt}\n---\n\n"
                    f"Task: {prompt}{lang_instruction}"
                )
                response = model.generate_content(fallback_prompt, request_options={"timeout": 90})
                sections[mode] = response.text.strip()
                logger.info(f"Gemini transcript fallback succeeded for mode '{mode}' url={gemini_url}")
            except Exception as fallback_error:
                logger.warning(f"Gemini transcript fallback failed for {gemini_url}: {fallback_error}")
                sections[mode] = f"[Error generating {mode}: {native_error}]"

    return sections


def _summarise_gemini_transcript(transcript: str, context_label: str, model_name: str, output_modes: list[str], language: str = "") -> dict:
    genai = _gemini_client()
    model_id = "gemini-2.5-pro" if model_name == "gemini-pro" else "gemini-2.5-flash"
    model = genai.GenerativeModel(model_id)
    transcript_excerpt = _transcript_text_excerpt(transcript)
    lang_instruction = f"\n\nIMPORTANT: Write your entire response in {language}." if language else ""

    sections = {}
    for mode in output_modes:
        prompt = MODE_PROMPTS.get(mode, f"Provide {mode} for this video.")
        response = model.generate_content(
            (
                f"Here is the transcript of a YouTube video segment ({context_label}):\n\n"
                f"---\n{transcript_excerpt}\n---\n\n"
                f"Task: {prompt}{lang_instruction}"
            ),
            request_options={"timeout": 90},
        )
        sections[mode] = (response.text or "").strip()
    return sections


# ── Transcript extraction (for Claude path) ──

def _extract_transcript(url: str) -> str:
    """Extract transcript text using youtube-transcript-api."""
    entries = _extract_transcript_entries(url)
    return _format_transcript_entries(entries)


def _extract_transcript_entries(url: str) -> list[dict]:
    """Extract transcript entries with timing using youtube-transcript-api."""
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

    entries = []
    for entry in transcript_list:
        entries.append(
            {
                "start": float(entry.get("start") or 0.0),
                "duration": float(entry.get("duration") or 0.0),
                "text": str(entry.get("text") or "").strip(),
            }
        )
    return entries


def _format_transcript_entries(entries: list[dict]) -> str:
    lines = []
    for entry in entries:
        text = str(entry.get("text") or "").strip()
        if not text:
            continue
        lines.append(f"[{_format_seconds_label(int(entry.get('start') or 0))}] {text}")
    return "\n".join(lines)


def _slice_transcript_entries(
    entries: list[dict],
    start_time_seconds: int = 0,
    end_time_seconds: int = 0,
) -> list[dict]:
    if not entries:
        return []

    start_bound = max(0.0, float(start_time_seconds or 0))
    end_bound = float(end_time_seconds or 0)
    sliced = []
    for entry in entries:
        entry_start = float(entry.get("start") or 0.0)
        entry_duration = max(0.0, float(entry.get("duration") or 0.0))
        entry_end = entry_start + entry_duration
        if entry_end <= start_bound:
            continue
        if end_bound and entry_start >= end_bound:
            continue
        sliced.append(entry)
    return sliced


def _chunk_transcript_entries(
    entries: list[dict],
    chunk_duration_seconds: int,
    chunk_overlap_seconds: int = 0,
    start_time_seconds: int = 0,
    end_time_seconds: int = 0,
) -> list[dict]:
    scoped_entries = _slice_transcript_entries(entries, start_time_seconds, end_time_seconds)
    if not scoped_entries:
        return []

    if chunk_duration_seconds <= 0:
        actual_start = max(0, int(start_time_seconds or scoped_entries[0].get("start") or 0))
        actual_end = int(end_time_seconds or (scoped_entries[-1].get("start") or 0) + (scoped_entries[-1].get("duration") or 0))
        return [{
            "index": 1,
            "start_seconds": actual_start,
            "end_seconds": max(actual_start, actual_end),
            "entries": scoped_entries,
        }]

    chunks = []
    window_start = max(0, int(start_time_seconds or scoped_entries[0].get("start") or 0))
    natural_end = int((scoped_entries[-1].get("start") or 0) + (scoped_entries[-1].get("duration") or 0))
    final_end = int(end_time_seconds or natural_end)
    step = max(1, chunk_duration_seconds - max(0, chunk_overlap_seconds))
    index = 1

    while window_start < final_end:
        window_end = min(final_end, window_start + chunk_duration_seconds)
        chunk_entries = _slice_transcript_entries(scoped_entries, window_start, window_end)
        if chunk_entries:
            chunks.append(
                {
                    "index": index,
                    "start_seconds": window_start,
                    "end_seconds": window_end,
                    "entries": chunk_entries,
                }
            )
            index += 1
        window_start += step

    return chunks


# ── Claude path ──

def _summarise_claude(transcript: str, url: str, model_name: str, output_modes: list[str], language: str = "") -> dict:
    """Summarise using Claude with an extracted transcript."""
    client = _anthropic_client()

    model_id = (
        "claude-sonnet-4-6" if model_name == "claude-sonnet"
        else "claude-haiku-4-5-20251001"
    )

    # Truncate long transcripts (100k chars ≈ ~75k tokens, well within context)
    transcript_excerpt = _transcript_text_excerpt(transcript)

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


def _summarise_transcript_chunks(
    transcript_entries: list[dict],
    url: str,
    api_choice: str,
    output_modes: list[str],
    language: str,
    youtube_options: dict,
) -> list[dict]:
    chunks = _chunk_transcript_entries(
        transcript_entries,
        youtube_options.get("chunk_duration_seconds", 0),
        youtube_options.get("chunk_overlap_seconds", 0),
        youtube_options.get("start_time_seconds", 0),
        youtube_options.get("end_time_seconds", 0),
    )
    if not chunks:
        return []

    chunk_results = []
    for chunk in chunks:
        transcript_text = _format_transcript_entries(chunk["entries"])
        context_label = (
            f"{url} from {_format_seconds_label(chunk['start_seconds'])} "
            f"to {_format_seconds_label(chunk['end_seconds'])}"
        )
        if api_choice.startswith("gemini"):
            sections = _summarise_gemini_transcript(transcript_text, context_label, api_choice, output_modes, language)
        else:
            sections = _summarise_claude(transcript_text, context_label, api_choice, output_modes, language)
        chunk_results.append(
            {
                "index": chunk["index"],
                "start_seconds": chunk["start_seconds"],
                "end_seconds": chunk["end_seconds"],
                "sections": _sanitize_sections(sections),
                "transcript_entry_count": len(chunk["entries"]),
            }
        )
    return chunk_results


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
        chunk_results = result.get("chunk_results") or []
        lines.append(f"## {report_title}")
        if video_title:
            lines.append(f"**Clip title:** {video_title}")
        if author:
            lines.append(f"**Author / channel:** {author}")
        lines.append(f"**URL:** {url}")
        if chunk_results:
            lines.append(f"**Processed as:** {len(chunk_results)} time chunk(s)")
        lines.append("")

        if chunk_results:
            for chunk in chunk_results:
                lines.append(
                    "### Chunk "
                    f"{chunk.get('index', '?')} "
                    f"({_format_seconds_label(chunk.get('start_seconds', 0))}"
                    f" - {_format_seconds_label(chunk.get('end_seconds', 0))})"
                )
                lines.append("")
                sections = chunk.get("sections", {})
                for mode in output_modes:
                    label = MODE_LABELS.get(mode, mode)
                    content = sections.get(mode, "[Not generated]")
                    lines.append(f"#### {label}")
                    lines.append(content)
                    lines.append("")
        else:
            sections = result.get("sections", {})
            for mode in output_modes:
                label = MODE_LABELS.get(mode, mode)
                content = sections.get(mode, "[Not generated]")
                lines.append(f"### {label}")
                lines.append(content)
                lines.append("")

        extra_context = result.get("extra_context") or {}
        description = (extra_context.get("description") or "").strip()
        tags = extra_context.get("tags") or []
        urls = extra_context.get("urls") or []
        playlist_urls = extra_context.get("playlist_urls") or []
        comments = extra_context.get("comments") or []

        if description or tags or urls:
            lines.append("### Video Description & Links")
            if description:
                lines.append("#### Description")
                lines.append(description)
                lines.append("")
            if tags:
                lines.append("#### Tags")
                lines.append(", ".join(f"`{tag}`" for tag in tags))
                lines.append("")
            if urls:
                lines.append("#### URLs")
                for item in urls:
                    lines.append(f"- {item}")
                lines.append("")
            if playlist_urls:
                lines.append("#### YouTube Playlist URLs")
                for item in playlist_urls:
                    lines.append(f"- {item}")
                lines.append("")

        if comments:
            lines.append("### Top Comments")
            for index, comment in enumerate(comments[:3], 1):
                author = comment.get("author") or "YouTube commenter"
                like_count = comment.get("like_count")
                like_suffix = f" · {like_count} like(s)" if like_count is not None else ""
                published = comment.get("published_at")
                published_suffix = f" · {published}" if published else ""
                lines.append(f"{index}. **{author}**{like_suffix}{published_suffix}")
                lines.append("")
                lines.append(comment.get("text", "").strip())
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


def _safe_filename(value: str) -> str:
    value = re.sub(r"[^\w\s.-]+", "", str(value or ""), flags=re.UNICODE).strip()
    value = re.sub(r"\s+", "-", value)
    return value[:VAULT_NOTE_FILENAME_MAX].strip(".-") or "youtube-summary"


def _has_successful_content(results: list[dict]) -> bool:
    """True if any result produced a real (non-error) section. Used to gate KB /
    vault publishing so failed or skipped jobs never pollute the knowledge base."""
    def _ok(sections: dict) -> bool:
        return any(
            v and not str(v).strip().startswith("[Error")
            for v in (sections or {}).values()
        )
    for r in results:
        if _ok(r.get("sections")):
            return True
        for chunk in (r.get("chunk_results") or []):
            if _ok(chunk.get("sections")):
                return True
    return False


def _write_vault_lab_note(content: str, title: str, job: dict) -> None:
    """Persist email-origin YouTube summaries where the wiki ingest can see them."""
    try:
        today = date.today().isoformat()
        VAULT_LAB_NOTES_DIR.mkdir(parents=True, exist_ok=True)
        path = VAULT_LAB_NOTES_DIR / f"{today}-{_safe_filename(title)}.md"
        if path.exists():
            logger.info("vault lab note already exists: %s", path)
            return
        path.write_text(content.rstrip() + "\n", encoding="utf-8")
        logger.info("wrote vault lab note for job %s: %s", job.get("id", "job"), path)
    except Exception as e:
        logger.warning("write vault lab note failed: %s", e)


# ── Main handler ──

def handle(input_path, input_data: dict, job: dict):
    """
    Entry point called by the queue worker.

    Returns:
        {"output_data": dict, "output_file": Path | None}
    """
    input_data = validate_youtube_summarise_input(input_data)
    urls = input_data.get("urls", [])
    api_choice = input_data.get("api_choice", "gemini-flash")
    output_modes = input_data.get("output_modes", ["summary"])
    push_to_kb = input_data.get("push_to_kb", False)
    kb_category = input_data.get("kb_category", "")
    language = input_data.get("language", "")
    youtube_options = input_data.get("youtube_options", {})

    if not urls:
        raise ValueError("No YouTube URLs provided in input_data")

    use_gemini = api_choice.startswith("gemini")
    use_chunked_transcript = any(
        youtube_options.get(key, 0) > 0
        for key in ("start_time_seconds", "end_time_seconds", "chunk_duration_seconds")
    )
    results = []
    errors  = []

    for url in urls:
        if not _is_youtube_url(url):
            logger.warning("Skipping non-YouTube URL (not sent to Gemini): %s", url)
            errors.append({"url": url, "error": "Not a valid YouTube URL — skipped"})
            continue
        logger.info(f"Processing: {url} via {api_choice}")
        metadata = _fetch_youtube_metadata(url)
        extra_context = _fetch_youtube_extra_context(url)
        try:
            chunk_results = []
            if use_chunked_transcript:
                transcript_entries = _extract_transcript_entries(url)
                chunk_results = _summarise_transcript_chunks(
                    transcript_entries,
                    url,
                    api_choice,
                    output_modes,
                    language,
                    youtube_options,
                )
                if not chunk_results:
                    raise RuntimeError("No transcript content found in the requested time range")
                sections = {}
            elif use_gemini:
                sections = _summarise_gemini(url, api_choice, output_modes, language)
            else:
                transcript = _extract_transcript(url)
                sections = _summarise_claude(transcript, url, api_choice, output_modes, language)
            sections = _sanitize_sections(sections)
            title_sections = chunk_results[0]["sections"] if chunk_results else sections
            report_title = _generate_report_title(url, api_choice, metadata, title_sections, len(results) + 1, language)
            results.append({
                "url": url,
                "sections": sections,
                "chunk_results": chunk_results,
                "video_title": metadata.get("video_title", ""),
                "author": metadata.get("author", ""),
                "author_url": metadata.get("author_url", ""),
                "report_title": report_title,
                "extra_context": extra_context,
            })
        except Exception as e:
            logger.error(f"Failed to process {url}: {e}")
            errors.append({"url": url, "error": str(e)})
            results.append({
                "url": url,
                "sections": {m: f"[Error: {e}]" for m in output_modes},
                "chunk_results": [],
                "video_title": metadata.get("video_title", ""),
                "author": metadata.get("author", ""),
                "author_url": metadata.get("author_url", ""),
                "report_title": _fallback_report_title(metadata, len(results) + 1),
                "extra_context": extra_context,
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
        "chunked": use_chunked_transcript,
        "chunk_options": youtube_options,
        "chunk_count": sum(len(item.get("chunk_results") or []) for item in results),
        "report_title": results[0].get("report_title", "YouTube Summary Report") if len(results) == 1 else f"YouTube Summary Report - {len(results)} Videos",
        "videos": [
            {
                "url": item.get("url", ""),
                "clip_title": item.get("video_title", ""),
                "author": item.get("author", ""),
                "report_title": item.get("report_title", ""),
                "chunk_count": len(item.get("chunk_results") or []),
                "description_available": bool((item.get("extra_context") or {}).get("description")),
                "top_comment_count": len((item.get("extra_context") or {}).get("comments") or []),
            }
            for item in results
        ],
        "errors": errors,
    }

    publishable = _has_successful_content(results)
    if not publishable:
        logger.warning("No successful content for job %s — skipping KB/vault publish (errors=%s)",
                       job.get("id", "job"), len(errors))
    if push_to_kb and publishable:
        _push_to_kb(report_md, kb_category, job)
    if publishable and str(input_data.get("source_system", "")).lower() == "email":
        _write_vault_lab_note(report_md, output_data["report_title"], job)

    return {"output_data": output_data, "output_file": output_path}
