"""
Signal Episode Audio Handler
Generates podcast audio for a SIGNal episode by calling ElevenLabs TTS
for each script line, stitching the PCM chunks with silence gaps, and
outputting an MP3 (via ffmpeg) or WAV fallback.

input_data schema:
    episode_id          int
    episode_slug        str
    script_json         list[{speaker: str, text: str}]
    paul_voice_id       str   — ElevenLabs voice ID for Paul
    nova_voice_id       str   — ElevenLabs voice ID for Nova
    elevenlabs_api_key  str
"""

import json
import logging
import shutil
import struct
import subprocess
import tempfile
import time
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

# ElevenLabs TTS endpoint (PCM output, no header)
EL_TTS_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
EL_PARAMS   = "?output_format=pcm_44100"

SAMPLE_RATE   = 44100
CHANNELS      = 1
SAMPLE_WIDTH  = 2          # 16-bit
SILENCE_SECS  = 0.6        # gap between speaker turns

# Voice settings
VOICE_SETTINGS = {
    "stability":        0.5,
    "similarity_boost": 0.75,
    "style":            0.0,
    "use_speaker_boost": True,
}


def _silence_bytes(seconds: float) -> bytes:
    """Generate silence as raw PCM bytes."""
    n_samples = int(SAMPLE_RATE * seconds)
    return b'\x00' * (n_samples * CHANNELS * SAMPLE_WIDTH)


def _tts_line(text: str, voice_id: str, api_key: str, retries: int = 3) -> bytes:
    """Call ElevenLabs TTS, return raw 16-bit mono PCM at 44100 Hz."""
    url     = EL_TTS_URL.format(voice_id=voice_id) + EL_PARAMS
    headers = {
        "xi-api-key":   api_key,
        "Content-Type": "application/json",
        "Accept":       "audio/pcm",
    }
    payload = {
        "text":           text,
        "model_id":       "eleven_multilingual_v2",
        "voice_settings": VOICE_SETTINGS,
    }

    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
            if resp.status_code == 200:
                return resp.content
            elif resp.status_code == 429:
                wait = 10 * attempt
                logger.warning("ElevenLabs rate limit — waiting %ds (attempt %d)", wait, attempt)
                time.sleep(wait)
            else:
                logger.error("ElevenLabs error %d: %s", resp.status_code, resp.text[:300])
                if attempt == retries:
                    raise RuntimeError(f"ElevenLabs returned {resp.status_code}: {resp.text[:200]}")
                time.sleep(5)
        except requests.RequestException as e:
            if attempt == retries:
                raise
            logger.warning("ElevenLabs request error (attempt %d): %s", attempt, e)
            time.sleep(5)

    return b''


def _build_wav_header(data_len: int) -> bytes:
    """Build a RIFF/WAVE header for 16-bit mono PCM at 44100 Hz."""
    byte_rate   = SAMPLE_RATE * CHANNELS * SAMPLE_WIDTH
    block_align = CHANNELS * SAMPLE_WIDTH
    return struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF',
        36 + data_len,      # file size - 8
        b'WAVE',
        b'fmt ',
        16,                 # fmt chunk size
        1,                  # PCM format
        CHANNELS,
        SAMPLE_RATE,
        byte_rate,
        block_align,
        16,                 # bits per sample
        b'data',
        data_len,
    )


def _pcm_to_mp3(wav_path: Path, mp3_path: Path) -> bool:
    """Convert WAV to MP3 via ffmpeg. Returns True on success."""
    if not shutil.which("ffmpeg"):
        logger.info("ffmpeg not found — falling back to WAV output")
        return False
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", str(wav_path), "-b:a", "128k", str(mp3_path)],
            capture_output=True, timeout=120,
        )
        if result.returncode == 0:
            return True
        logger.error("ffmpeg error: %s", result.stderr.decode()[-500:])
        return False
    except Exception as e:
        logger.error("ffmpeg exception: %s", e)
        return False


def handle(input_path, input_data: dict, job: dict) -> dict:
    """
    Main handler. Returns {'output_data': dict, 'output_file': Path | None}.
    """
    episode_id   = input_data.get("episode_id")
    episode_slug = input_data.get("episode_slug", f"ep{episode_id}")
    script       = input_data.get("script_json", [])
    paul_voice   = input_data.get("paul_voice_id", "")
    nova_voice   = input_data.get("nova_voice_id", "")
    api_key      = input_data.get("elevenlabs_api_key", "")

    if not script:
        raise ValueError("script_json is empty")
    if not paul_voice or not nova_voice:
        raise ValueError("paul_voice_id and nova_voice_id are required")
    if not api_key:
        raise ValueError("elevenlabs_api_key is required")

    logger.info("Generating audio for episode %s (%d lines)", episode_slug, len(script))

    pcm_chunks  = []
    prev_speaker = None

    for i, line in enumerate(script):
        speaker = line.get("speaker", "paul").lower()
        text    = line.get("text", "").strip()
        if not text:
            continue

        voice_id = nova_voice if speaker == "nova" else paul_voice

        logger.info("  Line %d/%d [%s]: %d chars", i + 1, len(script), speaker, len(text))

        # Add silence gap on speaker change (not before first line)
        if prev_speaker is not None and speaker != prev_speaker:
            pcm_chunks.append(_silence_bytes(SILENCE_SECS))
        elif prev_speaker is not None:
            # Brief intra-speaker gap (0.15s)
            pcm_chunks.append(_silence_bytes(0.15))

        pcm = _tts_line(text, voice_id, api_key)
        if pcm:
            pcm_chunks.append(pcm)
        else:
            logger.warning("  Empty PCM for line %d — skipping", i + 1)

        prev_speaker = speaker

    if not pcm_chunks:
        raise RuntimeError("No audio generated — all TTS calls failed")

    raw_pcm   = b''.join(pcm_chunks)
    data_len  = len(raw_pcm)
    duration_secs = data_len / (SAMPLE_RATE * CHANNELS * SAMPLE_WIDTH)
    mins  = int(duration_secs // 60)
    secs  = int(duration_secs % 60)
    duration_str = f"{mins}:{secs:02d}"

    logger.info("PCM assembled: %.1f MB, duration: %s", data_len / 1_048_576, duration_str)

    # Write to temp WAV
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        wav_path = Path(tf.name)
        tf.write(_build_wav_header(data_len))
        tf.write(raw_pcm)

    # Attempt MP3 conversion
    mp3_path  = wav_path.with_suffix(".mp3")
    use_mp3   = _pcm_to_mp3(wav_path, mp3_path)

    if use_mp3:
        output_path = mp3_path
        wav_path.unlink(missing_ok=True)
        logger.info("MP3 output: %s (%.2f MB)", output_path.name, output_path.stat().st_size / 1_048_576)
    else:
        output_path = wav_path
        logger.info("WAV output: %s (%.2f MB)", output_path.name, output_path.stat().st_size / 1_048_576)

    output_data = {
        "episode_id":       episode_id,
        "episode_slug":     episode_slug,
        "duration":         duration_str,
        "duration_seconds": round(duration_secs, 1),
        "lines_count":      len(script),
        "format":           "mp3" if use_mp3 else "wav",
    }

    return {
        "output_data": output_data,
        "output_file": output_path,
    }
