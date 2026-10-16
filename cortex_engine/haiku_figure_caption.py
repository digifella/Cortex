"""Caption a document figure with local Qwen vision, falling back to Haiku.

Distinct from textifier.py's photo-captioning prompt,
which is written for photographs and explicitly tells the model to omit
logos/icons — this prompt is written for the opposite case: we've already
filtered out small decorative graphics by bounding-box area before this
function is ever called (see enhanced_ingest_cortex.py), so every image
reaching here is presumed to be genuine document content worth describing
in full, including any text, axis labels, or data visible in it.
"""
import base64
import os

from .lmstudio_vision import describe_image as describe_with_loaded_qwen

MODEL = "claude-haiku-5-5"
_last_haiku_failure = ""


def last_haiku_failure() -> str:
    return _last_haiku_failure


class BillingExhaustedError(RuntimeError):
    """Anthropic rejected a caption request for lack of credit."""

_PROMPT = (
    "This image is a figure extracted from a business/consulting document "
    "(a chart, diagram, table screenshot, matrix, or similar). Describe it "
    "for a search index: what kind of figure it is, any axis labels or "
    "headings visible, and the key data or relationships it shows. "
    "Transcribe any short text labels verbatim if legible. "
    "Write 2-4 plain sentences. Do not use markdown, headings, or bullet points."
)


def _load_api_key() -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if api_key:
        return api_key
    env_path = os.path.expanduser("~/cortex_suite/.env")
    if not os.path.exists(env_path):
        return ""
    for line in open(env_path, encoding="utf-8").read().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key == "ANTHROPIC_API_KEY" and val:
            return val
    return ""


def caption_figure_with_qwen(image_bytes: bytes, context_hint: str = "") -> str:
    """Use the already-loaded LM Studio Qwen VLM; never load a model on demand."""
    prompt = _PROMPT + (" " + context_hint.strip() if context_hint else "")
    return describe_with_loaded_qwen(image_bytes, mime_type="image/png", prompt=prompt)


def caption_figure_with_haiku(image_bytes: bytes, context_hint: str = "") -> str:
    global _last_haiku_failure
    _last_haiku_failure = ""
    api_key = _load_api_key()
    if not api_key:
        _last_haiku_failure = "missing_api_key"
        return ""
    try:
        import anthropic
    except ImportError:
        _last_haiku_failure = "anthropic_package_missing"
        return ""

    prompt = _PROMPT
    if context_hint:
        prompt += " " + context_hint.strip()

    try:
        client = anthropic.Anthropic(api_key=api_key)
        encoded = base64.b64encode(image_bytes).decode("utf-8")
        response = client.messages.create(
            model=MODEL,
            max_tokens=450,
            output_config={"effort": "low"},
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64",
                                                  "media_type": "image/png",
                                                  "data": encoded}},
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        for block in response.content or []:
            if hasattr(block, "text") and block.text:
                return block.text.strip()
        _last_haiku_failure = "empty_model_response"
        return ""
    except Exception as exc:
        status = getattr(exc, "status_code", None)
        message = str(exc).lower()
        if status == 402 or ("credit" in message and ("balance" in message or "billing" in message)):
            _last_haiku_failure = "billing_exhausted"
            raise BillingExhaustedError("Anthropic vision credit is exhausted") from exc
        _last_haiku_failure = f"http_{status}" if isinstance(status, int) else type(exc).__name__
        return ""
