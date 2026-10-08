"""Caption a document figure (chart, diagram, screenshot, table image) using
Claude Haiku vision. Distinct from textifier.py's photo-captioning prompt,
which is written for photographs and explicitly tells the model to omit
logos/icons — this prompt is written for the opposite case: we've already
filtered out small decorative graphics by bounding-box area before this
function is ever called (see enhanced_ingest_cortex.py), so every image
reaching here is presumed to be genuine document content worth describing
in full, including any text, axis labels, or data visible in it.
"""
import base64
import os

MODEL = "claude-haiku-5-5"

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


def caption_figure_with_haiku(image_bytes: bytes, context_hint: str = "") -> str:
    api_key = _load_api_key()
    if not api_key:
        return ""
    try:
        import anthropic
    except ImportError:
        return ""

    prompt = _PROMPT
    if context_hint:
        prompt += " " + context_hint.strip()

    try:
        client = anthropic.Anthropic(api_key=api_key)
        encoded = base64.b64encode(image_bytes).decode("utf-8")
        response = client.messages.create(
            model=MODEL,
            max_tokens=200,
            thinking={"type": "disabled"},
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
        return ""
    except Exception:
        return ""
