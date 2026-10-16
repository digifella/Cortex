"""Use an already-loaded LM Studio Qwen vision model without proxy routing."""

import base64
import json
import os
import urllib.request
from pathlib import Path
from typing import Any

_last_completion_failure = ""


class LocalVisionUnavailableError(RuntimeError):
    """A required document figure could not be captioned locally or by fallback."""


def last_completion_failure() -> str:
    return _last_completion_failure


def base_url() -> str:
    configured = os.environ.get("CORTEX_LMSTUDIO_BASE_URL", "").strip()
    if not configured:
        env_file = Path(__file__).resolve().parents[1] / ".env"
        if env_file.exists():
            for line in env_file.read_text(encoding="utf-8").splitlines():
                key, _, value = line.partition("=")
                if key.strip() == "CORTEX_LMSTUDIO_BASE_URL":
                    configured = value.strip().strip('"').strip("'")
                    break
    return (configured or "http://localhost:1234/v1").rstrip("/")


def _opener():
    # The Windows tailnet address must not be routed through HTTP(S)_PROXY.
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _models(path: str) -> dict[str, Any]:
    root = base_url().removesuffix("/v1")
    with _opener().open(root + path, timeout=8) as response:
        return json.load(response)


def loaded_qwen_model() -> str | None:
    """Select only a Qwen model with loaded vision-capable weights."""
    try:
        for entry in _models("/api/v1/models").get("models", []):
            key = str(entry.get("key") or "")
            if "qwen" not in key.lower() or not entry.get("capabilities", {}).get("vision"):
                continue
            for instance in entry.get("loaded_instances", []):
                model = str(instance.get("id") or key)
                if model:
                    return model
    except (OSError, ValueError, TypeError, KeyError):
        pass

    # Older LM Studio releases expose loaded VLMs only through v0.
    try:
        for entry in _models("/api/v0/models").get("data", []):
            model = str(entry.get("id") or "")
            vision = (entry.get("type") == "vlm" or entry.get("vision") is True
                      or entry.get("capabilities", {}).get("vision") is True)
            if "qwen" in model.lower() and entry.get("state") == "loaded" and vision:
                return model
    except (OSError, ValueError, TypeError, KeyError):
        pass
    return None


def describe_encoded_image(
    encoded_image: str, *, mime_type: str, prompt: str, model: str | None = None,
    max_tokens: int = 300, timeout: float = 120.0,
) -> str:
    global _last_completion_failure
    _last_completion_failure = ""
    selected = model or loaded_qwen_model()
    if not selected:
        _last_completion_failure = "no_loaded_qwen_vision_model"
        return ""
    payload = {
        "model": selected,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {
                "url": f"data:{mime_type};base64,{encoded_image}"}},
        ]}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "reasoning_effort": "none",
    }
    request = urllib.request.Request(
        base_url() + "/chat/completions", data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with _opener().open(request, timeout=timeout) as response:
            answer = json.load(response)
        content = answer["choices"][0]["message"]["content"]
        if isinstance(content, str) and content.strip():
            return content.strip()
        _last_completion_failure = "empty_model_response"
        return ""
    except (OSError, ValueError, TypeError, KeyError, IndexError) as exc:
        status = getattr(exc, "code", None)
        _last_completion_failure = f"http_{status}" if isinstance(status, int) else type(exc).__name__
        return ""


def describe_image(image_bytes: bytes, *, mime_type: str, prompt: str) -> str:
    return describe_encoded_image(
        base64.b64encode(image_bytes).decode("ascii"), mime_type=mime_type, prompt=prompt)


def safe_status() -> dict[str, Any]:
    """Non-secret diagnostics for the currently configured native endpoint."""
    result: dict[str, Any] = {"base_url": base_url(), "model_api_reachable": False,
                              "loaded_qwen_vision_models": []}
    try:
        models = _models("/api/v1/models").get("models", [])
        result["model_api_reachable"] = True
        result["loaded_qwen_vision_models"] = [
            str(instance.get("id") or entry.get("key"))
            for entry in models
            if "qwen" in str(entry.get("key") or "").lower()
            and entry.get("capabilities", {}).get("vision")
            for instance in entry.get("loaded_instances", [])
        ]
    except (OSError, ValueError, TypeError, KeyError) as exc:
        result["error_type"] = type(exc).__name__
    return result
