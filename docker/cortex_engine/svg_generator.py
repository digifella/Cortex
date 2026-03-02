"""
Gemini-backed SVG generation utility.

Supports direct image-to-SVG generation and a two-stage workflow:
1. Create a simplified sketch/poster intermediate image.
2. Optionally convert that intermediate into SVG markup.
"""

import base64
import json
import os
import re
from typing import Any, Dict, List, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from xml.etree import ElementTree as ET

from cortex_engine.utils.logging_utils import get_logger

logger = get_logger(__name__)


class GeminiSvgGenerator:
    """Generate SVG markup from an input image using the Gemini API."""

    DEFAULT_MODEL = "gemini-3-flash-preview"

    MODE_HINTS = {
        "Trace Sketch": "Prioritize clean vector line work, simple fills, and faithful geometric tracing.",
        "Diagram to SVG": "Prioritize semantic diagram structure, labels, arrows, boxes, and clean technical layout.",
        "Logo/Icon Vectorize": "Prioritize bold minimal shapes, crisp edges, and compact reusable icon-style geometry.",
        "Stylized Photo Poster": "Create a simplified poster-style vector interpretation with large color regions and reduced detail.",
    }

    SKETCH_MODELS = (
        "gemini-3-pro-image-preview",
        "gemini-2.5-flash-image",
        "gemini-3.1-flash-image-preview",
    )

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        self.api_key = (api_key or os.getenv("GEMINI_API_KEY") or "").strip()
        self.model_name = (model_name or os.getenv("CORTEX_SVG_GEMINI_MODEL") or self.DEFAULT_MODEL).strip()
        self.api_url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent"
        )

    @staticmethod
    def _extract_text_parts(response_json: Dict[str, Any]) -> str:
        parts_out: List[str] = []
        for candidate in response_json.get("candidates", []) or []:
            content = candidate.get("content", {}) or {}
            for part in content.get("parts", []) or []:
                text = part.get("text")
                if text:
                    parts_out.append(str(text))
        return "\n".join(parts_out).strip()

    @staticmethod
    def _extract_svg_block(text: str) -> str:
        raw = str(text or "").strip()
        raw = re.sub(r"```(?:svg|xml)?\s*", "", raw, flags=re.IGNORECASE)
        raw = raw.replace("```", "").strip()
        match = re.search(r"(<svg\b.*?</svg>)", raw, flags=re.IGNORECASE | re.DOTALL)
        return match.group(1).strip() if match else raw

    @staticmethod
    def _sanitize_svg(svg_text: str) -> Dict[str, Any]:
        """Validate and sanitize SVG markup before rendering/downloading."""
        warnings: List[str] = []
        candidate = str(svg_text or "").strip()
        if not candidate:
            return {"success": False, "error": "Gemini returned empty SVG text", "warnings": warnings}

        candidate = re.sub(r"^\s*<\?xml.*?\?>", "", candidate, flags=re.DOTALL).strip()
        candidate = re.sub(r"<script\b.*?</script>", "", candidate, flags=re.IGNORECASE | re.DOTALL)

        try:
            root = ET.fromstring(candidate)
        except ET.ParseError as e:
            return {"success": False, "error": f"Invalid SVG/XML: {e}", "warnings": warnings}

        tag_name = root.tag.split("}", 1)[-1].lower()
        if tag_name != "svg":
            return {"success": False, "error": "Generated markup is not an <svg> document", "warnings": warnings}

        if "xmlns" not in root.attrib:
            root.set("xmlns", "http://www.w3.org/2000/svg")
            warnings.append("Added missing SVG namespace.")

        if "viewBox" not in root.attrib:
            width = root.attrib.get("width")
            height = root.attrib.get("height")
            try:
                if width and height:
                    width_val = float(re.sub(r"[^0-9.]+", "", str(width)))
                    height_val = float(re.sub(r"[^0-9.]+", "", str(height)))
                    if width_val > 0 and height_val > 0:
                        root.set("viewBox", f"0 0 {int(width_val)} {int(height_val)}")
                        warnings.append("Added viewBox from width/height.")
            except Exception:
                pass

        removed_attrs = 0
        for elem in root.iter():
            for attr_name in list(elem.attrib.keys()):
                if attr_name.lower().startswith("on"):
                    del elem.attrib[attr_name]
                    removed_attrs += 1
        if removed_attrs:
            warnings.append(f"Removed {removed_attrs} inline event handler attribute(s).")

        sanitized = ET.tostring(root, encoding="unicode")
        return {"success": True, "svg": sanitized, "warnings": warnings}

    def _build_prompt(self, mode: str, user_prompt: str, detail_level: int = 5) -> str:
        mode_hint = self.MODE_HINTS.get(mode, self.MODE_HINTS["Trace Sketch"])
        custom = (user_prompt or "").strip()
        custom_block = (
            f"\nAdditional style instructions from user:\n{custom}\n"
            if custom
            else ""
        )
        return (
            "You convert the provided image into a clean SVG illustration.\n"
            "Return ONLY valid SVG markup. Do not use markdown fences. Do not explain your work.\n"
            "Requirements:\n"
            "- Output a complete <svg>...</svg> document.\n"
            "- Use simple reusable vector shapes where possible.\n"
            "- Keep the design visually clean and intentional.\n"
            "- Avoid scripts, external references, embedded raster images, CSS imports, or foreignObject.\n"
            "- Include width, height, and viewBox when possible.\n"
            "- If the source is a photo, create a stylized vector poster interpretation, not photorealistic detail.\n"
            f"- Mode guidance: {mode_hint}\n"
            f"- Detail guidance: {self._detail_hint(detail_level)}\n"
            f"{custom_block}"
        )

    @staticmethod
    def _detail_hint(detail_level: int) -> str:
        level = max(1, min(int(detail_level or 5), 10))
        if level <= 3:
            return "Use very low detail: major outlines only, minimal internal lines, broad simple shapes."
        if level <= 7:
            return "Use medium detail: main contours, key internal edges, restrained texture, selective line accents."
        return "Use high detail: preserve smaller contours, richer internal line work, and more nuanced shading cues."

    def _call_generate_content(self, model_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
        req = Request(
            api_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "X-goog-api-key": self.api_key,
            },
            method="POST",
        )
        with urlopen(req, timeout=180) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        return json.loads(raw) if raw.strip() else {}

    @staticmethod
    def _extract_inline_image(response_json: Dict[str, Any]) -> Dict[str, Any]:
        for candidate in response_json.get("candidates", []) or []:
            content = candidate.get("content", {}) or {}
            for part in content.get("parts", []) or []:
                blob = part.get("inlineData") or part.get("inline_data")
                if isinstance(blob, dict) and blob.get("data"):
                    mime = blob.get("mimeType") or blob.get("mime_type") or "image/png"
                    try:
                        data = base64.b64decode(blob["data"])
                    except Exception:
                        continue
                    return {"bytes": data, "mime_type": mime}
        return {}

    def generate_intermediate_sketch(
        self,
        image_bytes: bytes,
        mime_type: str,
        color_mode: str = "B&W Line Art",
        detail_level: int = 5,
        user_prompt: str = "",
        model_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate a simplified image first, suitable as a useful output and as SVG input."""
        if not self.api_key:
            return {"success": False, "error": "GEMINI_API_KEY is not configured"}
        if not image_bytes:
            return {"success": False, "error": "No image data provided"}

        chosen_model = (model_name or os.getenv("CORTEX_SVG_GEMINI_SKETCH_MODEL") or self.SKETCH_MODELS[0]).strip()
        style_hint = (
            "Convert the image into clean black-and-white line art with white background, crisp outlines, and no photoreal texture."
            if color_mode == "B&W Line Art"
            else "Convert the image into a clean stylized sketch with flat color fills, simplified shapes, and minimal texture."
        )
        custom = (user_prompt or "").strip()
        custom_block = f"\nAdditional style instructions:\n{custom}\n" if custom else ""
        prompt = (
            "Transform the provided image into a simplified illustrative reference image.\n"
            f"{style_hint}\n"
            f"{self._detail_hint(detail_level)}\n"
            "- Preserve the main subject and overall composition.\n"
            "- Remove clutter, tiny details, and photographic noise.\n"
            "- Do not add text overlays or watermarks.\n"
            f"{custom_block}"
        )
        payload = {
            "contents": [{
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": mime_type or "image/png",
                            "data": base64.b64encode(image_bytes).decode("utf-8"),
                        }
                    },
                ]
            }]
        }
        try:
            response_json = self._call_generate_content(chosen_model, payload)
        except HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            logger.warning(f"Gemini sketch generation HTTP error {e.code}: {body[:300]}")
            return {"success": False, "error": f"Gemini API HTTP {e.code}", "response_preview": body[:300]}
        except URLError as e:
            return {"success": False, "error": f"Gemini API connection failed: {e}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

        image_part = self._extract_inline_image(response_json)
        if not image_part:
            text = self._extract_text_parts(response_json)
            return {
                "success": False,
                "error": "Gemini did not return an intermediate image",
                "response_preview": text[:300],
            }
        return {
            "success": True,
            "model_used": chosen_model,
            "image_bytes": image_part["bytes"],
            "mime_type": image_part["mime_type"],
        }

    def generate_from_image(
        self,
        image_bytes: bytes,
        mime_type: str,
        mode: str = "Trace Sketch",
        user_prompt: str = "",
        detail_level: int = 5,
    ) -> Dict[str, Any]:
        """Generate an SVG string from an input image."""
        if not self.api_key:
            return {"success": False, "error": "GEMINI_API_KEY is not configured"}
        if not image_bytes:
            return {"success": False, "error": "No image data provided"}

        prompt = self._build_prompt(mode, user_prompt, detail_level=detail_level)
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                        {
                            "inline_data": {
                                "mime_type": mime_type or "image/png",
                                "data": base64.b64encode(image_bytes).decode("utf-8"),
                            }
                        },
                    ]
                }
            ]
        }
        req = Request(
            self.api_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "X-goog-api-key": self.api_key,
            },
            method="POST",
        )

        try:
            with urlopen(req, timeout=180) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
                status = int(getattr(resp, "status", 200) or 200)
        except HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            logger.warning(f"Gemini SVG generation HTTP error {e.code}: {body[:300]}")
            return {"success": False, "error": f"Gemini API HTTP {e.code}", "response_preview": body[:300]}
        except URLError as e:
            logger.warning(f"Gemini SVG generation connection error: {e}")
            return {"success": False, "error": f"Gemini API connection failed: {e}"}
        except Exception as e:
            logger.warning(f"Gemini SVG generation failed: {e}")
            return {"success": False, "error": str(e)}

        try:
            response_json = json.loads(raw) if raw.strip() else {}
        except json.JSONDecodeError:
            return {
                "success": False,
                "error": "Gemini API returned non-JSON response",
                "response_preview": raw[:300],
            }

        text = self._extract_text_parts(response_json)
        svg_candidate = self._extract_svg_block(text)
        sanitized = self._sanitize_svg(svg_candidate)
        if not sanitized.get("success"):
            return {
                "success": False,
                "error": sanitized.get("error", "Could not validate generated SVG"),
                "response_preview": text[:300],
                "http_status": status,
            }

        return {
            "success": True,
            "model_used": self.model_name,
            "mode": mode,
            "detail_level": max(1, min(int(detail_level or 5), 10)),
            "svg": sanitized["svg"],
            "warnings": sanitized.get("warnings", []),
            "http_status": status,
            "response_preview": text[:300],
        }
