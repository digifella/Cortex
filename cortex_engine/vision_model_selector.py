"""Choose the best local vision model that fits the machine's free VRAM.

The same Cortex install runs on an 8GB laptop GPU and a 48GB RTX 8000. Hardcoding
one model is wrong on both: the laptop silently spills to CPU and runs 10-100x
slower, while the workstation leaves most of its capability unused.

This module picks the highest-quality *installed* model that fits in the VRAM
actually free right now, so the choice adapts to the machine — and to whether
Lightroom happens to be open.

Measured on an RTX 4060 Laptop (8188 MiB), 2026-07-28, six-photo benchmark with
identical prompts. "clean" = caption passed all style checks:

    gemma4:e2b-it-qat   1.6 GB   80s   5/6 clean   best speed/size; cucumber-for-jalapeno
    qwen3-vl:8b         7.4 GB  127s   4/6 clean   BEST accuracy (only model to read
                                                   the jalapeno garnish correctly)
    qwen3-vl:4b         4.7 GB  159s   3/6 clean   3 empty — worst of both
    llava:7b            4.9 GB   90s   2/6 clean   fast but weakest content accuracy

Quality ordering below reflects *content accuracy*, which is what ends up in the
catalog as keywords. Style problems are recoverable by re-running; a wrong noun
is not, because nobody re-reads 10,000 captions.
"""
import shutil
import subprocess
from typing import Dict, List, Optional, Tuple

from .utils import get_logger

logger = get_logger(__name__)

# Headroom above the model's resident size. Ollama needs room for the KV cache
# and the vision encoder's activations; running right at the limit is what
# produced "llama runner terminated, exit status 2" during testing.
VRAM_HEADROOM_MB = 1024


class VisionModelProfile:
    """What a model costs and what it is good at."""

    def __init__(self, name: str, vram_mb: int, quality: int,
                 reasoning: bool, num_predict: int, note: str = ""):
        self.name = name
        self.vram_mb = vram_mb          # measured resident size, not download size
        self.quality = quality          # content-accuracy rank; higher is better
        self.reasoning = reasoning      # emits chain-of-thought before answering
        self.num_predict = num_predict  # token budget needed to reach an answer
        self.note = note

    def __repr__(self) -> str:
        return f"<{self.name} {self.vram_mb}MB q={self.quality}>"


# Ordered best-quality-first. Add new models here; nothing else needs changing.
# vram_mb comes from `ollama ps` after a real inference, NOT from `ollama list`
# (which reports the download size and understates the 8b by ~1.3GB).
VISION_MODEL_PROFILES: List[VisionModelProfile] = [
    VisionModelProfile("qwen3-vl:32b", 22000, 100, True, 640,
                       "untested here; expected best on a large-VRAM machine"),
    VisionModelProfile("qwen3-vl:8b", 7400, 90, True, 640,
                       "best measured content accuracy; slow (127s/photo on a 4060)"),
    VisionModelProfile("minicpm-v:8b", 6000, 70, False, 160,
                       "untested here — vram_mb and quality are estimates"),
    VisionModelProfile("gemma4:e4b-it-qat", 3200, 75, True, 640,
                       "untested; larger sibling of the e2b"),
    VisionModelProfile("llava:7b", 4900, 40, False, 160,
                       "fast, weakest accuracy — 'lime and a pickle' for a jalapeno"),
    VisionModelProfile("qwen3-vl:4b", 4700, 50, True, 640,
                       "worst of both: slow and often empty"),
    VisionModelProfile("gemma4:e2b-it-qat", 1600, 65, True, 640,
                       "best practical choice on a small GPU; coexists with Lightroom"),
]

PROFILES_BY_NAME: Dict[str, VisionModelProfile] = {p.name: p for p in VISION_MODEL_PROFILES}


def free_vram_mb() -> Optional[int]:
    """Free VRAM in MiB, or None when there is no readable NVIDIA GPU."""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode != 0:
            return None
        return max(int(v.strip()) for v in out.stdout.split("\n") if v.strip())
    except Exception as exc:
        logger.debug("Could not read GPU memory: %s", exc)
        return None


def installed_models() -> List[str]:
    """Model names present in `ollama list`. Empty when Ollama is unreachable."""
    if not shutil.which("ollama"):
        return []
    try:
        out = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=15)
        if out.returncode != 0:
            return []
        return [ln.split()[0] for ln in out.stdout.splitlines()[1:] if ln.strip()]
    except Exception as exc:
        logger.debug("Could not list ollama models: %s", exc)
        return []


def select_vision_model(
    available: Optional[List[str]] = None,
    free_mb: Optional[int] = None,
    headroom_mb: int = VRAM_HEADROOM_MB,
) -> Tuple[Optional[str], str]:
    """Pick the best installed model that fits in free VRAM.

    Returns (model_name, reason). model_name is None when nothing is installed.

    When VRAM cannot be read (no GPU, or CPU-only host) the smallest installed
    profile wins — on CPU, size dominates speed and a large model is unusable.
    """
    available = installed_models() if available is None else available
    if not available:
        return None, "no Ollama models installed"

    known = [p for p in VISION_MODEL_PROFILES if p.name in available]
    if not known:
        return None, f"none of the {len(available)} installed models is a known vision model"

    if free_mb is None:
        free_mb = free_vram_mb()

    if free_mb is None:
        pick = min(known, key=lambda p: p.vram_mb)
        return pick.name, "no readable GPU — chose the smallest model for CPU inference"

    budget = free_mb - headroom_mb
    fits = [p for p in known if p.vram_mb <= budget]
    if not fits:
        pick = min(known, key=lambda p: p.vram_mb)
        return (pick.name,
                f"nothing fits in {free_mb}MB free (need {pick.vram_mb}+{headroom_mb}MB) — "
                f"chose smallest; expect CPU spill. Close other GPU apps (Lightroom "
                f"holds ~3.5GB) and retry")

    pick = max(fits, key=lambda p: p.quality)
    return (pick.name,
            f"best quality fitting {free_mb}MB free "
            f"({pick.vram_mb}MB + {headroom_mb}MB headroom); {pick.note}" if pick.note
            else f"best quality fitting {free_mb}MB free")


def num_predict_for(model: str, default: int = 200) -> int:
    """Token budget for a model, from its profile."""
    profile = PROFILES_BY_NAME.get((model or "").strip())
    return profile.num_predict if profile else default


def describe_selection() -> str:
    """Human-readable summary for the UI."""
    free = free_vram_mb()
    model, reason = select_vision_model(free_mb=free)
    free_txt = f"{free}MB free VRAM" if free is not None else "no GPU detected"
    if not model:
        return f"{free_txt} — {reason}"
    return f"{free_txt} → {model} ({reason})"
