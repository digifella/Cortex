from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from cortex_engine.textifier import DocumentTextifier


def test_halftone_low_strength_config_is_gentle():
    cfg = DocumentTextifier._build_halftone_repair_config(5.0)

    assert cfg["scale"] > 0.99
    assert cfg["median"] == 1
    assert cfg["gaussian"] == 1
    assert cfg["denoise"] < 0.5
    assert cfg["sharpen_amount"] < 0.02


def test_repair_halftone_image_preserves_dimensions_and_outputs_file(tmp_path):
    width, height = 160, 120
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[..., 1] = 90
    canvas[..., 2] = 40

    # Create a simple dot-grid pattern to simulate print/screen artefacts.
    for y in range(0, height, 4):
        for x in range(0, width, 4):
            canvas[y:y + 2, x:x + 2] = [210, 180, 120]

    source_path = tmp_path / "halftone.png"
    Image.fromarray(canvas, mode="RGB").save(source_path)

    result = DocumentTextifier(use_vision=False).repair_halftone_image(
        str(source_path),
        strength=42,
        preserve_color=True,
    )

    assert result["file_name"] == "halftone.png"
    assert Path(result["output_path"]).exists()

    info = result["halftone_repair_info"]
    assert info["repaired"] is True
    assert info["preset"] == "medium"
    assert info["strength"] == 42.0
    assert info["preserve_color"] is True
    assert info["original_width"] == width
    assert info["original_height"] == height

    with Image.open(result["output_path"]) as repaired:
        assert repaired.size == (width, height)
        assert repaired.mode in {"RGB", "RGBA"}
