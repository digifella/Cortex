from __future__ import annotations

from pathlib import Path
from typing import Optional

from cortex_engine.anonymizer import AnonymizationMapping, DocumentAnonymizer


def handle(input_path: Optional[Path], input_data: dict, job: dict) -> dict:
    """
    PDF anonymiser handler.

    Returns:
        {
            "output_data": dict,
            "output_file": Path | None
        }
    """
    if input_path is None:
        raise ValueError("pdf_anonymise requires an input file")
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    confidence_threshold = float((input_data or {}).get("confidence_threshold", 0.3))
    anonymizer = DocumentAnonymizer()
    mapping = AnonymizationMapping()
    output_path, final_mapping = anonymizer.anonymize_single_file(
        input_path=input_path,
        output_path=None,
        mapping=mapping,
        confidence_threshold=confidence_threshold,
    )

    output_data = {
        "summary": "Processed via cortex_engine.anonymizer.DocumentAnonymizer",
        "entities_replaced": len(final_mapping.mappings),
        "confidence_threshold": confidence_threshold,
    }
    return {"output_data": output_data, "output_file": Path(output_path)}
