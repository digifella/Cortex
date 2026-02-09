"""Model-aware similarity threshold utilities for search paths."""

from __future__ import annotations

import os
from typing import Optional

from .logging_utils import get_logger

logger = get_logger(__name__)


def get_model_aware_threshold(db_path: Optional[str] = None) -> float:
    """
    Get similarity threshold adjusted for the embedding model dimension.

    Lower-dimension models (2B/2048D) produce lower raw similarity scores
    than higher-dimension models (8B/4096D) for equivalent semantic matches.
    """
    dimension_thresholds = {
        2048: 0.30,  # 2B model
        4096: 0.40,  # 8B model
    }
    default_threshold = 0.35

    try:
        if db_path:
            chroma_path = os.path.join(db_path, "knowledge_hub_db")
            db_file = os.path.join(chroma_path, "chroma.sqlite3")
            if os.path.exists(db_file):
                import sqlite3

                conn = sqlite3.connect(db_file)
                cursor = conn.cursor()
                cursor.execute("SELECT dimension FROM collections LIMIT 1")
                row = cursor.fetchone()
                conn.close()

                if row and row[0]:
                    dimension = row[0]
                    threshold = dimension_thresholds.get(dimension, default_threshold)
                    logger.info(
                        f"📊 Model-aware threshold: {threshold:.2f} (detected {dimension}D embeddings)"
                    )
                    return threshold

        from cortex_engine.config import QWEN3_VL_MODEL_SIZE

        if QWEN3_VL_MODEL_SIZE == "2B":
            logger.info("📊 Model-aware threshold: 0.30 (2B model from config)")
            return 0.30
        if QWEN3_VL_MODEL_SIZE == "8B":
            logger.info("📊 Model-aware threshold: 0.40 (8B model from config)")
            return 0.40

    except Exception as e:
        logger.warning(f"Could not detect embedding dimension: {e}")

    logger.info(f"📊 Using default threshold: {default_threshold}")
    return default_threshold
