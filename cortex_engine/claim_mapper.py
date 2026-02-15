"""
Claim-to-citation mapping utilities.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from .collection_manager import WorkingCollectionManager
from .config import COLLECTION_NAME
from .embedding_service import embed_query
from .utils import get_logger

logger = get_logger(__name__)


@dataclass
class Claim:
    id: str
    text: str


@dataclass
class Evidence:
    doc_id: str
    snippet: str
    source_file: str
    score: float
    distance: float
    metadata: Dict


def extract_claims(text: str, min_chars: int = 40) -> List[Claim]:
    """Extract candidate claim sentences from draft text."""
    content = (text or "").strip()
    if not content:
        return []

    candidates = re.split(r"(?<=[.!?])\s+|\n+", content)
    claims: List[Claim] = []
    for idx, raw in enumerate(candidates, start=1):
        sentence = raw.strip()
        if len(sentence) < min_chars:
            continue
        if len(sentence.split()) < 7:
            continue
        claims.append(Claim(id=f"c{idx}", text=sentence))
    return claims


def _distance_to_score(distance: float) -> float:
    """Convert vector distance to normalized confidence score (0..1)."""
    try:
        dist = float(distance)
    except Exception:
        return 0.0
    return 1.0 / (1.0 + max(dist, 0.0))


def _allowed_doc_ids(selected_collections: Optional[List[str]]) -> Optional[set]:
    if not selected_collections:
        return None
    mgr = WorkingCollectionManager()
    allowed = set()
    for name in selected_collections:
        allowed.update(mgr.get_doc_ids_by_name(name))
    return allowed if allowed else set()


def map_claims_to_evidence(
    db_path: str,
    text: str,
    top_k: int = 3,
    selected_collections: Optional[List[str]] = None,
    support_threshold: float = 0.42,
) -> Dict:
    """
    Map extracted claims to top evidence chunks from Chroma.
    Uses query_embeddings from embedding_service to keep embedding consistency.
    """
    claims = extract_claims(text)
    if not claims:
        return {"claims": [], "summary": {"supported": 0, "weak": 0, "unsupported": 0}}

    client = chromadb.PersistentClient(path=db_path, settings=ChromaSettings(anonymized_telemetry=False))
    collection = client.get_collection(COLLECTION_NAME)
    allowed_ids = _allowed_doc_ids(selected_collections)

    mapped_claims: List[Dict] = []
    supported = weak = unsupported = 0

    for claim in claims:
        emb = embed_query(claim.text)
        raw = collection.query(
            query_embeddings=[emb],
            n_results=max(1, int(top_k) * 5),
            include=["documents", "metadatas", "distances"],
        )

        documents = (raw.get("documents") or [[]])[0]
        metadatas = (raw.get("metadatas") or [[]])[0]
        distances = (raw.get("distances") or [[]])[0]

        evidence_rows: List[Evidence] = []
        for doc_text, meta, dist in zip(documents, metadatas, distances):
            m = meta or {}
            doc_id = str(m.get("doc_id", ""))
            if allowed_ids is not None and doc_id not in allowed_ids:
                continue
            snippet = (doc_text or "").strip().replace("\n", " ")
            if len(snippet) > 320:
                snippet = snippet[:320].rstrip() + "..."
            score = _distance_to_score(dist)
            evidence_rows.append(
                Evidence(
                    doc_id=doc_id,
                    snippet=snippet,
                    source_file=str(m.get("file_name", "")),
                    score=score,
                    distance=float(dist),
                    metadata=m,
                )
            )

        evidence_rows.sort(key=lambda e: e.score, reverse=True)
        evidence_rows = evidence_rows[: max(1, int(top_k))]
        best = evidence_rows[0].score if evidence_rows else 0.0

        if best >= support_threshold:
            status = "supported"
            supported += 1
        elif best >= max(0.25, support_threshold * 0.65):
            status = "weak"
            weak += 1
        else:
            status = "unsupported"
            unsupported += 1

        mapped_claims.append(
            {
                "claim_id": claim.id,
                "claim_text": claim.text,
                "status": status,
                "best_score": round(best, 4),
                "evidence": [asdict(e) for e in evidence_rows],
            }
        )

    return {
        "claims": mapped_claims,
        "summary": {"supported": supported, "weak": weak, "unsupported": unsupported},
    }
