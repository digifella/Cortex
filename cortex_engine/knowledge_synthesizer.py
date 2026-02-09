"""
Knowledge synthesizer and ideation workflows.

Default workflow is a fast 2-step ideation mode:
1) discover themes
2) generate ideas

An optional classic 4-stage workflow is also exposed for deeper sessions.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from .collection_manager import WorkingCollectionManager
from .config import COLLECTION_NAME
from .idea_generator import IdeaGenerator
from .llm_service import create_llm_service
from .utils import convert_to_docker_mount_path, resolve_db_root_path
from .utils.logging_utils import get_logger

logger = get_logger(__name__)


def _parse_json_payload(text: str, fallback_key: str) -> Dict[str, Any]:
    if not text:
        return {fallback_key: []}

    cleaned = text.strip()
    cleaned = re.sub(r"^```json\s*", "", cleaned)
    cleaned = re.sub(r"^```\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list):
            return {fallback_key: parsed}
    except Exception:
        pass

    m_obj = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if m_obj:
        try:
            parsed = json.loads(m_obj.group(0))
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

    m_arr = re.search(r"\[.*\]", cleaned, flags=re.DOTALL)
    if m_arr:
        try:
            parsed = json.loads(m_arr.group(0))
            if isinstance(parsed, list):
                return {fallback_key: parsed}
        except Exception:
            pass

    return {fallback_key: []}


def _tokenize(text: str) -> List[str]:
    return [t for t in re.findall(r"[a-zA-Z0-9][a-zA-Z0-9\-]+", (text or "").lower()) if len(t) > 2]


def _score_text(query: str, text: str) -> int:
    if not query or not text:
        return 0
    q_terms = _tokenize(query)
    if not q_terms:
        return 0
    low = text.lower()
    return sum(low.count(term) for term in q_terms)


def _get_chroma_collection() -> Any:
    mgr = WorkingCollectionManager()
    db_root = Path(resolve_db_root_path(Path(mgr.collections_file).parent) or Path(mgr.collections_file).parent)
    chroma_dir = os.path.join(convert_to_docker_mount_path(str(db_root)), "knowledge_hub_db")
    settings = ChromaSettings(anonymized_telemetry=False)
    client = chromadb.PersistentClient(path=chroma_dir, settings=settings)
    return client.get_collection(COLLECTION_NAME)


def _metadata_passes_credibility(meta: Dict[str, Any], min_tier: int) -> bool:
    if min_tier <= 0:
        return True
    try:
        value = int(meta.get("credibility_tier_value", 0) or 0)
    except Exception:
        value = 0
    return value >= min_tier


def _load_ranked_chunks(
    collection_name: str,
    query: str,
    top_k: int,
    min_credibility_tier: int = 0,
) -> List[Dict[str, Any]]:
    collection_mgr = WorkingCollectionManager()
    doc_ids = set(collection_mgr.get_doc_ids_by_name(collection_name))
    if not doc_ids:
        return []

    collection = _get_chroma_collection()
    count = collection.count()
    if count <= 0:
        return []

    rows: List[Dict[str, Any]] = []
    page_size = 500
    for offset in range(0, count, page_size):
        batch = collection.get(limit=page_size, offset=offset, include=["metadatas", "documents"])
        ids = batch.get("ids", [])
        metadatas = batch.get("metadatas", [])
        docs = batch.get("documents", [])
        for cid, meta, doc in zip(ids, metadatas, docs):
            if not isinstance(meta, dict):
                continue
            if meta.get("doc_id") not in doc_ids:
                continue
            if not _metadata_passes_credibility(meta, min_credibility_tier):
                continue
            text = doc or ""
            score = _score_text(query, f"{meta.get('file_name', '')}\n{meta.get('summary', '')}\n{text}")
            if score <= 0 and query:
                continue
            rows.append(
                {
                    "id": cid,
                    "doc_id": meta.get("doc_id"),
                    "file_name": meta.get("file_name", "Unknown"),
                    "summary": meta.get("summary", ""),
                    "content": text,
                    "metadata": meta,
                    "score": score,
                }
            )

    rows.sort(key=lambda r: r["score"], reverse=True)
    return rows[:top_k]


def discover_themes(
    collection_name: str,
    seed_topic: str,
    llm_provider: str,
    min_credibility_tier: int = 0,
    top_k_chunks: int = 15,
) -> Dict[str, Any]:
    chunks = _load_ranked_chunks(
        collection_name=collection_name,
        query=seed_topic,
        top_k=top_k_chunks,
        min_credibility_tier=min_credibility_tier,
    )
    if not chunks:
        return {"status": "error", "error": "No relevant chunks found for the selected filters."}

    context_lines = []
    for i, chunk in enumerate(chunks, start=1):
        snippet = (chunk["content"] or "")[:500].replace("\n", " ").strip()
        context_lines.append(
            f"[{i}] {chunk['file_name']}: {chunk.get('summary', '')}\nSnippet: {snippet}"
        )
    context = "\n\n".join(context_lines)

    prompt = f"""
You are an innovation theme extractor.
Given the seed topic and source context, return 6-10 high-value ideation themes.

Seed topic: {seed_topic}

Context:
{context}

Return STRICT JSON only:
{{
  "themes": [
    {{"name": "Theme name", "description": "1-2 sentence explanation"}}
  ]
}}
"""
    try:
        llm = create_llm_service("ideation", llm_provider).get_llm()
        raw = str(llm.complete(prompt))
        parsed = _parse_json_payload(raw, "themes")
        themes = parsed.get("themes", [])
        if not isinstance(themes, list):
            themes = []
        clean_themes = []
        for t in themes:
            if isinstance(t, dict):
                name = str(t.get("name", "")).strip()
                desc = str(t.get("description", "")).strip()
                if name:
                    clean_themes.append({"name": name, "description": desc})
            elif isinstance(t, str) and t.strip():
                clean_themes.append({"name": t.strip(), "description": ""})
        return {
            "status": "success",
            "themes": clean_themes[:10],
            "source_chunks": [{"id": c["id"], "doc_id": c["doc_id"], "file_name": c["file_name"]} for c in chunks],
        }
    except Exception as e:
        logger.error(f"Theme discovery failed: {e}", exc_info=True)
        return {"status": "error", "error": str(e)}


def generate_ideas_by_theme(
    collection_name: str,
    themes: List[str],
    innovation_goals: str,
    llm_provider: str,
    min_credibility_tier: int = 0,
    chunks_per_theme: int = 3,
) -> Dict[str, Any]:
    if not themes:
        return {"status": "error", "error": "No themes selected."}

    theme_context: Dict[str, List[Dict[str, Any]]] = {}
    all_chunks: List[Dict[str, Any]] = []
    seen_ids = set()
    for theme in themes:
        ranked = _load_ranked_chunks(
            collection_name=collection_name,
            query=theme,
            top_k=max(chunks_per_theme, 1),
            min_credibility_tier=min_credibility_tier,
        )
        theme_context[theme] = ranked
        for c in ranked:
            if c["id"] not in seen_ids:
                seen_ids.add(c["id"])
                all_chunks.append(c)

    if not all_chunks:
        return {"status": "error", "error": "No relevant chunks found for selected themes/filters."}

    context_lines = []
    for i, chunk in enumerate(all_chunks, start=1):
        snippet = (chunk["content"] or "")[:600].replace("\n", " ").strip()
        context_lines.append(
            f"[{i}] {chunk['file_name']} | doc_id={chunk['doc_id']}\n{chunk.get('summary', '')}\nSnippet: {snippet}"
        )
    context = "\n\n".join(context_lines)

    theme_list = "\n".join(f"- {t}" for t in themes)
    prompt = f"""
You are an innovation strategist.
Generate 2-3 innovative ideas per theme, grounded in the supplied KB context.

Themes:
{theme_list}

Innovation goals:
{innovation_goals or "None provided"}

KB Context:
{context}

Return STRICT JSON only:
{{
  "theme_groups": [
    {{
      "theme": "Theme name",
      "ideas": [
        {{
          "title": "Idea title",
          "description": "2-3 sentences grounded in KB context",
          "impact": "1 sentence impact statement"
        }}
      ]
    }}
  ]
}}
"""
    try:
        llm = create_llm_service("ideation", llm_provider).get_llm()
        raw = str(llm.complete(prompt))
        parsed = _parse_json_payload(raw, "theme_groups")
        groups = parsed.get("theme_groups", [])
        if not isinstance(groups, list):
            groups = []
        clean_groups = []
        for g in groups:
            if not isinstance(g, dict):
                continue
            theme = str(g.get("theme", "")).strip()
            ideas = g.get("ideas", [])
            if not theme or not isinstance(ideas, list):
                continue
            clean_ideas = []
            for idea in ideas:
                if not isinstance(idea, dict):
                    continue
                title = str(idea.get("title", "")).strip()
                desc = str(idea.get("description", "")).strip()
                impact = str(idea.get("impact", "")).strip()
                if title:
                    clean_ideas.append({"title": title, "description": desc, "impact": impact})
            clean_groups.append({"theme": theme, "ideas": clean_ideas[:3]})
        return {
            "status": "success",
            "theme_groups": clean_groups,
            "used_chunks": [{"id": c["id"], "doc_id": c["doc_id"], "file_name": c["file_name"]} for c in all_chunks],
        }
    except Exception as e:
        logger.error(f"Idea generation failed: {e}", exc_info=True)
        return {"status": "error", "error": str(e)}


def run_four_stage_ideation(
    collection_name: str,
    seed_topic: str,
    innovation_goals: str,
    llm_provider: str,
    selected_themes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Classic 4-stage wrapper using the existing IdeaGenerator APIs.
    """
    try:
        idea_gen = IdeaGenerator(vector_index=None, graph_manager=None)

        discovery = {
            "themes": selected_themes or [],
            "seed_topic": seed_topic,
            "innovation_goals": innovation_goals,
            "collection_name": collection_name,
        }

        if not discovery["themes"]:
            theme_result = idea_gen.generate_intelligent_themes(
                idea_gen._get_collection_content(collection_name),
                llm_provider=llm_provider,
            )
            if "error" in theme_result:
                return {"status": "error", "error": theme_result["error"]}
            discovery["themes"] = theme_result.get("themes", [])[:8]

        define = idea_gen.generate_problem_statements(
            themes=discovery["themes"],
            innovation_goals=innovation_goals or seed_topic,
            constraints="",
            llm_provider=llm_provider,
        )
        if "error" in define:
            return {"status": "error", "error": define["error"]}

        problems = []
        for group in define.get("problem_statements", []):
            problems.extend(group.get("problems", []))
        problems = problems[:8]

        develop = idea_gen.generate_ideas_from_problems(
            problem_statements=problems,
            collection_name=collection_name,
            themes=discovery["themes"],
            num_ideas_per_problem=3,
            creativity_level="Balanced",
            focus_areas=[],
            include_implementation=False,
            llm_provider=llm_provider,
        )
        if "error" in develop:
            return {"status": "error", "error": develop["error"]}

        # Lightweight deliver summary.
        total_ideas = sum(len(g.get("ideas", [])) for g in develop.get("idea_groups", []))
        deliver = {
            "summary": f"Generated {total_ideas} ideas from {len(problems)} problem statements across {len(discovery['themes'])} themes.",
            "recommendation": "Prioritize top 3 ideas by feasibility and expected impact, then run pilot design.",
        }

        return {
            "status": "success",
            "discovery": discovery,
            "define": define,
            "develop": develop,
            "deliver": deliver,
        }
    except Exception as e:
        logger.error(f"Four-stage ideation failed: {e}", exc_info=True)
        return {"status": "error", "error": str(e)}


def run_synthesis(collection_name: str, seed_ideas: str, llm_provider: str) -> str:
    """
    Backward-compatible synthesis entrypoint.

    This preserves older page imports while routing through the new 2-stage workflow.
    """
    discovery = discover_themes(
        collection_name=collection_name,
        seed_topic=seed_ideas,
        llm_provider=llm_provider,
        min_credibility_tier=0,
        top_k_chunks=15,
    )
    if discovery.get("status") != "success":
        return f"Synthesis failed during theme discovery: {discovery.get('error', 'unknown error')}"

    theme_names = [
        t.get("name", "").strip()
        for t in discovery.get("themes", [])
        if isinstance(t, dict) and t.get("name")
    ]
    if not theme_names:
        return "Synthesis failed: no themes discovered."

    generated = generate_ideas_by_theme(
        collection_name=collection_name,
        themes=theme_names,
        innovation_goals="",
        llm_provider=llm_provider,
        min_credibility_tier=0,
        chunks_per_theme=3,
    )
    return export_ideation_markdown(generated, title="Knowledge Synthesizer")


def export_ideation_markdown(payload: Dict[str, Any], title: str = "Ideation Results") -> str:
    lines = [f"# {title}", ""]
    if payload.get("status") != "success":
        lines.append(f"Error: {payload.get('error', 'Unknown error')}")
        return "\n".join(lines)

    if "theme_groups" in payload:
        for group in payload.get("theme_groups", []):
            lines.append(f"## {group.get('theme', 'Theme')}")
            for idea in group.get("ideas", []):
                lines.append(f"- **{idea.get('title', 'Idea')}**")
                lines.append(f"  - {idea.get('description', '')}")
                lines.append(f"  - Impact: {idea.get('impact', '')}")
            lines.append("")
        return "\n".join(lines)

    if "develop" in payload:
        lines.append("## Discovery Themes")
        for theme in payload.get("discovery", {}).get("themes", []):
            lines.append(f"- {theme}")
        lines.append("")
        lines.append("## Ideas")
        for group in payload.get("develop", {}).get("idea_groups", []):
            lines.append(f"### {group.get('problem_statement', 'Problem')}")
            for idea in group.get("ideas", []):
                lines.append(f"- **{idea.get('title', 'Idea')}**: {idea.get('description', '')}")
            lines.append("")
        deliver = payload.get("deliver", {})
        lines.append("## Delivery Summary")
        lines.append(deliver.get("summary", ""))
        lines.append("")
        lines.append(deliver.get("recommendation", ""))
    return "\n".join(lines)
