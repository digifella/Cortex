from typing import Dict, List, Tuple, Callable, Optional
from .utils.logging_utils import get_logger

logger = get_logger(__name__)

StatusCb = Optional[Callable[[str], None]]


def _log(cb: StatusCb, msg: str):
    if cb:
        cb(msg)
    logger.info(msg)


def agent_foundational_query_crafter(topic: str, status_callback: StatusCb = None) -> Dict:
    _log(status_callback, f"Crafting foundational queries for: {topic}")
    return {"queries": [f"site:scholar.google.com {topic}", f"best practices {topic}"]}


def agent_exploratory_query_crafter(topic: str, status_callback: StatusCb = None) -> Dict:
    _log(status_callback, f"Crafting exploratory queries for: {topic}")
    return {"queries": [f"overview {topic}", f"case studies {topic}"]}


def step1_fetch_foundational_sources(queries: Dict, status_callback: StatusCb = None) -> List[Dict]:
    _log(status_callback, "Fetching foundational sources (stub)")
    return [
        {"title": "Foundational Paper A", "url": "https://example.com/a", "source_type": "web", "citations": 42},
        {"title": "Foundational Paper B", "url": "https://example.com/b", "source_type": "web", "citations": 17},
    ]


def step2_fetch_exploratory_sources(queries: Dict, status_callback: StatusCb = None) -> List[Dict]:
    _log(status_callback, "Fetching exploratory sources (stub)")
    return [
        {"title": "Exploratory Article X", "url": "https://example.com/x", "source_type": "web"},
        {"title": "Exploratory Article Y", "url": "https://example.com/y", "source_type": "web"},
    ]


def agent_thematic_analyser(sources: List[Dict], status_callback: StatusCb = None) -> List[str]:
    _log(status_callback, "Analysing themes (stub)")
    return ["Strategy", "Implementation", "Risks"]


def step3_go_deeper(themes: List[str], status_callback: StatusCb = None) -> Dict:
    _log(status_callback, "Going deeper on selected themes (stub)")
    return {"notes": {t: f"Key insights for {t}" for t in themes}}


def build_context_from_sources(sources: List[Dict]) -> str:
    return "\n".join(f"- {s.get('title','(untitled)')} ({s.get('url','')})" for s in sources)


def step4_run_synthesis(context: str, status_callback: StatusCb = None) -> Tuple[str, Dict]:
    _log(status_callback, "Running initial synthesis (stub)")
    note = f"Synthesis Summary\n\n{context}\n\nConclusions: ..."
    concept_map = {"nodes": ["Strategy", "Implementation", "Risks"], "edges": [("Strategy","Implementation")]} 
    return note, concept_map


def step5_run_deep_synthesis(context: str, status_callback: StatusCb = None) -> str:
    _log(status_callback, "Running deep synthesis (stub)")
    return f"Deep Research Note\n\n{context}\n\nDetailed findings..."


def save_outputs_to_custom_dir(output_dir: str, note_content: str, map_data: Dict) -> Tuple[str, str]:
    from pathlib import Path
    import json
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    note_path = out / "research_note.md"
    map_path = out / "concept_map.json"
    note_path.write_text(note_content, encoding="utf-8")
    map_path.write_text(json.dumps(map_data, indent=2), encoding="utf-8")
    return str(note_path), str(map_path)

