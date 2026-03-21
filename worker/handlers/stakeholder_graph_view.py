from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from cortex_engine.handoff_contract import validate_stakeholder_graph_view_input
from cortex_engine.stakeholder_signal_store import StakeholderSignalStore


def handle(
    input_path: Optional[Path],
    input_data: dict,
    job: dict,
    progress_cb: Optional[Callable[[float, str, Optional[str]], None]] = None,
    is_cancelled_cb: Optional[Callable[[], bool]] = None,
) -> dict:
    del input_path, job

    payload = validate_stakeholder_graph_view_input(input_data or {})
    if progress_cb:
        progress_cb(15, "Building stakeholder graph view", "collect")
    if is_cancelled_cb and is_cancelled_cb():
        raise RuntimeError("Cancelled before graph view generation")

    store = StakeholderSignalStore()
    graph_view = store.build_graph_view(
        org_name=payload["org_name"],
        view_mode=payload.get("view_mode", "watch_network"),
        profile_keys=payload.get("profile_keys") or [],
        child_profile_keys=payload.get("child_profile_keys") or [],
        focus_profile_key=payload.get("focus_profile_key", ""),
        focus_org_name=payload.get("focus_org_name", ""),
        since_ts=payload.get("since_ts", ""),
        max_hops=int(payload.get("max_hops", 2)),
        max_nodes=int(payload.get("max_nodes", 100)),
        max_edges=int(payload.get("max_edges", 200)),
        include_signals=bool(payload.get("include_signals", True)),
        include_sources=bool(payload.get("include_sources", False)),
        include_lab_members=bool(payload.get("include_lab_members", True)),
        include_alumni=bool(payload.get("include_alumni", True)),
        include_unwatched_bridges=bool(payload.get("include_unwatched_bridges", False)),
        edge_types=payload.get("edge_types") or [],
        min_edge_weight=float(payload.get("min_edge_weight", 0.2)),
        min_confidence=float(payload.get("min_confidence", 5.0)),
        layout_hint=payload.get("layout_hint", "force"),
        top_k_paths=int(payload.get("top_k_paths", 5)),
    )

    if progress_cb:
        progress_cb(
            100,
            f"Generated graph view with {graph_view['node_count']} nodes and {graph_view['edge_count']} edges",
            "done",
        )

    return {
        "output_data": {
            "status": "generated",
            "graph_id": graph_view["graph_id"],
            "org_name": graph_view["org_name"],
            "view_mode": graph_view["view_mode"],
            "node_count": graph_view["node_count"],
            "edge_count": graph_view["edge_count"],
            "generated_at": graph_view["generated_at"],
            "focus_node_ids": graph_view.get("focus_node_ids", []),
            "has_paths": graph_view.get("has_paths", False),
            "has_signal_overlay": graph_view.get("has_signal_overlay", False),
            "summary": graph_view.get("summary", {}),
            "output_path": graph_view["output_path"],
        },
        "output_file": Path(graph_view["output_path"]),
    }
