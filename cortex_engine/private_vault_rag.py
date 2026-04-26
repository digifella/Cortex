"""Private vault RAG helpers for Cortex Streamlit.

This module deliberately keeps private vault search local to the workstation:
it reuses the NemoClaw vault Chroma index and builds a lightweight Obsidian
graph from markdown links, tags, and folders. It does not touch the public wiki
publish path.
"""

from __future__ import annotations

import datetime as dt
import glob
import json
import os
import pickle
import re
import sqlite3
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

import networkx as nx

HOME = Path.home()

for site_packages in glob.glob(str(HOME / "venvs" / "vault-rag" / "lib" / "python3*" / "site-packages")):
    if site_packages not in sys.path:
        sys.path.insert(0, site_packages)


PRIVATE_VAULT = Path("/mnt/c/Users/paul/OneDrive - VentraIP Australia/Vault_OneDrive")
PUBLIC_VAULT = Path("/mnt/c/Users/paul/Documents/AI-Vault")
VAULT_DB_PATH = HOME / "vault-rag-db"
PRIVATE_GRAPH_PATH = VAULT_DB_PATH / "private_vault_graph.gpickle"
PUBLIC_GRAPH_PATH = VAULT_DB_PATH / "public_vault_graph.gpickle"
VAULT_QUERY_PYTHON = HOME / "venvs" / "vault-rag" / "bin" / "python3"
VAULT_QUERY_SCRIPT = HOME / "nemoclaw-vault-query.py"
VAULT_INDEXER_SCRIPT = HOME / "nemoclaw-vault-indexer.py"

VAULTS = {
    "private": {
        "root": PRIVATE_VAULT,
        "graph_path": PRIVATE_GRAPH_PATH,
        "collection": "vault_private",
        "obsidian_vault": "Vault_OneDrive",
        "display": "Private",
    },
    "public": {
        "root": PUBLIC_VAULT,
        "graph_path": PUBLIC_GRAPH_PATH,
        "collection": "vault_public",
        "obsidian_vault": "AI-Vault",
        "display": "Public",
    },
}

WIKILINK_RE = re.compile(r"\[\[([^\]|#]+)(?:#[^\]|]+)?(?:\|[^\]]+)?\]\]")
TAG_RE = re.compile(r"(?<!\w)#([A-Za-z0-9_/\-]+)")
FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?", re.DOTALL)


@dataclass
class VaultSearchResult:
    score: float
    source_file: str
    vault: str
    title: str
    snippet: str
    graph_context: list[str]


@dataclass
class GraphNeighbour:
    node_id: str
    node_type: str
    title: str
    relationship: str
    weight: float
    source_file: str | None = None


@dataclass
class VaultNote:
    source_file: str
    abs_path: str
    title: str
    metadata: dict[str, str]
    body: str
    wikilinks: list[str]
    tags: list[str]
    obsidian_uri: str
    neighbours: list[GraphNeighbour]
    related_documents: list[GraphNeighbour]


def _normalise_title(value: str) -> str:
    cleaned = re.sub(r"[_\-]+", " ", value).strip().lower()
    return re.sub(r"\s+", " ", cleaned)


def _parse_frontmatter(content: str) -> tuple[dict[str, str], str]:
    match = FRONTMATTER_RE.match(content)
    if not match:
        return {}, content
    metadata: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            metadata[key.strip().lower()] = value.strip().strip("'\"")
    return metadata, content[match.end() :]


def markdown_files(vault_path: Path = PRIVATE_VAULT) -> list[Path]:
    pattern = str(vault_path / "**" / "*.md")
    files = [Path(p) for p in glob.glob(pattern, recursive=True)]
    return sorted(p for p in files if "/.git/" not in str(p) and "/.obsidian/" not in str(p))


def _vault_key(vault: str) -> str:
    if vault in {"wiki", "public"}:
        return "public"
    if vault == "private":
        return "private"
    raise ValueError(f"Unsupported vault: {vault}")


def _vault_root(vault: str) -> Path:
    return VAULTS[_vault_key(vault)]["root"]


def _vault_graph_path(vault: str) -> Path:
    return VAULTS[_vault_key(vault)]["graph_path"]


def build_vault_graph(vault: str = "private") -> dict[str, Any]:
    """Build a local graph from Obsidian markdown files in one vault."""
    vault_key = _vault_key(vault)
    vault_path = _vault_root(vault_key)
    graph_path = _vault_graph_path(vault_key)
    graph = nx.Graph()
    files = markdown_files(vault_path)
    title_to_rel: dict[str, str] = {}

    for path in files:
        rel = path.relative_to(vault_path).as_posix()
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        metadata, body = _parse_frontmatter(content)
        title = metadata.get("title") or path.stem.replace("-", " ")
        title_to_rel[_normalise_title(path.stem)] = rel
        title_to_rel[_normalise_title(title)] = rel
        graph.add_node(
            rel,
            node_type="document",
            title=title,
            path=str(path),
            folder=path.parent.relative_to(vault_path).as_posix(),
            mtime=path.stat().st_mtime,
        )

    unresolved_links = 0
    tag_edges = 0
    folder_edges = 0
    link_edges = 0

    for path in files:
        rel = path.relative_to(vault_path).as_posix()
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        metadata, body = _parse_frontmatter(content)

        folder = path.parent.relative_to(vault_path).as_posix()
        folder_node = f"folder:{folder or 'root'}"
        graph.add_node(folder_node, node_type="folder", title=folder or "root")
        graph.add_edge(rel, folder_node, relationship="in_folder", weight=0.35)
        folder_edges += 1

        tag_blob = " ".join([metadata.get("tags", ""), body[:4000]])
        for tag in sorted(set(TAG_RE.findall(tag_blob))):
            tag_node = f"tag:{tag.lower()}"
            graph.add_node(tag_node, node_type="tag", title=tag)
            graph.add_edge(rel, tag_node, relationship="tagged", weight=0.45)
            tag_edges += 1

        for raw_link in WIKILINK_RE.findall(body):
            target = title_to_rel.get(_normalise_title(raw_link))
            if target:
                graph.add_edge(rel, target, relationship="links_to", weight=1.0)
                link_edges += 1
            else:
                concept_node = f"concept:{_normalise_title(raw_link)}"
                graph.add_node(concept_node, node_type="concept", title=raw_link.strip())
                graph.add_edge(rel, concept_node, relationship="mentions", weight=0.25)
                unresolved_links += 1

    graph_path.parent.mkdir(parents=True, exist_ok=True)
    with graph_path.open("wb") as handle:
        pickle.dump(graph, handle)

    return {
        "graph_path": str(graph_path),
        "vault": vault_key,
        "documents": len(files),
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "link_edges": link_edges,
        "tag_edges": tag_edges,
        "folder_edges": folder_edges,
        "unresolved_links": unresolved_links,
        "built_at": dt.datetime.now().isoformat(timespec="seconds"),
    }


def build_private_graph(vault_path: Path = PRIVATE_VAULT, graph_path: Path = PRIVATE_GRAPH_PATH) -> dict[str, Any]:
    """Backward-compatible private graph builder."""
    if vault_path != PRIVATE_VAULT or graph_path != PRIVATE_GRAPH_PATH:
        # Keep the old direct-call behaviour available for tests/manual use.
        graph = nx.Graph()
        files = markdown_files(vault_path)
        for path in files:
            rel = path.relative_to(vault_path).as_posix()
            graph.add_node(rel, node_type="document", title=path.stem.replace("-", " "))
        graph_path.parent.mkdir(parents=True, exist_ok=True)
        with graph_path.open("wb") as handle:
            pickle.dump(graph, handle)
        return {"graph_path": str(graph_path), "vault": "custom", "documents": len(files), "nodes": graph.number_of_nodes(), "edges": graph.number_of_edges()}
    return build_vault_graph("private")


def load_vault_graph(vault: str = "private") -> nx.Graph:
    graph_path = _vault_graph_path(vault)
    if not graph_path.exists():
        return nx.Graph()
    with graph_path.open("rb") as handle:
        return pickle.load(handle)


def load_private_graph(graph_path: Path = PRIVATE_GRAPH_PATH) -> nx.Graph:
    if graph_path != PRIVATE_GRAPH_PATH:
        if not graph_path.exists():
            return nx.Graph()
        with graph_path.open("rb") as handle:
            return pickle.load(handle)
    return load_vault_graph("private")


def vault_graph_stats(vault: str = "private") -> dict[str, Any]:
    vault_key = _vault_key(vault)
    graph_path = _vault_graph_path(vault_key)
    graph = load_vault_graph(vault_key)
    if graph.number_of_nodes() == 0:
        return {
            "exists": graph_path.exists(),
            "vault": vault_key,
            "graph_path": str(graph_path),
            "documents": 0,
            "nodes": 0,
            "edges": 0,
            "mtime": None,
        }
    documents = sum(1 for _, attrs in graph.nodes(data=True) if attrs.get("node_type") == "document")
    mtime = dt.datetime.fromtimestamp(graph_path.stat().st_mtime).isoformat(timespec="seconds")
    return {
        "exists": True,
        "vault": vault_key,
        "graph_path": str(graph_path),
        "documents": documents,
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "mtime": mtime,
    }


def private_graph_stats() -> dict[str, Any]:
    return vault_graph_stats("private")


def vault_index_stats(vault: str = "private") -> dict[str, Any]:
    vault_key = _vault_key(vault)
    try:
        import chromadb

        client = chromadb.PersistentClient(path=str(VAULT_DB_PATH))
        collection = client.get_collection(VAULTS[vault_key]["collection"])
        return {"exists": True, "vault": vault_key, "chunks": collection.count(), "path": str(VAULT_DB_PATH)}
    except Exception as exc:
        fallback = _sqlite_collection_count(VAULTS[vault_key]["collection"])
        if fallback is not None:
            return {
                "exists": True,
                "vault": vault_key,
                "chunks": fallback,
                "path": str(VAULT_DB_PATH),
                "warning": f"Chroma count unavailable; using SQLite fallback: {exc}",
            }
        return {"exists": False, "vault": vault_key, "chunks": 0, "path": str(VAULT_DB_PATH), "error": str(exc)}


def _sqlite_collection_count(collection_name: str) -> int | None:
    sqlite_path = VAULT_DB_PATH / "chroma.sqlite3"
    if not sqlite_path.exists():
        return None
    try:
        with sqlite3.connect(sqlite_path) as con:
            row = con.execute(
                """
                select count(e.id)
                from collections c
                join segments s on s.collection = c.id and s.scope = 'METADATA'
                join embeddings e on e.segment_id = s.id
                where c.name = ?
                """,
                (collection_name,),
            ).fetchone()
        return int(row[0]) if row else None
    except Exception:
        return None


def private_index_stats() -> dict[str, Any]:
    return vault_index_stats("private")


def related_context(source_file: str, limit: int = 6, vault: str = "private") -> list[str]:
    return [f"{n.node_type}: {n.title}" for n in graph_neighbours(source_file, limit=limit, vault=vault)]


def graph_neighbours(source_file: str, limit: int = 12, vault: str = "private") -> list[GraphNeighbour]:
    graph = load_vault_graph(vault)
    if source_file not in graph:
        return []
    neighbours: list[GraphNeighbour] = []
    for neighbour in graph.neighbors(source_file):
        attrs = graph.nodes[neighbour]
        edge = graph.get_edge_data(source_file, neighbour, default={})
        weight = float(edge.get("weight", 0.1))
        label = attrs.get("title") or neighbour
        node_type = attrs.get("node_type", "node")
        neighbours.append(
            GraphNeighbour(
                node_id=neighbour,
                node_type=node_type,
                title=label,
                relationship=edge.get("relationship", "related_to"),
                weight=weight,
                source_file=neighbour if node_type == "document" else None,
            )
        )
    neighbours.sort(key=lambda item: item.weight, reverse=True)
    return neighbours[:limit]


def related_documents(source_file: str, limit: int = 10, vault: str = "private") -> list[GraphNeighbour]:
    graph = load_vault_graph(vault)
    if source_file not in graph:
        return []

    scored: dict[str, tuple[float, str]] = {}
    for connector in graph.neighbors(source_file):
        connector_attrs = graph.nodes[connector]
        connector_type = connector_attrs.get("node_type", "node")
        if connector_type == "document":
            edge = graph.get_edge_data(source_file, connector, default={})
            scored[connector] = (float(edge.get("weight", 1.0)), "direct link")
            continue

        connector_title = connector_attrs.get("title") or connector
        first_edge = graph.get_edge_data(source_file, connector, default={})
        first_weight = float(first_edge.get("weight", 0.1))
        for candidate in graph.neighbors(connector):
            if candidate == source_file:
                continue
            candidate_attrs = graph.nodes[candidate]
            if candidate_attrs.get("node_type") != "document":
                continue
            second_edge = graph.get_edge_data(connector, candidate, default={})
            score = first_weight * float(second_edge.get("weight", 0.1))
            previous = scored.get(candidate)
            if previous is None or score > previous[0]:
                scored[candidate] = (score, f"via {connector_type}: {connector_title}")

    related: list[GraphNeighbour] = []
    for candidate, (score, relationship) in scored.items():
        attrs = graph.nodes[candidate]
        related.append(
            GraphNeighbour(
                node_id=candidate,
                node_type="document",
                title=attrs.get("title") or candidate,
                relationship=relationship,
                weight=score,
                source_file=candidate,
            )
        )
    related.sort(key=lambda item: item.weight, reverse=True)
    return related[:limit]


def _obsidian_uri(source_file: str, vault: str = "private") -> str:
    vault_name = VAULTS[_vault_key(vault)]["obsidian_vault"]
    return f"obsidian://open?vault={quote(vault_name)}&file=" + quote(source_file)


def load_vault_note(source_file: str, vault: str = "private") -> VaultNote:
    vault_key = _vault_key(vault)
    if not source_file or source_file.startswith("/") or ".." in Path(source_file).parts:
        raise ValueError("Invalid vault note path")

    path = _vault_root(vault_key) / source_file
    if not path.exists():
        raise FileNotFoundError(source_file)

    content = path.read_text(encoding="utf-8", errors="replace")
    metadata, body = _parse_frontmatter(content)
    title = metadata.get("title") or path.stem.replace("-", " ")
    wikilinks = sorted(set(link.strip() for link in WIKILINK_RE.findall(body)))
    tag_blob = " ".join([metadata.get("tags", ""), body[:8000]])
    tags = sorted(set(TAG_RE.findall(tag_blob)))
    return VaultNote(
        source_file=source_file,
        abs_path=str(path),
        title=title,
        metadata=metadata,
        body=body.strip(),
        wikilinks=wikilinks,
        tags=tags,
        obsidian_uri=_obsidian_uri(source_file, vault_key),
        neighbours=graph_neighbours(source_file, vault=vault_key),
        related_documents=related_documents(source_file, vault=vault_key),
    )


def load_private_note(source_file: str) -> VaultNote:
    return load_vault_note(source_file, "private")


def markdown_for_streamlit(body: str) -> str:
    """Render Obsidian wikilinks as readable markdown labels in Streamlit."""
    def replace(match: re.Match[str]) -> str:
        raw = match.group(0)
        target = match.group(1).strip()
        alias_match = re.search(r"\|([^\]]+)\]\]$", raw)
        label = alias_match.group(1).strip() if alias_match else target
        return f"**{label}**"

    return WIKILINK_RE.sub(replace, body)


def search_vault(query: str, target: str = "private", mode: str = "rapid", timeout: int = 60) -> dict[str, Any]:
    if target not in {"public", "private", "both"}:
        raise ValueError(f"Unsupported search target: {target}")
    command = [
        str(VAULT_QUERY_PYTHON),
        str(VAULT_QUERY_SCRIPT),
        "--target",
        target,
        "--mode",
        mode,
        "--json",
        query,
    ]
    env = {
        **os.environ,
        "HF_HOME": "/mnt/f/hf-home",
        "TOKENIZERS_PARALLELISM": "false",
        "VAULT_QUERY_MODE": mode,
    }
    completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout, env=env)
    if completed.returncode != 0:
        raise RuntimeError((completed.stderr or completed.stdout or "vault query failed").strip())
    payload = json.loads(completed.stdout)
    results = []
    for item in payload.get("results", []):
        source_file = item.get("source_file", "")
        result_vault = _vault_key(item.get("vault", target if target != "both" else "private"))
        results.append(
            VaultSearchResult(
                score=float(item.get("score", 0.0)),
                source_file=source_file,
                vault=result_vault,
                title=item.get("title") or source_file,
                snippet=item.get("snippet", ""),
                graph_context=related_context(source_file, vault=result_vault),
            )
        )
    payload["results"] = results
    return payload


def search_private_vault(query: str, mode: str = "rapid", timeout: int = 60) -> dict[str, Any]:
    return search_vault(query, "private", mode, timeout)


def run_vault_indexer(vault: str = "private", full: bool = False, timeout: int = 1800) -> subprocess.CompletedProcess[str]:
    vault_key = _vault_key(vault)
    command = [str(VAULT_QUERY_PYTHON), str(VAULT_INDEXER_SCRIPT)]
    command.append("--private-only" if vault_key == "private" else "--public-only")
    if full:
        command.append("--full")
    env = {
        **os.environ,
        "HF_HOME": "/mnt/f/hf-home",
        "TOKENIZERS_PARALLELISM": "false",
    }
    return subprocess.run(command, capture_output=True, text=True, timeout=timeout, env=env)


def run_private_indexer(full: bool = False, timeout: int = 1800) -> subprocess.CompletedProcess[str]:
    return run_vault_indexer("private", full, timeout)
