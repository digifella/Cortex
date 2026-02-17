from __future__ import annotations

from pathlib import Path

from worker.handlers.cortex_sync import _load_doc_id_map, _partition_doc_ids_by_collision


def test_load_doc_id_map_json_format(tmp_path):
    log = tmp_path / "ingested_files.log"
    log.write_text(
        '{"\/tmp\/a.md":"doc_a","\/tmp\/b.md":["doc_b"]}',
        encoding="utf-8",
    )
    parsed = _load_doc_id_map(str(log))
    assert parsed[str(Path("/tmp/a.md"))] == "doc_a"
    assert parsed[str(Path("/tmp/b.md"))] == "doc_b"


def test_load_doc_id_map_legacy_line_format(tmp_path):
    log = tmp_path / "ingested_files.log"
    log.write_text(
        "/tmp/a.md | doc_a\n"
        "/tmp/b.md | doc_b\n",
        encoding="utf-8",
    )
    parsed = _load_doc_id_map(str(log))
    assert parsed[str(Path("/tmp/a.md"))] == "doc_a"
    assert parsed[str(Path("/tmp/b.md"))] == "doc_b"


def test_partition_doc_ids_by_collision():
    normal, collisions = _partition_doc_ids_by_collision(
        ["a", "b", "a", "c", ""],
        {"b", "z"},
    )
    assert normal == ["a", "c"]
    assert collisions == ["b"]
