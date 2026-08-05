"""Security-policy tests for the recoverable Hermes SP4 artifacts."""

from __future__ import annotations

import importlib.util
from importlib.machinery import SourceFileLoader
from pathlib import Path
import tempfile


ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, path: Path):
    loader = SourceFileLoader(name, str(path))
    spec = importlib.util.spec_from_loader(name, loader)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def test_client_accepts_only_regular_tmp_kb_files() -> None:
    client = _load(
        "hermes_delivery_sp4_client",
        ROOT / "ops/hermes-delivery/sp4/kb-email-me",
    )
    with tempfile.NamedTemporaryFile(prefix="kb-hermes-test-", suffix=".txt", dir="/tmp") as handle:
        handle.write(b"test")
        handle.flush()
        assert client.validate_file(handle.name) == Path(handle.name)

    with tempfile.NamedTemporaryFile(prefix="not-kb-", suffix=".txt", dir="/tmp") as handle:
        handle.write(b"test")
        handle.flush()
        try:
            client.validate_file(handle.name)
        except ValueError as exc:
            assert "only regular /tmp/kb-*" in str(exc)
        else:
            raise AssertionError("non-kb filename was accepted")


def test_plugin_requires_per_call_approval_and_blocks_direct_access() -> None:
    plugin = _load(
        "hermes_delivery_approval_plugin",
        ROOT / "ops/hermes-delivery/sp4/email-delivery-approval/__init__.py",
    )
    command = "/home/paul/.local/bin/kb-email-me /tmp/kb-report.pdf"
    first = plugin._pre_tool_call("terminal", {"command": command}, "call-one")
    second = plugin._pre_tool_call("terminal", {"command": command}, "call-two")
    assert first["action"] == "approve"
    assert first["rule_key"] != second["rule_key"]

    blocked = plugin._pre_tool_call(
        "terminal",
        {"command": "curl http://100.118.92.17:7341/health"},
        "call-three",
    )
    assert blocked["action"] == "block"
    assert plugin._pre_tool_call("terminal", {"command": "date"}, "call-four") is None
