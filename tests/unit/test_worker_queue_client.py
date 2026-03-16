from __future__ import annotations

import json
import io
from pathlib import Path

import pytest
import requests

from worker.worker import Config, QueueClient


class FakeResponse:
    def __init__(self, *, status_code: int = 200, headers: dict | None = None, payload=None, chunks=None):
        self.status_code = status_code
        self.headers = headers or {"Content-Type": "application/json"}
        self._payload = payload if payload is not None else {}
        self._chunks = list(chunks or [])

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload

    def iter_content(self, chunk_size: int = 0):
        del chunk_size
        for chunk in self._chunks:
            yield chunk


class FakeSession:
    def __init__(self, responses: list[FakeResponse]):
        self.responses = list(responses)
        self.calls = []
        self.headers = {}

    def request(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


def make_cfg() -> Config:
    return Config(
        server_url="https://example.com/admin/queue_worker_api.php",
        secret_key="secret",
        poll_interval=15,
        worker_id="worker-test",
        worker_type="cortex_worker",
        supported_types="youtube_summarise",
        log_level="INFO",
        temp_dir=Path("/tmp"),
        heartbeat_interval=60,
        request_timeout=30,
        queue_monitor_state_path=Path("/tmp/queue_monitor_state.json"),
        cortex_api_url="http://127.0.0.1:8000",
        cortex_tunnel_url="",
        cortex_meta_sync_interval=300,
    )


def test_complete_posts_action_and_id_in_form_body(monkeypatch, tmp_path):
    session = FakeSession([FakeResponse(payload={"success": True})])
    monkeypatch.setattr("worker.worker.requests.Session", lambda: session)

    client = QueueClient(make_cfg())
    output_file = tmp_path / "result.md"
    output_file.write_text("summary", encoding="utf-8")

    client.complete(61, {"trace_id": "trace-123", "ok": True}, output_file)

    assert len(session.calls) == 1
    call = session.calls[0]
    assert call["method"] == "POST"
    assert call["url"] == "https://example.com/admin/queue_worker_api.php"
    assert call["params"] == {}
    assert call["data"]["action"] == "complete"
    assert call["data"]["id"] == "61"
    assert json.loads(call["data"]["output_data"]) == {"trace_id": "trace-123", "ok": True}
    assert call["files"]["file"][0] == "result.md"


def test_worker_checkin_posts_form_encoded_status(monkeypatch):
    session = FakeSession([FakeResponse(payload={"success": True, "server_time": "2026-03-16T12:00:00Z"})])
    monkeypatch.setattr("worker.worker.requests.Session", lambda: session)

    client = QueueClient(make_cfg())
    client.worker_checkin("processing job #349")

    assert len(session.calls) == 1
    call = session.calls[0]
    assert call["method"] == "POST"
    assert call["url"] == "https://example.com/admin/queue_worker_api.php"
    assert call["params"] == {}
    assert call["data"]["action"] == "worker_checkin"
    assert call["data"]["worker_id"] == "worker-test"
    assert call["data"]["worker_type"] == "cortex_worker"
    assert call["data"]["supported_types"] == "youtube_summarise"
    assert call["data"]["status"] == "processing job #349"


def test_complete_raises_on_non_json_response(monkeypatch):
    session = FakeSession([FakeResponse(headers={"Content-Type": "text/html"})])
    monkeypatch.setattr("worker.worker.requests.Session", lambda: session)

    client = QueueClient(make_cfg())

    with pytest.raises(RuntimeError, match="unexpected content type"):
        client.complete(61, {"ok": True}, output_file=None)


def test_complete_raises_on_json_error_payload(monkeypatch):
    session = FakeSession([FakeResponse(payload={"error": "Unknown action"})])
    monkeypatch.setattr("worker.worker.requests.Session", lambda: session)

    client = QueueClient(make_cfg())

    with pytest.raises(RuntimeError, match="Unknown action"):
        client.complete(61, {"ok": True}, output_file=None)


def test_download_input_accepts_binary_response(monkeypatch, tmp_path):
    response = requests.Response()
    response.status_code = 200
    response.headers["Content-Type"] = "application/octet-stream"
    response.raw = io.BytesIO(b"hello world")
    session = FakeSession(
        [
            response
        ]
    )
    monkeypatch.setattr("worker.worker.requests.Session", lambda: session)

    client = QueueClient(make_cfg())
    destination = tmp_path / "input.txt"

    result = client.download_input(77, destination)

    assert result == destination
    assert destination.read_text(encoding="utf-8") == "hello world"
    assert session.calls[0]["method"] == "GET"
    assert session.calls[0]["params"] == {"action": "download_input", "id": "77"}
