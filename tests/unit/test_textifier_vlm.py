from cortex_engine.textifier import DocumentTextifier


class _FakeModel:
    def __init__(self, name: str):
        self.name = name


class _FakeListResponse:
    def __init__(self, names):
        self.models = [_FakeModel(name) for name in names]


class _FakeClient:
    def __init__(self, names):
        self._names = names
        self.show_calls = []

    def list(self):
        return _FakeListResponse(self._names)

    def show(self, model):
        self.show_calls.append(model)
        raise AssertionError("show() should not be used when list() is available")


def test_ollama_model_available_uses_list_cache():
    textifier = DocumentTextifier()
    client = _FakeClient(["qwen3-vl:8b", "mistral:latest"])

    assert textifier._ollama_model_available(client, "qwen3-vl:8b") is True
    assert textifier._ollama_model_available(client, "llava:7b") is False
    assert client.show_calls == []


def test_init_vlm_falls_back_to_llava_latest(monkeypatch):
    textifier = DocumentTextifier()
    client = _FakeClient(["llava:latest"])

    class _FakeOllamaModule:
        @staticmethod
        def Client(timeout=120):
            return client

    import sys

    monkeypatch.setitem(sys.modules, "ollama", _FakeOllamaModule())
    textifier._init_vlm()

    assert textifier._vlm_client is client
    assert textifier._vlm_model == "llava:latest"
