from types import SimpleNamespace

import pytest

from cortex_engine.llm_interface import DEFAULT_LMSTUDIO_MODEL, LLMInterface
from cortex_engine.setup_manager import SetupManager, SetupStatus
from cortex_engine.system_status import (
    ModelStatus,
    PlatformInfo,
    ServiceStatus,
    SystemStatusChecker,
)


def _platform_info() -> PlatformInfo:
    return PlatformInfo(
        platform_name="Linux",
        architecture="x86_64",
        gpu_type="NVIDIA GPU",
        optimization="CUDA Acceleration",
        docker_env=False,
    )


def test_lmstudio_is_the_shared_llm_default(monkeypatch):
    captured = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="ready"))]
            )

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=FakeCompletions())
    )

    def fake_openai(**kwargs):
        captured["client"] = kwargs
        return fake_client

    monkeypatch.delenv("CORTEX_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("CORTEX_LMSTUDIO_MODEL", raising=False)
    monkeypatch.setattr("cortex_engine.llm_interface.OpenAI", fake_openai)

    llm = LLMInterface()

    assert llm.provider == "lmstudio"
    assert llm.model == DEFAULT_LMSTUDIO_MODEL
    assert llm.generate("ping") == "ready"
    assert captured["model"] == DEFAULT_LMSTUDIO_MODEL
    assert captured["extra_body"] == {"reasoning_effort": "none"}


def test_registered_lmstudio_model_is_ready_without_being_loaded(monkeypatch):
    checker = SystemStatusChecker()
    checker.llm_provider = "lmstudio"
    monkeypatch.setattr(checker, "detect_platform_info", _platform_info)
    monkeypatch.setattr(checker, "check_lmstudio_status", lambda: ServiceStatus.RUNNING)
    monkeypatch.setattr(
        checker,
        "check_ollama_status",
        lambda: (_ for _ in ()).throw(AssertionError("startup must not probe Ollama")),
    )
    monkeypatch.setattr(checker, "check_api_status", lambda: ServiceStatus.RUNNING)
    monkeypatch.setattr(
        checker,
        "get_lmstudio_models",
        lambda: [
            {
                "id": checker.default_lmstudio_model,
                "type": "llm",
                "state": "not-loaded",
            }
        ],
    )

    progress = checker.get_setup_progress()

    assert progress["setup_complete"] is True
    assert progress["lmstudio_running"] is True
    assert progress["models"] == [
        {
            "name": checker.default_lmstudio_model,
            "status": ModelStatus.AVAILABLE.value,
            "size_gb": 0.0,
            "available": True,
            "loaded": False,
        }
    ]
    assert "loads on first use" in progress["status_message"]
    assert "Downloading" not in progress["status_message"]
    assert [backend["name"] for backend in progress["backends"]] == ["LM Studio"]


def test_missing_lmstudio_model_is_not_reported_as_downloading(monkeypatch):
    checker = SystemStatusChecker()
    monkeypatch.setattr(checker, "detect_platform_info", _platform_info)
    monkeypatch.setattr(checker, "check_lmstudio_status", lambda: ServiceStatus.RUNNING)
    monkeypatch.setattr(checker, "check_ollama_status", lambda: ServiceStatus.RUNNING)
    monkeypatch.setattr(checker, "check_api_status", lambda: ServiceStatus.RUNNING)
    monkeypatch.setattr(
        checker,
        "get_lmstudio_models",
        lambda: [{"id": "another-model", "state": "loaded"}],
    )
    monkeypatch.setattr(checker, "get_installed_models", lambda: [])

    progress = checker.get_setup_progress()

    assert progress["setup_complete"] is False
    assert "not registered" in progress["status_message"]
    assert "Downloading" not in progress["status_message"]


@pytest.mark.asyncio
async def test_setup_skips_ollama_downloads_for_lmstudio(monkeypatch):
    monkeypatch.setenv("CORTEX_LLM_PROVIDER", "lmstudio")
    monkeypatch.setenv("CORTEX_LMSTUDIO_MODEL", DEFAULT_LMSTUDIO_MODEL)
    manager = object.__new__(SetupManager)

    result = await manager._step_model_installation({})

    assert result.status == SetupStatus.COMPLETED
    assert result.details["provider"] == "lmstudio"
    assert result.details["installed_models"] == []
    assert "No Ollama models will be downloaded" in result.message
