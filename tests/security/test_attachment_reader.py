"""attachment_reader defences against the harmless corpus in make_attachment_corpus.py.

Blocked cases never reach the model. The rest need LM Studio's Qwen and are skipped
when it is unreachable.  Run:  venv/bin/python -m pytest tests/security -q
"""
import sys
from pathlib import Path

import pytest
import requests

sys.path.insert(0, str(Path(__file__).parent))
from make_attachment_corpus import EXPECT, main as build  # noqa: E402

from cortex_engine.attachment_reader import LMS_URL, read_attachment  # noqa: E402


def _qwen_up():
    try:
        return requests.get(LMS_URL.replace("/chat/completions", "/models"), timeout=3).ok
    except requests.RequestException:
        return False


def _clamd_up():
    import subprocess
    return subprocess.run(["clamdscan", "--ping", "1"], capture_output=True).returncode == 0


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    out = tmp_path_factory.mktemp("corpus")
    build(out)
    return out


@pytest.mark.parametrize("name,want", sorted(EXPECT.items()))
def test_risk_level(corpus, name, want):
    if want != "blocked" and not _qwen_up():
        pytest.skip("LM Studio Qwen unreachable")
    if not _clamd_up():
        pytest.skip("clamd not running (sudo systemctl enable --now clamav-daemon)")
    r = read_attachment(corpus / name, name)
    assert r["risk"]["level"] == want, r["risk"]
    if want in ("blocked", "high"):
        assert not r["raw_text_included"] and not r["text"]


def test_hidden_text_not_returned(corpus):
    if not _qwen_up():
        pytest.skip("LM Studio Qwen unreachable")
    r = read_attachment(corpus / "hidden_text.pdf", "hidden_text.pdf")
    assert "approve payment" not in (r["text"] + r["summary"]).lower()


def test_links_defanged(corpus):
    if not _qwen_up():
        pytest.skip("LM Studio Qwen unreachable")
    r = read_attachment(corpus / "phish_bank.pdf", "phish_bank.pdf")
    assert r["links"] and all(link.startswith("hxxp") and "[.]" in link for link in r["links"])
    assert "https://" not in r["summary"]
