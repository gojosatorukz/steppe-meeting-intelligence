"""Tests for the LLM half of pipeline.py (structure() and model resolution).

Fixture-driven tests need a live, reachable Ollama with one of the supported
models pulled; they are skipped (not failed) when that isn't the case. The
model-resolution tests are pure unit tests against a mocked `ollama.list()`
and always run.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

import pipeline

FIXTURES_DIR = Path(__file__).parent / "fixtures"
KAZAKH_LETTERS = set("әғқңөұүі")


def _ollama_ready() -> bool:
    try:
        available = {
            m.get("model") or m.get("name")
            for m in pipeline.ollama.list().get("models", [])
        }
    except Exception:
        return False
    return any(candidate in available for candidate in pipeline.LLM_MODEL_CANDIDATES)


OLLAMA_READY = _ollama_ready()

requires_ollama = pytest.mark.skipif(
    not OLLAMA_READY, reason="Ollama unreachable at localhost:11434 or no supported model pulled"
)


def _load_fixture(name: str) -> str:
    return (FIXTURES_DIR / name).read_text(encoding="utf-8")


def _assert_valid_protocol(protocol: dict, fixture_text: str) -> None:
    assert set(pipeline.EMPTY.keys()) <= set(protocol.keys()), protocol.keys()
    assert 3 <= len(protocol["summary"]) <= 5, protocol["summary"]
    assert len(protocol["action_items"]) > 0, "expected at least one action item"
    for item in protocol["action_items"]:
        owner = item["owner"]
        assert owner == "не назначен" or owner in fixture_text, f"invented owner: {owner!r}"


@requires_ollama
def test_structure_ru_fixture():
    text = _load_fixture("transcript_ru.txt")
    t0 = time.perf_counter()
    protocol = pipeline.structure(text, language_hint="ru")
    dt = time.perf_counter() - t0
    print(f"\n[timing] RU fixture: {dt:.2f}s")

    _assert_valid_protocol(protocol, text)
    assert any("25" in d for d in protocol["decisions"]), protocol["decisions"]

    joined = json.dumps(protocol, ensure_ascii=False)
    assert any(("а" <= ch <= "я") or ("А" <= ch <= "Я") for ch in joined), "expected Cyrillic output"


@requires_ollama
def test_structure_kk_fixture():
    text = _load_fixture("transcript_kk.txt")
    t0 = time.perf_counter()
    protocol = pipeline.structure(text, language_hint="kk")
    dt = time.perf_counter() - t0
    print(f"\n[timing] KK fixture: {dt:.2f}s")

    _assert_valid_protocol(protocol, text)
    assert any("30" in d for d in protocol["decisions"]), protocol["decisions"]

    joined = json.dumps(protocol, ensure_ascii=False)
    assert any(ch.lower() in KAZAKH_LETTERS for ch in joined), "expected a Kazakh-specific letter"


def test_resolved_llm_model_prefers_first_available(monkeypatch):
    """First candidate present in ollama.list() wins; result is cached."""
    pipeline._resolved_llm_model = None
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.setattr(
        pipeline.ollama,
        "list",
        lambda: {
            "models": [
                {"model": "qwen3:4b-instruct-2507-q4_K_M"},
                {"model": "qwen2.5:7b-instruct"},
            ]
        },
    )

    assert pipeline.resolved_llm_model() == "qwen3:4b-instruct-2507-q4_K_M"

    # Cached: a second call must not re-query ollama.list() at all.
    def _boom():
        raise AssertionError("resolved_llm_model() should use the cached value")

    monkeypatch.setattr(pipeline.ollama, "list", _boom)
    assert pipeline.resolved_llm_model() == "qwen3:4b-instruct-2507-q4_K_M"

    pipeline._resolved_llm_model = None


def test_resolved_llm_model_env_override(monkeypatch):
    pipeline._resolved_llm_model = None
    monkeypatch.setenv("LLM_MODEL", "custom:model")
    assert pipeline.resolved_llm_model() == "custom:model"
    pipeline._resolved_llm_model = None


def test_resolved_llm_model_raises_when_nothing_pulled(monkeypatch):
    pipeline._resolved_llm_model = None
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.setattr(
        pipeline.ollama, "list", lambda: {"models": [{"model": "some-unrelated-model"}]}
    )

    with pytest.raises(RuntimeError, match="ollama pull"):
        pipeline.resolved_llm_model()

    pipeline._resolved_llm_model = None


# --------------------------------------------------------------------------
# _ask() think=False catch/retry — mocked ollama.chat, no live server needed.
#
# Covers fix round 1: the model must be tagged think-unsupported only once
# the no-`think` retry has actually succeeded, only a 400 naming "think" is
# treated as "this model rejects think", and everything else (a generic
# exception, or a retry that itself fails) must propagate unchanged and must
# NOT tag the model.
# --------------------------------------------------------------------------

def test_ask_retries_without_think_and_tags_model_after_retry_succeeds(monkeypatch):
    pipeline._think_unsupported.discard("fake-model")
    calls = []

    def fake_chat(**kwargs):
        calls.append(kwargs)
        if "think" in kwargs:
            raise pipeline.ollama.ResponseError('json: unknown field "think"', 400)
        return {"message": {"content": json.dumps(pipeline.EMPTY)}}

    monkeypatch.setattr(pipeline.ollama, "chat", fake_chat)

    result = pipeline._ask("hello", "fake-model")

    assert result == pipeline.EMPTY
    assert len(calls) == 2
    assert "think" in calls[0], "first call must try think=False"
    assert "think" not in calls[1], "retry must omit think entirely"
    assert "fake-model" in pipeline._think_unsupported

    pipeline._think_unsupported.discard("fake-model")


def test_ask_generic_exception_propagates_and_does_not_tag_model(monkeypatch):
    pipeline._think_unsupported.discard("fake-model")

    def fake_chat(**kwargs):
        raise TimeoutError("connection timed out")

    monkeypatch.setattr(pipeline.ollama, "chat", fake_chat)

    with pytest.raises(TimeoutError):
        pipeline._ask("hello", "fake-model")

    assert "fake-model" not in pipeline._think_unsupported


def test_ask_non_think_response_error_propagates_and_does_not_tag_model(monkeypatch):
    """A ResponseError that isn't about `think` (e.g. a 500 from an OOM) must
    not be misread as think-unsupported."""
    pipeline._think_unsupported.discard("fake-model")

    def fake_chat(**kwargs):
        raise pipeline.ollama.ResponseError("internal server error", 500)

    monkeypatch.setattr(pipeline.ollama, "chat", fake_chat)

    with pytest.raises(pipeline.ollama.ResponseError):
        pipeline._ask("hello", "fake-model")

    assert "fake-model" not in pipeline._think_unsupported


def test_ask_failed_retry_after_think_rejection_does_not_tag_model(monkeypatch):
    """If the no-think retry itself fails (transient error), the model must
    not be permanently mis-tagged as think-unsupported."""
    pipeline._think_unsupported.discard("fake-model")

    def fake_chat(**kwargs):
        if "think" in kwargs:
            raise pipeline.ollama.ResponseError('json: unknown field "think"', 400)
        raise TimeoutError("still broken on retry")

    monkeypatch.setattr(pipeline.ollama, "chat", fake_chat)

    with pytest.raises(TimeoutError):
        pipeline._ask("hello", "fake-model")

    assert "fake-model" not in pipeline._think_unsupported
