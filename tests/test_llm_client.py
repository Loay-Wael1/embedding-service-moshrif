from __future__ import annotations

import importlib

import pytest

from app.llm import LLMConfigurationError, OpenAICompatibleLLMClient, parse_llm_json


def test_openai_compatible_client_uses_gemini_defaults_when_configured_explicitly():
    client = OpenAICompatibleLLMClient(
        api_key="test-key",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        model="gemini-2.5-flash",
        provider_name="gemini",
    )

    assert client.provider_name == "gemini"
    assert client.base_url == "https://generativelanguage.googleapis.com/v1beta/openai"
    assert client.model == "gemini-2.5-flash"


def test_openai_compatible_client_requires_api_key_before_request():
    client = OpenAICompatibleLLMClient(
        api_key="",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        model="gemini-2.5-flash",
        provider_name="gemini",
    )

    with pytest.raises(LLMConfigurationError, match="API key is not configured"):
        client.chat_completion(messages=[{"role": "user", "content": "hello"}])


def test_fallback_provider_resolution_uses_groq_aliases(monkeypatch):
    import app.settings as settings_module

    monkeypatch.delenv("LLM_FALLBACK_API_KEY", raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "groq-key")
    try:
        reloaded = importlib.reload(settings_module)
        cfg = reloaded.Settings()
        assert cfg.llm_fallback_provider_name == "groq"
        assert cfg.llm_fallback_api_key == "groq-key"
        assert cfg.llm_fallback_base_url == "https://api.groq.com/openai/v1"
        assert cfg.llm_fallback_model == "llama-3.3-70b-versatile"
    finally:
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        importlib.reload(settings_module)


def test_fallback_provider_resolution_generic_key_overrides_groq(monkeypatch):
    import app.settings as settings_module

    monkeypatch.setenv("GROQ_API_KEY", "groq-key")
    monkeypatch.setenv("LLM_FALLBACK_API_KEY", "fallback-key")
    monkeypatch.setenv("LLM_FALLBACK_PROVIDER_NAME", "groq")
    try:
        reloaded = importlib.reload(settings_module)
        cfg = reloaded.Settings()
        assert cfg.llm_fallback_api_key == "fallback-key"
        assert cfg.llm_fallback_base_url == "https://api.groq.com/openai/v1"
        assert cfg.llm_fallback_model == "llama-3.3-70b-versatile"
    finally:
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        monkeypatch.delenv("LLM_FALLBACK_API_KEY", raising=False)
        monkeypatch.delenv("LLM_FALLBACK_PROVIDER_NAME", raising=False)
        importlib.reload(settings_module)


def test_parse_llm_json_accepts_plain_json():
    payload = parse_llm_json('{"answer_from_sources":"source","final_answer":"answer"}')

    assert payload["answer_from_sources"] == "source"
    assert payload["final_answer"] == "answer"


def test_parse_llm_json_accepts_fenced_json():
    payload = parse_llm_json(
        """
        ```json
        {"answer_from_sources":"source","final_answer":"answer"}
        ```
        """
    )

    assert payload["answer_from_sources"] == "source"
    assert payload["final_answer"] == "answer"


def test_parse_llm_json_extracts_object_from_surrounding_text():
    payload = parse_llm_json(
        'Here is the JSON:\n{"answer_from_sources":"source","final_answer":"answer"}\nDone.'
    )

    assert payload["answer_from_sources"] == "source"
    assert payload["final_answer"] == "answer"
