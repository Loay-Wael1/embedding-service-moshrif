from __future__ import annotations

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

    with pytest.raises(LLMConfigurationError, match="LLM_API_KEY or GEMINI_API_KEY"):
        client.chat_completion(messages=[{"role": "user", "content": "hello"}])


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
