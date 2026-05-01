"""Tests for LLM JSON parsing, validation, and JSON mode support."""
from __future__ import annotations

import json

import pytest

from app.llm import LLMConfigurationError, OpenAICompatibleLLMClient, parse_llm_json, clean_generated_text
from app.llm.openai_compatible_client import validate_answer_payload


# ---------------------------------------------------------------------------
# 1. Valid JSON with answer_from_sources and final_answer
# ---------------------------------------------------------------------------

def test_parse_llm_json_valid_full_payload():
    raw = json.dumps({
        "answer_from_sources": "المادة 1 من قانون العمل.",
        "final_answer": "جواب مباشر.",
        "warning": None,
    })
    result = parse_llm_json(raw)
    assert result["answer_from_sources"] == "المادة 1 من قانون العمل."
    assert result["final_answer"] == "جواب مباشر."
    assert result["warning"] is None


def test_validate_valid_full_payload():
    parsed = {
        "answer_from_sources": "source text",
        "final_answer": "answer text",
        "warning": None,
    }
    out = validate_answer_payload(parsed)
    assert out["schema_error"] is None
    assert out["payload"]["answer_from_sources"] == "source text"
    assert out["payload"]["final_answer"] == "answer text"


# ---------------------------------------------------------------------------
# 2. JSON with only final_answer — mode-aware behaviour
# ---------------------------------------------------------------------------

def test_validate_only_final_answer_grounded_requires_sources():
    """Grounded mode: missing answer_from_sources IS a schema error."""
    parsed = {"final_answer": "جواب مباشر عن عقد العمل."}
    out = validate_answer_payload(parsed, answer_mode="grounded")
    assert out["schema_error"] == "missing_required_fields: answer_from_sources"
    assert out["payload"]["answer_from_sources"] == "جواب مباشر عن عقد العمل."


def test_validate_only_final_answer_external_assisted_no_schema_error():
    """external_assisted mode: missing answer_from_sources is NOT a schema error."""
    parsed = {"final_answer": "شرح عام عن الحضانة."}
    out = validate_answer_payload(parsed, answer_mode="external_assisted")
    assert out["schema_error"] is None
    assert out["payload"]["answer_from_sources"] is None
    assert out["payload"]["final_answer"] == "شرح عام عن الحضانة."


def test_validate_only_answer_from_sources_normalizes():
    parsed = {"answer_from_sources": "المادة 1."}
    out = validate_answer_payload(parsed)
    assert out["schema_error"] == "missing_required_fields: final_answer"
    assert out["payload"]["final_answer"] == "المادة 1."


# ---------------------------------------------------------------------------
# 3. Invalid JSON with literal newline — diagnostics
# ---------------------------------------------------------------------------

def test_parse_llm_json_literal_newline_inside_string():
    raw = '{"final_answer": "سطر أول\nسطر ثاني", "answer_from_sources": "مصدر"}'
    result = parse_llm_json(raw)
    assert "سطر أول" in result["final_answer"]
    assert "سطر ثاني" in result["final_answer"]


def test_parse_llm_json_literal_newline_diagnostics_on_total_failure():
    raw = "this is not json at all {broken"
    with pytest.raises(ValueError, match="LLM did not return valid JSON") as exc_info:
        parse_llm_json(raw)
    error_msg = str(exc_info.value)
    assert "raw_response_repr_preview=" in error_msg
    assert "error_type=" in error_msg


# ---------------------------------------------------------------------------
# 4. Fenced JSON still works
# ---------------------------------------------------------------------------

def test_parse_llm_json_fenced_json():
    raw = '```json\n{"answer_from_sources": "source", "final_answer": "answer"}\n```'
    result = parse_llm_json(raw)
    assert result["answer_from_sources"] == "source"
    assert result["final_answer"] == "answer"


# ---------------------------------------------------------------------------
# 5. Extra text around JSON still works
# ---------------------------------------------------------------------------

def test_parse_llm_json_extra_text_before_and_after():
    raw = (
        'Here is the response:\n'
        '{"answer_from_sources": "المادة 1.", "final_answer": "جواب."}\n'
        'I hope this helps!'
    )
    result = parse_llm_json(raw)
    assert result["answer_from_sources"] == "المادة 1."
    assert result["final_answer"] == "جواب."


# ---------------------------------------------------------------------------
# 6. Grounded labor-law partial JSON — uses LLM answer, not fallback
# ---------------------------------------------------------------------------

def test_grounded_labor_law_partial_gemini_no_fallback():
    from app.answering import LegalAnswerService
    from app.llm import LLMCompletion

    class PartialGeminiLLM:
        model = "gemini-2.5-flash"
        provider_name = "gemini"
        web_search_enabled = False

        def chat_completion(self, *, messages, temperature=0.0, max_tokens=None):
            payload = {"final_answer": "ينظم قانون العمل حقوق العامل."}
            return LLMCompletion(
                content=json.dumps(payload, ensure_ascii=False),
                model=self.model,
                provider=self.provider_name,
                usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
                raw_response=None,
            )

    retriever = _FakeRetriever(_grounded_retrieval_result())
    service = LegalAnswerService(retriever=retriever, llm_client=PartialGeminiLLM())
    response = service.answer("ما هي أحكام عقد العمل الفردي؟")

    assert response.answer_mode == "grounded"
    assert response.llm.succeeded is True
    assert response.llm.parse_error is None
    assert response.llm.schema_error == "missing_required_fields: answer_from_sources"
    assert "ينظم قانون العمل حقوق العامل" in response.final_answer
    assert "تعذر توليد" not in response.final_answer


# ---------------------------------------------------------------------------
# 7. external_assisted — only final_answer should NOT produce schema_error
# ---------------------------------------------------------------------------

def test_external_assisted_no_schema_error():
    from app.answering import LegalAnswerService
    from app.llm import LLMCompletion

    class ExternalLLM:
        model = "gemini-2.5-flash"
        provider_name = "gemini"
        web_search_enabled = False

        def chat_completion(self, *, messages, temperature=0.0, max_tokens=None):
            payload = {"final_answer": "الحضانة في القانون المصري تنظمها قوانين الأحوال الشخصية."}
            return LLMCompletion(
                content=json.dumps(payload, ensure_ascii=False),
                model=self.model,
                provider=self.provider_name,
                usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
                raw_response=None,
            )

    # This query will be shortcircuited — no retriever needed.
    service = LegalAnswerService(retriever=None, llm_client=ExternalLLM())
    response = service.answer("ما هي أحكام الحضانة؟")

    assert response.answer_mode == "external_assisted"
    assert response.llm.succeeded is True
    assert response.llm.parse_error is None
    assert response.llm.schema_error is None  # NOT a schema error for external_assisted
    assert response.is_out_of_internal_corpus is True
    assert response.is_out_of_domain is False  # Legal Egyptian question, not out-of-domain


# ---------------------------------------------------------------------------
# 8. external_assisted confident query — no LegalRetriever instantiation
# ---------------------------------------------------------------------------

def test_external_assisted_shortcircuit_no_retriever():
    """Confident personal-status query should NOT instantiate LegalRetriever."""
    from app.answering import LegalAnswerService
    from app.llm import LLMCompletion

    class SimpleLLM:
        model = "gemini-2.5-flash"
        provider_name = "gemini"
        web_search_enabled = False

        def chat_completion(self, *, messages, temperature=0.0, max_tokens=None):
            payload = {"final_answer": "شرح عام عن النفقة."}
            return LLMCompletion(
                content=json.dumps(payload, ensure_ascii=False),
                model=self.model,
                provider=self.provider_name,
                usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
                raw_response=None,
            )

    # Pass retriever=None — if service tries to access it, it would fail.
    service = LegalAnswerService(retriever=None, llm_client=SimpleLLM())
    response = service.answer("ما هي أحكام النفقة؟")

    assert response.answer_mode == "external_assisted"
    assert response.retrieval_summary.top_k_used == 0
    assert response.retrieval_summary.result_count == 0
    assert response.internal_sources == []
    assert response.sources == []


# ---------------------------------------------------------------------------
# 9. Warning should not contain technical text
# ---------------------------------------------------------------------------

def test_warning_no_technical_text():
    from app.answering import LegalAnswerService
    from app.llm import LLMCompletion

    class SimpleLLM:
        model = "gemini-2.5-flash"
        provider_name = "gemini"
        web_search_enabled = False

        def chat_completion(self, *, messages, temperature=0.0, max_tokens=None):
            payload = {"final_answer": "شرح عام."}
            return LLMCompletion(
                content=json.dumps(payload, ensure_ascii=False),
                model=self.model,
                provider=self.provider_name,
                usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
                raw_response=None,
            )

    service = LegalAnswerService(retriever=None, llm_client=SimpleLLM())
    response = service.answer("ما هي أحكام الحضانة؟")

    warning = response.warning or ""
    assert "schema_error" not in warning
    assert "partial JSON" not in warning
    assert "reused final_answer" not in warning


# ---------------------------------------------------------------------------
# 10. Grounded still requires answer_from_sources and final_answer
# ---------------------------------------------------------------------------

def test_grounded_requires_both_fields():
    out = validate_answer_payload({"final_answer": "x"}, answer_mode="grounded")
    assert out["schema_error"] is not None
    assert "answer_from_sources" in out["schema_error"]


# ---------------------------------------------------------------------------
# 11. Identity does not call LLM or retriever
# ---------------------------------------------------------------------------

def test_identity_no_retriever_no_llm():
    from app.answering import LegalAnswerService

    service = LegalAnswerService(retriever=None, llm_client=None)
    response = service.answer("اسمك إيه؟")

    assert response.answer_mode == "identity"
    assert response.llm.called is False
    assert "المستشار" in response.final_answer


# ---------------------------------------------------------------------------
# 12. clean_generated_text tests
# ---------------------------------------------------------------------------

def test_clean_repeated_arabic_letters():
    assert clean_generated_text("هههههه") == "هه"
    assert clean_generated_text("normal text") == "normal text"


def test_clean_excessive_blank_lines():
    text = "line1\n\n\n\n\nline2"
    result = clean_generated_text(text)
    assert result.count("\n") <= 3


def test_clean_repeated_spaces():
    assert clean_generated_text("كلمة    كلمة") == "كلمة كلمة"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_parse_llm_json_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        parse_llm_json("")


def test_validate_neither_field_present():
    parsed = {"warning": "something"}
    out = validate_answer_payload(parsed)
    assert "answer_from_sources" in out["schema_error"]
    assert "final_answer" in out["schema_error"]


def test_openai_compatible_client_json_mode_default():
    client = OpenAICompatibleLLMClient(
        api_key="test-key",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        model="gemini-2.5-flash",
        provider_name="gemini",
    )
    assert client.json_mode is True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeRetriever:
    def __init__(self, result: dict) -> None:
        self.result = result

    def search(self, query: str, *, top_k=None, filters=None):
        return self.result | {"query": query}


def _grounded_retrieval_result() -> dict:
    return {
        "normalized_query": "ما هي احكام عقد العمل الفردي",
        "query_analysis": {
            "out_of_domain": False,
            "suggested_domain": "labor_law",
        },
        "results": [
            {
                "id": "labor-1",
                "rerank_score": 0.91,
                "score": 0.78,
                "law_name": "قانون العمل المصري",
                "law_number": "14",
                "law_year": "2025",
                "article_number": "1",
                "title": "المادة 1 - قانون العمل المصري",
                "legal_domain": "labor_law",
                "section_level": "أحكام عامة",
                "source_url": "https://example.com/labor/1",
                "summary": "تنظيم عقد العمل الفردي وحقوق العامل.",
                "content": "ينظم قانون العمل حقوق العامل والتزامات صاحب العمل في عقد العمل الفردي.",
                "rank_explanation": ["strong_summary_overlap"],
            },
            {
                "id": "labor-2",
                "rerank_score": 0.84,
                "score": 0.70,
                "law_name": "قانون العمل المصري",
                "law_number": "14",
                "law_year": "2025",
                "article_number": "2",
                "title": "المادة 2 - قانون العمل المصري",
                "legal_domain": "labor_law",
                "section_level": "أحكام عامة",
                "source_url": "https://example.com/labor/2",
                "summary": "تعريف العامل وصاحب العمل وعقد العمل.",
                "content": "تتضمن المادة تعريفات مرتبطة بالعامل وصاحب العمل وعلاقة العمل.",
                "rank_explanation": ["strong_title_overlap"],
            },
        ],
    }
