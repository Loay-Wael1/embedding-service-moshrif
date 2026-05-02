"""Tests for intent router and service integration."""
from __future__ import annotations

import json

import pytest

from app.answering.intent_router import IntentType, route_intent


# ---------------------------------------------------------------------------
# Intent Router unit tests
# ---------------------------------------------------------------------------

class TestRouteIntent:
    def test_greeting_salam(self):
        d = route_intent("السلام عليكم")
        assert d.intent == IntentType.CONVERSATION
        assert d.is_legal_question is False
        assert d.confidence >= 0.9

    def test_thanks(self):
        d = route_intent("شكرا")
        assert d.intent == IntentType.CONVERSATION
        assert d.is_legal_question is False

    def test_identity(self):
        d = route_intent("اسمك إيه؟")
        assert d.intent == IntentType.IDENTITY
        assert d.confidence >= 0.95

    def test_identity_who_are_you(self):
        d = route_intent("انت مين؟")
        assert d.intent == IntentType.IDENTITY

    def test_legal_labor(self):
        d = route_intent("ما هي أحكام عقد العمل الفردي؟")
        assert d.intent == IntentType.LEGAL_RETRIEVAL
        assert d.is_legal_question is True
        assert d.suggested_domain == "labor_law"

    def test_external_assisted_custody(self):
        d = route_intent("ما هي أحكام الحضانة؟")
        assert d.intent == IntentType.EXTERNAL_ASSISTED
        assert d.is_out_of_internal_corpus is True
        assert d.is_legal_question is True

    def test_external_assisted_divorce(self):
        d = route_intent("ما هي إجراءات الطلاق؟")
        assert d.intent == IntentType.EXTERNAL_ASSISTED

    def test_non_legal_restaurant(self):
        d = route_intent("ما أفضل مطعم؟")
        assert d.intent == IntentType.NON_LEGAL
        assert d.is_legal_question is False

    def test_legal_criminal(self):
        d = route_intent("ما عقوبة السرقة؟")
        assert d.intent == IntentType.LEGAL_RETRIEVAL
        assert d.is_legal_question is True

    def test_greeting_ahlan(self):
        d = route_intent("اهلا")
        assert d.intent == IntentType.CONVERSATION

    def test_capability_prompt(self):
        d = route_intent("ممكن تساعدني؟")
        assert d.intent == IntentType.CONVERSATION

    def test_short_legal_query(self):
        d = route_intent("قانون العمل")
        assert d.intent == IntentType.LEGAL_RETRIEVAL


@pytest.mark.parametrize(
    ("query", "domain"),
    [
        ("ما ضمانات الحرية الشخصية في الدستور المصري؟", "constitutional_law"),
        ("ما مبدأ المساواة وعدم التمييز في الدستور المصري؟", "constitutional_law"),
        ("ماذا ينص الدستور المصري بشأن حرية الرأي؟", "constitutional_law"),
        ("ما الحق في التقاضي في الدستور المصري؟", "constitutional_law"),
        ("ما ضمانات المحاكمة العادلة في الدستور المصري؟", "constitutional_law"),
        ("ما المقصود بالحقوق والحريات في الدستور المصري؟", "constitutional_law"),
        ("ما هو عقد العمل؟", "labor_law"),
        ("ما حقوق العامل في قانون العمل المصري؟", "labor_law"),
        ("متى يجوز لصاحب العمل إنهاء عقد العمل؟", "labor_law"),
        ("ما التزامات المنشأة بشأن السلامة والصحة المهنية؟", "labor_law"),
        ("ما شروط صحة العقد في القانون المدني؟", "civil_law"),
        ("ما هي المسؤولية التقصيرية؟", "civil_law"),
        ("متى يجوز فسخ العقد؟", "civil_law"),
        ("ما التزامات البائع في عقد البيع؟", "civil_law"),
        ("ما عقوبة السرقة في قانون العقوبات؟", "criminal_law"),
        ("ما هي جريمة الرشوة؟", "criminal_law"),
        ("ما حكم الشروع في الجريمة؟", "criminal_law"),
        ("ما الفرق بين الجناية والجنحة والمخالفة؟", "criminal_law"),
    ],
)
def test_router_legal_retrieval_domains(query, domain):
    d = route_intent(query)
    assert d.intent == IntentType.LEGAL_RETRIEVAL
    assert d.is_legal_question is True
    assert d.suggested_domain == domain


@pytest.mark.parametrize(
    "query",
    [
        "ما أحكام الحضانة؟",
        "ما أحكام النفقة؟",
        "ما أحكام الطلاق؟",
        "ما أحكام الخلع؟",
        "ما أحكام الميراث؟",
    ],
)
def test_router_external_assisted_personal_status(query):
    d = route_intent(query)
    assert d.intent == IntentType.EXTERNAL_ASSISTED
    assert d.is_legal_question is True
    assert d.is_out_of_internal_corpus is True


@pytest.mark.parametrize("query", ["ما أفضل مطعم؟", "ما حالة الطقس؟", "احكي لي نكتة", "اكتب كود بايثون"])
def test_router_non_legal_obvious(query):
    d = route_intent(query)
    assert d.intent == IntentType.NON_LEGAL
    assert d.is_legal_question is False


@pytest.mark.parametrize("query", ["السلام عليكم", "شكرا", "صباح الخير"])
def test_router_conversation_obvious(query):
    d = route_intent(query)
    assert d.intent == IntentType.CONVERSATION
    assert d.is_legal_question is False


@pytest.mark.parametrize("query", ["اسمك إيه؟", "من المستشار؟", "مين طورك؟"])
def test_router_identity_obvious(query):
    d = route_intent(query)
    assert d.intent == IntentType.IDENTITY
    assert d.is_legal_question is False


@pytest.mark.parametrize("query", ["ما هي؟", "وضح لي", "العقد"])
def test_router_ambiguous_low_signal(query):
    d = route_intent(query)
    assert d.intent == IntentType.AMBIGUOUS
    assert d.is_legal_question is False


def test_explicit_domain_is_hint_not_command():
    d = route_intent("وضح لي", explicit_domain="civil_law")
    assert d.intent == IntentType.AMBIGUOUS
    assert d.is_legal_question is False


# ---------------------------------------------------------------------------
# Service integration tests
# ---------------------------------------------------------------------------

class TestServiceConversation:
    def test_greeting_returns_conversation(self):
        from app.answering import LegalAnswerService
        svc = LegalAnswerService(retriever=None, llm_client=_FakeLLM())
        resp = svc.answer("السلام عليكم")
        assert resp.answer_mode == "conversation"
        assert resp.llm.called is False
        assert resp.retrieval_summary.top_k_used == 0
        assert resp.internal_sources == []
        assert resp.sources == []
        assert resp.is_legal_question is False
        assert resp.is_supported_by_internal_sources is False
        assert "وعليكم السلام" in resp.final_answer

    def test_thanks_returns_conversation(self):
        from app.answering import LegalAnswerService
        svc = LegalAnswerService(retriever=None, llm_client=_FakeLLM())
        resp = svc.answer("شكرا")
        assert resp.answer_mode == "conversation"
        assert "العفو" in resp.final_answer

    def test_non_legal_returns_non_legal(self):
        from app.answering import LegalAnswerService
        svc = LegalAnswerService(retriever=None, llm_client=_FakeLLM())
        resp = svc.answer("ما أفضل مطعم؟")
        assert resp.answer_mode == "non_legal"
        assert resp.llm.called is False
        assert resp.internal_sources == []
        assert resp.is_legal_question is False

    def test_identity_returns_identity(self):
        from app.answering import LegalAnswerService
        svc = LegalAnswerService(retriever=None, llm_client=_FakeLLM())
        resp = svc.answer("اسمك إيه؟")
        assert resp.answer_mode == "identity"
        assert "المستشار" in resp.final_answer
        assert resp.llm.called is False
        assert resp.retrieval_summary.top_k_used == 0

    def test_identity_does_not_instantiate_retriever(self, monkeypatch):
        import app.answering.service as answering_service_module
        from app.answering import LegalAnswerService

        constructed = False

        class ExplodingRetriever:
            def __init__(self, *args, **kwargs):
                nonlocal constructed
                constructed = True
                raise AssertionError("LegalRetriever must not be constructed for identity queries")

        monkeypatch.setattr(answering_service_module, "LegalRetriever", ExplodingRetriever)
        svc = LegalAnswerService(retriever=None, llm_client=_FakeLLM())
        resp = svc.answer("مين طورك؟")

        assert constructed is False
        assert resp.answer_mode == "identity"
        assert resp.llm.called is False
        assert resp.retrieval_summary.top_k_used == 0
        assert resp.router is not None
        assert resp.router.intent == "identity"

    def test_external_assisted_skips_retriever(self):
        from app.answering import LegalAnswerService
        from app.llm import LLMCompletion

        class ExternalLLM:
            model = "gemini-2.5-flash"
            provider_name = "gemini"
            web_search_enabled = False
            def chat_completion(self, *, messages, temperature=0.0, max_tokens=None):
                return LLMCompletion(
                    content=json.dumps({"final_answer": "شرح عام."}, ensure_ascii=False),
                    model=self.model, provider=self.provider_name,
                    usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
                    raw_response=None,
                )

        svc = LegalAnswerService(retriever=None, llm_client=ExternalLLM())
        resp = svc.answer("ما هي أحكام الحضانة؟")
        assert resp.answer_mode == "external_assisted"
        assert resp.retrieval_summary.top_k_used == 0
        assert resp.is_out_of_internal_corpus is True
        assert resp.is_out_of_domain is False

    def test_grounded_legal_uses_retrieval(self):
        from app.answering import LegalAnswerService
        from app.llm import LLMCompletion

        class GroundedLLM:
            model = "gemini-2.5-flash"
            provider_name = "gemini"
            web_search_enabled = False
            def chat_completion(self, *, messages, temperature=0.0, max_tokens=None):
                return LLMCompletion(
                    content=json.dumps({
                        "answer_from_sources": "source",
                        "final_answer": "answer",
                        "warning": None,
                    }, ensure_ascii=False),
                    model=self.model, provider=self.provider_name,
                    usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
                    raw_response=None,
                )

        retriever = _FakeRetriever(_grounded_retrieval_result())
        svc = LegalAnswerService(retriever=retriever, llm_client=GroundedLLM())
        resp = svc.answer("ما هي أحكام عقد العمل الفردي؟")
        assert resp.answer_mode in ("grounded", "assisted")
        assert resp.llm.called is True
        assert retriever.calls >= 1


# ---------------------------------------------------------------------------
# Source sufficiency overlap guard
# ---------------------------------------------------------------------------

def test_sufficiency_no_grounded_when_zero_overlap_no_legal_intent():
    from app.answering.source_sufficiency import assess_source_sufficiency
    result = _grounded_retrieval_result()
    # Zero out all overlaps by using a query that shares no terms
    result["normalized_query"] = "xxxxxxx"
    result["query"] = "xxxxxxx"
    decision = assess_source_sufficiency(
        result, has_legal_intent=False, explicit_domain=None,
    )
    # With no overlap AND no legal intent, should NOT be grounded
    assert decision.answer_mode != "grounded" or decision.metrics.get("top_overlap", 0) > 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeLLM:
    model = "fake"
    provider_name = "fake"
    web_search_enabled = False
    def chat_completion(self, *, messages, temperature=0.0, max_tokens=None):
        raise RuntimeError("LLM should not be called")


class _FakeRetriever:
    def __init__(self, result: dict) -> None:
        self.result = result
        self.calls = 0
    def search(self, query: str, *, top_k=None, filters=None):
        self.calls += 1
        return self.result | {"query": query}


def _grounded_retrieval_result() -> dict:
    return {
        "normalized_query": "ما هي احكام عقد العمل الفردي",
        "query_analysis": {"out_of_domain": False, "suggested_domain": "labor_law"},
        "results": [
            {
                "id": "labor-1", "rerank_score": 0.91, "score": 0.78,
                "law_name": "قانون العمل المصري", "law_number": "14", "law_year": "2025",
                "article_number": "1", "title": "المادة 1 - قانون العمل المصري",
                "legal_domain": "labor_law", "section_level": "أحكام عامة",
                "source_url": "https://example.com/labor/1",
                "summary": "تنظيم عقد العمل الفردي وحقوق العامل.",
                "content": "ينظم قانون العمل حقوق العامل والتزامات صاحب العمل في عقد العمل الفردي.",
                "rank_explanation": ["strong_summary_overlap"],
            },
            {
                "id": "labor-2", "rerank_score": 0.84, "score": 0.70,
                "law_name": "قانون العمل المصري", "law_number": "14", "law_year": "2025",
                "article_number": "2", "title": "المادة 2 - قانون العمل المصري",
                "legal_domain": "labor_law", "section_level": "أحكام عامة",
                "source_url": "https://example.com/labor/2",
                "summary": "تعريف العامل وصاحب العمل وعقد العمل.",
                "content": "تتضمن المادة تعريفات مرتبطة بالعامل وصاحب العمل وعلاقة العمل.",
                "rank_explanation": ["strong_title_overlap"],
            },
        ],
    }
