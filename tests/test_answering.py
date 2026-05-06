from __future__ import annotations

import json

from app.answering import LegalAnswerService
import app.answering.service as answer_service_module
from app.llm import LLMCompletion, LLMConfigurationError


class FakeRetriever:
    def __init__(self, result: dict) -> None:
        self.result = result
        self.calls = 0

    def search(self, query: str, *, top_k=None, filters=None):
        self.calls += 1
        return self.result | {"query": query}


class FakeLLM:
    model = "fake-legal-llm"
    provider_name = "gemini"
    web_search_enabled = False

    def __init__(self, payload: dict | None = None, error: Exception | None = None, raw_response: dict | None = None) -> None:
        self.payload = payload or {}
        self.error = error
        self.raw_response = raw_response
        self.messages = None
        self.max_tokens = None
        self.calls = 0

    def chat_completion(self, *, messages, temperature=0.0, max_tokens=None):
        self.calls += 1
        self.messages = messages
        self.max_tokens = max_tokens
        if self.error:
            raise self.error
        return LLMCompletion(
            content=json.dumps(self.payload, ensure_ascii=False),
            model=self.model,
            provider=self.provider_name,
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            raw_response=self.raw_response,
        )


def test_answer_service_returns_grounded_mode_for_sufficient_sources():
    retriever = FakeRetriever(_grounded_retrieval_result())
    service = LegalAnswerService(
        retriever=retriever,
        llm_client=FakeLLM(
            {
                "final_answer": "جواب مباشر.\n\nالسند القانوني\nالمادة 1.\n\nالمصادر\nقانون العمل.",
                "answer_from_sources": "المادة 1 من قانون العمل.",
                "external_or_assisted_explanation": None,
                "warning": None,
            }
        ),
    )

    response = service.answer("ما هي أحكام عقد العمل الفردي؟")

    assert response.answer_mode == "grounded"
    assert response.internal_grounding_sufficient is True
    assert response.is_out_of_internal_corpus is False
    assert response.llm.succeeded is True
    assert response.llm.provider == "gemini"
    assert response.answer_parts is not None
    assert response.answer_parts.legal_basis is not None
    assert len(response.internal_sources) == 2
    assert response.internal_sources[0].law_name == "قانون العمل المصري"
    assert response.external_sources == []
    assert retriever.calls == 1


def test_answer_service_returns_external_assisted_for_family_law_outside_internal_corpus():
    retriever = FakeRetriever(
        {
            "normalized_query": "ما هي احكام الحضانة",
            "query_analysis": {
                "out_of_domain": True,
                "out_of_domain_reason": "query appears to target personal-status/family-law topics not covered by the corpus",
                "suggested_domain": None,
            },
            "results": [],
        }
    )
    service = LegalAnswerService(
        retriever=retriever,
        llm_client=FakeLLM(
            {
                "final_answer": "هذا السؤال خارج نطاق قاعدة المصادر الداخلية المتاحة حاليًا. شرح عام مساعد عن الحضانة.",
                "answer_from_sources": None,
                "external_or_assisted_explanation": "شرح عام مساعد عن الحضانة في سياق القانون المصري بدون أرقام مواد.",
                "warning": "لا توجد مصادر داخلية موثقة لهذا السؤال.",
            }
        ),
    )

    response = service.answer("ما هي أحكام الحضانة؟")

    assert response.answer_mode == "external_assisted"
    assert response.is_out_of_internal_corpus is True
    assert response.internal_grounding_sufficient is False
    assert response.internal_sources == []
    assert response.external_sources == []
    assert response.external_sources_verified_by_system is False
    assert "خارج قاعدة المصادر الداخلية" in (response.warning or "")
    assert retriever.calls == 0  # Shortcircuited — retriever not called.


def test_answer_service_falls_back_when_llm_is_not_configured():
    service = LegalAnswerService(
        retriever=FakeRetriever(_grounded_retrieval_result()),
        llm_client=FakeLLM(error=LLMConfigurationError("missing key")),
    )

    response = service.answer("ما هي أحكام عقد العمل الفردي؟")

    assert response.answer_mode == "grounded"
    assert response.llm.succeeded is False
    assert response.llm.provider == "gemini"
    assert "تعذر استدعاء نموذج اللغة" in (response.warning or "")
    assert response.internal_sources


def test_groq_only_success_returns_groq_metadata_and_answer_parts():
    groq = FakeLLM(
        {
            "final_answer": (
                "Groq legal answer.\n\n"
                "\u0623\u0647\u0645 \u0627\u0644\u0623\u062d\u0643\u0627\u0645:\n"
                "- Source-backed point.\n\n"
                "\u0627\u0644\u0633\u0646\u062f \u0627\u0644\u0642\u0627\u0646\u0648\u0646\u064a:\n"
                "\u0627\u0633\u062a\u0646\u062f\u062a \u0627\u0644\u0625\u062c\u0627\u0628\u0629 \u0625\u0644\u0649 \u0627\u0644\u0645\u0627\u062f\u0629 1 \u0645\u0646 \u0642\u0627\u0646\u0648\u0646 \u0627\u0644\u0639\u0645\u0644 \u0627\u0644\u0645\u0635\u0631\u064a."
            ),
            "answer_from_sources": "Groq source answer.",
            "warning": None,
        }
    )
    groq.provider_name = "groq"
    groq.model = "llama-3.3-70b-versatile"
    service = LegalAnswerService(
        retriever=FakeRetriever(_grounded_retrieval_result()),
        llm_client=groq,
        fallback_llm_client=None,
    )

    response = service.answer("\u0645\u0627 \u0647\u0648 \u0639\u0642\u062f \u0627\u0644\u0639\u0645\u0644\u061f", concise=True)

    assert response.llm.succeeded is True
    assert response.llm.provider == "groq"
    assert response.llm.model == "llama-3.3-70b-versatile"
    assert response.llm.fallback_used is False
    assert response.answer_parts is not None
    assert response.answer_parts.bullets


def test_groq_only_failure_without_fallback_returns_safe_source_fallback():
    groq = FakeLLM(error=LLMConfigurationError("groq returned HTTP 429: rate limit GROQ_API_KEY"))
    groq.provider_name = "groq"
    groq.model = "llama-3.3-70b-versatile"
    service = LegalAnswerService(
        retriever=FakeRetriever(_grounded_retrieval_result()),
        llm_client=groq,
        fallback_llm_client=None,
    )

    response = service.answer("\u0645\u0627 \u0647\u0648 \u0639\u0642\u062f \u0627\u0644\u0639\u0645\u0644\u061f", concise=True)

    assert response.llm.called is True
    assert response.llm.succeeded is False
    assert response.llm.provider == "groq"
    assert response.llm.fallback_provider is None
    assert response.internal_sources
    assert response.answer_parts is not None
    combined = " ".join(value for value in (response.warning, response.final_answer) if value)
    assert "GROQ_API_KEY" not in combined
    assert "429" not in combined
    assert "rate limit" not in combined


def test_llm_fallback_not_called_when_gemini_succeeds():
    primary = FakeLLM(
        {
            "final_answer": "Gemini answer.",
            "answer_from_sources": "Gemini sources.",
            "warning": None,
        }
    )
    fallback = FakeLLM({"final_answer": "Groq answer.", "answer_from_sources": "Groq sources."})
    fallback.provider_name = "groq"
    fallback.model = "llama-3.3-70b-versatile"
    service = LegalAnswerService(
        retriever=FakeRetriever(_grounded_retrieval_result()),
        llm_client=primary,
        fallback_llm_client=fallback,
    )

    response = service.answer("\u0645\u0627 \u0647\u0648 \u0639\u0642\u062f \u0627\u0644\u0639\u0645\u0644\u061f")

    assert response.llm.succeeded is True
    assert response.llm.provider == "gemini"
    assert primary.calls == 1
    assert fallback.calls == 0


def test_llm_fallback_uses_groq_when_gemini_rate_limited():
    primary = FakeLLM(error=LLMConfigurationError("gemini returned HTTP 429: quota exceeded"))
    fallback = FakeLLM(
        {
            "final_answer": "Groq answer.",
            "answer_from_sources": "Groq sources.",
            "warning": None,
        }
    )
    fallback.provider_name = "groq"
    fallback.model = "llama-3.3-70b-versatile"
    service = LegalAnswerService(
        retriever=FakeRetriever(_grounded_retrieval_result()),
        llm_client=primary,
        fallback_llm_client=fallback,
    )

    response = service.answer("\u0645\u0627 \u0647\u0648 \u0639\u0642\u062f \u0627\u0644\u0639\u0645\u0644\u061f")

    assert response.llm.succeeded is True
    assert response.llm.provider == "groq"
    assert response.llm.model == "llama-3.3-70b-versatile"
    assert response.llm.fallback_used is True
    assert primary.calls == 1
    assert fallback.calls == 1
    assert response.final_answer == "Groq answer."


def test_llm_fallback_uses_groq_when_gemini_schema_error():
    primary = FakeLLM({"final_answer": "Incomplete Gemini answer."})
    fallback = FakeLLM(
        {
            "final_answer": "Groq complete answer.",
            "answer_from_sources": "Groq complete sources.",
            "warning": None,
        }
    )
    fallback.provider_name = "groq"
    fallback.model = "llama-3.3-70b-versatile"
    service = LegalAnswerService(
        retriever=FakeRetriever(_grounded_retrieval_result()),
        llm_client=primary,
        fallback_llm_client=fallback,
    )

    response = service.answer("\u0645\u0627 \u0647\u0648 \u0639\u0642\u062f \u0627\u0644\u0639\u0645\u0644\u061f")

    assert response.llm.succeeded is True
    assert response.llm.provider == "groq"
    assert response.llm.fallback_used is True
    assert fallback.calls == 1
    assert response.final_answer == "Groq complete answer."


def test_public_chat_concise_prompt_and_answer_are_accepted():
    llm = FakeLLM(
        {
            "final_answer": (
                "ينظم قانون العمل عقد العمل الفردي باعتباره الإطار الذي يحدد علاقة العامل بصاحب العمل وحقوق كل طرف.\n\n"
                "أهم الأحكام:\n"
                "- يحدد العلاقة القانونية بين العامل وصاحب العمل من بداية التعاقد.\n"
                "- يوضح الحقوق والالتزامات الأساسية لكل طرف في علاقة العمل.\n"
                "- يرتبط بالأجر وساعات العمل والإجازات بحسب ما يرد في العقد والقانون.\n"
                "- يساعد على تحديد الأساس القانوني عند النزاع على شروط العمل.\n\n"
                "السند القانوني:\n"
                "استندت الإجابة إلى المادة 1 من قانون العمل المصري."
            ),
            "answer_from_sources": "المادة 1 من قانون العمل المصري.",
            "warning": None,
        }
    )
    service = LegalAnswerService(retriever=FakeRetriever(_grounded_retrieval_result()), llm_client=llm)

    response = service.answer("ما هي أحكام عقد العمل الفردي؟", concise=True)

    assert response.answer_mode == "grounded"
    assert response.llm.succeeded is True
    assert "السند القانوني" in response.final_answer
    assert "أهم الأحكام:" in response.final_answer
    assert "المصادر:" not in response.final_answer
    assert "S1" not in response.final_answer
    assert "S2" not in response.final_answer
    assert response.answer_parts is not None
    assert 4 <= len(response.answer_parts.bullets) <= 6
    assert response.answer_parts.legal_basis
    assert response.warning is None
    prompt_text = "\n".join(message["content"] for message in llm.messages or [])
    assert "نمط الإخراج العام المتوازن لـ /chat" in prompt_text
    assert "مختصر لكنه كافٍ ومفيد" in prompt_text
    assert "answer_parts" in prompt_text
    assert "4 إلى 6" in prompt_text
    assert "answer_detail_level" in prompt_text
    assert "لا تضع داخل final_answer قسمًا بعنوان \"المصادر\"" in prompt_text
    assert llm.max_tokens is not None
    assert llm.max_tokens <= 1536


def test_public_chat_balanced_grounded_procedural_answer_is_not_minimal():
    llm = FakeLLM(
        {
            "final_answer": (
                "إذا كان لك مبلغ مالي ثابت في ذمة شخص وامتنع عن رده، فالفكرة القانونية الأساسية هي إثبات الدين ثم مطالبته بالوفاء قبل اللجوء للقضاء عند الحاجة.\n\n"
                "الخطوات العملية:\n"
                "- اجمع ما يثبت الدين، مثل إيصال أو عقد أو رسائل أو أي دليل كتابي متاح.\n"
                "- وجّه مطالبة واضحة بالسداد، واحتفظ بما يثبت تاريخ المطالبة ومضمونها.\n"
                "- إذا لم يتم السداد، يمكن اللجوء لدعوى مدنية للمطالبة بالمبلغ المستحق.\n"
                "- رتّب المستندات التي توضح قيمة المبلغ وسبب استحقاقه وامتناع المدين عن الرد.\n"
                "- لا تعتمد على الادعاء المجرد وحده إذا كانت المصادر المتاحة تشترط وجود سند أو دليل.\n\n"
                "السند القانوني:\n"
                "استندت الإجابة إلى المادة 1 والمادة 2 من القانون المدني المصري."
            ),
            "answer_from_sources": "المادة 1 والمادة 2 من القانون المدني المصري.",
            "answer_parts": {
                "intro": "إذا كان لك مبلغ مالي ثابت في ذمة شخص وامتنع عن رده، فالفكرة القانونية الأساسية هي إثبات الدين ثم مطالبته بالوفاء قبل اللجوء للقضاء عند الحاجة.",
                "section_title": "الخطوات العملية:",
                "bullets": [
                    "اجمع ما يثبت الدين، مثل إيصال أو عقد أو رسائل أو أي دليل كتابي متاح.",
                    "وجّه مطالبة واضحة بالسداد، واحتفظ بما يثبت تاريخ المطالبة ومضمونها.",
                    "إذا لم يتم السداد، يمكن اللجوء لدعوى مدنية للمطالبة بالمبلغ المستحق.",
                    "رتّب المستندات التي توضح قيمة المبلغ وسبب استحقاقه وامتناع المدين عن الرد.",
                    "لا تعتمد على الادعاء المجرد وحده إذا كانت المصادر المتاحة تشترط وجود سند أو دليل.",
                ],
                "legal_basis": "استندت الإجابة إلى المادة 999 من قانون غير مسترجع.",
                "note": None,
            },
            "warning": None,
        }
    )
    service = LegalAnswerService(retriever=FakeRetriever(_civil_debt_retrieval_result()), llm_client=llm)

    response = service.answer(
        "إذا أقرضت شخصًا مبلغًا من المال ولم يقم برده، ما هي الخطوات القانونية لاسترداد أموالي؟",
        concise=True,
    )

    assert response.answer_mode == "grounded"
    assert response.answer_parts is not None
    assert response.answer_parts.section_title == "الخطوات العملية:"
    assert 4 <= len(response.answer_parts.bullets) <= 6
    assert len(response.final_answer.split()) >= 65
    assert response.final_answer.count("\n- ") >= 4
    assert response.answer_parts.legal_basis == "استندت الإجابة إلى المادة 1 من القانون المدني المصري والمادة 2 من القانون المدني المصري."
    assert "999" not in (response.answer_parts.legal_basis or "")


def test_full_legal_answer_prompt_and_budget_remain_large():
    llm = FakeLLM(
        {
            "final_answer": "جواب كامل.\n\nالسند القانوني\nالمادة 1.\n\nالمصادر\nقانون العمل.",
            "answer_from_sources": "المادة 1 من قانون العمل المصري.",
            "warning": None,
        }
    )
    service = LegalAnswerService(retriever=FakeRetriever(_grounded_retrieval_result()), llm_client=llm)

    response = service.answer("ما هي أحكام عقد العمل الفردي؟")

    assert response.answer_mode == "grounded"
    assert response.llm.succeeded is True
    prompt_text = "\n".join(message["content"] for message in llm.messages or [])
    assert "نمط الإخراج العام لـ /chat" not in prompt_text
    assert llm.max_tokens is not None
    assert llm.max_tokens > 1536


def test_external_assisted_concise_warning_is_user_friendly():
    llm = FakeLLM(
        {
            "final_answer": (
                "هذا السؤال خارج نطاق قاعدة المصادر الداخلية المتاحة حاليًا.\n\n"
                "شرح عام:\n"
                "- الحضانة تتعلق برعاية الصغير وتنظيم شؤونه.\n"
                "- تختلف التفاصيل بحسب السن والظروف ومصلحة الطفل.\n"
                "- يلزم الرجوع للنصوص الرسمية عند التطبيق.\n\n"
                "ملاحظة:\n"
                "هذه إجابة عامة غير موثقة من مصادر التطبيق الداخلية، ويُفضّل مراجعة محامٍ مختص أو النصوص الرسمية."
            ),
            "answer_from_sources": None,
            "warning": None,
        }
    )
    service = LegalAnswerService(retriever=None, llm_client=llm)

    response = service.answer("ما أحكام الحضانة؟", concise=True)

    assert response.answer_mode == "external_assisted"
    assert response.llm.succeeded is True
    assert "خارج مصادر التطبيق الداخلية" in (response.warning or "")
    assert "schema_error" not in (response.warning or "")
    assert "شرح عام:" in response.final_answer
    assert "ملاحظة:" in response.final_answer
    assert "هذه إجابة عامة غير موثقة من مصادر التطبيق الداخلية" in response.final_answer
    assert "المادة " not in response.final_answer
    assert "المصادر:" not in response.final_answer
    assert response.answer_parts is not None
    assert response.answer_parts.section_title == "شرح عام:"
    assert response.answer_parts.note
    assert llm.max_tokens is not None
    assert llm.max_tokens <= 1536


def test_identity_query_returns_branding_without_retrieval_or_llm():
    retriever = FakeRetriever(_grounded_retrieval_result())
    llm = FakeLLM({"final_answer": "should not be used"})
    service = LegalAnswerService(retriever=retriever, llm_client=llm)

    response = service.answer("اسمك إيه؟")

    assert response.answer_mode == "identity"
    assert response.llm.called is False
    assert response.retrieval_summary.top_k_used == 0
    assert "أنا المستشار" in response.final_answer
    assert "لؤي وائل" in response.final_answer
    assert "Claude" not in response.final_answer
    assert "محام" not in response.final_answer
    assert retriever.calls == 0
    assert llm.calls == 0


def test_identity_query_does_not_construct_default_legal_retriever(monkeypatch):
    constructed = {"count": 0}

    def fail_if_constructed(*args, **kwargs):
        constructed["count"] += 1
        raise AssertionError("LegalRetriever must not be constructed for identity queries")

    monkeypatch.setattr(answer_service_module, "LegalRetriever", fail_if_constructed)
    llm = FakeLLM({"final_answer": "should not be used"})
    service = answer_service_module.LegalAnswerService(llm_client=llm)

    response = service.answer("اسمك إيه؟")

    assert response.answer_mode == "identity"
    assert response.llm.called is False
    assert response.retrieval_summary.top_k_used == 0
    assert response.internal_sources == []
    assert response.external_sources == []
    assert constructed["count"] == 0
    assert llm.calls == 0


def test_developer_identity_query_returns_branding_without_retrieval_or_llm():
    retriever = FakeRetriever(_grounded_retrieval_result())
    llm = FakeLLM({"final_answer": "should not be used"})
    service = LegalAnswerService(retriever=retriever, llm_client=llm)

    response = service.answer("مين طورك؟")

    assert response.answer_mode == "identity"
    assert "تم تصميمي وتطويري بواسطة لؤي وائل" in response.final_answer
    assert retriever.calls == 0
    assert llm.calls == 0


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


def _civil_debt_retrieval_result() -> dict:
    return {
        "normalized_query": "اذا اقرضت شخصا مبلغا من المال ولم يقم برده",
        "query_analysis": {
            "out_of_domain": False,
            "suggested_domain": "civil_law",
        },
        "results": [
            {
                "id": "civil-1",
                "rerank_score": 0.91,
                "score": 0.82,
                "law_name": "القانون المدني المصري",
                "law_number": "131",
                "law_year": "1948",
                "article_number": "1",
                "title": "إثبات الالتزام والمطالبة بالدين",
                "legal_domain": "civil_law",
                "section_level": "الالتزامات",
                "source_url": "https://example.com/civil/1",
                "summary": "قواعد مدنية عن الالتزام والدين والمطالبة بالمال المستحق.",
                "content": "إذا أقرض الدائن شخصًا مبلغًا من المال ولم يرده المدين، يجوز إثبات الدين والمطالبة برد المال.",
                "rank_explanation": ["strong_summary_overlap"],
            },
            {
                "id": "civil-2",
                "rerank_score": 0.84,
                "score": 0.76,
                "law_name": "القانون المدني المصري",
                "law_number": "131",
                "law_year": "1948",
                "article_number": "2",
                "title": "المطالبة القضائية بالالتزام",
                "legal_domain": "civil_law",
                "section_level": "الالتزامات",
                "source_url": "https://example.com/civil/2",
                "summary": "إجراءات مطالبة المدين بتنفيذ الالتزام أو رد المال.",
                "content": "للدائن عند عدم سداد الدين أن يطالب المدين بالوفاء وأن يلجأ إلى المحكمة لاسترداد أمواله.",
                "rank_explanation": ["strong_title_overlap"],
            },
        ],
    }
