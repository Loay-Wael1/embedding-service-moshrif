import json


SAFE_PUBLIC_LLM_WARNING = "تعذر توليد الصياغة النهائية حاليًا، وتم عرض إجابة مستندة إلى المصادر الداخلية المتاحة."


def _install_constitutional_answer_service(app_client, llm_client, fallback_llm_client=None):
    from app.answering import LegalAnswerService

    class FakeRetriever:
        def search(self, query: str, *, top_k=None, filters=None):
            return {
                "query": query,
                "normalized_query": query,
                "query_analysis": {"out_of_domain": False, "suggested_domain": "constitutional_law"},
                "results": [
                    {
                        "id": "const-54",
                        "rerank_score": 0.91,
                        "score": 0.86,
                        "law_name": "دستور جمهورية مصر العربية",
                        "article_number": "54",
                        "title": "الحرية الشخصية",
                        "legal_domain": "constitutional_law",
                        "source_url": "https://example.com/constitution/54",
                        "summary": "تضمن المادة 54 ضمانات الحرية الشخصية.",
                        "content": (
                            "الحرية الشخصية حق طبيعي وهي مصونة لا تمس، ولا يجوز القبض أو التفتيش "
                            "أو الحبس إلا بأمر قضائي مسبب، ويجب إبلاغ من تقيد حريته بأسباب ذلك."
                        ),
                        "rank_explanation": ["strong_summary_overlap"],
                    },
                    {
                        "id": "const-92",
                        "rerank_score": 0.84,
                        "score": 0.79,
                        "law_name": "دستور جمهورية مصر العربية",
                        "article_number": "92",
                        "title": "الحقوق والحريات",
                        "legal_domain": "constitutional_law",
                        "source_url": "https://example.com/constitution/92",
                        "summary": "الحقوق والحريات اللصيقة بشخص المواطن لا تقبل تعطيلا ولا انتقاصا.",
                        "content": "الحقوق والحريات اللصيقة بشخص المواطن لا تقبل تعطيلا ولا انتقاصا.",
                        "rank_explanation": ["strong_title_overlap"],
                    },
                ],
            }

    app_client.app.state.chat_cache.clear()
    app_client.app.state.legal_answer_service = LegalAnswerService(
        retriever=FakeRetriever(),
        llm_client=llm_client,
        fallback_llm_client=fallback_llm_client,
    )


def test_info_endpoint_returns_contract(app_client):
    response = app_client.get("/info")
    payload = response.json()

    assert response.status_code == 200
    assert payload["model_name"] == "fake-bge-m3"
    assert payload["embedding_dimension"] == 4
    assert payload["supported_outputs"]["dense"] is True


def test_single_embed_endpoint_returns_structured_result(app_client):
    response = app_client.post(
        "/embed",
        json={
            "text": "ما هي أحكام عقد العمل؟",
            "mode": "query",
            "normalize": True,
            "return_dense": True,
            "return_sparse": True,
        },
    )
    payload = response.json()

    assert response.status_code == 200
    assert payload["mode"] == "query"
    assert payload["normalized"] is True
    assert payload["sparse_available"] is False
    assert len(payload["results"]) == 1
    assert payload["results"][0]["dense"] is not None
    assert payload["results"][0]["sparse"] is None


def test_batch_embed_endpoint_preserves_order(app_client):
    response = app_client.post(
        "/embed/batch",
        json={
            "texts": ["عقد العمل", "القانون المدني"],
            "mode": "document",
            "normalize": True,
            "return_dense": True,
            "return_sparse": False,
        },
    )
    payload = response.json()

    assert response.status_code == 200
    assert [item["text"] for item in payload["results"]] == ["عقد العمل", "القانون المدني"]


def test_chat_endpoint_conversation_intent(app_client):
    response = app_client.post("/chat", json={"query": "السلام عليكم"})
    payload = response.json()
    assert response.status_code == 200
    assert payload["answer_mode"] == "conversation"
    assert payload["llm"]["called"] is False
    assert "retrieval_summary" not in payload
    assert response.headers["X-Cache-Hit"] == "false"
    assert response.headers["X-Answer-Mode"] == "conversation"
    assert response.headers["X-LLM-Called"] == "false"
    assert response.headers["X-Sources-Count"] == "0"
    assert response.headers["X-Router-Intent"] == "conversation"


def test_chat_endpoint_identity_intent(app_client):
    response = app_client.post("/chat", json={"query": "اسمك إيه؟"})
    payload = response.json()
    assert response.status_code == 200
    assert payload["answer_mode"] == "identity"
    assert payload["llm"]["called"] is False
    assert "retrieval_summary" not in payload


def test_chat_cached_response_repairs_missing_answer_parts(app_client):
    from app.answering.schemas import ChatResponse, CompactLLMMetadata
    from app.preprocessing import normalize_legal_arabic

    query = "ما هي أحكام عقد العمل؟"
    normalized = normalize_legal_arabic(query)
    app_client.app.state.chat_cache.clear()
    app_client.app.state.chat_cache[normalized] = (
        ChatResponse(
            answer_mode="grounded",
            final_answer=(
                "ينظم القانون أحكام عقد العمل بصورة عامة.\n\n"
                "أهم الأحكام:\n"
                "- يحدد العقد حقوق العامل وصاحب العمل.\n"
                "- يجب الالتزام بما ورد في القانون.\n\n"
                "السند القانوني:\n"
                "استندت الإجابة إلى المادة 1 من قانون العمل المصري."
            ),
            answer_parts=None,
            is_out_of_internal_corpus=False,
            llm=CompactLLMMetadata(called=True, succeeded=True, provider="gemini", model="gemini-2.5-flash"),
        ),
        {"X-Answer-Mode": "grounded"},
    )

    response = app_client.post("/chat", json={"query": query})
    payload = response.json()

    assert response.status_code == 200
    assert response.headers["X-Cache-Hit"] == "true"
    assert payload["answer_parts"] is not None
    assert payload["answer_parts"]["section_title"] == "أهم الأحكام:"
    assert payload["answer_parts"]["bullets"]
    assert payload["answer_parts"]["legal_basis"] == "استندت الإجابة إلى المادة 1 من قانون العمل المصري."


def test_chat_requires_internal_token_when_enabled(app_client):
    from app.settings import settings

    old_required = settings.require_internal_api_token
    old_token = settings.internal_api_token
    old_header = settings.internal_api_token_header
    try:
        object.__setattr__(settings, "require_internal_api_token", True)
        object.__setattr__(settings, "internal_api_token", "test-token")
        object.__setattr__(settings, "internal_api_token_header", "X-Internal-Service-Token")

        missing = app_client.post("/chat", json={"query": "اسمك إيه؟"})
        wrong = app_client.post(
            "/chat",
            json={"query": "اسمك إيه؟"},
            headers={"X-Internal-Service-Token": "wrong"},
        )
        valid = app_client.post(
            "/chat",
            json={"query": "اسمك إيه؟"},
            headers={"X-Internal-Service-Token": "test-token"},
        )

        assert missing.status_code == 401
        assert missing.json() == {"detail": "Unauthorized"}
        assert wrong.status_code == 401
        assert valid.status_code == 200
        assert valid.json()["answer_mode"] == "identity"
    finally:
        object.__setattr__(settings, "require_internal_api_token", old_required)
        object.__setattr__(settings, "internal_api_token", old_token)
        object.__setattr__(settings, "internal_api_token_header", old_header)


def test_health_remains_public_when_internal_token_enabled(app_client):
    from app.settings import settings

    old_required = settings.require_internal_api_token
    old_token = settings.internal_api_token
    try:
        object.__setattr__(settings, "require_internal_api_token", True)
        object.__setattr__(settings, "internal_api_token", "test-token")

        response = app_client.get("/health")

        assert response.status_code == 200
        assert response.json()["status"] == "ok"
    finally:
        object.__setattr__(settings, "require_internal_api_token", old_required)
        object.__setattr__(settings, "internal_api_token", old_token)


def test_warmup_requires_internal_token_when_enabled(app_client):
    from app.settings import settings

    old_required = settings.require_internal_api_token
    old_token = settings.internal_api_token
    try:
        object.__setattr__(settings, "require_internal_api_token", True)
        object.__setattr__(settings, "internal_api_token", "test-token")

        response = app_client.post("/warmup")

        assert response.status_code == 401
        assert response.json() == {"detail": "Unauthorized"}
    finally:
        object.__setattr__(settings, "require_internal_api_token", old_required)
        object.__setattr__(settings, "internal_api_token", old_token)


def test_legal_answer_requires_internal_token_when_enabled(app_client):
    from app.settings import settings

    old_required = settings.require_internal_api_token
    old_token = settings.internal_api_token
    try:
        object.__setattr__(settings, "require_internal_api_token", True)
        object.__setattr__(settings, "internal_api_token", "test-token")

        response = app_client.post("/legal-answer", json={"query": "اسمك إيه؟"})

        assert response.status_code == 401
        assert response.json() == {"detail": "Unauthorized"}
    finally:
        object.__setattr__(settings, "require_internal_api_token", old_required)
        object.__setattr__(settings, "internal_api_token", old_token)


def test_embed_requires_internal_token_when_enabled(app_client):
    from app.settings import settings

    old_required = settings.require_internal_api_token
    old_token = settings.internal_api_token
    try:
        object.__setattr__(settings, "require_internal_api_token", True)
        object.__setattr__(settings, "internal_api_token", "test-token")

        response = app_client.post(
            "/embed",
            json={"text": "عقد العمل", "mode": "query", "normalize": True},
        )

        assert response.status_code == 401
        assert response.json() == {"detail": "Unauthorized"}
    finally:
        object.__setattr__(settings, "require_internal_api_token", old_required)
        object.__setattr__(settings, "internal_api_token", old_token)


def test_embed_batch_requires_internal_token_when_enabled(app_client):
    from app.settings import settings

    old_required = settings.require_internal_api_token
    old_token = settings.internal_api_token
    try:
        object.__setattr__(settings, "require_internal_api_token", True)
        object.__setattr__(settings, "internal_api_token", "test-token")

        response = app_client.post(
            "/embed/batch",
            json={"texts": ["عقد العمل"], "mode": "query", "normalize": True},
        )

        assert response.status_code == 401
        assert response.json() == {"detail": "Unauthorized"}
    finally:
        object.__setattr__(settings, "require_internal_api_token", old_required)
        object.__setattr__(settings, "internal_api_token", old_token)


def test_info_requires_internal_token_when_enabled(app_client):
    from app.settings import settings

    old_required = settings.require_internal_api_token
    old_token = settings.internal_api_token
    try:
        object.__setattr__(settings, "require_internal_api_token", True)
        object.__setattr__(settings, "internal_api_token", "test-token")

        response = app_client.get("/info")

        assert response.status_code == 401
        assert response.json() == {"detail": "Unauthorized"}
    finally:
        object.__setattr__(settings, "require_internal_api_token", old_required)
        object.__setattr__(settings, "internal_api_token", old_token)


def test_docs_are_disabled_when_public_docs_disabled(fake_embedding_service):
    from fastapi.testclient import TestClient

    from app.api import create_app
    from app.settings import settings

    old_enabled = settings.enable_public_docs
    try:
        object.__setattr__(settings, "enable_public_docs", False)
        client = TestClient(create_app(fake_embedding_service))

        assert client.get("/docs").status_code == 404
        assert client.get("/redoc").status_code == 404
        assert client.get("/openapi.json").status_code == 404
    finally:
        object.__setattr__(settings, "enable_public_docs", old_enabled)


def test_docs_work_when_public_docs_enabled(fake_embedding_service):
    from fastapi.testclient import TestClient

    from app.api import create_app
    from app.settings import settings

    old_enabled = settings.enable_public_docs
    try:
        object.__setattr__(settings, "enable_public_docs", True)
        client = TestClient(create_app(fake_embedding_service))

        assert client.get("/docs").status_code == 200
        assert client.get("/redoc").status_code == 200
        assert client.get("/openapi.json").status_code == 200
    finally:
        object.__setattr__(settings, "enable_public_docs", old_enabled)


def test_chat_endpoint_non_legal_intent(app_client):
    response = app_client.post("/chat", json={"query": "ما أفضل مطعم؟"})
    payload = response.json()
    assert response.status_code == 200
    assert payload["answer_mode"] == "non_legal"
    assert payload["llm"]["called"] is False
    assert "retrieval_summary" not in payload


def test_chat_endpoint_ambiguous_intent_proceeds_to_retrieval(app_client):
    """Ambiguous queries must proceed to retrieval, not return a clarification block."""
    from app.answering import LegalAnswerService
    from unittest.mock import MagicMock

    mock_retriever = MagicMock()
    mock_retriever.search.return_value = {
        "query": "ما هي",
        "normalized_query": "ما هي",
        "rewritten_query": "ما هي",
        "query_analysis": {},
        "results": [],
    }
    mock_llm = MagicMock()
    mock_llm.provider_name = "test"
    mock_llm.model = "test"
    mock_llm.web_search_enabled = False
    mock_llm.chat_completion.side_effect = Exception("no key")

    app_client.app.state.chat_cache.clear()
    app_client.app.state.legal_answer_service = LegalAnswerService(
        retriever=mock_retriever,
        llm_client=mock_llm,
    )

    response = app_client.post("/chat", json={"query": "ما هي؟"})
    payload = response.json()
    assert response.status_code == 200
    # Must NOT return the old blocking message
    assert payload["final_answer"] != "من فضلك وضّح سؤالك القانوني أو اذكر المجال القانوني المطلوب."
    # Retriever must have been called
    mock_retriever.search.assert_called()



def test_chat_endpoint_external_assisted_intent(app_client):
    # Depending on mock, this should trigger external_assisted because of "الحضانة"
    response = app_client.post("/chat", json={"query": "ما هي أحكام الحضانة؟"})
    payload = response.json()
    assert response.status_code == 200
    assert payload["answer_mode"] == "external_assisted"


def test_chat_endpoint_grounded_intent(app_client, monkeypatch):
    import app.answering.service as answering_service_module
    
    class FakeRetriever:
        def __init__(self, *args, **kwargs):
            pass
        def search(self, *args, **kwargs):
            return {
                "normalized_query": "ما هي أحكام عقد العمل الفردي؟",
                "query_analysis": {"out_of_domain": False, "suggested_domain": "labor_law"},
                "results": [
                    {
                        "content": "عقد العمل الفردي هو...",
                        "score": 0.9,
                        "law_name": "قانون العمل",
                        "keywords": ["عقد", "العمل"],
                    }
                ],
            }
            
    monkeypatch.setattr(answering_service_module, "LegalRetriever", FakeRetriever)
    
    response = app_client.post("/chat", json={"query": "ما هي أحكام عقد العمل الفردي؟"})
    payload = response.json()
    assert response.status_code == 200
    assert payload["answer_mode"] in {"grounded", "assisted", "insufficient"}
    assert "internal_sources" not in payload
    assert "timing" not in payload

def test_chat_endpoint_debug_mode(app_client):
    import app.settings as app_settings
    object.__setattr__(app_settings.settings, 'debug_response_metadata', True)
    
    try:
        response = app_client.post("/chat?debug=true", json={"query": "السلام عليكم"})
        payload = response.json()
        assert response.status_code == 200
        assert payload["answer_mode"] == "conversation"
        # Should include full response fields
        assert "retrieval_summary" in payload
        assert "timing" in payload
    finally:
        object.__setattr__(app_settings.settings, 'debug_response_metadata', False)


def test_chat_endpoint_debug_query_includes_router_without_env_flag(app_client):
    response = app_client.post("/chat?debug=true", json={"query": "السلام عليكم"})
    payload = response.json()
    assert response.status_code == 200
    assert payload["answer_mode"] == "conversation"
    assert "retrieval_summary" in payload
    assert "timing" in payload
    assert payload["router"]["intent"] == "conversation"

def test_chat_endpoint_cache(app_client):
    import app.settings as app_settings

    original_size = app_settings.settings.chat_response_cache_size
    object.__setattr__(app_settings.settings, "chat_response_cache_size", 128)
    app_client.app.state.chat_cache.clear()

    try:
        response1 = app_client.post("/chat", json={"query": "اسمك إيه؟"})
        assert response1.status_code == 200

        cache = app_client.app.state.chat_cache
        assert len(cache) == 1

        response2 = app_client.post("/chat", json={"query": "اسمك إيه؟"})
        assert response2.status_code == 200

        assert response1.json() == response2.json()
        assert response2.headers["X-Cache-Hit"] == "true"
    finally:
        object.__setattr__(app_settings.settings, "chat_response_cache_size", original_size)
        app_client.app.state.chat_cache.clear()


def test_chat_endpoint_non_legal_is_not_cached(app_client):
    import app.settings as app_settings

    original_size = app_settings.settings.chat_response_cache_size
    object.__setattr__(app_settings.settings, "chat_response_cache_size", 128)
    app_client.app.state.chat_cache.clear()

    try:
        response = app_client.post("/chat", json={"query": "ما أفضل مطعم؟"})
        payload = response.json()
        assert response.status_code == 200
        assert payload["answer_mode"] == "non_legal"
        assert len(app_client.app.state.chat_cache) == 0
        assert response.headers["X-Cache-Hit"] == "false"
    finally:
        object.__setattr__(app_settings.settings, "chat_response_cache_size", original_size)
        app_client.app.state.chat_cache.clear()


def test_chat_endpoint_constitutional_legal_question_routes_to_retrieval(app_client):
    from app.answering import LegalAnswerService
    from app.llm import LLMCompletion

    captured = {}

    class FakeLLM:
        model = "gemini-2.5-flash"
        provider_name = "gemini"
        web_search_enabled = False

        def chat_completion(self, *, messages, temperature=0.0, max_tokens=None):
            captured["messages"] = messages
            captured["max_tokens"] = max_tokens
            return LLMCompletion(
                content=json.dumps(
                    {
                        "answer_from_sources": "المادة 54 من دستور جمهورية مصر العربية.",
                        "final_answer": (
                            "يوضح الدستور المصري أن الحرية الشخصية حق طبيعي مصون لا يجوز المساس به إلا بضمانات قانونية وقضائية محددة.\n\n"
                            "أهم الضمانات:\n"
                            "- لا يجوز القبض على الشخص أو تفتيشه أو حبسه إلا بأمر قضائي مسبب، عدا حالة التلبس.\n"
                            "- يجب إبلاغ من تقيد حريته بأسباب ذلك وحقوقه كتابةً فورًا.\n"
                            "- يجب تمكينه من الاتصال بذويه ومحاميه فورًا.\n"
                            "- يجب عرضه على سلطة التحقيق خلال 24 ساعة.\n\n"
                            "السند القانوني:\n"
                            "استندت الإجابة إلى المادة 54 من دستور جمهورية مصر العربية."
                        ),
                        "warning": None,
                    },
                    ensure_ascii=False,
                ),
                model=self.model,
                provider=self.provider_name,
                usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
                raw_response=None,
            )

    class FakeRetriever:
        def __init__(self):
            self.calls = 0

        def search(self, query: str, *, top_k=None, filters=None):
            self.calls += 1
            assert filters is not None
            assert filters.legal_domain == "constitutional_law"
            return {
                "query": query,
                "normalized_query": query,
                "query_analysis": {"out_of_domain": False, "suggested_domain": "constitutional_law"},
                "results": [
                    {
                        "id": "const-54",
                        "rerank_score": 0.91,
                        "score": 0.86,
                        "law_name": "دستور جمهورية مصر العربية",
                        "article_number": "54",
                        "title": "الحرية الشخصية",
                        "legal_domain": "constitutional_law",
                        "source_url": "https://example.com/constitution/54",
                        "summary": "الحرية الشخصية حق طبيعي وهي مصونة لا تمس.",
                        "content": "الحرية الشخصية حق طبيعي، وهي مصونة لا تمس.",
                        "rank_explanation": ["strong_summary_overlap"],
                    },
                    {
                        "id": "const-92",
                        "rerank_score": 0.84,
                        "score": 0.79,
                        "law_name": "دستور جمهورية مصر العربية",
                        "article_number": "92",
                        "title": "الحقوق والحريات",
                        "legal_domain": "constitutional_law",
                        "source_url": "https://example.com/constitution/92",
                        "summary": "الحقوق والحريات اللصيقة بشخص المواطن لا تقبل تعطيلا ولا انتقاصا.",
                        "content": "الحقوق والحريات اللصيقة بشخص المواطن لا تقبل تعطيلا ولا انتقاصا.",
                        "rank_explanation": ["strong_title_overlap"],
                    },
                ],
            }

    retriever = FakeRetriever()
    app_client.app.state.chat_cache.clear()
    app_client.app.state.legal_answer_service = LegalAnswerService(retriever=retriever, llm_client=FakeLLM())

    response = app_client.post("/chat", json={"query": "ما ضمانات الحرية الشخصية في الدستور المصري؟"})
    payload = response.json()

    assert response.status_code == 200
    assert payload["answer_mode"] in {"grounded", "assisted"}
    assert payload["is_legal_question"] is True
    assert payload["llm"]["called"] is True
    assert payload["answer_parts"] is not None
    assert payload["answer_parts"]["intro"]
    assert isinstance(payload["answer_parts"]["bullets"], list)
    assert len(payload["answer_parts"]["bullets"]) >= 1
    assert payload["answer_parts"]["legal_basis"]
    assert payload["final_answer"]
    assert "السند القانوني:" in payload["final_answer"]
    assert "أهم الضمانات:" in payload["final_answer"]
    assert "المصادر:" not in payload["final_answer"]
    assert "S1" not in payload["final_answer"]
    assert "S2" not in payload["final_answer"]
    assert len(payload["sources"]) > 0
    assert retriever.calls == 1
    assert response.headers["X-Router-Intent"] == "legal_retrieval"
    assert response.headers["X-Router-Domain"] == "constitutional_law"
    assert response.headers["X-LLM-Called"] == "true"
    assert captured["max_tokens"] <= 1536
    prompt_text = "\n".join(message["content"] for message in captured["messages"])
    assert "نمط الإخراج العام المتوازن لـ /chat" in prompt_text
    assert "مختصر لكنه كافٍ ومفيد" in prompt_text
    assert "answer_parts" in prompt_text
    assert "4 إلى 6" in prompt_text
    assert "لا تضع داخل final_answer قسمًا بعنوان \"المصادر\"" in prompt_text


def test_public_chat_llm_429_uses_safe_warning(app_client):
    from app.llm import LLMRequestError

    class RateLimitedLLM:
        model = "gemini-2.5-flash"
        provider_name = "gemini"
        web_search_enabled = False

        def chat_completion(self, *, messages, temperature=0.0, max_tokens=None):
            raise LLMRequestError(
                "gemini returned HTTP 429: quota exceeded. See https://example.com/quota. GEMINI_API_KEY"
            )

    _install_constitutional_answer_service(app_client, RateLimitedLLM())

    response = app_client.post("/chat", json={"query": "ما ضمانات الحرية الشخصية في الدستور المصري؟"})
    payload = response.json()
    diagnostic_text = " ".join(
        value for value in (payload.get("warning"), payload.get("final_answer"), payload.get("llm", {}).get("error")) if value
    )

    assert response.status_code == 200
    assert payload["answer_mode"] in {"grounded", "assisted"}
    assert payload["llm"]["called"] is True
    assert payload["llm"]["succeeded"] is False
    assert payload["warning"] == SAFE_PUBLIC_LLM_WARNING
    assert len(payload["sources"]) > 0
    for leaked in ("429", "quota", "rate limit", "gemini returned", "GEMINI_API_KEY", "https://example.com/quota", "HTTP"):
        assert leaked not in diagnostic_text


def test_public_chat_invalid_json_uses_safe_warning(app_client):
    from app.llm import LLMCompletion

    class InvalidJsonLLM:
        model = "gemini-2.5-flash"
        provider_name = "gemini"
        web_search_enabled = False

        def chat_completion(self, *, messages, temperature=0.0, max_tokens=None):
            return LLMCompletion(
                content="```json\nnot valid json\n```",
                model=self.model,
                provider=self.provider_name,
                usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                raw_response=None,
            )

    _install_constitutional_answer_service(app_client, InvalidJsonLLM())

    response = app_client.post("/chat", json={"query": "ما ضمانات الحرية الشخصية في الدستور المصري؟"})
    payload = response.json()
    combined = json.dumps(payload, ensure_ascii=False)

    assert response.status_code == 200
    assert payload["answer_mode"] in {"grounded", "assisted"}
    assert payload["llm"]["called"] is True
    assert payload["llm"]["succeeded"] is True
    assert payload["warning"] == SAFE_PUBLIC_LLM_WARNING
    assert "valid json" not in combined.lower()
    assert "parse_error" not in combined


def test_public_chat_gemini_429_uses_groq_fallback(app_client):
    from app.llm import LLMCompletion, LLMRequestError

    class RateLimitedGemini:
        model = "gemini-2.5-flash"
        provider_name = "gemini"
        web_search_enabled = False
        calls = 0

        def chat_completion(self, *, messages, temperature=0.0, max_tokens=None):
            self.calls += 1
            raise LLMRequestError("gemini returned HTTP 429: quota exceeded")

    class GroqFallback:
        model = "llama-3.3-70b-versatile"
        provider_name = "groq"
        web_search_enabled = False
        calls = 0

        def chat_completion(self, *, messages, temperature=0.0, max_tokens=None):
            self.calls += 1
            return LLMCompletion(
                content=json.dumps(
                    {
                        "answer_from_sources": "Groq sources.",
                        "final_answer": "Groq final answer.",
                        "warning": None,
                    },
                    ensure_ascii=False,
                ),
                model=self.model,
                provider=self.provider_name,
                usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                raw_response=None,
            )

    primary = RateLimitedGemini()
    fallback = GroqFallback()
    _install_constitutional_answer_service(app_client, primary, fallback)

    response = app_client.post("/chat", json={"query": "ما ضمانات الحرية الشخصية في الدستور المصري؟"})
    payload = response.json()

    assert response.status_code == 200
    assert payload["llm"]["called"] is True
    assert payload["llm"]["succeeded"] is True
    assert payload["llm"]["provider"] == "groq"
    assert payload["llm"]["model"] == "llama-3.3-70b-versatile"
    assert payload["final_answer"] == "Groq final answer."
    assert payload["warning"] is None
    assert primary.calls == 1
    assert fallback.calls == 1


def test_public_chat_both_llm_providers_fail_safely(app_client):
    from app.llm import LLMRequestError

    class FailingGemini:
        model = "gemini-2.5-flash"
        provider_name = "gemini"
        web_search_enabled = False

        def chat_completion(self, *, messages, temperature=0.0, max_tokens=None):
            raise LLMRequestError("gemini returned HTTP 429: quota exceeded GEMINI_API_KEY")

    class FailingGroq:
        model = "llama-3.3-70b-versatile"
        provider_name = "groq"
        web_search_enabled = False

        def chat_completion(self, *, messages, temperature=0.0, max_tokens=None):
            raise LLMRequestError("groq returned HTTP 429: rate limit GROQ_API_KEY")

    _install_constitutional_answer_service(app_client, FailingGemini(), FailingGroq())

    response = app_client.post("/chat", json={"query": "ما ضمانات الحرية الشخصية في الدستور المصري؟"})
    payload = response.json()
    diagnostic_text = " ".join(
        value for value in (payload.get("warning"), payload.get("final_answer"), payload.get("llm", {}).get("error")) if value
    )

    assert response.status_code == 200
    assert payload["llm"]["called"] is True
    assert payload["llm"]["succeeded"] is False
    assert payload["llm"]["provider"] == "gemini"
    assert payload["warning"] == SAFE_PUBLIC_LLM_WARNING
    for leaked in ("429", "quota", "rate limit", "GROQ", "GEMINI", "api key", "API_KEY", "stack"):
        assert leaked not in diagnostic_text


def test_legal_answer_llm_error_details_only_with_debug_metadata(app_client):
    import app.settings as app_settings
    from app.llm import LLMRequestError

    class RateLimitedLLM:
        model = "gemini-2.5-flash"
        provider_name = "gemini"
        web_search_enabled = False

        def chat_completion(self, *, messages, temperature=0.0, max_tokens=None):
            raise LLMRequestError("gemini returned HTTP 429: quota exceeded. GEMINI_API_KEY")

    original_debug = app_settings.settings.debug_response_metadata

    try:
        object.__setattr__(app_settings.settings, "debug_response_metadata", False)
        _install_constitutional_answer_service(app_client, RateLimitedLLM())
        response = app_client.post(
            "/legal-answer",
            json={"query": "ما ضمانات الحرية الشخصية في الدستور المصري؟"},
        )
        payload = response.json()
        public_text = json.dumps(
            {
                "warning": payload.get("warning"),
                "final_answer": payload.get("final_answer"),
                "answer_from_sources": payload.get("answer_from_sources"),
                "external_or_assisted_explanation": payload.get("external_or_assisted_explanation"),
                "llm_error": (payload.get("llm") or {}).get("error"),
            },
            ensure_ascii=False,
        )

        assert response.status_code == 200
        assert payload["warning"] == SAFE_PUBLIC_LLM_WARNING
        assert payload["llm"]["error"] is None
        assert "429" not in public_text
        assert "GEMINI_API_KEY" not in public_text

        object.__setattr__(app_settings.settings, "debug_response_metadata", True)
        _install_constitutional_answer_service(app_client, RateLimitedLLM())
        debug_response = app_client.post(
            "/legal-answer",
            json={"query": "ما ضمانات الحرية الشخصية في الدستور المصري؟"},
        )
        debug_payload = debug_response.json()

        assert debug_response.status_code == 200
        assert "429" in debug_payload["llm"]["error"]
        assert "quota exceeded" in debug_payload["llm"]["error"]
        assert "429" in debug_payload["llm"]["primary_error"]
    finally:
        object.__setattr__(app_settings.settings, "debug_response_metadata", original_debug)


def test_legal_answer_endpoint_keeps_full_prompt_and_budget(app_client):
    from app.answering import LegalAnswerService
    from app.llm import LLMCompletion

    captured = {}

    class FakeLLM:
        model = "gemini-2.5-flash"
        provider_name = "gemini"
        web_search_enabled = False

        def chat_completion(self, *, messages, temperature=0.0, max_tokens=None):
            captured["messages"] = messages
            captured["max_tokens"] = max_tokens
            return LLMCompletion(
                content=json.dumps(
                    {
                        "answer_from_sources": "المادة 54 من دستور جمهورية مصر العربية.",
                        "final_answer": "جواب كامل.\n\nالسند القانوني\nالمادة 54.\n\nالمصادر\nدستور جمهورية مصر العربية.",
                        "warning": None,
                    },
                    ensure_ascii=False,
                ),
                model=self.model,
                provider=self.provider_name,
                usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
                raw_response=None,
            )

    class FakeRetriever:
        def search(self, query: str, *, top_k=None, filters=None):
            return {
                "query": query,
                "normalized_query": query,
                "query_analysis": {"out_of_domain": False, "suggested_domain": "constitutional_law"},
                "results": [
                    {
                        "id": "const-54",
                        "rerank_score": 0.91,
                        "score": 0.86,
                        "law_name": "دستور جمهورية مصر العربية",
                        "article_number": "54",
                        "title": "الحرية الشخصية",
                        "legal_domain": "constitutional_law",
                        "source_url": "https://example.com/constitution/54",
                        "summary": "تضمن المادة 54 ضمانات الحرية الشخصية.",
                        "content": "الحرية الشخصية حق طبيعي وهي مصونة لا تمس، ولا يجوز القبض أو التفتيش أو الحبس إلا بأمر قضائي.",
                        "rank_explanation": ["strong_summary_overlap"],
                    },
                    {
                        "id": "const-92",
                        "rerank_score": 0.84,
                        "score": 0.79,
                        "law_name": "دستور جمهورية مصر العربية",
                        "article_number": "92",
                        "title": "الحقوق والحريات",
                        "legal_domain": "constitutional_law",
                        "source_url": "https://example.com/constitution/92",
                        "summary": "لا تقبل الحقوق والحريات اللصيقة بشخص المواطن تعطيلا ولا انتقاصا.",
                        "content": "الحقوق والحريات اللصيقة بشخص المواطن لا تقبل تعطيلا ولا انتقاصا.",
                        "rank_explanation": ["strong_title_overlap"],
                    },
                ],
            }

    app_client.app.state.legal_answer_service = LegalAnswerService(retriever=FakeRetriever(), llm_client=FakeLLM())

    response = app_client.post("/legal-answer", json={"query": "ما ضمانات الحرية الشخصية في الدستور المصري؟"})
    payload = response.json()

    assert response.status_code == 200
    assert payload["answer_mode"] in {"grounded", "assisted"}
    assert "retrieval_summary" in payload
    assert captured["max_tokens"] > 1536
    prompt_text = "\n".join(message["content"] for message in captured["messages"])
    assert "نمط الإخراج العام لـ /chat" not in prompt_text
