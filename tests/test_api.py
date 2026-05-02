import json


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


def test_chat_endpoint_non_legal_intent(app_client):
    response = app_client.post("/chat", json={"query": "ما أفضل مطعم؟"})
    payload = response.json()
    assert response.status_code == 200
    assert payload["answer_mode"] == "non_legal"
    assert payload["llm"]["called"] is False
    assert "retrieval_summary" not in payload


def test_chat_endpoint_ambiguous_intent(app_client):
    response = app_client.post("/chat", json={"query": "ما هي؟"})
    payload = response.json()
    assert response.status_code == 200
    assert payload["answer_mode"] == "insufficient"
    assert payload["llm"]["called"] is False
    assert "retrieval_summary" not in payload


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
    # First call
    response1 = app_client.post("/chat", json={"query": "ما أفضل مطعم؟"})
    assert response1.status_code == 200
    
    # Check cache state
    cache = app_client.app.state.chat_cache
    assert len(cache) > 0
    
    # Second call
    response2 = app_client.post("/chat", json={"query": "ما أفضل مطعم؟"})
    assert response2.status_code == 200
    
    # The cache hit should be identical to the first response
    assert response1.json() == response2.json()
    assert response2.headers["X-Cache-Hit"] == "true"


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
    assert "نمط الإخراج العام المختصر لـ /chat" in prompt_text
    assert "لا تضع داخل final_answer قسمًا بعنوان \"المصادر\"" in prompt_text


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
