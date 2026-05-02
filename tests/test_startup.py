def test_openapi_schema(app_client):
    response = app_client.get("/openapi.json")
    schema = response.json()
    chat_request_schema = schema["components"]["schemas"]["ChatRequest"]
    assert "session_id" not in chat_request_schema["properties"]
    assert "conversation_id" not in chat_request_schema["properties"]
    assert "query" in chat_request_schema["properties"]

def test_health_ultra_light(app_client):
    # Ensure answer service is not loaded
    app = app_client.app
    assert app.state.legal_answer_service is None
    
    response = app_client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert app.state.legal_answer_service is None

def test_legal_info_ultra_light(app_client, monkeypatch):
    import app.answering.service as answering_service
    
    retriever_instantiated = False
    class FakeRetriever:
        def __init__(self, *args, **kwargs):
            nonlocal retriever_instantiated
            retriever_instantiated = True
            
    monkeypatch.setattr(answering_service, "LegalRetriever", FakeRetriever)
    
    response = app_client.get("/legal-info")
    assert response.status_code == 200
    assert not retriever_instantiated

def test_chat_fast_path_does_not_instantiate_retriever(app_client, monkeypatch):
    import app.answering.service as answering_service
    
    retriever_instantiated = False
    class FakeRetriever:
        def __init__(self, *args, **kwargs):
            nonlocal retriever_instantiated
            retriever_instantiated = True
            
    monkeypatch.setattr(answering_service, "LegalRetriever", FakeRetriever)
    
    response = app_client.post("/chat", json={"query": "السلام عليكم"})
    assert response.status_code == 200
    assert not retriever_instantiated

def test_warmup_loads_retriever(app_client, monkeypatch):
    import app.answering.service as answering_service
    import app.embeddings.service as embedding_service
    
    retriever_instantiated = False
    class FakeRetriever:
        def __init__(self, *args, **kwargs):
            nonlocal retriever_instantiated
            retriever_instantiated = True
            
    monkeypatch.setattr(answering_service, "LegalRetriever", FakeRetriever)
    
    # Also mock _ensure_model
    original_ensure_model = embedding_service.EmbeddingService._ensure_model
    model_loaded = False
    
    def fake_ensure_model(self):
        nonlocal model_loaded
        model_loaded = True
        # Do not call original
        
    monkeypatch.setattr(embedding_service.EmbeddingService, "_ensure_model", fake_ensure_model)
    
    response = app_client.post("/warmup")
    assert response.status_code == 200
    payload = response.json()
    assert payload["retriever_loaded"] is True
    assert retriever_instantiated
