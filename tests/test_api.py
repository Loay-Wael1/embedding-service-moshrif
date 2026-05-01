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
