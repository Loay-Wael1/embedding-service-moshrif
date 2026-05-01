from __future__ import annotations

from fastapi import FastAPI, HTTPException, Request

from app.answering import LegalAnswerRequest, LegalAnswerResponse, LegalAnswerService
from app.api.schemas import EmbedBatchRequest, EmbedRequest, EmbedResponse, EmbeddingResultResponse, ServiceInfoResponse
from app.embeddings.service import EmbeddingService, get_default_embedding_service
from app.models import RetrievalFilters
from app.retrieval import LegalRetriever


def create_app(
    embedding_service: EmbeddingService | None = None,
    answer_service: LegalAnswerService | None = None,
) -> FastAPI:
    app = FastAPI(
        title="Egyptian Laws Embedding Service",
        version="2.0.0",
        description="Embedding API for Egyptian-law retrieval with legal-text normalization.",
    )

    app.state.embedding_service = embedding_service or get_default_embedding_service()
    app.state.legal_answer_service = answer_service

    def _get_service(request: Request) -> EmbeddingService:
        return request.app.state.embedding_service

    def _get_answer_service(request: Request) -> LegalAnswerService:
        service = request.app.state.legal_answer_service
        if service is None:
            embedding = _get_service(request)
            retriever = LegalRetriever(
                embedding_service=embedding,
                config=getattr(embedding, "settings", None),
            )
            service = LegalAnswerService(retriever=retriever, config=getattr(embedding, "settings", None))
            request.app.state.legal_answer_service = service
        return service

    @app.get("/health")
    def health(request: Request) -> dict[str, object]:
        service = _get_service(request)
        return {
            "status": "ok",
            "model_loaded": service.is_loaded,
            "supports_sparse": service.supports_sparse,
        }

    @app.get("/info", response_model=ServiceInfoResponse)
    def info(request: Request) -> ServiceInfoResponse:
        service = _get_service(request)
        return ServiceInfoResponse(**service.get_info())

    @app.get("/legal-info")
    def legal_info() -> dict[str, object]:
        return {
            "service": "almostashar-legal-rag",
            "app_name": "المستشار",
            "status": "ok",
            "description": "Legal RAG API for Egyptian law",
            "llm_provider": "gemini",
            "retrieval_backend": "qdrant",
            "embedding_backend": "flagembedding_bgem3",
            "answer_modes": [
                "identity",
                "conversation",
                "grounded",
                "assisted",
                "external_assisted",
                "insufficient",
                "non_legal",
            ],
            "supported_internal_domains": [
                "labor_law",
                "civil_law",
                "criminal_law",
                "constitutional_law",
            ],
            "out_of_internal_corpus_examples": [
                "الحضانة",
                "النفقة",
                "الطلاق",
                "الخلع",
                "الميراث",
            ],
        }

    @app.post("/embed", response_model=EmbedResponse)
    def embed(request_body: EmbedRequest, request: Request) -> EmbedResponse:
        service = _get_service(request)
        if not request_body.text.strip():
            raise HTTPException(status_code=400, detail="text must not be empty")

        results, warnings = service.embed_texts(
            [request_body.text],
            mode=request_body.mode,
            normalize=request_body.normalize,
            return_dense=request_body.return_dense,
            return_sparse=request_body.return_sparse,
        )
        return _to_embed_response(service, request_body.mode, request_body.normalize, results, warnings)

    @app.post("/embed/batch", response_model=EmbedResponse)
    def embed_batch(request_body: EmbedBatchRequest, request: Request) -> EmbedResponse:
        service = _get_service(request)
        if not request_body.texts:
            raise HTTPException(status_code=400, detail="texts must not be empty")
        if any(not text.strip() for text in request_body.texts):
            raise HTTPException(status_code=400, detail="texts must not contain empty items")

        results, warnings = service.embed_texts(
            request_body.texts,
            mode=request_body.mode,
            normalize=request_body.normalize,
            return_dense=request_body.return_dense,
            return_sparse=request_body.return_sparse,
        )
        return _to_embed_response(service, request_body.mode, request_body.normalize, results, warnings)

    @app.post("/legal-answer", response_model=LegalAnswerResponse)
    @app.post("/ask-legal", response_model=LegalAnswerResponse)
    def legal_answer(request_body: LegalAnswerRequest, request: Request) -> LegalAnswerResponse:
        if not request_body.query.strip():
            raise HTTPException(status_code=400, detail="query must not be empty")

        filters = RetrievalFilters(
            legal_domain=None if request_body.legal_domain in (None, "all") else request_body.legal_domain,
            law_number=request_body.law_number,
            law_year=request_body.law_year,
            status_normalized=request_body.status_normalized,
            exclude_repealed=request_body.exclude_repealed,
        )
        try:
            return _get_answer_service(request).answer(
                request_body.query,
                top_k=request_body.top_k,
                filters=filters,
                include_retrieval=request_body.include_retrieval,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    return app


def _to_embed_response(
    service: EmbeddingService,
    mode: str,
    normalize: bool,
    results,
    warnings: list[str],
) -> EmbedResponse:
    return EmbedResponse(
        model=service.model_name,
        dim=service.embedding_dimension,
        mode=mode,
        normalized=normalize,
        sparse_available=service.supports_sparse,
        warnings=warnings,
        results=[
            EmbeddingResultResponse(
                text=result.text,
                normalized_text=result.normalized_text,
                dense=result.dense,
                sparse=None
                if result.sparse is None
                else {
                    "indices": result.sparse.indices,
                    "values": result.sparse.values,
                },
                metadata=result.metadata,
            )
            for result in results
        ],
    )


app = create_app()
