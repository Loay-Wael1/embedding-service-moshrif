from __future__ import annotations

import hmac
import re
import time

from collections import OrderedDict

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from app.answering import ChatRequest, LegalAnswerRequest, LegalAnswerResponse, LegalAnswerService
from app.answering.schemas import AnswerParts, ChatResponse, CompactSourceCitation, CompactLLMMetadata
from app.api.schemas import EmbedBatchRequest, EmbedRequest, EmbedResponse, EmbeddingResultResponse, ServiceInfoResponse
from app.embeddings.service import EmbeddingService, get_default_embedding_service
from app.models import RetrievalFilters
from app.runtime_assets import ensure_runtime_assets
from app.settings import settings


def create_app(
    embedding_service: EmbeddingService | None = None,
    answer_service: LegalAnswerService | None = None,
) -> FastAPI:
    import logging
    from contextlib import asynccontextmanager

    logger = logging.getLogger("api.startup")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("API startup begin")
        
        if settings.preload_retriever:
            logger.info("PRELOAD_RETRIEVER=true: Loading retriever and embedding models...")
            ensure_runtime_assets()
            embedding_svc = app.state.embedding_service
            # Eagerly load model
            if not embedding_svc.is_loaded:
                embedding_svc._ensure_model()
            
            answer_svc = app.state.legal_answer_service
            if answer_svc is None:
                answer_svc = LegalAnswerService(config=getattr(embedding_svc, "settings", None))
                app.state.legal_answer_service = answer_svc
            # Force retriever instantiation
            _ = answer_svc.retriever
            logger.info("Retriever and models fully loaded.")
        else:
            logger.info("Retriever preload disabled. Models will load lazily on first legal query.")
            
        logger.info("API startup complete")
        yield

    app = FastAPI(
        title="Egyptian Laws Embedding Service",
        version="2.0.0",
        description="Embedding API for Egyptian-law retrieval with legal-text normalization.",
        lifespan=lifespan,
        docs_url="/docs" if settings.enable_public_docs else None,
        redoc_url="/redoc" if settings.enable_public_docs else None,
        openapi_url="/openapi.json" if settings.enable_public_docs else None,
    )

    # CORS for frontend/mobile access.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.embedding_service = embedding_service or get_default_embedding_service()
    app.state.legal_answer_service = answer_service
    app.state.chat_cache = OrderedDict()

    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        start_time = time.perf_counter()
        response = await call_next(request)
        process_time = time.perf_counter() - start_time
        response.headers["X-Process-Time-Ms"] = str(round(process_time * 1000, 2))
        return response

    def _get_service(request: Request) -> EmbeddingService:
        return request.app.state.embedding_service

    def _get_answer_service(request: Request) -> LegalAnswerService:
        """Lazy singleton — does NOT eagerly create LegalRetriever or load Qdrant."""
        service = request.app.state.legal_answer_service
        if service is None:
            # Retriever is lazy inside LegalAnswerService.retriever property.
            service = LegalAnswerService(config=getattr(_get_service(request), "settings", None))
            request.app.state.legal_answer_service = service
        return service

    def _require_internal_api_token(request: Request) -> None:
        if not settings.require_internal_api_token:
            return

        expected_token = settings.internal_api_token
        provided_token = request.headers.get(settings.internal_api_token_header)
        if (
            not expected_token
            or not provided_token
            or not hmac.compare_digest(provided_token, expected_token)
        ):
            raise HTTPException(status_code=401, detail="Unauthorized")

    @app.get("/health")
    def health(request: Request) -> dict[str, object]:
        return {
            "status": "ok",
            "service": "almostashar-legal-rag",
        }

    @app.post("/warmup")
    def warmup(request: Request) -> dict[str, object]:
        """Loads models if they are not already loaded (e.g. after a cold start)."""
        _require_internal_api_token(request)
        answer_svc = _get_answer_service(request)
        embedding_svc = _get_service(request)
        ensure_runtime_assets(config=getattr(embedding_svc, "settings", None))
        if not embedding_svc.is_loaded:
            embedding_svc._ensure_model()
        # Force retriever init
        _ = answer_svc.retriever
        return {
            "status": "ok",
            "retriever_loaded": True,
        }

    @app.get("/info", response_model=ServiceInfoResponse, include_in_schema=False)
    def info(request: Request) -> ServiceInfoResponse:
        _require_internal_api_token(request)
        service = _get_service(request)
        return ServiceInfoResponse(**service.get_info())

    @app.get("/legal-info")
    def legal_info(request: Request) -> dict[str, object]:
        if settings.protect_legal_info:
            _require_internal_api_token(request)
        return {
            "service": "almostashar-legal-rag",
            "app_name": "المستشار",
            "status": "ok",
            "description": "Legal RAG API for Egyptian law",
            "llm_provider": settings.llm_provider_name,
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

    @app.post("/embed", response_model=EmbedResponse, include_in_schema=False)
    def embed(request_body: EmbedRequest, request: Request) -> EmbedResponse:
        _require_internal_api_token(request)
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

    @app.post("/embed/batch", response_model=EmbedResponse, include_in_schema=False)
    def embed_batch(request_body: EmbedBatchRequest, request: Request) -> EmbedResponse:
        _require_internal_api_token(request)
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

    @app.post("/chat", response_model=ChatResponse | LegalAnswerResponse)
    def chat(
        request_body: ChatRequest,
        request: Request,
        http_response: Response,
        debug: bool = Query(False, include_in_schema=False)
    ) -> ChatResponse | LegalAnswerResponse:
        from app.preprocessing import normalize_legal_arabic

        _require_internal_api_token(request)
        
        normalized_query = normalize_legal_arabic(request_body.query)
        cache: OrderedDict = request.app.state.chat_cache
        include_debug = bool(debug or settings.debug_response_metadata)
        
        # Check cache if not in debug mode
        if not include_debug and normalized_query in cache:
            cache.move_to_end(normalized_query)
            cached_value = cache[normalized_query]
            if isinstance(cached_value, tuple):
                cached_response, cached_headers = cached_value
            else:
                cached_response, cached_headers = cached_value, {}
            cached_response = _repair_chat_answer_parts(cached_response)
            if isinstance(cached_value, tuple):
                cache[normalized_query] = (cached_response, cached_headers)
            else:
                cache[normalized_query] = cached_response
            _set_chat_headers(http_response, cached_headers, cache_hit=True)
            return cached_response

        try:
            answer = _get_answer_service(request).answer(
                request_body.query,
                top_k=settings.chat_answer_top_k,
                concise=settings.chat_concise_answers,
            )
            headers = _chat_header_values(answer)
            _set_chat_headers(http_response, headers, cache_hit=False)
            
            if include_debug:
                return _sanitize_response(answer, include_debug=True, expose_llm_errors=False)
                
            # Map to Compact ChatResponse
            compact_sources = [
                CompactSourceCitation(
                    law_name=s.law_name,
                    article_number=s.article_number,
                    title=s.title,
                    source_url=s.source_url,
                    legal_domain=s.legal_domain,
                ) for s in answer.sources
            ]
            
            chat_resp = ChatResponse(
                answer_mode=answer.answer_mode,
                final_answer=answer.final_answer,
                answer_parts=answer.answer_parts,
                warning=answer.warning,
                is_legal_question=answer.is_legal_question,
                is_supported_by_internal_sources=answer.is_supported_by_internal_sources,
                is_out_of_internal_corpus=answer.is_out_of_internal_corpus,
                sources=compact_sources,
                llm=CompactLLMMetadata(
                    called=answer.llm.called,
                    succeeded=answer.llm.succeeded,
                    provider=answer.llm.provider,
                    model=answer.llm.model,
                ),
            )
            chat_resp = _repair_chat_answer_parts(chat_resp)
            
            # Cache only deterministic or successfully generated safe responses.
            if not include_debug and is_cacheable_chat_response(chat_resp) and settings.chat_response_cache_size > 0:
                cache[normalized_query] = (chat_resp, headers)
                cache.move_to_end(normalized_query)
                if len(cache) > settings.chat_response_cache_size:
                    cache.popitem(last=False)
                    
            return chat_resp
            
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post("/legal-answer", response_model=LegalAnswerResponse, include_in_schema=False)
    def legal_answer(request_body: LegalAnswerRequest, request: Request) -> LegalAnswerResponse:
        _require_internal_api_token(request)
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
            response = _get_answer_service(request).answer(
                request_body.query,
                top_k=request_body.top_k,
                filters=filters,
                include_retrieval=request_body.include_retrieval,
            )
            return _sanitize_response(response, expose_llm_errors=settings.debug_response_metadata)
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


def _chat_header_values(response: LegalAnswerResponse) -> dict[str, str]:
    router = response.router
    return {
        "X-Answer-Mode": response.answer_mode,
        "X-LLM-Called": str(bool(response.llm.called)).lower(),
        "X-Sources-Count": str(len(response.sources)),
        "X-Router-Intent": router.intent if router else "",
        "X-Router-Confidence": str(router.confidence if router else ""),
        "X-Router-Domain": router.suggested_domain if router and router.suggested_domain else "",
    }


def _set_chat_headers(http_response: Response, headers: dict[str, str], *, cache_hit: bool) -> None:
    http_response.headers["X-Cache-Hit"] = str(cache_hit).lower()
    for name, value in headers.items():
        http_response.headers[name] = value


def is_cacheable_chat_response(response: ChatResponse) -> bool:
    if response.answer_mode in {"identity", "conversation"}:
        return True
    if response.answer_mode in {"grounded", "assisted", "external_assisted"}:
        return bool(response.llm.succeeded)
    return False


def _repair_chat_answer_parts(response: ChatResponse) -> ChatResponse:
    if response.answer_parts is not None or not response.final_answer:
        return response
    response.answer_parts = _answer_parts_from_final_answer(
        response.final_answer,
        mode=response.answer_mode,
        warning=response.warning,
    )
    return response


def _answer_parts_from_final_answer(final_answer: str, *, mode: str, warning: str | None = None) -> AnswerParts:
    lines = [line.strip() for line in final_answer.splitlines() if line.strip()]
    if not lines:
        return AnswerParts(intro=final_answer, bullets=[])

    heading_index = _first_answer_heading_index(lines)
    intro = " ".join(lines[:heading_index]).strip() if heading_index is not None and heading_index > 0 else lines[0]
    section_title = _normalise_answer_heading(lines[heading_index]) if heading_index is not None else None

    bullets: list[str] = []
    legal_basis_lines: list[str] = []
    note_lines: list[str] = []
    active: str | None = None
    start = heading_index + 1 if heading_index is not None else 1

    for line in lines[start:]:
        normalized = line.rstrip(":").strip()
        if _is_bullet_line(line):
            bullet = _clean_bullet_line(line)
            if bullet:
                bullets.append(bullet)
            continue
        if normalized == "السند القانوني":
            active = "legal_basis"
            continue
        if normalized in {"ملاحظة", "تنبيه"}:
            active = "note"
            continue
        if active == "legal_basis":
            legal_basis_lines.append(line)
        elif active == "note":
            note_lines.append(line)

    if mode == "external_assisted":
        section_title = section_title or "شرح عام:"
        legal_basis = None
        note = " ".join(note_lines).strip() or warning or "هذه إجابة عامة وليست مستندة إلى مصادر داخلية موثقة."
    else:
        legal_basis = " ".join(legal_basis_lines).strip() or None
        note = " ".join(note_lines).strip() or None

    return AnswerParts(
        intro=intro,
        section_title=section_title,
        bullets=bullets[:6],
        legal_basis=legal_basis,
        note=note,
    )


def _first_answer_heading_index(lines: list[str]) -> int | None:
    for index, line in enumerate(lines):
        if line.rstrip(":").strip() in {"أهم الأحكام", "أهم الضمانات", "النقاط الأساسية", "الخطوات العملية", "ما يمكنك فعله", "شرح عام"}:
            return index
    return None


def _normalise_answer_heading(value: str) -> str:
    text = value.strip()
    return text if text.endswith(":") else f"{text}:"


def _is_bullet_line(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith(("-", "•", "*")) or bool(re.match(r"^[0-9٠-٩١٢٣٤٥٦٧٨٩]+[.)-]\s+", stripped))


def _clean_bullet_line(line: str) -> str:
    stripped = line.strip().lstrip("-•* ").strip()
    stripped = re.sub(r"^[0-9٠-٩١٢٣٤٥٦٧٨٩]+[.)-]\s+", "", stripped).strip()
    return stripped


def _sanitize_response(
    response: LegalAnswerResponse,
    *,
    include_debug: bool | None = None,
    expose_llm_errors: bool = False,
) -> LegalAnswerResponse:
    """Strip debug/diagnostic fields in production."""
    keep_debug = settings.debug_response_metadata if include_debug is None else include_debug
    if not keep_debug or not expose_llm_errors:
        if _has_llm_diagnostic(response) and not response.llm.succeeded:
            response.warning = _safe_llm_warning(response)
        response.llm.error = None
        response.llm.parse_error = None
        response.llm.schema_error = None
        response.llm.raw_response_preview = None
        response.llm.raw_response_repr_preview = None
        response.llm.usage = None
        response.llm.primary_provider = None
        response.llm.primary_model = None
        response.llm.primary_error = None
        response.llm.fallback_provider = None
        response.llm.fallback_model = None
        response.llm.fallback_used = False
        response.llm.fallback_error = None
    return response


def _has_llm_diagnostic(response: LegalAnswerResponse) -> bool:
    return bool(
        response.llm.error
        or response.llm.parse_error
        or response.llm.schema_error
        or response.llm.raw_response_preview
        or response.llm.raw_response_repr_preview
        or response.llm.primary_error
        or response.llm.fallback_error
    )


def _safe_llm_warning(response: LegalAnswerResponse) -> str:
    if response.answer_mode in {"grounded", "assisted"} and response.sources:
        return "تعذر توليد الصياغة النهائية حاليًا، وتم عرض إجابة مستندة إلى المصادر الداخلية المتاحة."
    if response.answer_mode == "external_assisted":
        return "هذا السؤال خارج مصادر التطبيق الداخلية المتاحة حاليًا، لذلك لا أستطيع توثيق الإجابة منها."
    return "تعذر توليد الصياغة النهائية حاليًا."


app = create_app()
