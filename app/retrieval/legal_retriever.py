from __future__ import annotations

from typing import Any

from qdrant_client import QdrantClient, models

from app.embeddings import EmbeddingService
from app.models import RetrievalFilters, make_point_id
from app.retrieval.query_understanding import QueryAnalysis, analyze_legal_query
from app.retrieval.rerank import BaseReranker, build_reranker
from app.settings import Settings, settings


class LegalRetriever:
    def __init__(
        self,
        *,
        embedding_service: EmbeddingService | None = None,
        client: QdrantClient | None = None,
        config: Settings | None = None,
        reranker: BaseReranker | None = None,
    ) -> None:
        import logging
        logger = logging.getLogger("api.retriever")
        logger.info("Loading LegalRetriever (This may take a moment to initialize models and connections)...")

        self.settings = config or settings
        self.embedding_service = embedding_service or EmbeddingService(self.settings)
        self.client = client or QdrantClient(path=self.settings.qdrant_path)
        self.collection_name = self.settings.collection_name
        self.hybrid_enabled = self.settings.enable_sparse and self.embedding_service.supports_sparse
        if reranker is not None:
            self.reranker = reranker
        else:
            self.reranker = build_reranker(self.settings.retrieval_reranker, self.settings)
        logger.info("LegalRetriever initialization complete.")

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        filters: RetrievalFilters | None = None,
    ) -> dict[str, Any]:
        if not query.strip():
            raise ValueError("query must not be empty")
        if not self.client.collection_exists(self.collection_name):
            raise RuntimeError(
                f"Qdrant collection '{self.collection_name}' was not found. Run the legal index build first."
            )

        filters = filters or RetrievalFilters()
        final_k = top_k or self.settings.retrieval_top_k
        query_analysis = analyze_legal_query(query)
        effective_domain = filters.legal_domain or query_analysis.suggested_domain
        effective_filters = RetrievalFilters(
            legal_domain=effective_domain,
            law_number=filters.law_number,
            law_year=filters.law_year,
            status_normalized=filters.status_normalized,
            exclude_repealed=filters.exclude_repealed,
        )

        if query_analysis.out_of_domain:
            return {
                "query": query,
                "normalized_query": query_analysis.normalized_query,
                "rewritten_query": query_analysis.rewritten_query,
                "query_analysis": {
                    "intent": query_analysis.intent,
                    "key_phrases": query_analysis.key_phrases,
                    "legal_keywords": query_analysis.legal_keywords,
                    "domain_hints": query_analysis.domain_hints,
                    "prefer_specific_articles": query_analysis.prefer_specific_articles,
                    "constitutional_rights_query": query_analysis.constitutional_rights_query,
                    "preamble_related": query_analysis.preamble_related,
                    "suggested_domain": query_analysis.suggested_domain,
                    "domain_scores": query_analysis.domain_scores,
                    "out_of_domain": query_analysis.out_of_domain,
                    "out_of_domain_reason": query_analysis.out_of_domain_reason,
                    "criminal_offense_query": query_analysis.criminal_offense_query,
                    "criminal_offense_terms": query_analysis.criminal_offense_terms,
                    "multi_offense_query": query_analysis.multi_offense_query,
                },
                "filters_applied": {
                    "legal_domain": effective_filters.legal_domain,
                    "law_number": effective_filters.law_number,
                    "law_year": effective_filters.law_year,
                    "status_normalized": effective_filters.status_normalized,
                    "exclude_repealed": effective_filters.exclude_repealed or self.settings.retrieval_exclude_non_current,
                },
                "warnings": [query_analysis.out_of_domain_reason or "query is outside the current corpus scope"],
                "retrieval_backend": {
                    "hybrid_enabled": False,
                    "fusion": self.settings.retrieval_hybrid_fusion,
                    "dense_backend": None,
                    "sparse_backend": None,
                    "reranker": self.settings.retrieval_reranker,
                    "auto_domain_filter": effective_filters.legal_domain,
                    "short_circuit_reason": "out_of_domain_detected",
                    "limits": {
                        "chunk_limit": 0,
                        "article_limit": 0,
                        "law_limit": 0,
                        "candidate_limit": 0,
                    },
                },
                "routing_law_hits": [],
                "results": [],
            }
        chunk_limit, article_limit, law_limit, candidate_limit = self._candidate_limits(final_k, query_analysis)

        embeddings, warnings = self.embedding_service.embed_texts(
            [query_analysis.rewritten_query],
            mode="query",
            normalize=True,
            return_dense=True,
            return_sparse=self.hybrid_enabled,
        )
        query_embedding = embeddings[0]

        chunk_hits = self._search_record_kind(
            record_kind="article_chunk",
            query_vector=query_embedding.dense or [],
            sparse_vector=query_embedding.sparse,
            limit=chunk_limit,
            filters=effective_filters,
        )
        article_hits = self._search_record_kind(
            record_kind="article",
            query_vector=query_embedding.dense or [],
            sparse_vector=query_embedding.sparse,
            limit=article_limit,
            filters=effective_filters,
        )
        law_hits = self._search_record_kind(
            record_kind="law",
            query_vector=query_embedding.dense or [],
            sparse_vector=query_embedding.sparse,
            limit=law_limit,
            filters=effective_filters,
        )

        candidates = self._merge_article_candidates(chunk_hits, article_hits)[:candidate_limit]
        reranked = self.reranker.rerank(
            query_analysis.normalized_query,
            candidates,
            query_analysis=query_analysis,
        )

        return {
            "query": query,
            "normalized_query": query_analysis.normalized_query,
            "rewritten_query": query_analysis.rewritten_query,
            "query_analysis": {
                "intent": query_analysis.intent,
                "key_phrases": query_analysis.key_phrases,
                "legal_keywords": query_analysis.legal_keywords,
                "domain_hints": query_analysis.domain_hints,
                "prefer_specific_articles": query_analysis.prefer_specific_articles,
                "constitutional_rights_query": query_analysis.constitutional_rights_query,
                "preamble_related": query_analysis.preamble_related,
                "suggested_domain": query_analysis.suggested_domain,
                "domain_scores": query_analysis.domain_scores,
                "out_of_domain": query_analysis.out_of_domain,
                "out_of_domain_reason": query_analysis.out_of_domain_reason,
                "criminal_offense_query": query_analysis.criminal_offense_query,
                "criminal_offense_terms": query_analysis.criminal_offense_terms,
                "multi_offense_query": query_analysis.multi_offense_query,
            },
            "filters_applied": {
                "legal_domain": effective_filters.legal_domain,
                "law_number": effective_filters.law_number,
                "law_year": effective_filters.law_year,
                "status_normalized": effective_filters.status_normalized,
                "exclude_repealed": effective_filters.exclude_repealed or self.settings.retrieval_exclude_non_current,
            },
            "warnings": warnings,
            "retrieval_backend": {
                "hybrid_enabled": self.hybrid_enabled and query_embedding.sparse is not None,
                "fusion": self.settings.retrieval_hybrid_fusion,
                "dense_backend": "qdrant_dense",
                "sparse_backend": "qdrant_sparse" if self.hybrid_enabled else None,
                "reranker": self.settings.retrieval_reranker,
                "auto_domain_filter": None if filters.legal_domain else query_analysis.suggested_domain,
                "limits": {
                    "chunk_limit": chunk_limit,
                    "article_limit": article_limit,
                    "law_limit": law_limit,
                    "candidate_limit": candidate_limit,
                },
            },
            "routing_law_hits": [self._point_to_summary(hit) for hit in law_hits],
            "results": reranked[:final_k],
        }

    def _search_record_kind(
        self,
        *,
        record_kind: str,
        query_vector: list[float],
        sparse_vector: Any | None,
        limit: int,
        filters: RetrievalFilters,
    ) -> list[models.ScoredPoint]:
        query_filter = self._build_filter(record_kind=record_kind, filters=filters)
        if self.hybrid_enabled and sparse_vector is not None:
            prefetch_limit = max(limit, min(limit * 2, limit + 12))
            response = self.client.query_points(
                collection_name=self.collection_name,
                prefetch=[
                    models.Prefetch(
                        query=query_vector,
                        using="dense",
                        limit=prefetch_limit,
                        filter=query_filter,
                    ),
                    models.Prefetch(
                        query=models.SparseVector(
                            indices=sparse_vector.indices,
                            values=sparse_vector.values,
                        ),
                        using="sparse",
                        limit=prefetch_limit,
                        filter=query_filter,
                    ),
                ],
                query=models.FusionQuery(fusion=self._fusion_mode()),
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )
            return response.points if hasattr(response, "points") else response

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            using="dense",
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        return response.points if hasattr(response, "points") else response

    def _fusion_mode(self) -> models.Fusion:
        if self.settings.retrieval_hybrid_fusion.lower() == "dbsf":
            return models.Fusion.DBSF
        return models.Fusion.RRF

    def _merge_article_candidates(
        self,
        chunk_hits: list[models.ScoredPoint],
        article_hits: list[models.ScoredPoint],
    ) -> list[dict[str, Any]]:
        article_ids = {str(hit.id) for hit in article_hits}
        article_ids.update(
            make_point_id(str(hit.payload.get("parent_id")))
            for hit in chunk_hits
            if hit.payload.get("parent_id")
        )
        if not article_ids:
            return []

        article_records = self.client.retrieve(
            collection_name=self.collection_name,
            ids=list(article_ids),
            with_payload=True,
            with_vectors=False,
        )
        article_map = {str(record.id): record for record in article_records}

        grouped_chunks: dict[str, list[models.ScoredPoint]] = {}
        for hit in chunk_hits:
            parent_id = hit.payload.get("parent_id")
            if not parent_id:
                continue
            grouped_chunks.setdefault(make_point_id(str(parent_id)), []).append(hit)

        article_scores = {str(hit.id): float(hit.score) for hit in article_hits}
        candidates = []

        for article_id, article_record in article_map.items():
            payload = article_record.payload or {}
            supporting_chunks = sorted(
                grouped_chunks.get(article_id, []),
                key=lambda item: item.score,
                reverse=True,
            )

            matched_record_kind = "article" if article_id in article_scores else "article_chunk"
            best_chunk_score = supporting_chunks[0].score if supporting_chunks else 0.0
            direct_article_score = article_scores.get(article_id, 0.0)
            base_score = max(direct_article_score, best_chunk_score)
            if supporting_chunks:
                base_score += min(0.08, 0.02 * len(supporting_chunks))
            if matched_record_kind == "article":
                base_score += 0.03
            if payload.get("is_repealed_candidate"):
                base_score -= 0.12
            base_score -= float(payload.get("noise_score", 0.0)) * 0.1

            candidates.append(
                {
                    "id": payload.get("id", article_id),
                    "record_kind": payload.get("record_kind", "article"),
                    "matched_record_kind": matched_record_kind,
                    "score": round(base_score, 6),
                    "direct_article_score": round(direct_article_score, 6),
                    "best_chunk_score": round(best_chunk_score, 6),
                    "supporting_chunk_count": len(supporting_chunks),
                    "title": payload.get("title"),
                    "content": payload.get("content"),
                    "summary": payload.get("summary"),
                    "legal_domain": payload.get("legal_domain"),
                    "law_name": payload.get("law_name"),
                    "law_number": payload.get("law_number"),
                    "law_year": payload.get("law_year"),
                    "article_number": payload.get("article_number"),
                    "source_url": payload.get("source_url"),
                    "status": payload.get("status"),
                    "status_normalized": payload.get("status_normalized"),
                    "quality_flags": payload.get("quality_flags", []),
                    "quality_warnings": payload.get("quality_warnings", []),
                    "noise_score": payload.get("noise_score", 0.0),
                    "quality_score": payload.get("quality_score", 1.0),
                    "document_level": payload.get("document_level"),
                    "section_level": payload.get("section_level"),
                    "retrieval_text": payload.get("retrieval_text"),
                    "keywords": payload.get("keywords", []),
                    "semantic_tags": payload.get("semantic_tags", []),
                    "is_repealed_candidate": payload.get("is_repealed_candidate", False),
                    "supporting_chunks": [self._point_to_supporting_chunk(hit) for hit in supporting_chunks[:5]],
                }
            )

        return sorted(candidates, key=lambda item: item["score"], reverse=True)

    def _candidate_limits(self, final_k: int, query_analysis: QueryAnalysis) -> tuple[int, int, int, int]:
        chunk_limit = max(self.settings.retrieval_chunk_limit, final_k * 6)
        article_limit = max(self.settings.retrieval_article_limit, final_k * 3)
        law_limit = max(self.settings.retrieval_law_limit, min(5, final_k))
        candidate_limit = max(self.settings.retrieval_candidate_limit, final_k * 4)

        if query_analysis.prefer_specific_articles:
            chunk_limit = max(chunk_limit, final_k * 8)
            article_limit = max(article_limit, final_k * 4)
            candidate_limit = max(candidate_limit, final_k * 5)

        if query_analysis.key_phrases:
            chunk_limit += min(12, len(query_analysis.key_phrases) * 4)
            article_limit += min(6, len(query_analysis.key_phrases) * 2)

        return chunk_limit, article_limit, law_limit, candidate_limit

    def _build_filter(self, *, record_kind: str, filters: RetrievalFilters) -> models.Filter:
        must_conditions = [
            models.FieldCondition(
                key="record_kind",
                match=models.MatchValue(value=record_kind),
            )
        ]

        if filters.legal_domain:
            must_conditions.append(
                models.FieldCondition(
                    key="legal_domain",
                    match=models.MatchValue(value=filters.legal_domain),
                )
            )
        if filters.law_number:
            must_conditions.append(
                models.FieldCondition(
                    key="law_number",
                    match=models.MatchValue(value=filters.law_number),
                )
            )
        if filters.law_year:
            must_conditions.append(
                models.FieldCondition(
                    key="law_year",
                    match=models.MatchValue(value=filters.law_year),
                )
            )
        if filters.status_normalized:
            must_conditions.append(
                models.FieldCondition(
                    key="status_normalized",
                    match=models.MatchValue(value=filters.status_normalized),
                )
            )

        must_not = []
        if filters.exclude_repealed or self.settings.retrieval_exclude_non_current:
            must_not.append(
                models.FieldCondition(
                    key="is_repealed_candidate",
                    match=models.MatchValue(value=True),
                )
            )

        return models.Filter(must=must_conditions, must_not=must_not or None)

    @staticmethod
    def _point_to_summary(point: models.ScoredPoint) -> dict[str, Any]:
        payload = point.payload or {}
        return {
            "id": payload.get("id", str(point.id)),
            "record_kind": payload.get("record_kind"),
            "title": payload.get("title"),
            "law_name": payload.get("law_name"),
            "law_number": payload.get("law_number"),
            "law_year": payload.get("law_year"),
            "legal_domain": payload.get("legal_domain"),
            "score": round(point.score, 6),
        }

    @staticmethod
    def _point_to_supporting_chunk(point: models.ScoredPoint) -> dict[str, Any]:
        payload = point.payload or {}
        return {
            "id": payload.get("id", str(point.id)),
            "title": payload.get("title"),
            "content": payload.get("content"),
            "score": round(point.score, 6),
            "chunk_index": payload.get("chunk_index"),
            "chunk_total": payload.get("chunk_total"),
        }
