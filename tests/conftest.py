from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from app.api import create_app
from app.embeddings.service import EmbeddingResult, EmbeddingService
from app.indexing import LegalIndexBuilder
from app.preprocessing import normalize_legal_arabic
from app.settings import Settings


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class FakeEmbeddingService(EmbeddingService):
    def __init__(self) -> None:
        self.settings = Settings(
            model_name="fake-bge-m3",
            model_local_only=True,
            device_preference="cpu",
            max_length=512,
            batch_size=8,
            enable_sparse=False,
            query_prefix="",
            document_prefix="",
            qdrant_path="",
            collection_name="test_collection",
            dataset_path="",
            index_batch_size=8,
            include_law_records=True,
            retrieval_top_k=5,
            retrieval_chunk_limit=10,
            retrieval_article_limit=5,
            retrieval_law_limit=2,
            retrieval_candidate_limit=8,
            retrieval_exclude_non_current=False,
            retrieval_reranker="heuristic",
        )
        self._embedding_dimension = 4
        self._supports_sparse = False
        self._device = "cpu"

    @property
    def is_loaded(self) -> bool:
        return True

    @property
    def device(self) -> str:
        return "cpu"

    @property
    def embedding_dimension(self) -> int:
        return 4

    @property
    def model_name(self) -> str:
        return "fake-bge-m3"

    @property
    def supports_sparse(self) -> bool:
        return False

    def get_info(self) -> dict[str, object]:
        return {
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dimension,
            "device": self.device,
            "max_length": 512,
            "backend": "fake",
            "supports_sparse": False,
            "supported_modes": ["query", "document"],
            "supported_outputs": {"dense": True, "sparse": False},
            "normalization": {"text_normalization_default": True, "vector_l2_normalization": True, "rules": []},
            "mode_prefixes": {"query": "", "document": ""},
        }

    def embed_texts(
        self,
        texts: list[str],
        *,
        mode: str = "document",
        normalize: bool = True,
        return_dense: bool = True,
        return_sparse: bool = False,
    ):
        results = []
        warnings = ["Sparse embeddings are not enabled in the current backend."] if return_sparse else []
        for text in texts:
            normalized = normalize_legal_arabic(text) if normalize else text
            tokens = set(normalized.split())
            vector = [
                1.0 if {"عمل", "العمل", "عامل", "عقد"}.intersection(tokens) else 0.0,
                1.0 if {"مدني", "القانون", "العقود", "بيع"}.intersection(tokens) else 0.0,
                1.0 if {"عقوبة", "جريمة", "سرقة", "رشوة"}.intersection(tokens) else 0.0,
                1.0 if "المادة" in tokens or any(token.isdigit() for token in tokens) else 0.0,
            ]
            norm = sum(value * value for value in vector) ** 0.5 or 1.0
            dense = [value / norm for value in vector] if return_dense else None
            results.append(
                EmbeddingResult(
                    text=text,
                    normalized_text=normalized,
                    dense=dense,
                    sparse=None,
                    metadata={"mode": mode},
                )
            )
        return results, warnings


@pytest.fixture()
def fake_embedding_service() -> FakeEmbeddingService:
    return FakeEmbeddingService()


@pytest.fixture()
def app_client(fake_embedding_service: FakeEmbeddingService):
    from fastapi.testclient import TestClient

    app = create_app(fake_embedding_service)
    return TestClient(app)


@pytest.fixture()
def sample_dataset_path(tmp_path: Path) -> Path:
    dataset = [
        {
            "id": "labor_article_1",
            "content": "تسري أحكام هذا القانون على عقد العمل الفردي والأجر وساعات العمل.",
            "title": "المادة 1 - قانون العمل المصري",
            "summary": "تحديد نطاق عقد العمل الفردي.",
            "keywords": ["عقد", "عمل"],
            "embedding_text": "المادة 1 قانون العمل المصري عقد العمل الفردي",
            "retrieval_metadata": {
                "record_kind": "article",
                "legal_domain": "labor_law",
                "law_number": "14",
                "law_year": "2025",
                "source_url": "https://example.com/labor",
                "status": "سارية",
                "status_normalized": "current",
                "jurisdiction": "egypt",
                "language": "arabic",
            },
            "cross_references": [],
            "semantic_tags": ["employment"],
            "hierarchical_context": {
                "document_level": "قانون العمل المصري",
                "section_level": "أحكام عامة",
                "article_level": "1",
            },
            "retrieval_text": "قانون العمل المصري المادة 1 عقد العمل الفردي",
            "quality_flags": ["normalized_v2", "text_cleaned"],
        },
        {
            "id": "labor_chunk_1",
            "content": "يحدد القانون حقوق العامل والتزامات صاحب العمل في عقد العمل الفردي.",
            "title": "المادة 1 - قانون العمل المصري - مقطع 1",
            "summary": "حقوق العامل.",
            "keywords": ["عامل", "عمل"],
            "embedding_text": "المادة 1 قانون العمل مقطع حقوق العامل",
            "retrieval_metadata": {
                "record_kind": "article_chunk",
                "parent_id": "labor_article_1",
                "chunk_index": 1,
                "chunk_total": 1,
                "legal_domain": "labor_law",
                "law_number": "14",
                "law_year": "2025",
                "source_url": "https://example.com/labor",
                "status": "سارية",
                "status_normalized": "current",
                "jurisdiction": "egypt",
                "language": "arabic",
            },
            "cross_references": [],
            "semantic_tags": ["employment"],
            "hierarchical_context": {
                "document_level": "قانون العمل المصري",
                "section_level": "أحكام عامة",
                "article_level": "1",
            },
            "retrieval_text": "قانون العمل المصري المادة 1 حقوق العامل",
            "quality_flags": ["normalized_v2", "text_cleaned", "derived_chunk_v2"],
        },
        {
            "id": "civil_article_2",
            "content": "يتم العقد بمجرد أن يتبادل الطرفان التعبير عن إرادتين متطابقتين.",
            "title": "المادة 2 - القانون المدني المصري",
            "summary": "انعقاد العقد في القانون المدني.",
            "keywords": ["عقد", "مدني"],
            "embedding_text": "المادة 2 القانون المدني انعقاد العقد",
            "retrieval_metadata": {
                "record_kind": "article",
                "legal_domain": "civil_law",
                "law_number": "131",
                "law_year": "1948",
                "source_url": "https://example.com/civil",
                "status": "سارية",
                "status_normalized": "current",
                "jurisdiction": "egypt",
                "language": "arabic",
            },
            "cross_references": [],
            "semantic_tags": ["contract"],
            "hierarchical_context": {
                "document_level": "القانون المدني المصري",
                "section_level": "العقود",
                "article_level": "2",
            },
            "retrieval_text": "القانون المدني المصري المادة 2 انعقاد العقد",
            "quality_flags": ["normalized_v2", "text_cleaned"],
        },
        {
            "id": "civil_chunk_2",
            "content": "يتم العقد عندما يقترن الإيجاب بالقبول على وجه يثبت أثره في المعقود عليه.",
            "title": "المادة 2 - القانون المدني المصري - مقطع 1",
            "summary": "الإيجاب والقبول.",
            "keywords": ["إيجاب", "قبول"],
            "embedding_text": "المادة 2 القانون المدني الايجاب والقبول",
            "retrieval_metadata": {
                "record_kind": "article_chunk",
                "parent_id": "civil_article_2",
                "chunk_index": 1,
                "chunk_total": 1,
                "legal_domain": "civil_law",
                "law_number": "131",
                "law_year": "1948",
                "source_url": "https://example.com/civil",
                "status": "سارية",
                "status_normalized": "current",
                "jurisdiction": "egypt",
                "language": "arabic",
            },
            "cross_references": [],
            "semantic_tags": ["contract"],
            "hierarchical_context": {
                "document_level": "القانون المدني المصري",
                "section_level": "العقود",
                "article_level": "2",
            },
            "retrieval_text": "القانون المدني المصري المادة 2 الايجاب والقبول",
            "quality_flags": ["normalized_v2", "text_cleaned", "derived_chunk_v2"],
        },
        {
            "id": "criminal_article_3",
            "content": "تعد هذه المادة ملغاة. وكانت تتعلق بعقوبة سابقة.",
            "title": "المادة 3 - قانون العقوبات المصري",
            "summary": "مادة ملغاة.",
            "keywords": ["عقوبة"],
            "embedding_text": "المادة 3 قانون العقوبات ملغاة",
            "retrieval_metadata": {
                "record_kind": "article",
                "legal_domain": "criminal_law",
                "law_number": "58",
                "law_year": "1937",
                "source_url": "https://example.com/criminal",
                "status": "ملغاة",
                "status_normalized": "repealed",
                "jurisdiction": "egypt",
                "language": "arabic",
            },
            "cross_references": [],
            "semantic_tags": ["criminal"],
            "hierarchical_context": {
                "document_level": "قانون العقوبات المصري",
                "section_level": "أحكام عامة",
                "article_level": "3",
            },
            "retrieval_text": "قانون العقوبات المصري المادة 3 ملغاة",
            "quality_flags": ["normalized_v2"],
        },
    ]
    path = tmp_path / "sample_legal_dataset.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for record in dataset:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


@pytest.fixture()
def built_index(tmp_path: Path, sample_dataset_path: Path, fake_embedding_service: FakeEmbeddingService):
    qdrant_path = tmp_path / "qdrant"
    config = Settings(
        model_name="fake-bge-m3",
        model_local_only=True,
        device_preference="cpu",
        max_length=512,
        batch_size=8,
        enable_sparse=False,
        query_prefix="",
        document_prefix="",
        qdrant_path=str(qdrant_path),
        collection_name="egyptian_laws_test",
        dataset_path=str(sample_dataset_path),
        index_batch_size=2,
        include_law_records=True,
        retrieval_top_k=5,
        retrieval_chunk_limit=5,
        retrieval_article_limit=5,
        retrieval_law_limit=2,
        retrieval_candidate_limit=8,
        retrieval_exclude_non_current=False,
        retrieval_reranker="heuristic",
    )
    client = QdrantClient(path=str(qdrant_path))
    builder = LegalIndexBuilder(embedding_service=fake_embedding_service, client=client, config=config)
    summary = builder.build_from_path(sample_dataset_path)
    return {"builder": builder, "summary": summary, "config": config, "client": client}
