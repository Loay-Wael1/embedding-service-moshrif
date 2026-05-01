from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import NAMESPACE_URL, uuid5


@dataclass(slots=True)
class LegalRecord:
    record_id: str
    record_kind: str
    parent_id: str | None
    legal_domain: str
    law_name: str
    law_number: str | None
    law_year: str | None
    article_number: str | None
    title: str
    content: str
    summary: str
    source_url: str | None
    status: str | None
    status_normalized: str | None
    quality_flags: list[str] = field(default_factory=list)
    quality_warnings: list[str] = field(default_factory=list)
    quality_score: float = 1.0
    noise_score: float = 0.0
    section_level: str | None = None
    document_level: str | None = None
    retrieval_text: str = ""
    embedding_text: str = ""
    jurisdiction: str | None = None
    language: str | None = None
    keywords: list[str] = field(default_factory=list)
    semantic_tags: list[str] = field(default_factory=list)
    cross_references: list[str] = field(default_factory=list)
    chunk_index: int | None = None
    chunk_total: int | None = None
    is_repealed_candidate: bool = False

    def to_payload(self) -> dict[str, Any]:
        return {
            "id": self.record_id,
            "record_kind": self.record_kind,
            "parent_id": self.parent_id,
            "legal_domain": self.legal_domain,
            "law_name": self.law_name,
            "law_number": self.law_number,
            "law_year": self.law_year,
            "article_number": self.article_number,
            "title": self.title,
            "content": self.content,
            "summary": self.summary,
            "source_url": self.source_url,
            "status": self.status,
            "status_normalized": self.status_normalized,
            "quality_flags": self.quality_flags,
            "quality_warnings": self.quality_warnings,
            "quality_score": self.quality_score,
            "noise_score": self.noise_score,
            "section_level": self.section_level,
            "document_level": self.document_level,
            "retrieval_text": self.retrieval_text,
            "embedding_text": self.embedding_text,
            "jurisdiction": self.jurisdiction,
            "language": self.language,
            "keywords": self.keywords,
            "semantic_tags": self.semantic_tags,
            "cross_references": self.cross_references,
            "chunk_index": self.chunk_index,
            "chunk_total": self.chunk_total,
            "is_repealed_candidate": self.is_repealed_candidate,
        }


@dataclass(slots=True)
class RetrievalFilters:
    legal_domain: str | None = None
    law_number: str | None = None
    law_year: str | None = None
    status_normalized: str | None = None
    exclude_repealed: bool = False


def make_point_id(record_id: str) -> str:
    return str(uuid5(NAMESPACE_URL, record_id))
