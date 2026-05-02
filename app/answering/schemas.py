from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.answering.intent_router import sanitize_optional_value


AnswerMode = Literal["identity", "conversation", "non_legal", "grounded", "assisted", "external_assisted", "insufficient"]
LegalDomain = Literal["labor_law", "civil_law", "criminal_law", "constitutional_law", "all"]


class LegalAnswerRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1)
    legal_domain: LegalDomain | None = None
    law_number: str | None = None
    law_year: str | None = None
    status_normalized: str | None = None
    exclude_repealed: bool = False
    top_k: int | None = Field(default=None, ge=1, le=10)
    include_retrieval: bool = False

    @field_validator("legal_domain", "law_number", "law_year", "status_normalized", mode="before")
    @classmethod
    def _sanitize_placeholders(cls, value: object) -> object:
        return sanitize_optional_value(value)


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1)


class SourceCitation(BaseModel):
    source_type: Literal["internal", "external"] = "internal"
    verified_by_system: bool = True
    id: str | None = None
    law_name: str | None = None
    law_number: str | None = None
    law_year: str | None = None
    article_number: str | None = None
    title: str | None = None
    source_url: str | None = None
    section_level: str | None = None
    document_level: str | None = None
    legal_domain: str | None = None
    score: float | None = None
    summary_snippet: str | None = None
    quote_snippet: str | None = None


class CompactSourceCitation(BaseModel):
    law_name: str | None = None
    article_number: str | None = None
    title: str | None = None
    source_url: str | None = None
    legal_domain: str | None = None


class RetrievalSummary(BaseModel):
    domain: str | None = None
    law: str | None = None
    top_k_used: int
    result_count: int
    source_count: int
    internal_source_count: int = 0
    external_source_count: int = 0
    sufficiency_reasons: list[str] = Field(default_factory=list)
    sufficiency_metrics: dict[str, Any] = Field(default_factory=dict)


class LLMCallMetadata(BaseModel):
    provider: str = "gemini"
    model: str | None = None
    called: bool = False
    succeeded: bool = False
    error: str | None = None
    parse_error: str | None = None
    schema_error: str | None = None
    raw_response_preview: str | None = None
    raw_response_repr_preview: str | None = None
    usage: dict[str, Any] | None = None
    web_search_enabled: bool = False


class CompactLLMMetadata(BaseModel):
    called: bool = False
    succeeded: bool = False
    provider: str = "gemini"
    model: str | None = None


class TimingMetadata(BaseModel):
    intent_ms: float | None = None
    retrieval_ms: float | None = None
    llm_ms: float | None = None
    total_ms: float | None = None


class RouterMetadata(BaseModel):
    intent: str
    confidence: float
    suggested_domain: str | None = None
    is_legal_question: bool
    is_out_of_internal_corpus: bool
    reasons: list[str] = Field(default_factory=list)
    scores: dict[str, float] = Field(default_factory=dict)


class LegalAnswerResponse(BaseModel):
    query: str
    answer_mode: AnswerMode
    is_out_of_internal_corpus: bool
    internal_grounding_sufficient: bool
    final_answer: str
    answer_from_sources: str | None = None
    external_or_assisted_explanation: str | None = None
    warning: str | None = None
    internal_sources: list[SourceCitation] = Field(default_factory=list)
    external_sources: list[SourceCitation] = Field(default_factory=list)
    external_sources_verified_by_system: bool = False
    retrieval_summary: RetrievalSummary
    llm: LLMCallMetadata
    retrieval_result: dict[str, Any] | None = None

    # Compatibility fields for earlier clients of the first answer-layer version.
    is_out_of_domain: bool | None = None
    grounding_sufficient: bool | None = None
    assisted_explanation: str | None = None
    sources: list[SourceCitation] = Field(default_factory=list)

    # Semantic classification fields.
    is_legal_question: bool | None = None
    is_supported_by_internal_sources: bool | None = None
    timing: TimingMetadata | None = None
    router: RouterMetadata | None = None


class ChatResponse(BaseModel):
    answer_mode: AnswerMode
    final_answer: str
    warning: str | None = None
    is_legal_question: bool | None = None
    is_supported_by_internal_sources: bool | None = None
    is_out_of_internal_corpus: bool
    sources: list[CompactSourceCitation] = Field(default_factory=list)
    llm: CompactLLMMetadata
