from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class SparseEmbeddingResponse(BaseModel):
    indices: list[int]
    values: list[float]


class EmbeddingResultResponse(BaseModel):
    text: str
    normalized_text: str
    dense: list[float] | None = None
    sparse: SparseEmbeddingResponse | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class EmbedRequest(BaseModel):
    text: str
    mode: Literal["query", "document"] = "document"
    normalize: bool = True
    return_dense: bool = True
    return_sparse: bool = False


class EmbedBatchRequest(BaseModel):
    texts: list[str]
    mode: Literal["query", "document"] = "document"
    normalize: bool = True
    return_dense: bool = True
    return_sparse: bool = False


class EmbedResponse(BaseModel):
    model: str
    dim: int
    mode: Literal["query", "document"]
    normalized: bool
    sparse_available: bool
    warnings: list[str] = Field(default_factory=list)
    results: list[EmbeddingResultResponse]


class ServiceInfoResponse(BaseModel):
    model_name: str
    embedding_dimension: int
    device: str
    max_length: int
    backend: str
    supports_sparse: bool
    supported_modes: list[str]
    supported_outputs: dict[str, bool]
    normalization: dict[str, Any]
    mode_prefixes: dict[str, str]
    sparse_extraction: dict[str, Any] | None = None
