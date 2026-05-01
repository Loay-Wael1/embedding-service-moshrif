from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


TRUE_VALUES = {"1", "true", "yes", "on"}


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip()


def _env_optional(name: str) -> str | None:
    value = os.getenv(name)
    if value is None or not value.strip():
        return None
    return value.strip()


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in TRUE_VALUES


def _default_dataset_path() -> str:
    candidates = []
    env_path = os.getenv("LEGAL_DATASET_PATH")
    if env_path:
        candidates.append(Path(env_path))

    candidates.extend(
        [
            Path.home() / "Downloads" / "egypt_legal_corpus_v3_cleaned_with_constitution.jsonl",
            Path.home() / "Downloads" / "egypt_legal_corpus_v3_cleaned_with_constitution (1).jsonl",
            Path.cwd() / "egypt_legal_corpus_v3_cleaned_with_constitution.jsonl",
        ]
    )

    for candidate in candidates:
        if candidate and str(candidate) and candidate.exists():
            return str(candidate)
    return str(candidates[0])


@dataclass(frozen=True)
class Settings:
    model_name: str = os.getenv("EMBEDDING_MODEL_NAME", "./model/bge-m3")
    model_local_only: bool = _env_bool("EMBEDDING_MODEL_LOCAL_ONLY", True)
    device_preference: str = os.getenv("EMBEDDING_DEVICE", "cpu")
    max_length: int = int(os.getenv("EMBEDDING_MAX_LENGTH", "512"))
    batch_size: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "8"))
    enable_sparse: bool = _env_bool("EMBEDDING_ENABLE_SPARSE", True)
    query_prefix: str = os.getenv("EMBEDDING_QUERY_PREFIX", "")
    document_prefix: str = os.getenv("EMBEDDING_DOCUMENT_PREFIX", "")

    qdrant_path: str = os.getenv("QDRANT_PATH", "./qdrant_db_legal")
    collection_name: str = os.getenv("QDRANT_COLLECTION", "egyptian_laws_v2_legal")
    dataset_path: str = _default_dataset_path()
    index_batch_size: int = int(os.getenv("INDEX_BATCH_SIZE", "16"))
    include_law_records: bool = _env_bool("INDEX_INCLUDE_LAW_RECORDS", True)

    retrieval_top_k: int = int(os.getenv("RETRIEVAL_TOP_K", "5"))
    retrieval_chunk_limit: int = int(os.getenv("RETRIEVAL_CHUNK_LIMIT", "24"))
    retrieval_article_limit: int = int(os.getenv("RETRIEVAL_ARTICLE_LIMIT", "10"))
    retrieval_law_limit: int = int(os.getenv("RETRIEVAL_LAW_LIMIT", "4"))
    retrieval_candidate_limit: int = int(os.getenv("RETRIEVAL_CANDIDATE_LIMIT", "18"))
    retrieval_exclude_non_current: bool = _env_bool("RETRIEVAL_EXCLUDE_NON_CURRENT", False)
    retrieval_reranker: str = os.getenv("RETRIEVAL_RERANKER", "feature_based")
    retrieval_reranker_model_name: str = os.getenv(
        "RETRIEVAL_RERANKER_MODEL_NAME",
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
    )
    retrieval_reranker_local_only: bool = _env_bool("RETRIEVAL_RERANKER_LOCAL_ONLY", False)
    retrieval_hybrid_fusion: str = os.getenv("RETRIEVAL_HYBRID_FUSION", "rrf")

    llm_provider_name: str = _env_str("LLM_PROVIDER_NAME", "gemini")
    llm_api_key: str | None = _env_optional("LLM_API_KEY") or _env_optional("GEMINI_API_KEY")
    llm_base_url: str = _env_str(
        "LLM_BASE_URL",
        _env_str("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/"),
    )
    llm_model: str = _env_str("LLM_MODEL", _env_str("GEMINI_MODEL", "gemini-2.5-flash"))
    llm_timeout_seconds: float = float(_env_str("LLM_TIMEOUT_SECONDS", _env_str("GEMINI_TIMEOUT_SECONDS", "90")))
    llm_max_tokens: int = int(_env_str("LLM_MAX_TOKENS", _env_str("GEMINI_MAX_TOKENS", "8192")))
    llm_web_search_enabled: bool = _env_bool("LLM_WEB_SEARCH_ENABLED", _env_bool("GEMINI_WEB_SEARCH_ENABLED", False))
    llm_extra_body_json: str = _env_str("LLM_EXTRA_BODY_JSON", _env_str("GEMINI_EXTRA_BODY_JSON", ""))
    llm_json_mode: bool = _env_bool("LLM_JSON_MODE", True)

    legal_answer_top_k: int = int(os.getenv("LEGAL_ANSWER_TOP_K", "4"))
    legal_answer_context_char_limit: int = int(os.getenv("LEGAL_ANSWER_CONTEXT_CHAR_LIMIT", "12000"))
    legal_answer_source_char_limit: int = int(os.getenv("LEGAL_ANSWER_SOURCE_CHAR_LIMIT", "2400"))
    legal_answer_grounded_min_sources: int = int(os.getenv("LEGAL_ANSWER_GROUNDED_MIN_SOURCES", "2"))
    legal_answer_assisted_min_sources: int = int(os.getenv("LEGAL_ANSWER_ASSISTED_MIN_SOURCES", "1"))
    legal_answer_grounded_min_score: float = float(os.getenv("LEGAL_ANSWER_GROUNDED_MIN_SCORE", "0.62"))
    legal_answer_assisted_min_score: float = float(os.getenv("LEGAL_ANSWER_ASSISTED_MIN_SCORE", "0.28"))
    legal_answer_grounded_min_overlap: float = float(os.getenv("LEGAL_ANSWER_GROUNDED_MIN_OVERLAP", "0.16"))
    legal_answer_assisted_min_overlap: float = float(os.getenv("LEGAL_ANSWER_ASSISTED_MIN_OVERLAP", "0.07"))


settings = Settings()
