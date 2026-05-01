from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import torch
from transformers import AutoModel, AutoTokenizer

from app.preprocessing.legal_arabic import NORMALIZATION_RULES, normalize_legal_arabic
from app.settings import Settings, settings


@dataclass(slots=True)
class SparseEmbedding:
    indices: list[int]
    values: list[float]


@dataclass(slots=True)
class EmbeddingResult:
    text: str
    normalized_text: str
    dense: list[float] | None
    sparse: SparseEmbedding | None
    metadata: dict[str, Any]


class EmbeddingService:
    def __init__(self, config: Settings | None = None) -> None:
        self.settings = config or settings
        self._device = torch.device(
            "cuda"
            if self.settings.device_preference == "cuda" and torch.cuda.is_available()
            else "cpu"
        )
        self._tokenizer = None
        self._model = None
        self._bgem3_model = None
        self._embedding_dimension: int | None = None
        self._supports_sparse = False
        self._backend = "transformers_dense"

    @property
    def is_loaded(self) -> bool:
        return self._model is not None or self._bgem3_model is not None

    @property
    def model_name(self) -> str:
        return self.settings.model_name

    @property
    def device(self) -> str:
        return str(self._device)

    @property
    def max_length(self) -> int:
        return self.settings.max_length

    @property
    def embedding_dimension(self) -> int:
        self._ensure_model()
        assert self._embedding_dimension is not None
        return self._embedding_dimension

    @property
    def supports_sparse(self) -> bool:
        self._ensure_model()
        return self._supports_sparse

    def _ensure_model(self) -> None:
        if self._bgem3_model is not None or (self._model is not None and self._tokenizer is not None):
            return

        model_name = self.settings.model_name
        local_files_only = self.settings.model_local_only or model_name.startswith(".")

        if "bge-m3" in model_name.lower():
            try:
                from FlagEmbedding import BGEM3FlagModel
            except ImportError as exc:
                raise RuntimeError(
                    "Sparse BGE-M3 support requires the FlagEmbedding package. "
                    "Install it with `pip install FlagEmbedding`."
                ) from exc

            device_name = "cuda" if self._device.type == "cuda" else "cpu"
            self._bgem3_model = BGEM3FlagModel(
                model_name,
                use_fp16=self._device.type == "cuda",
                devices=device_name,
            )
            self._bgem3_model.model.eval()
            self._embedding_dimension = 1024
            self._supports_sparse = True
            self._backend = "flagembedding_bgem3"
            return

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            local_files_only=local_files_only,
            trust_remote_code=True,
        )
        self._model = AutoModel.from_pretrained(
            model_name,
            local_files_only=local_files_only,
            trust_remote_code=True,
        ).to(self._device)
        self._model.eval()
        self._embedding_dimension = int(getattr(self._model.config, "hidden_size", 1024))
        self._supports_sparse = False
        self._backend = "transformers_dense"

    @staticmethod
    def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        summed = torch.sum(last_hidden_state * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts

    def _prepare_text(self, text: str, mode: str, normalize: bool) -> tuple[str, str]:
        clean_text = normalize_legal_arabic(text) if normalize else text.strip()
        prefix = self.settings.query_prefix if mode == "query" else self.settings.document_prefix
        model_text = f"{prefix}{clean_text}".strip() if prefix else clean_text
        return clean_text, model_text

    @staticmethod
    def _lexical_weights_to_sparse(lexical_weights: dict[Any, Any]) -> SparseEmbedding | None:
        pairs: list[tuple[int, float]] = []
        for token_id, weight in lexical_weights.items():
            try:
                index = int(token_id)
                value = float(weight)
            except (TypeError, ValueError):
                continue
            if value > 0:
                pairs.append((index, value))

        if not pairs:
            return None

        pairs.sort(key=lambda item: item[0])
        return SparseEmbedding(
            indices=[index for index, _ in pairs],
            values=[value for _, value in pairs],
        )

    def _embed_with_bgem3(
        self,
        model_inputs: list[str],
        *,
        return_dense: bool,
        return_sparse: bool,
    ) -> tuple[list[list[float] | None], list[SparseEmbedding | None]]:
        batch_size = max(1, self.settings.batch_size)
        dense_vectors: list[list[float] | None] = [None] * len(model_inputs)
        sparse_vectors: list[SparseEmbedding | None] = [None] * len(model_inputs)

        for start in range(0, len(model_inputs), batch_size):
            batch_texts = model_inputs[start : start + batch_size]
            outputs = self._bgem3_model.encode(
                batch_texts,
                batch_size=len(batch_texts),
                max_length=self.settings.max_length,
                return_dense=return_dense,
                return_sparse=return_sparse,
                return_colbert_vecs=False,
            )

            if return_dense:
                for offset, vector in enumerate(outputs["dense_vecs"]):
                    dense_vectors[start + offset] = [float(value) for value in vector]

            if return_sparse:
                for offset, lexical_weights in enumerate(outputs["lexical_weights"]):
                    sparse_vectors[start + offset] = self._lexical_weights_to_sparse(lexical_weights)

        return dense_vectors, sparse_vectors

    def embed_texts(
        self,
        texts: list[str],
        *,
        mode: str = "document",
        normalize: bool = True,
        return_dense: bool = True,
        return_sparse: bool = False,
    ) -> tuple[list[EmbeddingResult], list[str]]:
        if mode not in {"query", "document"}:
            raise ValueError("mode must be either 'query' or 'document'")
        if not return_dense and not return_sparse:
            raise ValueError("At least one of return_dense or return_sparse must be true")
        if not texts:
            raise ValueError("texts must not be empty")

        self._ensure_model()
        warnings: list[str] = []
        prepared = [self._prepare_text(text, mode, normalize) for text in texts]
        model_inputs = [item[1] for item in prepared]
        dense_vectors: list[list[float] | None] = [None] * len(texts)
        sparse_vectors: list[SparseEmbedding | None] = [None] * len(texts)

        if self._bgem3_model is not None:
            dense_vectors, sparse_vectors = self._embed_with_bgem3(
                model_inputs,
                return_dense=return_dense,
                return_sparse=return_sparse,
            )
        elif return_dense:
            model_inputs = [item[1] for item in prepared]
            batch_size = max(1, self.settings.batch_size)

            for start in range(0, len(model_inputs), batch_size):
                batch_texts = model_inputs[start : start + batch_size]
                tokenized = self._tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.settings.max_length,
                ).to(self._device)

                with torch.no_grad():
                    outputs = self._model(**tokenized)
                    pooled = self._mean_pool(outputs.last_hidden_state, tokenized["attention_mask"])
                    normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)

                batch_vectors = normalized.cpu().tolist()
                for offset, vector in enumerate(batch_vectors):
                    dense_vectors[start + offset] = vector
        if return_sparse and not self.supports_sparse:
            warnings.append("Sparse embeddings are not enabled in the current backend.")

        results = []
        for original_text, (normalized_text, _) in zip(texts, prepared):
            index = len(results)
            results.append(
                EmbeddingResult(
                    text=original_text,
                    normalized_text=normalized_text,
                    dense=dense_vectors[index],
                    sparse=sparse_vectors[index],
                    metadata={
                        "mode": mode,
                        "input_length": len(original_text),
                        "normalized_length": len(normalized_text),
                    },
                )
            )

        return results, warnings

    def get_info(self) -> dict[str, Any]:
        self._ensure_model()
        return {
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dimension,
            "device": self.device,
            "max_length": self.max_length,
            "backend": self._backend,
            "supports_sparse": self.supports_sparse,
            "supported_modes": ["query", "document"],
            "supported_outputs": {
                "dense": True,
                "sparse": self.supports_sparse,
            },
            "normalization": {
                "text_normalization_default": True,
                "vector_l2_normalization": True,
                "rules": NORMALIZATION_RULES,
            },
            "mode_prefixes": {
                "query": self.settings.query_prefix,
                "document": self.settings.document_prefix,
            },
            "sparse_extraction": None
            if not self.supports_sparse
            else {
                "method": "BGEM3FlagModel.encode(return_sparse=True)",
                "representation": "lexical_weights token-id -> weight map",
                "qdrant_encoding": "SparseVector(indices=int(token_id), values=weight)",
            },
        }


class LegacyEmbeddingModelAdapter:
    def __init__(self, embedding_service: EmbeddingService) -> None:
        self._embedding_service = embedding_service

    def embed(self, text: str) -> list[float]:
        results, _ = self._embedding_service.embed_texts(
            [text],
            mode="document",
            normalize=True,
            return_dense=True,
            return_sparse=False,
        )
        return results[0].dense or []


@lru_cache(maxsize=1)
def get_default_embedding_service() -> EmbeddingService:
    return EmbeddingService(settings)
