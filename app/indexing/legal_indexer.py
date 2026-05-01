from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from qdrant_client import QdrantClient, models

from app.embeddings import EmbeddingService
from app.indexing.dataset import load_legal_dataset
from app.models import LegalRecord, make_point_id
from app.settings import Settings, settings


@dataclass(slots=True)
class IndexBuildSummary:
    collection_name: str
    total_records: int
    article_count: int
    chunk_count: int
    law_count: int
    qdrant_path: str


class LegalIndexBuilder:
    def __init__(
        self,
        *,
        embedding_service: EmbeddingService | None = None,
        client: QdrantClient | None = None,
        config: Settings | None = None,
    ) -> None:
        self.settings = config or settings
        self.embedding_service = embedding_service or EmbeddingService(self.settings)
        self.client = client or QdrantClient(path=self.settings.qdrant_path)
        self.collection_name = self.settings.collection_name
        self.use_sparse = self.settings.enable_sparse and self.embedding_service.supports_sparse

    def recreate_collection(self) -> None:
        vectors_config = {
            "dense": models.VectorParams(
                size=self.embedding_service.embedding_dimension,
                distance=models.Distance.COSINE,
            )
        }
        sparse_vectors_config = {"sparse": models.SparseVectorParams()} if self.use_sparse else None

        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=vectors_config,
            sparse_vectors_config=sparse_vectors_config,
            on_disk_payload=True,
            metadata={"schema": "egyptian_laws_v2_legal"},
        )
        self._create_payload_indexes()

    def build_from_path(self, dataset_path: str | Path | None = None, *, recreate: bool = True) -> IndexBuildSummary:
        path = Path(dataset_path or self.settings.dataset_path)
        records = load_legal_dataset(path, include_law_records=self.settings.include_law_records)
        if recreate:
            self.recreate_collection()
        self._upsert_records(records)

        article_count = sum(1 for record in records if record.record_kind == "article")
        chunk_count = sum(1 for record in records if record.record_kind == "article_chunk")
        law_count = sum(1 for record in records if record.record_kind == "law")

        return IndexBuildSummary(
            collection_name=self.collection_name,
            total_records=len(records),
            article_count=article_count,
            chunk_count=chunk_count,
            law_count=law_count,
            qdrant_path=self.settings.qdrant_path,
        )

    def _upsert_records(self, records: list[LegalRecord]) -> None:
        batch_size = max(1, self.settings.index_batch_size)
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            embeddings, _ = self.embedding_service.embed_texts(
                [record.embedding_text for record in batch],
                mode="document",
                normalize=True,
                return_dense=True,
                return_sparse=self.use_sparse,
            )

            points = []
            for record, embedding in zip(batch, embeddings):
                vector_payload: dict[str, object] = {"dense": embedding.dense}
                if self.use_sparse and embedding.sparse is not None:
                    vector_payload["sparse"] = models.SparseVector(
                        indices=embedding.sparse.indices,
                        values=embedding.sparse.values,
                    )
                points.append(
                    models.PointStruct(
                        id=make_point_id(record.record_id),
                        vector=vector_payload,
                        payload=record.to_payload(),
                    )
                )

            self.client.upsert(collection_name=self.collection_name, points=points, wait=True)

    def _create_payload_indexes(self) -> None:
        if ".local." in type(self.client._client).__module__:
            return

        keyword_fields = [
            "record_kind",
            "parent_id",
            "legal_domain",
            "law_number",
            "law_year",
            "status_normalized",
            "article_number",
        ]
        for field_name in keyword_fields:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field_name,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="is_repealed_candidate",
            field_schema=models.PayloadSchemaType.BOOL,
        )
