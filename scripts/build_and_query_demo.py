from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.indexing import LegalIndexBuilder
from app.models import RetrievalFilters
from app.retrieval import LegalRetriever


def main() -> None:
    builder = LegalIndexBuilder()
    summary = builder.build_from_path()
    print(f"Indexed {summary.total_records} records into {summary.collection_name}")

    retriever = LegalRetriever(client=builder.client, embedding_service=builder.embedding_service)
    queries = [
        ("ما هي حدود تطبيق قانون العمل المصري؟", RetrievalFilters(legal_domain="labor_law")),
        ("متى ينعقد البيع في القانون المدني؟", RetrievalFilters(legal_domain="civil_law")),
        ("ما هي عقوبة الرشوة؟", RetrievalFilters(legal_domain="criminal_law")),
    ]
    for query, filters in queries:
        print("-" * 100)
        print(query)
        result = retriever.search(query, filters=filters)
        for item in result["results"][:3]:
            print(f"{item['title']} | score={item.get('rerank_score', item['score'])}")


if __name__ == "__main__":
    main()
