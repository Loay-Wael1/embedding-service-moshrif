from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.models import RetrievalFilters
from app.retrieval import LegalRetriever


DEMO_QUERIES = [
    ("ما هي أحكام عقد العمل الفردي؟", RetrievalFilters(legal_domain="labor_law")),
    ("ما هي شروط العقد في القانون المدني المصري؟", RetrievalFilters(legal_domain="civil_law")),
    ("ما العقوبة المقررة لجريمة السرقة؟", RetrievalFilters(legal_domain="criminal_law")),
]


def main() -> None:
    retriever = LegalRetriever()
    for query, filters in DEMO_QUERIES:
        result = retriever.search(query, filters=filters)
        print("=" * 100)
        print(query)
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
