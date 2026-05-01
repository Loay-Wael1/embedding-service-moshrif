from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.indexing import LegalIndexBuilder


def main() -> None:
    builder = LegalIndexBuilder()
    summary = builder.build_from_path()
    print("Collection:", summary.collection_name)
    print("Total records:", summary.total_records)
    print("Articles:", summary.article_count)
    print("Chunks:", summary.chunk_count)
    print("Synthetic law records:", summary.law_count)
    print("Qdrant path:", summary.qdrant_path)


if __name__ == "__main__":
    main()
