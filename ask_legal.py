from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.models import RetrievalFilters
from app.retrieval import LegalRetriever


VALID_DOMAINS = ("labor_law", "civil_law", "criminal_law", "constitutional_law", "all")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ask the existing Egyptian-laws Qdrant index.")
    parser.add_argument("--query", help="Legal question to search for.")
    parser.add_argument(
        "--domain",
        choices=VALID_DOMAINS,
        default=None,
        help="Optional legal domain filter. Use 'all' to disable domain filtering.",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of final results to print.")
    parser.add_argument("--json", action="store_true", dest="as_json", help="Print raw JSON response.")
    parser.add_argument(
        "--show-chunks",
        action="store_true",
        help="Print supporting chunks under each result.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    query = args.query.strip() if args.query else _prompt_query()
    domain = args.domain or _prompt_domain()

    retriever = LegalRetriever()
    if not retriever.client.collection_exists(retriever.collection_name):
        print(
            f"Collection '{retriever.collection_name}' was not found in Qdrant. "
            "Run the legal index build first.",
            file=sys.stderr,
        )
        return 1

    filters = RetrievalFilters(
        legal_domain=None if domain == "all" else domain,
    )

    try:
        result = retriever.search(query, top_k=args.top_k, filters=filters)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if args.as_json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    _print_human_friendly(result, show_chunks=args.show_chunks)
    return 0


def _prompt_query() -> str:
    while True:
        query = input("Legal query: ").strip()
        if query:
            return query
        print("Please enter a non-empty question.")


def _prompt_domain() -> str:
    value = input(
        "Domain [labor_law/civil_law/criminal_law/constitutional_law/all] (default: all): "
    ).strip() or "all"
    if value not in VALID_DOMAINS:
        print("Unknown domain. Falling back to 'all'.")
        return "all"
    return value


def _print_human_friendly(result: dict, *, show_chunks: bool) -> None:
    backend = result.get("retrieval_backend", {})
    query_analysis = result.get("query_analysis", {})
    print("=" * 100)
    print(f"Query: {result.get('query')}")
    print(f"Normalized: {result.get('normalized_query')}")
    if result.get("rewritten_query") and result.get("rewritten_query") != result.get("normalized_query"):
        print(f"Rewritten: {result.get('rewritten_query')}")
    print(
        "Retrieval:"
        f" hybrid={backend.get('hybrid_enabled')}"
        f" fusion={backend.get('fusion')}"
        f" reranker={backend.get('reranker')}"
    )
    if query_analysis:
        print(
            "Query analysis:"
            f" intent={query_analysis.get('intent')}"
            f" key_phrases={query_analysis.get('key_phrases')}"
            f" legal_keywords={query_analysis.get('legal_keywords')}"
        )

    warnings = result.get("warnings") or []
    if warnings:
        print(f"Warnings: {warnings}")

    results = result.get("results") or []
    if not results:
        print("No results found.")
        return

    for index, item in enumerate(results, start=1):
        score = item.get("rerank_score", item.get("score"))
        excerpt = _excerpt(item.get("summary") or item.get("content") or "", limit=420)
        print("-" * 100)
        print(f"{index}. {item.get('title') or 'Untitled'}")
        print(
            f"   Article: {item.get('article_number') or '-'}"
            f" | Law: {item.get('law_name') or '-'}"
            f" | Domain: {item.get('legal_domain') or '-'}"
        )
        print(
            f"   Score: {item.get('score')} | Rerank: {score}"
            f" | Direct article score: {item.get('direct_article_score')}"
            f" | Best chunk score: {item.get('best_chunk_score')}"
        )
        print(f"   Summary: {excerpt}")
        print(f"   Source: {item.get('source_url') or '-'}")
        print(f"   Supporting chunks: {item.get('supporting_chunk_count', len(item.get('supporting_chunks', [])))}")

        quality_warnings = item.get("quality_warnings") or []
        if quality_warnings:
            print(f"   Quality warnings: {', '.join(quality_warnings)}")
        rank_explanation = item.get("rank_explanation") or []
        if rank_explanation:
            print(f"   Why ranked: {', '.join(rank_explanation)}")

        if show_chunks and item.get("supporting_chunks"):
            for chunk in item["supporting_chunks"]:
                chunk_excerpt = _excerpt(chunk.get("content") or "", limit=220)
                print(
                    f"     - Chunk {chunk.get('chunk_index') or '-'}"
                    f"/{chunk.get('chunk_total') or '-'}"
                    f" | score={chunk.get('score')}"
                )
                if chunk.get("title"):
                    print(f"       title: {chunk.get('title')}")
                if chunk_excerpt:
                    print(f"       text: {chunk_excerpt}")


def _excerpt(text: str, *, limit: int) -> str:
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


if __name__ == "__main__":
    raise SystemExit(main())
