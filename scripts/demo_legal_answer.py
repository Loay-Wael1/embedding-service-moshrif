from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.answering import LegalAnswerService
from app.models import RetrievalFilters


VALID_DOMAINS = ("labor_law", "civil_law", "criminal_law", "constitutional_law", "all")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ask the Egyptian-law retrieval system with LLM answer generation.")
    parser.add_argument("--query", help="Legal question to answer.")
    parser.add_argument(
        "--domain",
        choices=VALID_DOMAINS,
        default="all",
        help="Optional legal domain filter. Default disables explicit domain filtering.",
    )
    parser.add_argument("--law-number", default=None, help="Optional law number filter.")
    parser.add_argument("--law-year", default=None, help="Optional law year filter.")
    parser.add_argument("--top-k", type=int, default=None, help="Number of retrieval results to use.")
    parser.add_argument("--exclude-repealed", action="store_true", help="Exclude repealed candidate articles.")
    parser.add_argument("--include-retrieval", action="store_true", help="Include raw retrieval output in JSON mode.")
    parser.add_argument("--json", action="store_true", dest="as_json", help="Print the structured JSON response.")
    return parser.parse_args()


def main() -> int:
    _configure_stdio()
    args = parse_args()
    query = args.query.strip() if args.query else _prompt_query()
    filters = RetrievalFilters(
        legal_domain=None if args.domain == "all" else args.domain,
        law_number=args.law_number,
        law_year=args.law_year,
        exclude_repealed=args.exclude_repealed,
    )
    service = LegalAnswerService()
    try:
        response = service.answer(
            query,
            top_k=args.top_k,
            filters=filters,
            include_retrieval=args.include_retrieval,
        )
    except (RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if args.as_json:
        print(response.model_dump_json(indent=2, exclude_none=True))
    else:
        _print_human(response.model_dump(exclude_none=True))
    return 0


def _prompt_query() -> str:
    while True:
        query = input("Legal query: ").strip()
        if query:
            return query
        print("Please enter a non-empty question.")


def _print_human(payload: dict) -> None:
    print("=" * 100)
    print(f"Query: {payload.get('query')}")
    print(f"Answer mode: {payload.get('answer_mode')}")
    print(f"Internal grounding sufficient: {payload.get('internal_grounding_sufficient')}")
    print(f"Out of internal corpus: {payload.get('is_out_of_internal_corpus')}")
    if payload.get("warning"):
        print(f"Warning: {payload['warning']}")
    print("-" * 100)
    print(payload.get("final_answer") or "")
    print("-" * 100)
    print("Internal sources:")
    for index, source in enumerate(payload.get("internal_sources") or payload.get("sources") or [], start=1):
        print(
            f"{index}. {source.get('law_name') or '-'}"
            f" | Article: {source.get('article_number') or '-'}"
            f" | Title: {source.get('title') or '-'}"
        )
        if source.get("source_url"):
            print(f"   URL: {source['source_url']}")
        if source.get("quote_snippet"):
            print(f"   Snippet: {source['quote_snippet']}")
    external_sources = payload.get("external_sources") or []
    if external_sources:
        print("External sources:")
        for index, source in enumerate(external_sources, start=1):
            print(
                f"{index}. {source.get('title') or '-'}"
                f" | verified_by_system={source.get('verified_by_system')}"
            )
            if source.get("source_url"):
                print(f"   URL: {source['source_url']}")
    print("-" * 100)
    print(json.dumps(payload.get("retrieval_summary") or {}, ensure_ascii=False, indent=2))


def _configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
