from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.preprocessing import normalize_legal_arabic
from app.retrieval import LegalRetriever


EVAL_DOMAINS = ("labor_law", "civil_law", "criminal_law", "constitutional_law", "out_of_domain")
GENERIC_ARTICLE_CUES = ("احكام عامه", "احكام عامة", "أحكام عامة", "تعريفات", "تعريف", "الديباجة", "الديباجه")
PREAMBLE_CUES = ("الديباجة", "الديباجه")
WARN_REJECT_CUES = (
    "out_of_domain",
    "outside corpus",
    "outside scope",
    "not covered",
    "low confidence",
    "no relevant",
    "غير مدعوم",
    "خارج النطاق",
    "غير مغطى",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality against a benchmark JSON file.")
    parser.add_argument("--benchmark", required=True, help="Path to retrieval benchmark JSON.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to request from LegalRetriever.")
    parser.add_argument("--output", help="Optional report path. Use .json or .md.")
    parser.add_argument("--domain", choices=EVAL_DOMAINS, help="Optional benchmark-domain filter.")
    parser.add_argument(
        "--failures-only",
        action="store_true",
        help="Print only failed cases in the detailed console output.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    benchmark = load_benchmark(Path(args.benchmark))
    cases = benchmark["cases"]
    if args.domain:
        cases = [case for case in cases if case.get("domain") == args.domain]

    if not cases:
        print("No benchmark cases matched the selected filters.", file=sys.stderr)
        return 1

    retriever = LegalRetriever()
    if not retriever.client.collection_exists(retriever.collection_name):
        print(
            f"Collection '{retriever.collection_name}' was not found in Qdrant. "
            "Run the legal index build first.",
            file=sys.stderr,
        )
        return 1

    case_reports = [evaluate_case(case, retriever, args.top_k) for case in cases]
    summary = build_summary(case_reports)
    report = {
        "benchmark_name": benchmark.get("benchmark_name"),
        "benchmark_version": benchmark.get("version"),
        "evaluated_count": len(case_reports),
        "top_k": args.top_k,
        "summary": summary,
        "cases": case_reports if not args.failures_only else [item for item in case_reports if not item["passed"]],
    }

    print_summary(summary)
    print_case_details(report["cases"], failures_only=args.failures_only)

    if args.output:
        output_path = Path(args.output)
        output_path.write_text(render_report(report, output_path.suffix.lower()), encoding="utf-8")
        print(f"\nReport written to {output_path}")

    return 0


def load_benchmark(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return {"benchmark_name": path.stem, "version": "1.0", "cases": payload}
    if isinstance(payload, dict) and isinstance(payload.get("cases"), list):
        return payload
    raise ValueError("Benchmark JSON must be a list or an object with a 'cases' array.")


def normalized_expected_terms(values: list[str]) -> list[str]:
    return [normalize_legal_arabic(value) for value in values if value]


def term_match_rate(text: str, terms: list[str]) -> float:
    if not text or not terms:
        return 0.0
    matches = sum(1 for term in terms if term and term in text)
    return round(matches / max(1, len(terms)), 4)


def term_match_success(rate: float, terms: list[str], mode: str) -> bool:
    if not terms:
        return False
    active_mode = (mode or "any").lower()
    if active_mode == "all":
        return rate >= 1.0
    return rate > 0.0


def result_topic_match(result: dict[str, Any] | None, terms: list[str], mode: str) -> tuple[bool, float | None]:
    if not terms:
        return False, None
    if not result:
        return False, 0.0
    rate = term_match_rate(result_text_blob(result), terms)
    return term_match_success(rate, terms, mode), rate


def result_section_match(result: dict[str, Any] | None, terms: list[str], mode: str) -> tuple[bool, float | None]:
    if not terms:
        return False, None
    if not result:
        return False, 0.0
    section_text = normalize_legal_arabic(
        " ".join(
            part
            for part in (
                result.get("section_level") or "",
                result.get("document_level") or "",
                result.get("title") or "",
            )
            if part
        )
    )
    rate = term_match_rate(section_text, terms)
    return term_match_success(rate, terms, mode), rate


def best_result_match(
    results: list[dict[str, Any]],
    terms: list[str],
    mode: str,
    *,
    matcher,
) -> tuple[bool, float | None]:
    if not terms:
        return False, None
    best_rate = 0.0
    hit = False
    for result in results:
        matched, rate = matcher(result, terms, mode)
        hit = hit or matched
        if rate is not None:
            best_rate = max(best_rate, rate)
    return hit, round(best_rate, 4)


def context_success(
    *,
    topic_hit: bool | None,
    section_hit: bool | None,
    topic_terms: list[str],
    section_terms: list[str],
    policy: str,
) -> bool:
    require_topic = bool(topic_terms)
    require_section = bool(section_terms)
    if not require_topic and not require_section:
        return True
    if require_topic and require_section:
        if (policy or "topic_or_section").lower() == "all":
            return bool(topic_hit) and bool(section_hit)
        return bool(topic_hit) or bool(section_hit)
    if require_topic:
        return bool(topic_hit)
    return bool(section_hit)


def evaluate_case(case: dict[str, Any], retriever: LegalRetriever, top_k: int) -> dict[str, Any]:
    raw_result = retriever.search(case["query"], top_k=top_k)
    expected = case.get("expected", {})
    results = raw_result.get("results") or []
    target_window_size = min(int(expected.get("top_k_success", 3)), max(1, top_k))
    target_window = results[:target_window_size]
    top3 = results[:3]
    top1 = results[0] if results else None
    query = case["query"]
    expected_domain = expected.get("expected_domain")
    expected_law = normalize_legal_arabic(expected.get("expected_law_name") or "")
    expected_articles = [normalize_legal_arabic(value) for value in expected.get("expected_article_numbers", []) if value]
    expected_keywords = [normalize_legal_arabic(value) for value in expected.get("expected_keywords", []) if value]
    accepted_topic_terms = normalized_expected_terms(expected.get("accepted_topic_terms", []))
    accepted_topic_match_mode = expected.get("accepted_topic_match_mode", "any")
    accepted_section_terms = normalized_expected_terms(expected.get("accepted_section_terms", []))
    accepted_section_match_mode = expected.get("accepted_section_match_mode", "any")
    accepted_context_policy = expected.get("accepted_context_policy", "topic_or_section")
    context_requires_all = accepted_context_policy.lower() == "all"

    if expected_domain == "out_of_domain":
        report = evaluate_out_of_domain_case(
            case=case,
            raw_result=raw_result,
            top1=top1,
            top3=top3,
            target_window=target_window,
        )
        validate_case_report(report)
        return report

    exact_top1_domain = bool(top1 and top1.get("legal_domain") == expected_domain)
    exact_top3_domain = any(item.get("legal_domain") == expected_domain for item in top3)
    exact_top1_law = bool(top1 and law_matches(top1.get("law_name"), expected_law))
    exact_top3_law = any(law_matches(item.get("law_name"), expected_law) for item in top3)
    target_window_domain_hit = any(item.get("legal_domain") == expected_domain for item in target_window)
    target_window_law_hit = any(law_matches(item.get("law_name"), expected_law) for item in target_window)

    top1_keyword_hit = keyword_hit_ratio(top1, expected_keywords) if expected_keywords else None
    target_window_keyword_hit = max((keyword_hit_ratio(item, expected_keywords) for item in target_window), default=None)

    top1_article_match = article_matches(top1, expected_articles) if expected_articles else None
    target_window_article_hit = any(article_matches(item, expected_articles) for item in target_window) if expected_articles else None

    top1_topic_hit, top1_topic_rate = result_topic_match(top1, accepted_topic_terms, accepted_topic_match_mode)
    topk_topic_hit, topk_topic_rate = best_result_match(
        target_window,
        accepted_topic_terms,
        accepted_topic_match_mode,
        matcher=result_topic_match,
    )
    top1_section_hit, top1_section_rate = result_section_match(top1, accepted_section_terms, accepted_section_match_mode)
    topk_section_hit, topk_section_rate = best_result_match(
        target_window,
        accepted_section_terms,
        accepted_section_match_mode,
        matcher=result_section_match,
    )

    top1_context_success = context_success(
        topic_hit=top1_topic_hit,
        section_hit=top1_section_hit,
        topic_terms=accepted_topic_terms,
        section_terms=accepted_section_terms,
        policy=accepted_context_policy,
    )
    topk_context_hit = context_success(
        topic_hit=topk_topic_hit,
        section_hit=topk_section_hit,
        topic_terms=accepted_topic_terms,
        section_terms=accepted_section_terms,
        policy=accepted_context_policy,
    )

    strict_keyword_success = True if not expected_keywords else (top1_keyword_hit or 0.0) >= 0.34 or top1_context_success
    keyword_success = True if not expected_keywords else (target_window_keyword_hit or 0.0) >= 0.34 or topk_context_hit
    strict_article_success = True if not expected_articles else bool(top1_article_match)
    article_success = True if not expected_articles else bool(target_window_article_hit)

    benchmark_window_passed = (
        target_window_domain_hit
        and target_window_law_hit
        and keyword_success
        and article_success
        and topk_context_hit
    )
    passed = (
        exact_top1_domain
        and exact_top1_law
        and strict_keyword_success
        and strict_article_success
        and top1_context_success
    )

    why_passed: list[str] = []
    why_failed: list[str] = []
    if exact_top1_domain:
        why_passed.append(f"top1 domain matched expected domain '{expected_domain}'")
    else:
        why_failed.append(
            f"top1 domain '{top1.get('legal_domain') if top1 else None}' did not match expected domain '{expected_domain}'"
        )

    if exact_top1_law:
        why_passed.append("top1 law matched expected law")
    else:
        why_failed.append(
            f"top1 law '{top1.get('law_name') if top1 else None}' did not match expected law '{expected.get('expected_law_name')}'"
        )

    if expected_keywords:
        if strict_keyword_success:
            why_passed.append(f"top1 keyword hit ratio={top1_keyword_hit}")
        else:
            why_failed.append(
                f"top1 keyword hit ratio={top1_keyword_hit} was below threshold; best-in-window={target_window_keyword_hit}"
            )

    if expected_articles:
        if strict_article_success:
            why_passed.append("top1 matched expected article number")
        else:
            why_failed.append(
                f"top1 article did not match expected article numbers; window_article_hit={target_window_article_hit}"
            )

    if accepted_topic_terms:
        if top1_topic_hit:
            why_passed.append(f"top1 matched accepted topic terms with rate={top1_topic_rate}")
        elif not accepted_section_terms or context_requires_all:
            why_failed.append(
                f"top1 did not match accepted topic terms; topk_topic_hit={topk_topic_hit}, best_topic_rate={topk_topic_rate}"
            )

    if accepted_section_terms:
        if top1_section_hit:
            why_passed.append(f"top1 matched accepted section terms with rate={top1_section_rate}")
        elif not accepted_topic_terms or context_requires_all:
            why_failed.append(
                f"top1 did not match accepted section terms; topk_section_hit={topk_section_hit}, best_section_rate={topk_section_rate}"
            )

    if (accepted_topic_terms or accepted_section_terms) and not top1_context_success:
        why_failed.append("top1 did not satisfy the accepted topic/section specificity criteria")

    if benchmark_window_passed and not passed:
        why_failed.append("expected match appeared within target window but not at top1")

    failure_category = None if passed else classify_failure(
        case=case,
        top1=top1,
        top3=top3,
        target_window=target_window,
        target_window_keyword_hit=target_window_keyword_hit,
        top1_topic_hit=top1_topic_hit,
        topk_topic_hit=topk_topic_hit,
        top1_section_hit=top1_section_hit,
        topk_section_hit=topk_section_hit,
        top1_context_success=top1_context_success,
        topk_context_hit=topk_context_hit,
        benchmark_window_passed=benchmark_window_passed,
    )
    evaluation_reason = (
        "; ".join(why_passed)
        if passed
        else "; ".join(why_failed) or "retrieval did not satisfy strict top1 evaluation criteria"
    )

    report = {
        "id": case["id"],
        "domain": case["domain"],
        "query_type": case["query_type"],
        "query": query,
        "passed": passed,
        "evaluation_reason": evaluation_reason,
        "why_passed": why_passed,
        "why_failed": why_failed,
        "failure_category": failure_category,
        "expected": expected,
        "metrics": {
            "exact_top1_domain_accuracy": exact_top1_domain,
            "exact_top3_domain_hit_rate": exact_top3_domain,
            "exact_top1_law_accuracy": exact_top1_law,
            "exact_top3_law_hit_rate": exact_top3_law,
            "top1_expected_keyword_hit_rate": top1_keyword_hit,
            "topk_expected_keyword_hit_rate": target_window_keyword_hit,
            "top1_expected_article_match": top1_article_match,
            "topk_expected_article_hit": target_window_article_hit,
            "top1_accepted_topic_hit": top1_topic_hit if accepted_topic_terms else None,
            "topk_accepted_topic_hit": topk_topic_hit if accepted_topic_terms else None,
            "top1_accepted_topic_match_rate": top1_topic_rate if accepted_topic_terms else None,
            "topk_accepted_topic_match_rate": topk_topic_rate if accepted_topic_terms else None,
            "top1_accepted_section_hit": top1_section_hit if accepted_section_terms else None,
            "topk_accepted_section_hit": topk_section_hit if accepted_section_terms else None,
            "top1_accepted_section_match_rate": top1_section_rate if accepted_section_terms else None,
            "topk_accepted_section_match_rate": topk_section_rate if accepted_section_terms else None,
            "top1_context_success": top1_context_success,
            "topk_context_hit": topk_context_hit,
            "benchmark_window_domain_hit": target_window_domain_hit,
            "benchmark_window_law_hit": target_window_law_hit,
            "benchmark_window_passed": benchmark_window_passed,
            "out_of_domain_reject_rate": None,
        },
        "top_result": summarize_result(top1),
        "top3_results": [summarize_result(item) for item in top3],
        "target_window_results": [summarize_result(item) for item in target_window],
        "notes": case.get("notes"),
    }
    validate_case_report(report)
    return report


def evaluate_out_of_domain_case(
    *,
    case: dict[str, Any],
    raw_result: dict[str, Any],
    top1: dict[str, Any] | None,
    top3: list[dict[str, Any]],
    target_window: list[dict[str, Any]],
) -> dict[str, Any]:
    warnings = [normalize_legal_arabic(item) for item in raw_result.get("warnings", []) if item]
    explicit_reject = not target_window or any(any(cue in warning for cue in WARN_REJECT_CUES) for warning in warnings)
    why_passed: list[str] = []
    why_failed: list[str] = []

    if explicit_reject:
        why_passed.append("retriever returned explicit warning or no results for out-of-domain query")
    else:
        why_failed.append("retriever returned in-corpus legal results without explicit out-of-domain rejection")

    report = {
        "id": case["id"],
        "domain": case["domain"],
        "query_type": case["query_type"],
        "query": case["query"],
        "passed": explicit_reject,
        "evaluation_reason": "; ".join(why_passed) if explicit_reject else "; ".join(why_failed),
        "why_passed": why_passed,
        "why_failed": why_failed,
        "failure_category": None if explicit_reject else "out_of_domain_not_detected",
        "expected": case["expected"],
        "metrics": {
            "exact_top1_domain_accuracy": None,
            "exact_top3_domain_hit_rate": None,
            "exact_top1_law_accuracy": None,
            "exact_top3_law_hit_rate": None,
            "top1_expected_keyword_hit_rate": None,
            "topk_expected_keyword_hit_rate": None,
            "top1_expected_article_match": None,
            "topk_expected_article_hit": None,
            "top1_accepted_topic_hit": None,
            "topk_accepted_topic_hit": None,
            "top1_accepted_topic_match_rate": None,
            "topk_accepted_topic_match_rate": None,
            "top1_accepted_section_hit": None,
            "topk_accepted_section_hit": None,
            "top1_accepted_section_match_rate": None,
            "topk_accepted_section_match_rate": None,
            "top1_context_success": explicit_reject,
            "topk_context_hit": explicit_reject,
            "benchmark_window_domain_hit": False,
            "benchmark_window_law_hit": False,
            "benchmark_window_passed": explicit_reject,
            "out_of_domain_reject_rate": explicit_reject,
        },
        "top_result": summarize_result(top1),
        "top3_results": [summarize_result(item) for item in top3],
        "target_window_results": [summarize_result(item) for item in target_window],
        "notes": case.get("notes"),
    }
    return report


def build_summary(case_reports: list[dict[str, Any]]) -> dict[str, Any]:
    in_domain = [case for case in case_reports if case["domain"] != "out_of_domain"]
    ood = [case for case in case_reports if case["domain"] == "out_of_domain"]
    keyword_cases = [case for case in in_domain if case["metrics"]["topk_expected_keyword_hit_rate"] is not None]
    topic_cases = [case for case in in_domain if case["metrics"]["top1_accepted_topic_hit"] is not None]
    section_cases = [case for case in in_domain if case["metrics"]["top1_accepted_section_hit"] is not None]

    failure_counts = Counter(case["failure_category"] for case in case_reports if case["failure_category"])
    domain_breakdown = defaultdict(lambda: {"count": 0, "strict_passed": 0, "window_passed": 0})
    query_type_breakdown = defaultdict(lambda: {"count": 0, "strict_passed": 0, "window_passed": 0})

    for case in case_reports:
        domain_breakdown[case["domain"]]["count"] += 1
        query_type_breakdown[case["query_type"]]["count"] += 1
        if case["passed"]:
            domain_breakdown[case["domain"]]["strict_passed"] += 1
            query_type_breakdown[case["query_type"]]["strict_passed"] += 1
        if case["metrics"]["benchmark_window_passed"]:
            domain_breakdown[case["domain"]]["window_passed"] += 1
            query_type_breakdown[case["query_type"]]["window_passed"] += 1

    return {
        "exact_top1_domain_accuracy": ratio(
            sum(1 for case in in_domain if case["metrics"]["exact_top1_domain_accuracy"]),
            len(in_domain),
        ),
        "exact_top3_domain_hit_rate": ratio(
            sum(1 for case in in_domain if case["metrics"]["exact_top3_domain_hit_rate"]),
            len(in_domain),
        ),
        "exact_top1_law_accuracy": ratio(
            sum(1 for case in in_domain if case["metrics"]["exact_top1_law_accuracy"]),
            len(in_domain),
        ),
        "exact_top3_law_hit_rate": ratio(
            sum(1 for case in in_domain if case["metrics"]["exact_top3_law_hit_rate"]),
            len(in_domain),
        ),
        "topk_expected_keyword_hit_rate": round(
            sum(case["metrics"]["topk_expected_keyword_hit_rate"] for case in keyword_cases) / max(len(keyword_cases), 1),
            4,
        ),
        "accepted_topic_top1_hit_rate": ratio(
            sum(1 for case in topic_cases if case["metrics"]["top1_accepted_topic_hit"]),
            len(topic_cases),
        ),
        "accepted_topic_topk_hit_rate": ratio(
            sum(1 for case in topic_cases if case["metrics"]["topk_accepted_topic_hit"]),
            len(topic_cases),
        ),
        "accepted_section_top1_hit_rate": ratio(
            sum(1 for case in section_cases if case["metrics"]["top1_accepted_section_hit"]),
            len(section_cases),
        ),
        "accepted_section_topk_hit_rate": ratio(
            sum(1 for case in section_cases if case["metrics"]["topk_accepted_section_hit"]),
            len(section_cases),
        ),
        "out_of_domain_reject_rate": ratio(
            sum(1 for case in ood if case["metrics"]["out_of_domain_reject_rate"]),
            len(ood),
        ),
        "strict_pass_rate": ratio(sum(1 for case in case_reports if case["passed"]), len(case_reports)),
        "benchmark_window_pass_rate": ratio(
            sum(1 for case in case_reports if case["metrics"]["benchmark_window_passed"]),
            len(case_reports),
        ),
        "failure_categories": dict(sorted(failure_counts.items())),
        "by_domain": {
            domain: {
                "count": values["count"],
                "strict_pass_rate": ratio(values["strict_passed"], values["count"]),
                "benchmark_window_pass_rate": ratio(values["window_passed"], values["count"]),
            }
            for domain, values in sorted(domain_breakdown.items())
        },
        "by_query_type": {
            query_type: {
                "count": values["count"],
                "strict_pass_rate": ratio(values["strict_passed"], values["count"]),
                "benchmark_window_pass_rate": ratio(values["window_passed"], values["count"]),
            }
            for query_type, values in sorted(query_type_breakdown.items())
        },
    }


def classify_failure(
    *,
    case: dict[str, Any],
    top1: dict[str, Any] | None,
    top3: list[dict[str, Any]],
    target_window: list[dict[str, Any]],
    target_window_keyword_hit: float | None,
    top1_topic_hit: bool,
    topk_topic_hit: bool,
    top1_section_hit: bool,
    topk_section_hit: bool,
    top1_context_success: bool,
    topk_context_hit: bool,
    benchmark_window_passed: bool,
) -> str:
    if not top1:
        return "missed_specific_match"

    expected_domain = case["expected"]["expected_domain"]
    top1_domain = top1.get("legal_domain")
    title_norm = normalize_legal_arabic(top1.get("title") or "")
    section_norm = normalize_legal_arabic(top1.get("section_level") or "")
    query_norm = normalize_legal_arabic(case["query"])

    if case["domain"] == "constitutional_law":
        if any(cue in title_norm or cue in section_norm for cue in PREAMBLE_CUES) and not any(
            cue in query_norm for cue in PREAMBLE_CUES + ("مبادئ الدستور", "فلسفة الدستور", "فلسفه الدستور")
        ):
            return "constitutional_preamble_bias"

    if top1_domain != expected_domain:
        return "wrong_domain"
    if not law_matches(top1.get("law_name"), normalize_legal_arabic(case["expected"].get("expected_law_name") or "")):
        return "wrong_law"
    if case["domain"] == "criminal_law" and case["expected"].get("accepted_topic_terms"):
        if not top1_topic_hit:
            if topk_topic_hit:
                return "generic_article_bias"
            if any(item.get("quality_warnings") for item in target_window):
                return "noisy_text_issue"
            return "criminal_offense_miss"
    if case["domain"] == "constitutional_law" and case["expected"].get("accepted_section_terms"):
        if not top1_context_success and topk_context_hit:
            return "generic_article_bias"
    if is_generic_article(top1):
        return "generic_article_bias"
    if case["expected"].get("accepted_topic_terms") or case["expected"].get("accepted_section_terms"):
        if not topk_context_hit:
            if any(item.get("quality_warnings") for item in target_window):
                return "noisy_text_issue"
            if case["domain"] == "criminal_law":
                return "criminal_offense_miss"
            return "missed_specific_match"
    if case["domain"] == "criminal_law" and (target_window_keyword_hit or 0.0) < 0.34:
        return "criminal_offense_miss"
    if (target_window_keyword_hit or 0.0) < 0.34:
        if any(item.get("quality_warnings") for item in target_window):
            return "noisy_text_issue"
        return "missed_specific_match"
    if benchmark_window_passed:
        return "generic_article_bias"
    return "missed_specific_match"


def summarize_result(result: dict[str, Any] | None) -> dict[str, Any] | None:
    if not result:
        return None
    return {
        "title": result.get("title"),
        "article_number": result.get("article_number"),
        "law_name": result.get("law_name"),
        "legal_domain": result.get("legal_domain"),
        "score": result.get("score"),
        "rerank_score": result.get("rerank_score"),
        "section_level": result.get("section_level"),
        "quality_warnings": result.get("quality_warnings"),
        "rank_explanation": result.get("rank_explanation"),
    }


def law_matches(actual: str | None, expected: str) -> bool:
    if not expected:
        return True
    actual_norm = normalize_legal_arabic(actual or "")
    return expected == actual_norm or expected in actual_norm or actual_norm in expected


def article_matches(result: dict[str, Any] | None, expected_articles: list[str]) -> bool:
    if not result or not expected_articles:
        return False
    article_number = normalize_legal_arabic(result.get("article_number") or "")
    title = normalize_legal_arabic(result.get("title") or "")
    return any(article and (article == article_number or article in title) for article in expected_articles)


def keyword_hit_ratio(result: dict[str, Any] | None, keywords: list[str]) -> float:
    if not result or not keywords:
        return 0.0
    text = result_text_blob(result)
    matches = sum(1 for keyword in keywords if keyword and keyword in text)
    return round(matches / max(1, len(keywords)), 4)


def result_text_blob(result: dict[str, Any]) -> str:
    supporting_text = " ".join(
        " ".join(part for part in ((chunk.get("title") or ""), (chunk.get("content") or "")) if part)
        for chunk in (result.get("supporting_chunks") or [])
    )
    parts = [
        result.get("title") or "",
        result.get("summary") or "",
        result.get("content") or "",
        result.get("section_level") or "",
        result.get("document_level") or "",
        supporting_text,
    ]
    return normalize_legal_arabic(" ".join(part for part in parts if part))


def is_generic_article(result: dict[str, Any]) -> bool:
    title = normalize_legal_arabic(result.get("title") or "")
    section = normalize_legal_arabic(result.get("section_level") or "")
    combined = f"{title} {section}".strip()
    if any(cue in combined for cue in GENERIC_ARTICLE_CUES):
        return True
    article_number = str(result.get("article_number") or "").strip()
    return article_number.isdigit() and int(article_number) <= 5


def ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 4)


def validate_case_report(report: dict[str, Any]) -> None:
    metrics = report["metrics"]
    if report["domain"] != "out_of_domain":
        if report["passed"] and not metrics["exact_top1_domain_accuracy"]:
            raise AssertionError(f"{report['id']}: passed=True while exact_top1_domain_accuracy=False")
        if report["passed"] and not metrics["exact_top1_law_accuracy"]:
            raise AssertionError(f"{report['id']}: passed=True while exact_top1_law_accuracy=False")
        if (
            report["passed"]
            and (
                report["expected"].get("accepted_topic_terms")
                or report["expected"].get("accepted_section_terms")
            )
            and not metrics["top1_context_success"]
        ):
            raise AssertionError(f"{report['id']}: passed=True without required top1 topic/section specificity")
    if report["domain"] == "out_of_domain":
        if report["passed"] and not metrics["out_of_domain_reject_rate"]:
            raise AssertionError(f"{report['id']}: out_of_domain passed without explicit reject")
    if report["passed"] and report["failure_category"] is not None:
        raise AssertionError(f"{report['id']}: passed=True but failure_category is set")
    if not report["passed"] and not report["why_failed"]:
        raise AssertionError(f"{report['id']}: passed=False without why_failed details")


def print_summary(summary: dict[str, Any]) -> None:
    print("\nRetrieval Evaluation Summary")
    print("=" * 80)
    print(f"exact_top1_domain_accuracy : {summary['exact_top1_domain_accuracy']}")
    print(f"exact_top3_domain_hit_rate : {summary['exact_top3_domain_hit_rate']}")
    print(f"exact_top1_law_accuracy    : {summary['exact_top1_law_accuracy']}")
    print(f"exact_top3_law_hit_rate    : {summary['exact_top3_law_hit_rate']}")
    print(f"topk_expected_keyword_hit_rate: {summary['topk_expected_keyword_hit_rate']}")
    print(f"accepted_topic_top1_hit_rate  : {summary['accepted_topic_top1_hit_rate']}")
    print(f"accepted_topic_topk_hit_rate  : {summary['accepted_topic_topk_hit_rate']}")
    print(f"accepted_section_top1_hit_rate: {summary['accepted_section_top1_hit_rate']}")
    print(f"accepted_section_topk_hit_rate: {summary['accepted_section_topk_hit_rate']}")
    print(f"out_of_domain_reject_rate  : {summary['out_of_domain_reject_rate']}")
    print(f"strict_pass_rate           : {summary['strict_pass_rate']}")
    print(f"benchmark_window_pass_rate : {summary['benchmark_window_pass_rate']}")
    print("\nFailure categories")
    for category, count in summary["failure_categories"].items():
        print(f"  - {category}: {count}")


def print_case_details(cases: list[dict[str, Any]], *, failures_only: bool) -> None:
    if not cases:
        return
    print("\nCase details")
    print("=" * 80)
    for case in cases:
        if failures_only and case["passed"]:
            continue
        print(f"{case['id']} | domain={case['domain']} | type={case['query_type']} | passed={case['passed']}")
        print(f"  query: {case['query']}")
        print(f"  evaluation_reason: {case['evaluation_reason']}")
        print(f"  failure_category: {case['failure_category']}")
        print(f"  top_result: {json.dumps(case['top_result'], ensure_ascii=False)}")


def render_report(report: dict[str, Any], suffix: str) -> str:
    if suffix == ".md":
        return render_markdown_report(report)
    return json.dumps(report, ensure_ascii=False, indent=2)


def render_markdown_report(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        f"# Retrieval Evaluation Report: {report['benchmark_name']}",
        "",
        f"- Evaluated cases: {report['evaluated_count']}",
        f"- exact_top1_domain_accuracy: {summary['exact_top1_domain_accuracy']}",
        f"- exact_top3_domain_hit_rate: {summary['exact_top3_domain_hit_rate']}",
        f"- exact_top1_law_accuracy: {summary['exact_top1_law_accuracy']}",
        f"- exact_top3_law_hit_rate: {summary['exact_top3_law_hit_rate']}",
        f"- topk_expected_keyword_hit_rate: {summary['topk_expected_keyword_hit_rate']}",
        f"- accepted_topic_top1_hit_rate: {summary['accepted_topic_top1_hit_rate']}",
        f"- accepted_topic_topk_hit_rate: {summary['accepted_topic_topk_hit_rate']}",
        f"- accepted_section_top1_hit_rate: {summary['accepted_section_top1_hit_rate']}",
        f"- accepted_section_topk_hit_rate: {summary['accepted_section_topk_hit_rate']}",
        f"- out_of_domain_reject_rate: {summary['out_of_domain_reject_rate']}",
        f"- strict_pass_rate: {summary['strict_pass_rate']}",
        f"- benchmark_window_pass_rate: {summary['benchmark_window_pass_rate']}",
        "",
        "## Failure Categories",
    ]
    for category, count in summary["failure_categories"].items():
        lines.append(f"- {category}: {count}")

    lines.append("")
    lines.append("## Case Results")
    for case in report["cases"]:
        lines.append(
            f"- `{case['id']}` | domain={case['domain']} | type={case['query_type']} | passed={case['passed']} | failure={case['failure_category']}"
        )
        lines.append(f"  query: {case['query']}")
        lines.append(f"  evaluation_reason: {case['evaluation_reason']}")
        lines.append(f"  top_result: {json.dumps(case['top_result'], ensure_ascii=False)}")

    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
