# Retrieval Evaluation

This folder-less evaluation setup is meant to measure retrieval quality systematically without touching indexing code or tuning query-by-query.

## Files

- `retrieval_benchmark_dev.json`
  Development benchmark. Use this split while improving retrieval quality.
- `retrieval_benchmark_holdout.json`
  Holdout benchmark. Keep this split for comparison after changes are finalized on dev.
- `evaluate_retrieval.py`
  Runs benchmark queries through `LegalRetriever`, computes summary metrics, and emits JSON or Markdown reports.

## Benchmark Schema

Each case follows this shape:

```json
{
  "id": "criminal_001",
  "query": "ما هي عقوبة القذف والسب؟",
  "domain": "criminal_law",
  "query_type": "specific",
  "expected": {
    "expected_domain": "criminal_law",
    "expected_law_name": "قانون العقوبات المصري",
    "expected_article_numbers": [],
    "expected_keywords": ["القذف", "السب"],
    "accepted_topic_terms": ["القذف", "السب"],
    "accepted_topic_match_mode": "all",
    "top_k_success": 3
  },
  "notes": "..."
}
```

For out-of-domain cases:

```json
{
  "expected_domain": "out_of_domain",
  "expected_behavior": "reject_or_warn"
}
```

Optional expectation fields:

- `accepted_topic_terms`
  Topic terms that must appear in the returned legal context when domain/law correctness alone is too weak.
- `accepted_topic_match_mode`
  `any` or `all`. Use `all` for multi-offense queries such as `القذف والسب`.
- `accepted_section_terms`
  Section/document terms that are acceptable for broad thematic constitutional queries.
- `accepted_section_match_mode`
  `any` or `all`.
- `accepted_context_policy`
  When both topic and section terms are present, defaults to `topic_or_section`. Set to `all` only when both are required together.

## Metrics

`evaluate_retrieval.py` reports these metrics:

Specificity checks are computed from returned legal text and supporting chunks only. They do not count `rank_explanation` tags as evidence.

- `exact_top1_domain_accuracy`
  Share of in-domain cases whose first result belongs to the expected domain.
- `exact_top3_domain_hit_rate`
  Share of in-domain cases whose top 3 contains the expected domain.
- `exact_top1_law_accuracy`
  Share of in-domain cases whose first result belongs to the expected law.
- `exact_top3_law_hit_rate`
  Share of in-domain cases whose top 3 contains the expected law.
- `topk_expected_keyword_hit_rate`
  Mean of the best keyword-hit ratio inside each case’s `top_k_success` window.
- `accepted_topic_top1_hit_rate`
  Share of cases that define `accepted_topic_terms` and whose first result satisfies them.
- `accepted_topic_topk_hit_rate`
  Share of cases that define `accepted_topic_terms` and satisfy them somewhere inside the target window.
- `accepted_section_top1_hit_rate`
  Share of cases that define `accepted_section_terms` and whose first result satisfies them.
- `accepted_section_topk_hit_rate`
  Share of cases that define `accepted_section_terms` and satisfy them somewhere inside the target window.
- `out_of_domain_reject_rate`
  Strict metric. An out-of-domain case counts as handled only when the system returns no results or an explicit reject/warn signal.
- `strict_pass_rate`
  Case-level strict pass rate. A case passes only when the first result matches expected domain and law and satisfies top1 keyword/article checks.
- `benchmark_window_pass_rate`
  Softer metric. Measures whether the expected match appeared anywhere inside the benchmark’s target window.

## Failure Categories

The evaluator assigns one primary failure category per failed case:

- `wrong_domain`
  None of the top 3 results belongs to the expected domain.
- `wrong_law`
  Top results stay in-domain but miss the expected law.
- `generic_article_bias`
  Generic or introductory articles outrank more specific likely matches.
- `missed_specific_match`
  Expected specific terms never meaningfully surface in the evaluation window.
- `constitutional_preamble_bias`
  The constitution preamble outranks substantive constitutional-rights articles on a non-preamble query.
- `criminal_offense_miss`
  Criminal queries fail to surface the offense itself.
- `noisy_text_issue`
  Low keyword alignment is accompanied by quality-warning signals in retrieved results.
- `out_of_domain_not_detected`
  The system returned ordinary legal results instead of an explicit reject/warn for a query outside corpus coverage.

## Case-Level Semantics

Each case now contains:

- `passed`
  Strict top1 pass/fail.
- `evaluation_reason`
  One-line explanation of the final judgment.
- `why_passed`
  Flat list of conditions that were satisfied.
- `why_failed`
  Flat list of unmet conditions.
- `metrics.benchmark_window_passed`
  Indicates whether the correct answer appeared within the case’s allowed target window, even if `passed=false`.
- `metrics.top1_context_success`
  Indicates whether the first result satisfied any required topic/section specificity constraints.
- `metrics.topk_context_hit`
  Indicates whether the target window satisfied any required topic/section specificity constraints.

## Usage

Run the dev benchmark:

```powershell
python evaluate_retrieval.py --benchmark retrieval_benchmark_dev.json --top-k 5 --output retrieval_eval_dev.json
```

Run the holdout benchmark:

```powershell
python evaluate_retrieval.py --benchmark retrieval_benchmark_holdout.json --top-k 5 --output retrieval_eval_holdout.json
```

Evaluate one domain only:

```powershell
python evaluate_retrieval.py --benchmark retrieval_benchmark_dev.json --domain constitutional_law --top-k 5
```

Write Markdown instead of JSON:

```powershell
python evaluate_retrieval.py --benchmark retrieval_benchmark_dev.json --output retrieval_eval_dev.md
```

Show failed cases only:

```powershell
python evaluate_retrieval.py --benchmark retrieval_benchmark_dev.json --failures-only
```

## Reading The Report

- Start with the exact metrics:
  - `exact_top1_domain_accuracy`
  - `exact_top1_law_accuracy`
  - `out_of_domain_reject_rate`
- Then compare them with:
  - `exact_top3_domain_hit_rate`
  - `exact_top3_law_hit_rate`
  - `benchmark_window_pass_rate`
- If `benchmark_window_pass_rate` is much higher than `strict_pass_rate`, then the system often finds the right law/domain but ranks it too low.
- Then inspect `failure_categories` to see the dominant failure pattern.
- Then inspect per-case fields:
  - `evaluation_reason`
  - `why_failed`
  - `top_result`
  - `target_window_results`
- Use dev to guide improvements.
- Use holdout only after changes are stable.

## Recommended Workflow

1. Run dev.
2. Rank failure categories by count.
3. Fix broad retrieval behavior, not individual questions.
4. Re-run dev.
5. Once dev stabilizes, run holdout.
