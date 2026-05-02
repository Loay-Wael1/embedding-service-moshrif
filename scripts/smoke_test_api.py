#!/usr/bin/env python3
"""Smoke-test script for the Egyptian Laws Embedding Service API.

Usage:
    python scripts/smoke_test_api.py --base-url http://127.0.0.1:8000
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import requests


def main() -> None:
    parser = argparse.ArgumentParser(description="API smoke test")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--include-debug", action="store_true", help="Test hidden/debug endpoints like /info")
    args = parser.parse_args()
    base = args.base_url.rstrip("/")

    results: list[bool] = []

    results.append(_test_get(base, "/health", expect_keys=["status"]))
    if args.include_debug:
        results.append(
            _test_get(
                base,
                "/info",
                expect_keys=["model_name", "embedding_dimension", "supports_sparse", "supported_outputs"],
            )
        )
    results.append(_test_get(base, "/legal-info", expect_keys=["service", "status", "answer_modes"]))

    results.append(
        _test_legal(
            base,
            "/chat",
            "identity",
            query="اسمك إيه؟",
            checks={"answer_mode": "identity", "llm.called": False},
        )
    )
    results.append(
        _test_legal(
            base,
            "/chat",
            "conversation",
            query="السلام عليكم",
            checks={"answer_mode": "conversation", "llm.called": False},
        )
    )
    results.append(
        _test_legal(
            base,
            "/chat",
            "grounded",
            query="ما هي أحكام عقد العمل الفردي؟",
            checks={"answer_mode_in": ("grounded", "assisted"), "llm.called": True},
        )
    )
    results.append(
        _test_legal(
            base,
            "/chat",
            "external_assisted",
            query="ما هي أحكام الحضانة؟",
            checks={"answer_mode": "external_assisted", "is_out_of_internal_corpus": True},
        )
    )
    results.append(
        _test_legal(
            base,
            "/chat",
            "non_legal",
            query="ما أفضل مطعم؟",
            checks={"answer_mode_in": ("non_legal", "insufficient"), "llm.called": False},
        )
    )

    passed = sum(results)
    total = len(results)
    print(f"\n{'=' * 50}")
    print(f"Results: {passed}/{total} passed")
    if passed < total:
        print("FAIL - some tests did not pass.")
        sys.exit(1)
    print("ALL OK")
    sys.exit(0)


def _test_get(base: str, path: str, *, expect_keys: list[str]) -> bool:
    label = f"GET {path}"
    try:
        resp = requests.get(f"{base}{path}", timeout=30)
        if resp.status_code != 200:
            _print_result(label, resp.status_code, False)
            return False
        data = resp.json()
        missing = [key for key in expect_keys if key not in data]
        ok = not missing
        info = ""
        if missing:
            preview = json.dumps(list(data.keys())[:8], ensure_ascii=False)
            info = f"missing keys: {missing} | actual keys: {preview}"
        _print_result(label, resp.status_code, ok, info=info)
        return ok
    except Exception as exc:
        _print_result(label, 0, False, error=str(exc))
        return False


def _test_legal(
    base: str,
    endpoint: str,
    label: str,
    *,
    query: str,
    body_extra: dict[str, Any] | None = None,
    checks: dict[str, Any],
) -> bool:
    body: dict[str, Any] = {"query": query}
    if body_extra:
        body.update(body_extra)
    try:
        resp = requests.post(f"{base}{endpoint}", json=body, timeout=120)
        if resp.status_code != 200:
            _print_result(
                f"POST {endpoint} [{label}]",
                resp.status_code,
                False,
                info=resp.text[:200] if resp.text else "",
            )
            return False
        data = resp.json()
        ok = True
        details: list[str] = []
        for key, expected in checks.items():
            if key.endswith("_in"):
                real_key = key[:-3]
                actual = _deep_get(data, real_key)
                if actual not in expected:
                    ok = False
                    details.append(f"{real_key}={actual} NOT IN {expected}")
                else:
                    details.append(f"{real_key}={actual}")
            else:
                actual = _deep_get(data, key)
                if actual != expected:
                    ok = False
                    details.append(f"{key}={actual} (expected {expected})")
                else:
                    details.append(f"{key}={actual}")

        top_k = _deep_get(data, "retrieval_summary.top_k_used")
        sources = len(_deep_get(data, "internal_sources") or _deep_get(data, "sources") or [])
        details.append(f"top_k={top_k} sources={sources}")
        _print_result(f"POST {endpoint} [{label}]", resp.status_code, ok, info=" | ".join(details))
        return ok
    except Exception as exc:
        _print_result(f"POST {endpoint} [{label}]", 0, False, error=str(exc))
        return False


def _deep_get(data: dict[str, Any], dotted_key: str) -> Any:
    current: Any = data
    for part in dotted_key.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _print_result(label: str, status: int, ok: bool, *, info: str = "", error: str = "") -> None:
    tag = "OK" if ok else "FAIL"
    line = f"  [{tag:4s}] {label} (HTTP {status})"
    if info:
        line += f"  {info}"
    if error:
        line += f"  ERROR: {error}"
    print(line)


if __name__ == "__main__":
    main()
