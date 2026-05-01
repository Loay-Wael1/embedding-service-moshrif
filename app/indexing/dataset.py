from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from app.models import LegalRecord
from app.preprocessing.legal_arabic import assess_text_quality, is_repealed_text, normalize_legal_arabic


LAW_NAME_OVERRIDES = {
    "labor_law": "قانون العمل المصري",
    "civil_law": "القانون المدني المصري",
    "criminal_law": "قانون العقوبات المصري",
}


def load_legal_dataset(path: str | Path, *, include_law_records: bool = True) -> list[LegalRecord]:
    raw_records = _load_raw_records(Path(path))
    records = [_convert_record(item) for item in raw_records]
    if include_law_records:
        records.extend(_build_law_records(records))
    return records


def _load_raw_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
        if not isinstance(payload, list):
            raise ValueError("JSON dataset must contain a top-level list of records")
        return payload


def _convert_record(payload: dict[str, Any]) -> LegalRecord:
    metadata = payload.get("retrieval_metadata", {})
    context = payload.get("hierarchical_context", {})
    quality_flags = [str(flag) for flag in payload.get("quality_flags", [])]

    title = _clean_text(payload.get("title"))
    content = _clean_text(payload.get("content"))
    summary = _clean_text(payload.get("summary"))
    document_level = _clean_text(context.get("document_level"))
    section_level = _clean_text(context.get("section_level"))
    article_number = _clean_text(context.get("article_level"))
    record_kind = str(metadata.get("record_kind") or "article")
    legal_domain = str(metadata.get("legal_domain") or "unknown_law")
    law_number = _clean_text(metadata.get("law_number"))
    law_year = _clean_text(metadata.get("law_year"))
    law_name = _derive_law_name(legal_domain, document_level, title)
    retrieval_text = _clean_text(payload.get("retrieval_text")) or _compose_retrieval_text(
        legal_domain=legal_domain,
        law_name=law_name,
        law_number=law_number,
        law_year=law_year,
        article_number=article_number,
        title=title,
        content=content,
        summary=summary,
        section_level=section_level,
        document_level=document_level,
    )
    embedding_text = _clean_text(payload.get("embedding_text")) or retrieval_text

    quality_assessment = assess_text_quality(content or retrieval_text)
    quality_score = max(0.0, 1.0 - quality_assessment.noise_score)
    if "text_cleaned" in quality_flags:
        quality_score = min(1.0, quality_score + 0.05)
    if "normalized_v2" in quality_flags:
        quality_score = min(1.0, quality_score + 0.02)

    status = _clean_text(metadata.get("status"))
    status_normalized = _clean_text(metadata.get("status_normalized")) or "unknown"
    is_repealed_candidate = status_normalized not in {"current", ""} or is_repealed_text(
        " ".join(filter(None, [title, summary, content]))
    )

    return LegalRecord(
        record_id=str(payload["id"]),
        record_kind=record_kind,
        parent_id=_clean_text(metadata.get("parent_id")),
        legal_domain=legal_domain,
        law_name=law_name,
        law_number=law_number,
        law_year=law_year,
        article_number=article_number,
        title=title,
        content=content,
        summary=summary,
        source_url=_clean_text(metadata.get("source_url")),
        status=status,
        status_normalized=status_normalized,
        quality_flags=quality_flags,
        quality_warnings=quality_assessment.warnings,
        quality_score=round(quality_score, 4),
        noise_score=round(quality_assessment.noise_score, 4),
        section_level=section_level,
        document_level=document_level,
        retrieval_text=retrieval_text,
        embedding_text=embedding_text,
        jurisdiction=_clean_text(metadata.get("jurisdiction")),
        language=_clean_text(metadata.get("language")),
        keywords=[str(item) for item in payload.get("keywords", [])],
        semantic_tags=[str(item) for item in payload.get("semantic_tags", [])],
        cross_references=[str(item) for item in payload.get("cross_references", [])],
        chunk_index=_to_int(metadata.get("chunk_index")),
        chunk_total=_to_int(metadata.get("chunk_total")),
        is_repealed_candidate=is_repealed_candidate,
    )


def _build_law_records(records: Iterable[LegalRecord]) -> list[LegalRecord]:
    grouped: dict[tuple[str, str | None, str | None], list[LegalRecord]] = defaultdict(list)
    for record in records:
        if record.record_kind != "article":
            continue
        grouped[(record.legal_domain, record.law_number, record.law_year)].append(record)

    law_records: list[LegalRecord] = []
    for (legal_domain, law_number, law_year), items in grouped.items():
        if not items:
            continue

        article_titles = [item.title for item in items[:20] if item.title]
        section_titles = sorted({item.section_level for item in items if item.section_level})[:20]
        law_name = items[0].law_name
        record_id = f"law::{legal_domain}::{law_number or 'na'}::{law_year or 'na'}"
        summary = (
            f"{law_name} رقم {law_number or 'غير محدد'} لسنة {law_year or 'غير محددة'}."
            f" يحتوي على {len(items)} مادة مفهرسة للاسترجاع."
        )
        content = "\n".join(
            filter(
                None,
                [
                    summary,
                    "الأبواب أو الأقسام:",
                    "؛ ".join(section_titles),
                    "مواد بارزة:",
                    "؛ ".join(article_titles),
                ],
            )
        )
        retrieval_text = _compose_retrieval_text(
            legal_domain=legal_domain,
            law_name=law_name,
            law_number=law_number,
            law_year=law_year,
            article_number=None,
            title=law_name,
            content=content,
            summary=summary,
            section_level=None,
            document_level=law_name,
        )
        average_noise = sum(item.noise_score for item in items) / max(len(items), 1)
        average_quality = sum(item.quality_score for item in items) / max(len(items), 1)
        source_url = next((item.source_url for item in items if item.source_url), None)

        law_records.append(
            LegalRecord(
                record_id=record_id,
                record_kind="law",
                parent_id=None,
                legal_domain=legal_domain,
                law_name=law_name,
                law_number=law_number,
                law_year=law_year,
                article_number=None,
                title=law_name,
                content=content,
                summary=summary,
                source_url=source_url,
                status="سارية",
                status_normalized="current",
                quality_flags=["synthetic_law_record"],
                quality_warnings=[],
                quality_score=round(average_quality, 4),
                noise_score=round(average_noise, 4),
                section_level=None,
                document_level=law_name,
                retrieval_text=retrieval_text,
                embedding_text=retrieval_text,
                jurisdiction="egypt",
                language="arabic",
                keywords=[],
                semantic_tags=[],
                cross_references=[],
                chunk_index=None,
                chunk_total=None,
                is_repealed_candidate=False,
            )
        )

    return law_records


def _derive_law_name(legal_domain: str, document_level: str, title: str) -> str:
    if legal_domain in LAW_NAME_OVERRIDES:
        return LAW_NAME_OVERRIDES[legal_domain]
    if document_level:
        return document_level
    title_normalized = normalize_legal_arabic(title)
    if " - " in title_normalized:
        return title_normalized.split(" - ", 1)[-1]
    return title_normalized or legal_domain


def _compose_retrieval_text(
    *,
    legal_domain: str,
    law_name: str,
    law_number: str | None,
    law_year: str | None,
    article_number: str | None,
    title: str,
    content: str,
    summary: str,
    section_level: str | None,
    document_level: str | None,
) -> str:
    segments = [
        legal_domain,
        law_name,
        document_level,
        section_level,
        f"قانون رقم {law_number} لسنة {law_year}" if law_number and law_year else None,
        f"المادة {article_number}" if article_number else None,
        title,
        summary,
        content,
    ]
    return "\n".join(part for part in segments if part)


def _clean_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _to_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)
