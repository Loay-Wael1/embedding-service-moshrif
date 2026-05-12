from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.answering.schemas import AnswerMode, SourceCitation
from app.preprocessing import extract_search_terms, normalize_legal_arabic
from app.settings import Settings, settings


@dataclass(slots=True)
class EvaluatedSource:
    raw: dict[str, Any]
    citation: SourceCitation
    score: float
    overlap: float
    has_legal_content: bool
    has_law_identity: bool
    quality_warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class SourceSufficiencyDecision:
    answer_mode: AnswerMode
    internal_grounding_sufficient: bool
    is_out_of_internal_corpus: bool
    sources: list[EvaluatedSource]
    reasons: list[str]
    metrics: dict[str, Any]
    domain: str | None
    law: str | None


def assess_source_sufficiency(
    retrieval_result: dict[str, Any],
    *,
    config: Settings | None = None,
    top_k: int | None = None,
    explicit_domain: str | None = None,
    has_legal_intent: bool = True,
) -> SourceSufficiencyDecision:
    active = config or settings
    query = retrieval_result.get("normalized_query") or retrieval_result.get("query") or ""
    query_analysis = retrieval_result.get("query_analysis") or {}
    retrieval_out_of_scope = bool(query_analysis.get("out_of_domain"))
    results = (retrieval_result.get("results") or [])[:top_k or active.legal_answer_top_k]

    evaluated = [_evaluate_source(item, query) for item in results]
    evaluated = [source for source in evaluated if source.has_legal_content]
    evaluated.sort(key=lambda source: (source.score, source.overlap), reverse=True)

    partial_sources = [
        source
        for source in evaluated
        if source.score >= active.legal_answer_assisted_min_score
        or source.overlap >= active.legal_answer_assisted_min_overlap
        or source.raw.get("supporting_chunks")
    ]
    good_sources = [
        source
        for source in partial_sources
        if source.score >= active.legal_answer_grounded_min_score
        or source.overlap >= active.legal_answer_grounded_min_overlap
        or _has_strong_rank_signal(source.raw)
    ]

    top_score = evaluated[0].score if evaluated else 0.0
    top_overlap = evaluated[0].overlap if evaluated else 0.0
    avg_overlap = _average(source.overlap for source in evaluated[:3])
    domain = _pick_domain(query_analysis, evaluated)
    law = _pick_law(evaluated)
    domain_clear = domain is not None
    law_clear = law is not None
    conflict_detected = _has_clear_conflict(query_analysis, evaluated)
    exact_article_signal = _has_exact_article_signal(query, evaluated[:2])
    explicit_legal_source_signal = _has_explicit_legal_source_signal(query)

    reasons: list[str] = []
    is_out_of_internal_corpus = _is_egyptian_law_question_outside_internal_corpus(
        query=query,
        query_analysis=query_analysis,
        retrieval_out_of_scope=retrieval_out_of_scope,
        evaluated_sources=evaluated,
    )

    if retrieval_out_of_scope:
        reasons.append(query_analysis.get("out_of_domain_reason") or "query_outside_current_corpus_scope")
    if is_out_of_internal_corpus:
        reasons.append("egyptian_law_question_outside_internal_corpus")
    if not evaluated:
        reasons.append("no_usable_legal_sources")
    if not domain_clear:
        reasons.append("domain_not_clear")
    if not law_clear:
        reasons.append("law_not_clear")
    if conflict_detected:
        reasons.append("conflicting_top_sources")
    if top_score < active.legal_answer_assisted_min_score and top_overlap < active.legal_answer_assisted_min_overlap:
        reasons.append("weak_top_source_score_or_overlap")

    grounded_by_count = len(good_sources) >= active.legal_answer_grounded_min_sources
    grounded_by_exact_article = bool(good_sources and exact_article_signal and top_overlap >= active.legal_answer_assisted_min_overlap)

    # Overlap / intent guard: high vector score alone is NOT enough to ground.
    has_overlap_signal = (
        top_overlap > 0
        or avg_overlap > 0
        or exact_article_signal
        or explicit_legal_source_signal
    )

    # Safety guard: greetings / non-legal / vague queries must never become
    # grounded or assisted just because Qdrant returned random results.
    meaningful_legal_signal = _has_meaningful_legal_signal(
        query=query,
        query_analysis=query_analysis,
        has_legal_intent=has_legal_intent,
        explicit_legal_source_signal=explicit_legal_source_signal,
    )

    can_ground = (
        meaningful_legal_signal
        and not retrieval_out_of_scope
        and not conflict_detected
        and domain_clear
        and law_clear
        and bool(good_sources)
        and (grounded_by_count or grounded_by_exact_article)
        and has_overlap_signal
    )
    can_assist = (
        meaningful_legal_signal
        and not retrieval_out_of_scope
        and not conflict_detected
        and len(partial_sources) >= active.legal_answer_assisted_min_sources
    )

    if can_ground:
        mode: AnswerMode = "grounded"
        internal_grounding_sufficient = True
        reasons.append("grounding_sources_sufficient")
    elif can_assist:
        mode = "assisted"
        internal_grounding_sufficient = False
        reasons.append("partial_sources_available")
    elif is_out_of_internal_corpus and meaningful_legal_signal:
        mode = "external_assisted"
        internal_grounding_sufficient = False
        reasons.append("external_assisted_allowed_for_egyptian_law_scope")
    elif meaningful_legal_signal:
        mode = "external_assisted"
        internal_grounding_sufficient = False
        is_out_of_internal_corpus = True
        reasons.append("meaningful_legal_query_fallback_to_external_assisted")
    else:
        mode = "insufficient"
        internal_grounding_sufficient = False
        if "insufficient" not in reasons:
            reasons.append("insufficient")

    metrics = {
        "top_score": round(top_score, 6),
        "top_overlap": round(top_overlap, 6),
        "avg_top3_overlap": round(avg_overlap, 6),
        "usable_source_count": len(evaluated),
        "partial_source_count": len(partial_sources),
        "good_source_count": len(good_sources),
        "domain_clear": domain_clear,
        "law_clear": law_clear,
        "conflict_detected": conflict_detected,
        "exact_article_signal": exact_article_signal,
        "explicit_legal_source_signal": explicit_legal_source_signal,
        "has_legal_intent": has_legal_intent,
        "retrieval_out_of_scope": retrieval_out_of_scope,
        "is_out_of_internal_corpus": is_out_of_internal_corpus,
        "thresholds": {
            "grounded_min_sources": active.legal_answer_grounded_min_sources,
            "assisted_min_sources": active.legal_answer_assisted_min_sources,
            "grounded_min_score": active.legal_answer_grounded_min_score,
            "assisted_min_score": active.legal_answer_assisted_min_score,
            "grounded_min_overlap": active.legal_answer_grounded_min_overlap,
            "assisted_min_overlap": active.legal_answer_assisted_min_overlap,
        },
    }

    return SourceSufficiencyDecision(
        answer_mode=mode,
        internal_grounding_sufficient=internal_grounding_sufficient,
        is_out_of_internal_corpus=is_out_of_internal_corpus,
        sources=partial_sources if mode in {"grounded", "assisted"} else evaluated,
        reasons=_dedupe(reasons),
        metrics=metrics,
        domain=domain,
        law=law,
    )


def _is_egyptian_law_question_outside_internal_corpus(
    *,
    query: str,
    query_analysis: dict[str, Any],
    retrieval_out_of_scope: bool,
    evaluated_sources: list[EvaluatedSource],
) -> bool:
    reason = str(query_analysis.get("out_of_domain_reason") or "").lower()
    if retrieval_out_of_scope and ("personal-status" in reason or "family-law" in reason):
        return True

    query_norm = normalize_legal_arabic(query)
    if retrieval_out_of_scope and _contains_any(query_norm, OUT_OF_INTERNAL_CORPUS_LEGAL_CUES):
        return True

    # For zero-source results, require meaningful legal content
    # to avoid promoting vague queries.
    if not evaluated_sources:
        query_norm = normalize_legal_arabic(query)
        has_general_legal = _contains_any(query_norm, GENERAL_EGYPTIAN_LEGAL_CUES)
        has_scenario = _contains_any(query_norm, _SCENARIO_CUES)
        has_conceptual = _is_conceptual_legal_question(query_norm)
        if has_general_legal and (has_scenario or has_conceptual):
            return True

    return False


def _evaluate_source(item: dict[str, Any], query: str) -> EvaluatedSource:
    text = _source_text(item)
    query_terms = set(extract_search_terms(normalize_legal_arabic(query)))
    source_terms = set(extract_search_terms(normalize_legal_arabic(text)))
    overlap = len(query_terms & source_terms) / max(1, len(query_terms))
    score = _source_score(item)
    has_legal_content = bool(text.strip()) and (
        bool(item.get("law_name"))
        or bool(item.get("article_number"))
        or bool(item.get("legal_domain"))
        or bool(item.get("source_url"))
    )
    has_law_identity = bool(item.get("law_name") or item.get("law_number") or item.get("law_year"))
    return EvaluatedSource(
        raw=item,
        citation=_to_citation(item, score),
        score=score,
        overlap=overlap,
        has_legal_content=has_legal_content,
        has_law_identity=has_law_identity,
        quality_warnings=list(item.get("quality_warnings") or []),
    )


def _to_citation(item: dict[str, Any], score: float) -> SourceCitation:
    return SourceCitation(
        id=_string_or_none(item.get("id")),
        law_name=_string_or_none(item.get("law_name")),
        law_number=_string_or_none(item.get("law_number")),
        law_year=_string_or_none(item.get("law_year")),
        article_number=_string_or_none(item.get("article_number")),
        title=_string_or_none(item.get("title")),
        source_url=_string_or_none(item.get("source_url")),
        section_level=_string_or_none(item.get("section_level")),
        document_level=_string_or_none(item.get("document_level")),
        legal_domain=_string_or_none(item.get("legal_domain")),
        score=round(score, 6),
        summary_snippet=_snippet(item.get("summary") or item.get("retrieval_text") or ""),
        quote_snippet=_snippet(item.get("content") or _supporting_chunk_text(item.get("supporting_chunks") or [])),
    )


def _source_text(item: dict[str, Any]) -> str:
    parts = [
        item.get("law_name") or "",
        item.get("title") or "",
        item.get("section_level") or "",
        item.get("document_level") or "",
        item.get("summary") or "",
        item.get("content") or "",
        item.get("retrieval_text") or "",
        " ".join(str(value) for value in item.get("keywords") or []),
        " ".join(str(value) for value in item.get("semantic_tags") or []),
        _supporting_chunk_text(item.get("supporting_chunks") or []),
    ]
    return "\n".join(part for part in parts if part).strip()


def _supporting_chunk_text(chunks: list[dict[str, Any]]) -> str:
    return "\n".join(
        "\n".join(part for part in (chunk.get("title") or "", chunk.get("content") or "") if part)
        for chunk in chunks
    ).strip()


def _source_score(item: dict[str, Any]) -> float:
    value = item.get("rerank_score", item.get("score", 0.0))
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _pick_domain(query_analysis: dict[str, Any], sources: list[EvaluatedSource]) -> str | None:
    suggested = query_analysis.get("suggested_domain")
    if suggested:
        return str(suggested)
    return _dominant_value(source.raw.get("legal_domain") for source in sources[:3])


def _pick_law(sources: list[EvaluatedSource]) -> str | None:
    names = [
        source.raw.get("law_name")
        or " ".join(part for part in (source.raw.get("law_number"), source.raw.get("law_year")) if part)
        for source in sources[:3]
    ]
    return _dominant_value(names)


def _dominant_value(values) -> str | None:
    counts: dict[str, int] = {}
    for value in values:
        if not value:
            continue
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    if not counts:
        return None
    return sorted(counts.items(), key=lambda item: item[1], reverse=True)[0][0]


def _has_clear_conflict(query_analysis: dict[str, Any], sources: list[EvaluatedSource]) -> bool:
    if len(sources) < 2:
        return False
    suggested_domain = query_analysis.get("suggested_domain")
    top = sources[:3]
    domains = {source.raw.get("legal_domain") for source in top if source.raw.get("legal_domain")}
    if len(domains) <= 1:
        return False
    if suggested_domain and all(source.raw.get("legal_domain") == suggested_domain for source in top[:2]):
        return False
    score_gap = abs(top[0].score - top[1].score)
    return score_gap <= 0.05 and not suggested_domain


def _has_exact_article_signal(query: str, sources: list[EvaluatedSource]) -> bool:
    query_norm = normalize_legal_arabic(query)
    for source in sources:
        article_number = str(source.raw.get("article_number") or "").strip()
        if article_number and article_number in query_norm:
            return True
        law_number = str(source.raw.get("law_number") or "").strip()
        law_year = str(source.raw.get("law_year") or "").strip()
        if law_number and law_year and law_number in query_norm and law_year in query_norm:
            return True
    return False


def _has_explicit_legal_source_signal(query: str) -> bool:
    query_norm = normalize_legal_arabic(query)
    source_phrases = (
        "الدستور المصري",
        "دستور جمهورية مصر العربية",
        "قانون العمل المصري",
        "قانون العمل",
        "القانون المدني المصري",
        "القانون المدني",
        "قانون العقوبات المصري",
        "قانون العقوبات",
    )
    return any(normalize_legal_arabic(phrase) in query_norm for phrase in source_phrases)


def _has_strong_rank_signal(item: dict[str, Any]) -> bool:
    reasons = item.get("rank_explanation") or []
    return any(
        str(reason).startswith("phrase_match")
        or str(reason) in {"strong_title_overlap", "strong_summary_overlap", "exact_offense_match"}
        for reason in reasons
    )


def _average(values) -> float:
    collected = [float(value) for value in values]
    if not collected:
        return 0.0
    return sum(collected) / len(collected)


def _snippet(text: str, *, limit: int = 420) -> str | None:
    compact = " ".join((text or "").split())
    if not compact:
        return None
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


OUT_OF_INTERNAL_CORPUS_LEGAL_CUES = tuple(
    normalize_legal_arabic(value)
    for value in (
        "الأحوال الشخصية",
        "احوال شخصية",
        "الحضانة",
        "حضانة",
        "النفقة",
        "نفقة",
        "الطلاق",
        "طلاق",
        "الخلع",
        "خلع",
        "الميراث",
        "ميراث",
        "الزواج",
        "زواج",
        "محكمة الأسرة",
        "محكمة الاسرة",
        "رؤية الصغير",
        "الولاية التعليمية",
    )
)

GENERAL_EGYPTIAN_LEGAL_CUES = tuple(
    normalize_legal_arabic(value)
    for value in (
        "القانون المصري",
        "في مصر",
        "قانون",
        "القانون",
        "أحكام",
        "احكام",
        "حكم",
        "عقوبة",
        "عقوبات",
        "جريمة",
        "دعوى",
        "محكمة",
        "حقوق",
        "التزامات",
        "إجراءات",
        "اجراءات",
        "مادة",
        "المادة",
        "الحضانة",
        "حضانة",
        "نفقة",
        "ميراث",
        "طلاق",
        "خلع",
    )
)

# Personal-status / family-law cues used for confident pre-retrieval shortcircuit.
CONFIDENT_PERSONAL_STATUS_CUES = tuple(
    normalize_legal_arabic(value)
    for value in (
        "الحضانة",
        "حضانة",
        "النفقة",
        "نفقة",
        "الطلاق",
        "طلاق",
        "الخلع",
        "خلع",
        "الميراث",
        "ميراث",
        "مؤخر الصداق",
        "الرؤية",
        "رؤية الصغير",
        "الولاية التعليمية",
        "الأحوال الشخصية",
        "احوال شخصية",
    )
)


def _contains_any(text: str, cues: tuple[str, ...]) -> bool:
    """Return True if *text* contains any of the normalised *cues*."""
    return any(cue in text for cue in cues)


def _is_confident_out_of_internal_corpus(query: str) -> bool:
    """Pre-retrieval check: is this query confidently about a personal-status
    / family-law topic that our internal corpus does not cover?

    Only returns True for high-confidence matches so uncertain queries
    still go through full retrieval.
    """
    query_norm = normalize_legal_arabic(query)
    return _contains_any(query_norm, CONFIDENT_PERSONAL_STATUS_CUES)


# ---------------------------------------------------------------------------
# Generic query classification helpers
# ---------------------------------------------------------------------------

_MIN_MEANINGFUL_WORDS = 3


def _is_low_information_query(query_norm: str) -> bool:
    """Return True for empty, near-empty, punctuation-only, or generic
    help-request queries that carry no actionable legal content.

    This is a category-based check, NOT an exact-phrase list.
    """
    words = query_norm.split()

    # Empty or pure punctuation
    if not words:
        return True

    # Very short (1–2 words) with no legal concept
    if len(words) < _MIN_MEANINGFUL_WORDS:
        return True

    # Generic help/request patterns without a concrete subject
    # Catches: "محتاج مساعدة قانونية", "عندي مشكلة", "ممكن اعرف حقي",
    #          "فيه مشكلة قانونية", "عندي قضية اعمل ايه"
    _GENERIC_REQUEST_STEMS = (
        "محتاج", "ممكن", "عايز", "عندي",
    )
    _GENERIC_OBJECTS = (
        "مساعد", "مشكل", "قضي", "حق",
    )
    first_word = words[0] if words else ""
    if any(first_word.startswith(s) for s in _GENERIC_REQUEST_STEMS):
        # If no concrete scenario cue exists alongside, it is vague
        if not _contains_any(query_norm, _SCENARIO_CUES):
            return True
    # "فيه مشكلة ..." pattern
    if len(words) <= 4 and any(query_norm.startswith(normalize_legal_arabic(p)) for p in ("فيه ",)):
        if not _contains_any(query_norm, _SCENARIO_CUES):
            return True

    return False


# Concrete real-world scenario cues (compact category-based list).
_SCENARIO_CUES = tuple(
    normalize_legal_arabic(v)
    for v in (
        # Criminal actions
        "سرق", "سرقه", "سرقة", "نصب", "نصبوا", "ضرب", "ضربني",
        "هدد", "هددني", "تهديد", "قتل", "تزوير", "رشوة", "خطف",
        "اعتداء", "اتضربت", "اتسرق", "اتنصب",
        # Reports / complaints
        "بلاغ", "محضر", "ابلغ", "اشتكي", "ارفع",
        # Employment
        "مرتب", "مرتبي", "فصل", "فصلوني", "فصلني", "شغل", "الشغل",
        # Contracts / debts
        "عقد", "ايجار", "إيجار", "بيع", "شراء",
        "ايصال", "إيصال", "شيك", "دين", "ديون",
        # Penalties / legal outcomes
        "عقوبه", "عقوبة", "تعويض", "غرامه", "غرامة", "حبس", "سجن",
    )
)


def _is_concrete_legal_scenario(query_norm: str, query_analysis: dict[str, Any]) -> bool:
    """Return True when the query describes a real-world legal situation:
    theft, fraud, assault, employment dispute, contract issue, etc."""
    if _contains_any(query_norm, _SCENARIO_CUES):
        return True
    # Query analysis may flag a criminal offense or specific legal scenario
    if query_analysis.get("criminal_offense_query"):
        return True
    return False


# Question / conceptual patterns — must appear at the START of the query.
_CONCEPTUAL_QUESTION_PATTERNS = tuple(
    normalize_legal_arabic(v)
    for v in (
        "ما هو", "ما هي", "ما هو ال", "ما هي ال",
        "ما معنى", "ما المقصود", "ما الفرق",
        "اشرح", "عرفني", "تعريف",
        "ايه هو", "ايه هي", "يعني ايه",
    )
)

# Legal concept terms that make a conceptual question meaningful.
_LEGAL_CONCEPT_TERMS = tuple(
    normalize_legal_arabic(v)
    for v in (
        "القانون", "قانون", "الدستور", "دستور",
        "جريمه", "جريمة", "جنحه", "جنحة", "جنايه", "جناية", "مخالفه", "مخالفة",
        "عقوبه", "عقوبة", "عقد", "دعوى",
        "مسؤوليه", "مسؤولية", "تعويض",
        "حقوق",
        "مدني", "المدني", "جنائي", "الجنائي",
        "عمل", "العمل", "تجاري", "التجاري",
        "العقوبات", "عقوبات",
    )
)


def _is_conceptual_legal_question(query_norm: str) -> bool:
    """Return True when the query is a conceptual/definitional legal question,
    e.g. 'ما هو القانون المصري', 'ما الفرق بين الجنحة والجناية'.

    Requires BOTH:
    - a question/explanation pattern at the START of the query
    - a legal concept term anywhere in the query
    """
    has_question_pattern = any(
        query_norm.startswith(p) for p in _CONCEPTUAL_QUESTION_PATTERNS
    )
    has_legal_concept = _contains_any(query_norm, _LEGAL_CONCEPT_TERMS)
    return has_question_pattern and has_legal_concept


def _has_meaningful_legal_signal(
    *,
    query: str,
    query_analysis: dict[str, Any],
    has_legal_intent: bool,
    explicit_legal_source_signal: bool,
) -> bool:
    """Return True if the query carries enough legal substance to justify
    internal grounding or external-assisted general guidance.

    This is the single gate for both:
    - allowing grounded/assisted mode (safety guard against random Qdrant hits)
    - allowing external_assisted fallback when sources are insufficient

    Returns False for low-information queries regardless of has_legal_intent.
    """
    query_norm = normalize_legal_arabic(query)

    # Low-information queries never qualify, even if the router said legal
    if _is_low_information_query(query_norm):
        return False

    # Explicit legal source reference (e.g. "قانون العقوبات المصري")
    if explicit_legal_source_signal:
        return True

    # Concrete real-world scenario
    if _is_concrete_legal_scenario(query_norm, query_analysis):
        return True

    # Conceptual / definitional legal question
    if _is_conceptual_legal_question(query_norm):
        return True

    # Strong query_analysis signals from the retriever
    if query_analysis.get("suggested_domain"):
        return True
    if query_analysis.get("legal_keywords"):
        return True
    if query_analysis.get("criminal_offense_query"):
        return True

    # General legal cues with legal intent (e.g. "ما ضمانات الحرية الشخصية")
    if has_legal_intent and _contains_any(query_norm, GENERAL_EGYPTIAN_LEGAL_CUES):
        return True

    return False



