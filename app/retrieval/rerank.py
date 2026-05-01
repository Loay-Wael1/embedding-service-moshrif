from __future__ import annotations

from functools import lru_cache
from typing import Any

from app.preprocessing import extract_search_terms, normalize_legal_arabic
from app.retrieval.query_understanding import QueryAnalysis, analyze_legal_query
from app.settings import Settings, settings


GENERAL_ARTICLE_CUES = (
    "احكام عامه",
    "أحكام عامة",
    "احكام عامة",
    "تعاريف",
    "تعريفات",
    "تعريف",
    "نطاق التطبيق",
    "سريان",
    "الكتاب الاول",
    "الباب الاول",
    "مبادئ عامه",
    "مبادئ عامة",
)

PREAMBLE_CUES = (
    "الديباجة",
    "الديباجه",
)

CONSTITUTIONAL_RIGHTS_SECTION_CUES = (
    "الحقوق والحريات",
    "الحقوق والحريات والواجبات العامة",
    "الحقوق والحريات والواجبات العامه",
    "باب الحقوق والحريات",
    "الحقوق الاساسية",
    "الحقوق الأساسية",
)

CONSTITUTIONAL_CORE_RIGHTS_CUES = (
    "المساواة",
    "المساواه",
    "عدم التمييز",
    "الحرية",
    "الحرية",
    "الكرامة",
    "الكرامه",
    "الحق في",
    "حقوق الانسان",
    "حقوق الإنسان",
    "المواطنون لدى القانون",
    "تكافؤ الفرص",
    "الحرية الشخصية",
    "الحرية الشخصيه",
)

CONSTITUTIONAL_SUBTOPIC_CUES = (
    "النقابات",
    "النقابه",
    "الجمعيات",
    "الجمعيه",
    "المعلومات",
    "البيانات",
    "الصحافة",
    "الصحافه",
    "الاعلام",
    "الإعلام",
    "الأحزاب",
    "الاحزاب",
)

CRIMINAL_GENERIC_CUES = (
    "احكام عامة",
    "أحكام عامة",
    "العقوبات الاصلية",
    "العقوبات الأصلية",
    "العقوبات التبعية",
    "احكام تمهيدية",
    "أحكام تمهيدية",
    "المساهمة الجنائية",
)

class BaseReranker:
    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        *,
        query_analysis: QueryAnalysis | None = None,
    ) -> list[dict[str, Any]]:
        raise NotImplementedError


class NoOpReranker(BaseReranker):
    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        *,
        query_analysis: QueryAnalysis | None = None,
    ) -> list[dict[str, Any]]:
        return candidates


class FeatureBasedLegalReranker(BaseReranker):
    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        *,
        query_analysis: QueryAnalysis | None = None,
    ) -> list[dict[str, Any]]:
        analysis = query_analysis or analyze_legal_query(query)
        query_norm = analysis.normalized_query
        query_tokens = set(analysis.legal_keywords or extract_search_terms(query_norm))
        query_phrases = analysis.key_phrases

        for candidate in candidates:
            candidate_domain = candidate.get("legal_domain")
            title_norm = normalize_legal_arabic(candidate.get("title") or "")
            summary_norm = normalize_legal_arabic(candidate.get("summary") or "")
            content_norm = normalize_legal_arabic(candidate.get("content") or "")
            section_norm = normalize_legal_arabic(candidate.get("section_level") or "")
            document_norm = normalize_legal_arabic(candidate.get("document_level") or "")
            chunk_text_norm = normalize_legal_arabic(_supporting_chunk_text(candidate.get("supporting_chunks", [])))

            title_tokens = set(extract_search_terms(title_norm))
            summary_tokens = set(extract_search_terms(summary_norm))
            content_tokens = set(extract_search_terms(content_norm))
            section_tokens = set(extract_search_terms(section_norm))
            chunk_tokens = set(extract_search_terms(chunk_text_norm))
            candidate_tokens = title_tokens | summary_tokens | content_tokens | section_tokens | chunk_tokens

            overlap_title = _token_overlap(query_tokens, title_tokens)
            overlap_summary = _token_overlap(query_tokens, summary_tokens)
            overlap_content = _token_overlap(query_tokens, content_tokens)
            keyword_coverage = _token_overlap(query_tokens, candidate_tokens)

            phrase_hits = {
                "title": _phrase_match_score(query_phrases, title_norm),
                "summary": _phrase_match_score(query_phrases, summary_norm),
                "content": _phrase_match_score(query_phrases, content_norm),
                "section": _phrase_match_score(query_phrases, section_norm),
                "chunks": _phrase_match_score(query_phrases, chunk_text_norm),
            }

            matched_phrases = [
                phrase
                for phrase in query_phrases
                if any(
                    phrase in text
                    for text in (title_norm, summary_norm, content_norm, section_norm, chunk_text_norm)
                )
            ]

            article_number = normalize_legal_arabic(candidate.get("article_number") or "")
            law_name_norm = normalize_legal_arabic(candidate.get("law_name") or "")
            section_phrase_bonus = 0.10 * phrase_hits["section"]
            content_phrase_bonus = 0.14 * phrase_hits["content"]
            chunk_phrase_bonus = 0.12 * phrase_hits["chunks"]
            summary_phrase_bonus = 0.20 * phrase_hits["summary"]
            title_phrase_bonus = 0.28 * phrase_hits["title"]
            direct_article_bonus = 0.05 if candidate.get("matched_record_kind") == "article" else 0.0
            direct_article_bonus += 0.04 if float(candidate.get("direct_article_score", 0.0)) > 0 else 0.0
            strong_chunk_bonus = 0.04 if float(candidate.get("best_chunk_score", 0.0)) > 0 else 0.0
            support_bonus = min(0.10, 0.02 * len(candidate.get("supporting_chunks", [])))
            article_number_bonus = 0.12 if article_number and article_number in query_norm else 0.0
            law_bonus = 0.05 if law_name_norm and law_name_norm in query_norm else 0.0

            general_penalty = _general_article_penalty(
                analysis,
                candidate,
                title_norm,
                summary_norm,
                section_norm,
                document_norm,
                matched_phrases,
            )
            domain_adjustment = _domain_alignment_adjustment(analysis, candidate_domain)
            criminal_adjustment = _criminal_adjustment(
                analysis=analysis,
                candidate=candidate,
                title_norm=title_norm,
                summary_norm=summary_norm,
                content_norm=content_norm,
                section_norm=section_norm,
                chunk_text_norm=chunk_text_norm,
            )
            constitutional_adjustment = _constitutional_adjustment(
                analysis=analysis,
                candidate=candidate,
                title_norm=title_norm,
                summary_norm=summary_norm,
                content_norm=content_norm,
                section_norm=section_norm,
                document_norm=document_norm,
            )
            specificity_bonus = 0.0
            if analysis.prefer_specific_articles:
                specificity_bonus += title_phrase_bonus
                specificity_bonus += summary_phrase_bonus
                specificity_bonus += content_phrase_bonus
                specificity_bonus += section_phrase_bonus
                specificity_bonus += chunk_phrase_bonus
                specificity_bonus += 0.12 * keyword_coverage
                specificity_bonus += 0.10 if matched_phrases else 0.0
            else:
                specificity_bonus += 0.08 * keyword_coverage
                specificity_bonus += 0.08 * overlap_title
                specificity_bonus += 0.05 * overlap_summary

            status_penalty = -0.30 if candidate.get("is_repealed_candidate") else 0.0
            quality_penalty = float(candidate.get("noise_score", 0.0)) * 0.18

            rerank_score = (
                float(candidate.get("score", 0.0))
                + (0.22 * overlap_title)
                + (0.14 * overlap_summary)
                + (0.08 * overlap_content)
                + direct_article_bonus
                + strong_chunk_bonus
                + support_bonus
                + article_number_bonus
                + law_bonus
                + specificity_bonus
                + domain_adjustment["bonus"]
                + criminal_adjustment["bonus"]
                + constitutional_adjustment["bonus"]
                + status_penalty
                - quality_penalty
                - domain_adjustment["penalty"]
                - criminal_adjustment["penalty"]
                - general_penalty
                - constitutional_adjustment["penalty"]
            )

            reason_tags = _build_reason_tags(
                matched_phrases=matched_phrases,
                general_penalty=general_penalty,
                domain_reasons=domain_adjustment["reasons"],
                criminal_reasons=criminal_adjustment["reasons"],
                constitutional_reasons=constitutional_adjustment["reasons"],
                analysis=analysis,
                overlap_title=overlap_title,
                overlap_summary=overlap_summary,
                support_count=len(candidate.get("supporting_chunks", [])),
            )
            candidate["rerank_score"] = round(rerank_score, 6)
            candidate["rank_explanation"] = reason_tags
            candidate["rerank_metadata"] = {
                "strategy": "feature_based",
                "query_intent": analysis.intent,
                "query_key_phrases": query_phrases,
                "matched_phrases": matched_phrases,
                "overlap_title": round(overlap_title, 6),
                "overlap_summary": round(overlap_summary, 6),
                "overlap_content": round(overlap_content, 6),
                "keyword_coverage": round(keyword_coverage, 6),
                "phrase_hits": {key: round(value, 6) for key, value in phrase_hits.items()},
                "general_article_penalty": round(general_penalty, 6),
                "domain_alignment_bonus": round(domain_adjustment["bonus"], 6),
                "domain_alignment_penalty": round(domain_adjustment["penalty"], 6),
                "domain_reason_tags": domain_adjustment["reasons"],
                "criminal_bonus": round(criminal_adjustment["bonus"], 6),
                "criminal_penalty": round(criminal_adjustment["penalty"], 6),
                "criminal_reason_tags": criminal_adjustment["reasons"],
                "criminal_domain_hint": criminal_adjustment["criminal_domain_hint"],
                "exact_offense_match": criminal_adjustment["exact_offense_match"],
                "multi_offense_query": criminal_adjustment["multi_offense_query"],
                "generic_criminal_article_penalty": criminal_adjustment["generic_criminal_article_penalty"],
                "constitutional_bonus": round(constitutional_adjustment["bonus"], 6),
                "constitutional_penalty": round(constitutional_adjustment["penalty"], 6),
                "constitutional_reason_tags": constitutional_adjustment["reasons"],
                "rights_section_match": constitutional_adjustment["rights_section_match"],
                "preamble_candidate": constitutional_adjustment["preamble_candidate"],
                "core_rights_hits": constitutional_adjustment["core_rights_hits"],
                "supporting_chunk_count": len(candidate.get("supporting_chunks", [])),
                "reason_tags": reason_tags,
            }

        return sorted(candidates, key=lambda item: item.get("rerank_score", item.get("score", 0.0)), reverse=True)


class CrossEncoderLegalReranker(BaseReranker):
    def __init__(self, config: Settings | None = None) -> None:
        self.settings = config or settings
        self._model = None

    def _ensure_model(self):
        if self._model is not None:
            return self._model

        from sentence_transformers import CrossEncoder

        model_kwargs: dict[str, Any] = {"device": self.settings.device_preference}
        if self.settings.retrieval_reranker_local_only:
            model_kwargs["local_files_only"] = True

        self._model = CrossEncoder(self.settings.retrieval_reranker_model_name, **model_kwargs)
        return self._model

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        *,
        query_analysis: QueryAnalysis | None = None,
    ) -> list[dict[str, Any]]:
        if not candidates:
            return candidates

        model = self._ensure_model()
        pairs = [(query, _candidate_text(candidate)) for candidate in candidates]
        scores = model.predict(pairs)

        for candidate, score in zip(candidates, scores):
            rerank_score = float(score)
            if candidate.get("is_repealed_candidate"):
                rerank_score -= 0.20
            rerank_score -= float(candidate.get("noise_score", 0.0)) * 0.12
            candidate["rerank_score"] = round(rerank_score, 6)
            candidate["rank_explanation"] = ["cross_encoder_score"]
            candidate["rerank_metadata"] = {
                "strategy": "cross_encoder",
                "model_name": self.settings.retrieval_reranker_model_name,
            }

        return sorted(candidates, key=lambda item: item.get("rerank_score", item.get("score", 0.0)), reverse=True)


class HeuristicLegalReranker(FeatureBasedLegalReranker):
    pass


@lru_cache(maxsize=8)
def build_reranker(name: str, config: Settings | None = None) -> BaseReranker:
    active = config or settings
    if name == "none":
        return NoOpReranker()
    if name == "cross_encoder":
        return CrossEncoderLegalReranker(active)
    if name == "heuristic":
        return HeuristicLegalReranker()
    return FeatureBasedLegalReranker()


def _token_overlap(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / max(1, len(left))


def _phrase_match_score(phrases: list[str], text: str) -> float:
    if not phrases or not text:
        return 0.0
    matched = sum(1 for phrase in phrases if phrase and phrase in text)
    return matched / max(1, len(phrases))


def _general_article_penalty(
    analysis: QueryAnalysis,
    candidate: dict[str, Any],
    title_norm: str,
    summary_norm: str,
    section_norm: str,
    document_norm: str,
    matched_phrases: list[str],
) -> float:
    if analysis.intent == "definition":
        return 0.0

    generic_text = " ".join(part for part in (title_norm, summary_norm, section_norm, document_norm) if part)
    has_general_cue = any(cue in generic_text for cue in GENERAL_ARTICLE_CUES)
    if not has_general_cue:
        return 0.0
    if matched_phrases and not analysis.prefer_specific_articles:
        return 0.0

    penalty = 0.10 if analysis.prefer_specific_articles else 0.05
    article_number = str(candidate.get("article_number") or "").strip()
    if article_number.isdigit() and int(article_number) <= 5:
        penalty += 0.05
    return penalty


def _supporting_chunk_text(chunks: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for chunk in chunks:
        title = chunk.get("title") or ""
        content = chunk.get("content") or ""
        if title:
            parts.append(title)
        if content:
            parts.append(content)
    return " ".join(parts)


def _constitutional_adjustment(
    *,
    analysis: QueryAnalysis,
    candidate: dict[str, Any],
    title_norm: str,
    summary_norm: str,
    content_norm: str,
    section_norm: str,
    document_norm: str,
) -> dict[str, Any]:
    if candidate.get("legal_domain") != "constitutional_law":
        return {
            "bonus": 0.0,
            "penalty": 0.0,
            "reasons": [],
            "rights_section_match": False,
            "preamble_candidate": False,
            "core_rights_hits": 0,
        }

    candidate_text = " ".join(
        part for part in (title_norm, summary_norm, content_norm, section_norm, document_norm) if part
    )
    article_number = normalize_legal_arabic(candidate.get("article_number") or "")
    preamble_candidate = _is_preamble_candidate(title_norm, section_norm, article_number)
    rights_section_match = _contains_any(section_norm, CONSTITUTIONAL_RIGHTS_SECTION_CUES) or _contains_any(
        document_norm,
        CONSTITUTIONAL_RIGHTS_SECTION_CUES,
    )
    core_rights_hits = _count_hits(candidate_text, CONSTITUTIONAL_CORE_RIGHTS_CUES)
    subtopic_hits = _count_hits(candidate_text, CONSTITUTIONAL_SUBTOPIC_CUES)

    bonus = 0.0
    penalty = 0.0
    reasons: list[str] = []

    if analysis.preamble_related:
        if preamble_candidate:
            bonus += 0.22
            reasons.append("preamble_boost")
        return {
            "bonus": bonus,
            "penalty": penalty,
            "reasons": reasons,
            "rights_section_match": rights_section_match,
            "preamble_candidate": preamble_candidate,
            "core_rights_hits": core_rights_hits,
        }

    if preamble_candidate and analysis.intent != "constitutional_preamble":
        penalty += 0.34 if analysis.constitutional_rights_query else 0.18
        reasons.append("preamble_penalty")

    if analysis.constitutional_rights_query:
        if rights_section_match:
            bonus += 0.26
            reasons.append("rights_chapter_boost")
        if core_rights_hits:
            bonus += min(0.18, 0.045 * core_rights_hits)
            reasons.append("core_rights_boost")
        if subtopic_hits and not rights_section_match and core_rights_hits == 0:
            penalty += min(0.05, 0.015 * subtopic_hits)
            reasons.append("subtopic_penalty")

    return {
        "bonus": bonus,
        "penalty": penalty,
        "reasons": reasons,
        "rights_section_match": rights_section_match,
        "preamble_candidate": preamble_candidate,
        "core_rights_hits": core_rights_hits,
    }


def _domain_alignment_adjustment(analysis: QueryAnalysis, candidate_domain: str | None) -> dict[str, Any]:
    suggested_domain = analysis.suggested_domain
    if not suggested_domain or not candidate_domain:
        return {"bonus": 0.0, "penalty": 0.0, "reasons": []}
    if candidate_domain == suggested_domain:
        return {"bonus": 0.12, "penalty": 0.0, "reasons": ["domain_alignment_boost"]}
    return {"bonus": 0.0, "penalty": 0.24, "reasons": ["domain_alignment_penalty"]}


def _criminal_adjustment(
    *,
    analysis: QueryAnalysis,
    candidate: dict[str, Any],
    title_norm: str,
    summary_norm: str,
    content_norm: str,
    section_norm: str,
    chunk_text_norm: str,
) -> dict[str, Any]:
    if not analysis.criminal_offense_query:
        return {
            "bonus": 0.0,
            "penalty": 0.0,
            "reasons": [],
            "criminal_domain_hint": False,
            "exact_offense_match": False,
            "multi_offense_query": analysis.multi_offense_query,
            "generic_criminal_article_penalty": 0.0,
        }

    candidate_domain = candidate.get("legal_domain")
    offense_terms = [normalize_legal_arabic(term) for term in analysis.criminal_offense_terms]
    offense_title_hits = _count_hits(title_norm, tuple(offense_terms))
    offense_summary_hits = _count_hits(summary_norm, tuple(offense_terms))
    offense_content_hits = _count_hits(content_norm, tuple(offense_terms))
    offense_section_hits = _count_hits(section_norm, tuple(offense_terms))
    offense_chunk_hits = _count_hits(chunk_text_norm, tuple(offense_terms))
    exact_offense_match = any(
        count > 0
        for count in (
            offense_title_hits,
            offense_summary_hits,
            offense_content_hits,
            offense_section_hits,
            offense_chunk_hits,
        )
    )
    criminal_domain_hint = candidate_domain == "criminal_law"
    generic_criminal_article_penalty = 0.0
    reasons: list[str] = []
    bonus = 0.0
    penalty = 0.0

    if criminal_domain_hint:
        bonus += 0.08
        reasons.append("criminal_domain_hint")

    if exact_offense_match:
        bonus += min(
            0.34,
            0.16 * min(1, offense_title_hits)
            + 0.10 * min(1, offense_summary_hits)
            + 0.08 * min(1, offense_content_hits)
            + 0.06 * min(1, offense_chunk_hits),
        )
        reasons.append("exact_offense_match")
        if analysis.multi_offense_query and sum(
            1 for term in offense_terms if term and any(term in text for text in (title_norm, summary_norm, content_norm, chunk_text_norm))
        ) >= 2:
            bonus += 0.08
            reasons.append("multi_offense_query")
    else:
        penalty += 0.24

    combined = " ".join(part for part in (title_norm, summary_norm, section_norm) if part)
    article_number = str(candidate.get("article_number") or "").strip()
    if candidate_domain == "criminal_law" and (
        _contains_any(combined, CRIMINAL_GENERIC_CUES)
        or (article_number.isdigit() and int(article_number) <= 20 and not exact_offense_match)
    ):
        generic_criminal_article_penalty = 0.14
        penalty += generic_criminal_article_penalty
        reasons.append("generic_criminal_article_penalty")

    return {
        "bonus": bonus,
        "penalty": penalty,
        "reasons": reasons,
        "criminal_domain_hint": criminal_domain_hint,
        "exact_offense_match": exact_offense_match,
        "multi_offense_query": analysis.multi_offense_query,
        "generic_criminal_article_penalty": generic_criminal_article_penalty,
    }


def _is_preamble_candidate(title_norm: str, section_norm: str, article_number: str) -> bool:
    if article_number in PREAMBLE_CUES:
        return True
    return _contains_any(title_norm, PREAMBLE_CUES) or _contains_any(section_norm, PREAMBLE_CUES)


def _contains_any(text: str, cues: tuple[str, ...]) -> bool:
    if not text:
        return False
    return any(cue in text for cue in cues)


def _count_hits(text: str, cues: tuple[str, ...]) -> int:
    if not text:
        return 0
    return sum(1 for cue in cues if cue in text)


def _candidate_text(candidate: dict[str, Any]) -> str:
    parts = [
        candidate.get("law_name") or "",
        candidate.get("title") or "",
        candidate.get("summary") or "",
        candidate.get("content") or "",
    ]
    return "\n".join(part for part in parts if part).strip()


def _build_reason_tags(
    *,
    matched_phrases: list[str],
    general_penalty: float,
    domain_reasons: list[str],
    criminal_reasons: list[str],
    constitutional_reasons: list[str],
    analysis: QueryAnalysis,
    overlap_title: float,
    overlap_summary: float,
    support_count: int,
) -> list[str]:
    reasons: list[str] = []
    if matched_phrases:
        reasons.append(f"phrase_match:{', '.join(matched_phrases[:2])}")
    if analysis.prefer_specific_articles:
        reasons.append("specific_query_bias")
    if overlap_title >= 0.5:
        reasons.append("strong_title_overlap")
    if overlap_summary >= 0.5:
        reasons.append("strong_summary_overlap")
    if support_count:
        reasons.append(f"supporting_chunks:{support_count}")
    if general_penalty > 0:
        reasons.append("general_article_penalty")
    reasons.extend(reason for reason in domain_reasons if reason not in reasons)
    reasons.extend(reason for reason in criminal_reasons if reason not in reasons)
    reasons.extend(reason for reason in constitutional_reasons if reason not in reasons)
    return reasons
