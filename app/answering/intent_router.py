"""Hybrid intent router for pre-retrieval query classification.

This module must stay light: it runs before Qdrant, embeddings, and the LLM.
It uses deterministic Arabic normalization, local high-confidence intents, and
lexical domain profiles for the current internal corpus.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

from app.preprocessing import normalize_legal_arabic


class IntentType(str, Enum):
    IDENTITY = "identity"
    CONVERSATION = "conversation"
    LEGAL_RETRIEVAL = "legal_retrieval"
    EXTERNAL_ASSISTED = "external_assisted"
    NON_LEGAL = "non_legal"
    AMBIGUOUS = "ambiguous"


@dataclass(slots=True)
class IntentDecision:
    intent: IntentType
    confidence: float
    normalized_query: str
    is_legal_question: bool
    is_out_of_internal_corpus: bool
    suggested_domain: str | None
    reasons: list[str] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)


PLACEHOLDER_VALUES = {"", "string", "null", "none", "undefined"}
VALID_INTERNAL_DOMAINS = {"labor_law", "civil_law", "criminal_law", "constitutional_law"}


def sanitize_optional_value(value: object) -> str | None:
    """Normalize API placeholder strings to None."""
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in PLACEHOLDER_VALUES:
        return None
    return text or None


def route_intent(query: str, explicit_domain: str | None = None) -> IntentDecision:
    """Classify *query* before any retrieval is attempted.

    ``explicit_domain`` is only a hint. It can break close domain ties, but it
    never forces retrieval by itself.
    """
    norm = _normalize_for_routing(query)
    explicit_domain = sanitize_optional_value(explicit_domain)
    if explicit_domain == "all":
        explicit_domain = None
    if explicit_domain not in VALID_INTERNAL_DOMAINS:
        explicit_domain = None

    if _is_identity(norm, query):
        return IntentDecision(
            intent=IntentType.IDENTITY,
            confidence=0.98,
            normalized_query=norm,
            is_legal_question=False,
            is_out_of_internal_corpus=False,
            suggested_domain=None,
            reasons=["identity_cue_matched"],
            scores={"identity": 0.98},
        )

    conversation_score = _conversation_score(norm, query)
    if conversation_score >= 0.90:
        return IntentDecision(
            intent=IntentType.CONVERSATION,
            confidence=conversation_score,
            normalized_query=norm,
            is_legal_question=False,
            is_out_of_internal_corpus=False,
            suggested_domain=None,
            reasons=["conversation_cue_matched"],
            scores={"conversation": conversation_score},
        )

    if _is_obvious_non_legal(norm):
        return IntentDecision(
            intent=IntentType.NON_LEGAL,
            confidence=0.90,
            normalized_query=norm,
            is_legal_question=False,
            is_out_of_internal_corpus=False,
            suggested_domain=None,
            reasons=["obvious_non_legal_topic_detected"],
            scores={"non_legal": 0.90},
        )

    if _is_external_assisted(norm):
        return IntentDecision(
            intent=IntentType.EXTERNAL_ASSISTED,
            confidence=0.92,
            normalized_query=norm,
            is_legal_question=True,
            is_out_of_internal_corpus=True,
            suggested_domain=None,
            reasons=["personal_status_family_law_detected"],
            scores={"external_assisted": 0.92},
        )

    domain_scores = _score_domains(norm)
    suggested_domain, best_domain_score = _pick_domain(domain_scores, explicit_domain)
    legal_score, legal_reasons = _legal_likelihood(norm, best_domain_score)

    if legal_score >= 0.45:
        return IntentDecision(
            intent=IntentType.LEGAL_RETRIEVAL,
            confidence=min(0.99, max(0.55, legal_score)),
            normalized_query=norm,
            is_legal_question=True,
            is_out_of_internal_corpus=False,
            suggested_domain=suggested_domain,
            reasons=legal_reasons or ["legal_intent_score_medium_or_high"],
            scores={"legal": round(legal_score, 4), **domain_scores},
        )

    if legal_score >= 0.30 and suggested_domain:
        return IntentDecision(
            intent=IntentType.LEGAL_RETRIEVAL,
            confidence=min(0.85, legal_score + 0.20),
            normalized_query=norm,
            is_legal_question=True,
            is_out_of_internal_corpus=False,
            suggested_domain=suggested_domain,
            reasons=[*legal_reasons, "cautious_medium_legal_score_with_domain_profile"],
            scores={"legal": round(legal_score, 4), **domain_scores},
        )

    return IntentDecision(
        intent=IntentType.AMBIGUOUS,
        confidence=max(0.20, min(0.55, legal_score)),
        normalized_query=norm,
        is_legal_question=False,
        is_out_of_internal_corpus=False,
        suggested_domain=suggested_domain if best_domain_score >= 2.5 else None,
        reasons=["no_clear_legal_or_non_legal_signal"],
        scores={"legal": round(legal_score, 4), **domain_scores},
    )


def _normalize_for_routing(query: str) -> str:
    text = normalize_legal_arabic(query or "").lower()
    text = re.sub(r"[؟?،؛:,.!()\[\]{}<>\"'`~@#$%^&*_+=|\\/]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _tokens(text: str) -> list[str]:
    return re.findall(r"[\u0621-\u064A0-9]+", text)


def _token_key(token: str) -> str:
    if token.startswith("ال") and len(token) > 4:
        return token[2:]
    return token


def _contains_phrase(text: str, phrase: str) -> bool:
    return _normalize_for_routing(phrase) in text


def _has_any_phrase(text: str, phrases: tuple[str, ...]) -> bool:
    return any(_contains_phrase(text, phrase) for phrase in phrases)


def _is_identity(norm: str, raw: str) -> bool:
    lower = raw.strip().lower()
    if any(phrase in norm for phrase in _IDENTITY_DIRECT):
        return True
    has_identity_topic = any(term in norm for term in _IDENTITY_TOPICS)
    has_identity_question = any(term in norm for term in _IDENTITY_QUESTION_TERMS)
    if has_identity_topic and has_identity_question:
        return True
    english = ("who are you", "what is your name", "your name", "who developed you", "who built you")
    return any(cue in lower for cue in english)


def _conversation_score(norm: str, raw: str) -> float:
    lower = raw.strip().lower()
    tokens = norm.split()
    if not tokens:
        return 0.0
    if len(tokens) <= 5 and any(phrase == norm or norm.startswith(phrase + " ") for phrase in _CONVERSATION_EXACT):
        return 0.98
    if len(tokens) <= 5 and any(phrase in norm for phrase in _CONVERSATION_SHORT):
        return 0.94
    if len(tokens) <= 6 and any(phrase in norm for phrase in _CAPABILITY_SHORT):
        return 0.92
    if len(tokens) <= 4 and any(cue in lower for cue in ("hello", "hi", "hey", "thanks", "thank you")):
        return 0.93
    return 0.0


def _is_obvious_non_legal(norm: str) -> bool:
    if not _has_any_phrase(norm, _NON_LEGAL_TERMS):
        return False
    # If the user clearly asks about a legal violation involving a non-legal
    # noun, let legal scoring handle it.
    legal_hits = _legal_term_hits(norm)
    return legal_hits == 0 and not _has_any_phrase(norm, _STRONG_SOURCE_PHRASES)


def _is_external_assisted(norm: str) -> bool:
    return _has_any_phrase(norm, _PERSONAL_STATUS_TERMS)


def _legal_likelihood(norm: str, best_domain_score: float) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []

    if _has_any_phrase(norm, _STRONG_SOURCE_PHRASES):
        score += 0.58
        reasons.append("strong_legal_source_phrase")

    term_hits = _legal_term_hits(norm)
    if term_hits:
        score += min(0.42, term_hits * 0.08)
        reasons.append("legal_concept_terms")

    question_score = _question_pattern_score(norm)
    if question_score:
        score += question_score
        reasons.append("legal_question_pattern")

    if best_domain_score >= 4:
        score += 0.28
        reasons.append("strong_domain_profile_overlap")
    elif best_domain_score >= 2:
        score += 0.18
        reasons.append("domain_profile_overlap")
    elif best_domain_score >= 1:
        score += 0.08

    return min(score, 1.0), reasons


def _legal_term_hits(norm: str) -> int:
    token_keys = {_token_key(token) for token in _tokens(norm)}
    hits = 0
    for term in _LEGAL_CONCEPT_TERMS:
        term_norm = _normalize_for_routing(term)
        if " " in term_norm:
            hits += int(term_norm in norm)
        else:
            hits += int(_token_key(term_norm) in token_keys)
    return hits


def _question_pattern_score(norm: str) -> float:
    if _has_any_phrase(norm, _LEGAL_QUESTION_PATTERNS):
        return 0.24
    first = norm.split()[0] if norm.split() else ""
    if first in {"ما", "ماذا", "متى", "كيف", "هل"}:
        return 0.12
    return 0.0


def _score_domains(norm: str) -> dict[str, float]:
    query_tokens = {_token_key(token) for token in _tokens(norm)}
    scores: dict[str, float] = {}
    for domain, profile in _DOMAIN_PROFILES.items():
        profile_tokens = {_token_key(token) for token in _tokens(_normalize_for_routing(profile))}
        score = float(len(query_tokens & profile_tokens))
        for phrase in _DOMAIN_PHRASE_BOOSTS.get(domain, ()):
            if _contains_phrase(norm, phrase):
                score += 2.5
        scores[domain] = round(score, 4)
    return scores


def _pick_domain(domain_scores: dict[str, float], explicit_domain: str | None) -> tuple[str | None, float]:
    ordered = sorted(domain_scores.items(), key=lambda item: item[1], reverse=True)
    if not ordered:
        return explicit_domain, 0.0
    best_domain, best_score = ordered[0]
    second_score = ordered[1][1] if len(ordered) > 1 else 0.0
    if explicit_domain and domain_scores.get(explicit_domain, 0.0) >= max(1.0, best_score - 1.0):
        return explicit_domain, domain_scores[explicit_domain]
    if best_score >= 1.5 and best_score > second_score:
        return best_domain, best_score
    if best_score >= 4.0:
        return best_domain, best_score
    return None, best_score


def _n(value: str) -> str:
    return _normalize_for_routing(value)


_IDENTITY_DIRECT = tuple(_n(value) for value in (
    "اسمك ايه",
    "اسمك إيه",
    "ما اسمك",
    "من أنت",
    "من انت",
    "انت مين",
    "أنت مين",
    "مين انت",
    "مين أنت",
    "من المستشار",
    "ما هو المستشار",
    "مين طورك",
    "من طورك",
    "مين صممك",
    "من صممك",
    "مين عملك",
    "من عملك",
    "مين صمم التطبيق",
    "مين طور التطبيق",
    "انت تابع لتطبيق ايه",
    "أنت تابع لتطبيق إيه",
))

_IDENTITY_TOPICS = tuple(_n(value) for value in (
    "اسمك",
    "هويتك",
    "المستشار",
    "التطبيق",
    "طورك",
    "صممك",
    "عملك",
    "طور",
    "صمم",
))

_IDENTITY_QUESTION_TERMS = tuple(_n(value) for value in (
    "مين",
    "من",
    "ما",
    "ايه",
    "إيه",
    "انت",
    "أنت",
    "هو",
))

_CONVERSATION_EXACT = tuple(_n(value) for value in (
    "السلام عليكم",
    "وعليكم السلام",
    "سلام عليكم",
    "اهلا",
    "أهلا",
    "مرحبا",
    "صباح الخير",
    "مساء الخير",
    "مساء النور",
    "شكرا",
    "متشكر",
    "تسلم",
    "تمام",
    "ازيك",
    "إزيك",
    "عامل ايه",
    "عامل إيه",
))

_CONVERSATION_SHORT = tuple(_n(value) for value in (
    "سلام",
    "اهلا",
    "أهلا",
    "مرحبا",
    "صباح",
    "مساء",
    "شكرا",
    "متشكر",
    "تسلم",
    "تمام",
    "ازيك",
    "إزيك",
    "عامل ايه",
))

_CAPABILITY_SHORT = tuple(_n(value) for value in (
    "ممكن تساعدني",
    "تقدر تساعدني",
    "ماذا يمكنك",
    "تعرف تعمل ايه",
    "ايه اللي تقدر تعمله",
))

_NON_LEGAL_TERMS = tuple(_n(value) for value in (
    "مطعم",
    "أكل",
    "اكل",
    "الطقس",
    "رياضة",
    "نكتة",
    "برمجة",
    "ترجمة",
    "علاج طبي",
    "دواء",
    "سفر",
    "فيلم",
    "أغنية",
    "اغنية",
    "كود بايثون",
    "اكتب كود",
))

_PERSONAL_STATUS_TERMS = tuple(_n(value) for value in (
    "حضانة",
    "الحضانة",
    "نفقة",
    "النفقة",
    "طلاق",
    "الطلاق",
    "خلع",
    "الخلع",
    "الرؤية",
    "رؤية الصغير",
    "ميراث",
    "الميراث",
    "مؤخر الصداق",
    "قائمة المنقولات",
    "مسكن الزوجية",
    "تمكين الزوجة",
    "الأحوال الشخصية",
    "احوال شخصية",
))

_LEGAL_CONCEPT_TERMS = tuple(_n(value) for value in (
    "قانون",
    "الدستور",
    "دستوري",
    "دستورية",
    "لائحة",
    "مادة",
    "مواد",
    "محكمة",
    "دعوى",
    "حكم",
    "أحكام",
    "احكام",
    "عقوبة",
    "جريمة",
    "عقد",
    "التزام",
    "حق",
    "حقوق",
    "حرية",
    "حريات",
    "ضمانات",
    "تعويض",
    "مسؤولية",
    "ملكية",
    "حيازة",
    "عامل",
    "صاحب العمل",
    "أجر",
    "اجر",
    "فصل",
    "متهم",
    "مجني عليه",
    "جناية",
    "جنحة",
    "مخالفة",
))

_LEGAL_QUESTION_PATTERNS = tuple(_n(value) for value in (
    "ما حكم",
    "ما هي أحكام",
    "ما أحكام",
    "ما ضمانات",
    "ما حقوق",
    "ما واجبات",
    "ما شروط",
    "ما آثار",
    "ما عقوبة",
    "متى يجوز",
    "هل يجوز",
    "ماذا ينص",
    "ما المقصود",
    "ما الفرق بين",
))

_STRONG_SOURCE_PHRASES = tuple(_n(value) for value in (
    "الدستور المصري",
    "دستور جمهورية مصر العربية",
    "قانون العمل المصري",
    "القانون المدني المصري",
    "القانون المدني",
    "قانون العقوبات المصري",
    "قانون العقوبات",
))

_DOMAIN_PROFILES = {
    "constitutional_law": (
        "الدستور المصري الدستور الحقوق الحريات الحرية الشخصية حرية الاعتقاد حرية الرأي "
        "حرية التعبير المساواة عدم التمييز تكافؤ الفرص كرامة الإنسان المواطنة الحق في "
        "التعليم الحق في الصحة الحق في العمل الحق في التقاضي المحاكمة العادلة العدالة "
        "الاجتماعية واجبات الدولة"
    ),
    "labor_law": (
        "قانون العمل العمل العامل صاحب العمل عقد العمل عقد العمل الفردي الأجر الإجازة "
        "الإجازات الفصل إنهاء العقد علاقة العمل السلامة والصحة المهنية بيئة العمل "
        "إصابة العمل تشغيل العامل حقوق العامل واجبات العامل"
    ),
    "civil_law": (
        "القانون المدني العقد الالتزام الالتزامات البيع الإيجار التعويض المسؤولية "
        "التقصيرية المسؤولية المدنية الملكية الحيازة الفسخ البطلان الشرط الجزائي "
        "الدائن المدين الضرر الإخلال بالعقد"
    ),
    "criminal_law": (
        "قانون العقوبات الجريمة العقوبة العقوبات السرقة النصب الرشوة التزوير القتل "
        "الضرب خيانة الأمانة الشروع العود الجناية الجنحة المخالفة المتهم المجني عليه "
        "السجن الحبس الغرامة"
    ),
}

_DOMAIN_PHRASE_BOOSTS = {
    "constitutional_law": tuple(_n(value) for value in (
        "الدستور المصري",
        "الحقوق والحريات",
        "عدم التمييز",
        "حرية الاعتقاد",
        "حرية الرأي",
        "الحرية الشخصية",
        "المحاكمة العادلة",
        "الحق في التقاضي",
    )),
    "labor_law": tuple(_n(value) for value in (
        "عقد العمل",
        "صاحب العمل",
        "قانون العمل",
        "السلامة والصحة المهنية",
    )),
    "civil_law": tuple(_n(value) for value in (
        "القانون المدني",
        "المسؤولية التقصيرية",
        "الشرط الجزائي",
        "عقد البيع",
    )),
    "criminal_law": tuple(_n(value) for value in (
        "قانون العقوبات",
        "خيانة الأمانة",
        "الجناية والجنحة والمخالفة",
    )),
}
