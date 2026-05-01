"""Professional intent router for pre-retrieval query classification.

Runs *before* any access to LegalRetriever / Qdrant / embeddings.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from app.preprocessing import normalize_legal_arabic


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def route_intent(query: str, explicit_domain: str | None = None) -> IntentDecision:
    """Classify *query* into an intent before any retrieval."""
    norm = normalize_legal_arabic(query)
    raw = query.strip()

    # 1. Identity
    if _is_identity(norm, raw):
        return IntentDecision(
            intent=IntentType.IDENTITY,
            confidence=0.98,
            normalized_query=norm,
            is_legal_question=False,
            is_out_of_internal_corpus=False,
            suggested_domain=None,
            reasons=["identity_cue_matched"],
        )

    # 2. Conversation / small-talk  (must run before legal scoring)
    conv = _conversation_score(norm, raw)
    if conv >= 0.9:
        return IntentDecision(
            intent=IntentType.CONVERSATION,
            confidence=conv,
            normalized_query=norm,
            is_legal_question=False,
            is_out_of_internal_corpus=False,
            suggested_domain=None,
            reasons=["conversation_cue_matched"],
        )

    # 3. External-assisted (personal-status / family-law outside internal corpus)
    if _is_external_assisted(norm):
        return IntentDecision(
            intent=IntentType.EXTERNAL_ASSISTED,
            confidence=0.90,
            normalized_query=norm,
            is_legal_question=True,
            is_out_of_internal_corpus=True,
            suggested_domain=None,
            reasons=["personal_status_family_law_detected"],
        )

    # 4. Legal retrieval scoring
    legal_score, domain = _legal_score(norm, explicit_domain)
    if legal_score >= 0.35 or explicit_domain:
        return IntentDecision(
            intent=IntentType.LEGAL_RETRIEVAL,
            confidence=min(0.6 + legal_score * 0.4, 0.99),
            normalized_query=norm,
            is_legal_question=True,
            is_out_of_internal_corpus=False,
            suggested_domain=domain or explicit_domain,
            reasons=["legal_intent_score_high"],
        )

    # 5. Non-legal
    if _is_non_legal(norm):
        return IntentDecision(
            intent=IntentType.NON_LEGAL,
            confidence=0.85,
            normalized_query=norm,
            is_legal_question=False,
            is_out_of_internal_corpus=False,
            suggested_domain=None,
            reasons=["non_legal_topic_detected"],
        )

    # 6. Ambiguous — low confidence, let retrieval + sufficiency decide.
    return IntentDecision(
        intent=IntentType.AMBIGUOUS,
        confidence=0.4,
        normalized_query=norm,
        is_legal_question=False,
        is_out_of_internal_corpus=False,
        suggested_domain=None,
        reasons=["no_strong_signal"],
    )


# ---------------------------------------------------------------------------
# Layer helpers
# ---------------------------------------------------------------------------

def _is_identity(norm: str, raw: str) -> bool:
    lower = raw.lower()
    cues_norm = _IDENTITY_CUES_NORM
    if any(c in norm for c in cues_norm):
        return True
    english = ("who are you", "what is your name", "your name", "who developed you", "who built you")
    return any(c in lower for c in english)


def _conversation_score(norm: str, raw: str) -> float:
    """Return 0.0–1.0 conversation confidence."""
    lower = raw.strip().lower()
    # Exact or near-exact greeting match
    for cue in _GREETING_EXACT_NORM:
        if norm == cue or norm.startswith(cue + " ") or cue.startswith(norm):
            return 0.98
    # Thanks
    for cue in _THANKS_NORM:
        if cue in norm:
            return 0.95
    # Capability prompts
    for cue in _CAPABILITY_NORM:
        if cue in norm:
            return 0.92
    # Short message with greeting substring but NOT legal content
    if len(norm.split()) <= 4:
        for cue in _GREETING_SUBSTR_NORM:
            if cue in norm:
                return 0.93
    # English greetings
    for cue in ("hello", "hi", "hey", "thanks", "thank you", "good morning"):
        if cue in lower:
            return 0.92
    return 0.0


def _is_external_assisted(norm: str) -> bool:
    return any(cue in norm for cue in _PERSONAL_STATUS_NORM)


def _legal_score(norm: str, explicit_domain: str | None) -> tuple[float, str | None]:
    """Return (score 0–1, detected domain or None)."""
    score = 0.0
    # Legal question phrases
    for phrase in _LEGAL_PHRASES_NORM:
        if phrase in norm:
            score += 0.35
            break
    # Domain terms
    best_domain: str | None = None
    best_domain_hits = 0
    for domain, terms in _DOMAIN_TERMS.items():
        hits = sum(1 for t in terms if t in norm)
        if hits > best_domain_hits:
            best_domain_hits = hits
            best_domain = domain
    if best_domain_hits >= 2:
        score += 0.4
    elif best_domain_hits == 1:
        score += 0.25
    # Generic legal cues
    generic_hits = sum(1 for c in _GENERIC_LEGAL_NORM if c in norm)
    if generic_hits >= 2:
        score += 0.3
    elif generic_hits == 1:
        score += 0.15
    return min(score, 1.0), best_domain


def _is_non_legal(norm: str) -> bool:
    return any(cue in norm for cue in _NON_LEGAL_NORM)


# ---------------------------------------------------------------------------
# Cue lists (normalised once at import time)
# ---------------------------------------------------------------------------
_n = normalize_legal_arabic

_IDENTITY_CUES_NORM = tuple(_n(c) for c in (
    "اسمك", "اسمك ايه", "اسمك إيه", "انت مين", "أنت مين",
    "مين انت", "مين أنت", "من انت", "من أنت",
    "مين طورك", "من طورك", "مين صممك", "من صممك",
    "مين عملك", "من عملك",
    "انت تابع لتطبيق", "أنت تابع لتطبيق",
))

_GREETING_EXACT_NORM = tuple(_n(c) for c in (
    "السلام عليكم", "وعليكم السلام", "سلام عليكم",
    "اهلا", "أهلا", "اهلا بك", "أهلا بك",
    "مرحبا", "مرحبًا",
    "صباح الخير", "مساء الخير", "مساء النور",
    "هاي", "هالو",
    "ازيك", "إزيك", "ازاي", "ازايك", "إزايك",
    "عامل ايه", "عامل إيه",
))

_GREETING_SUBSTR_NORM = tuple(_n(c) for c in (
    "سلام", "اهلا", "أهلا", "مرحبا", "صباح", "مساء",
    "ازيك", "إزيك", "هاي",
))

_THANKS_NORM = tuple(_n(c) for c in (
    "شكرا", "شكراً", "متشكر", "تسلم", "تمام", "جزاك الله",
))

_CAPABILITY_NORM = tuple(_n(c) for c in (
    "تقدر تساعدني", "ممكن تساعدني", "ماذا يمكنك",
    "ايه اللي تقدر تعمله", "تعرف تعمل ايه",
))

_PERSONAL_STATUS_NORM = tuple(_n(c) for c in (
    "حضانة", "الحضانة", "نفقة", "النفقة",
    "طلاق", "الطلاق", "خلع", "الخلع",
    "ميراث", "الميراث", "مؤخر الصداق",
    "الرؤية", "رؤية الصغير", "الولاية التعليمية",
    "قائمة المنقولات", "تمكين الزوجة", "مسكن الزوجية",
    "الأحوال الشخصية", "احوال شخصية",
))

_LEGAL_PHRASES_NORM = tuple(_n(c) for c in (
    "ما حكم", "ما هي احكام", "ما هي أحكام", "هل يجوز",
    "ما العقوبة", "ما عقوبة", "ما حقوق", "ما التزامات",
    "متى يسقط", "شروط", "اجراءات", "إجراءات",
))

_GENERIC_LEGAL_NORM = tuple(_n(c) for c in (
    "دعوى", "عقد", "مادة", "المادة", "قانون", "القانون",
    "محكمة", "حكم", "حقوق", "التزامات",
))

_DOMAIN_TERMS: dict[str, tuple[str, ...]] = {
    "labor_law": tuple(_n(c) for c in (
        "عقد العمل", "العامل", "صاحب العمل", "الاجر", "الأجر",
        "الفصل", "اجازة", "إجازة", "تأمين بيئة العمل",
        "العمل الفردي", "العمل الجماعي", "قانون العمل",
    )),
    "civil_law": tuple(_n(c) for c in (
        "الالتزام", "البيع", "الايجار", "الإيجار",
        "التعويض", "المسؤولية", "الملكية", "الحيازة",
    )),
    "criminal_law": tuple(_n(c) for c in (
        "جريمة", "عقوبة", "سرقة", "نصب", "خيانة امانة", "خيانة أمانة",
        "رشوة", "تزوير", "ضرب", "قتل",
    )),
    "constitutional_law": tuple(_n(c) for c in (
        "الدستور", "الحريات", "المساواة",
        "حرية الاعتقاد", "المواطنة",
    )),
}

_NON_LEGAL_NORM = tuple(_n(c) for c in (
    "مطعم", "طبخ", "وصفة", "الطقس", "نكتة", "برمجة",
    "ترجمة", "اخبار رياضة", "أخبار رياضة", "علاج طبي",
    "كورة", "ماتش", "فيلم", "اغنية", "أغنية",
))
