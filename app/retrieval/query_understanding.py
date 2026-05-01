from __future__ import annotations

from dataclasses import dataclass, field

from app.preprocessing import extract_search_terms, normalize_legal_arabic


QUESTION_STOPWORDS = {
    "ما",
    "ماذا",
    "متي",
    "متى",
    "كيف",
    "هل",
    "هي",
    "هو",
    "عن",
    "في",
    "من",
    "على",
    "الي",
    "إلى",
    "الى",
    "او",
    "أو",
    "ثم",
    "مع",
    "بشأن",
    "بشان",
    "ماهي",
    "هي",
    "هو",
    "التي",
    "الذي",
    "ذلك",
    "هذه",
    "هذا",
}

DEFINITION_CUES = (
    "ما المقصود",
    "ما معني",
    "ما معنى",
    "عرف",
    "تعريف",
    "يقصد ب",
    "المقصود ب",
)

SPECIFIC_INTENT_TERMS = {
    "احكام",
    "شروط",
    "عقد",
    "بيع",
    "رشوة",
    "مسؤوليه",
    "مسئوليه",
    "التزامات",
    "اجور",
    "اجر",
    "انهاء",
    "انتهاء",
    "فصل",
    "جزاءات",
    "جزاء",
    "عقوبه",
    "عقوبات",
    "خدمه",
    "خدمة",
    "حقوق",
    "حريات",
    "دستور",
    "دستوري",
}

PREAMBLE_QUERY_CUES = (
    "الديباجة",
    "الديباجه",
    "مبادئ الدستور",
    "مبادىء الدستور",
    "فلسفة الدستور",
    "فلسفه الدستور",
)

CONSTITUTIONAL_RIGHTS_CUES = (
    "الحقوق والحريات",
    "حقوق وحريات",
    "الحريات العامة",
    "الحريات العامه",
    "الحقوق الاساسية",
    "الحقوق الأساسية",
    "حقوق الانسان",
    "حقوق الإنسان",
)

DOMAIN_TERM_HINTS: dict[str, tuple[str, ...]] = {
    "labor_law": (
        "عامل",
        "العامل",
        "العمال",
        "عمل",
        "العمل",
        "اجر",
        "الأجر",
        "اجر",
        "إجازة",
        "اجازة",
        "إجازات",
        "اجازات",
        "فصل",
        "خدمة",
        "إنهاء",
        "انهاء",
        "ساعات العمل",
    ),
    "civil_law": (
        "عقد",
        "العقد",
        "التزام",
        "الالتزام",
        "الالتزامات",
        "بيع",
        "البيع",
        "فسخ",
        "بطلان",
        "تعويض",
        "ملكية",
        "إيجاب",
        "قبول",
        "مسؤولية",
        "مسئولية",
    ),
    "criminal_law": (
        "عقوبة",
        "العقوبة",
        "عقوبات",
        "جريمة",
        "الجريمة",
        "الرشوة",
        "سرقة",
        "السرقة",
        "تزوير",
        "اختلاس",
        "دفاع شرعي",
        "خيانة الأمانة",
        "خيانة الامانة",
        "نصب",
        "القذف",
        "السب",
    ),
    "constitutional_law": (
        "دستور",
        "الدستور",
        "حرية",
        "الحرية",
        "مساواة",
        "المساواة",
        "تعليم",
        "التعليم",
        "نقابات",
        "النقابات",
        "جمعيات",
        "الجمعيات",
        "معلومات",
        "المعلومات",
        "كرامة",
        "الكرامة",
        "حقوق",
        "الحريات",
        "الديباجة",
    ),
}

OUT_OF_DOMAIN_FAMILY_CUES = (
    "زوجتي",
    "زوجة",
    "زوج",
    "الزواج",
    "زواج",
    "الطلاق",
    "طلاق",
    "الحضانة",
    "حضانة",
    "النفقة",
    "نفقة",
    "الميراث",
    "ميراث",
    "الخلع",
    "خلع",
    "مؤخر الصداق",
    "رؤية الصغير",
    "نسب",
)

CRIMINAL_OFFENSE_TERMS = (
    "الرشوة",
    "سرقة",
    "السرقة",
    "تزوير",
    "اختلاس",
    "خيانة الأمانة",
    "خيانة الامانة",
    "النصب",
    "نصب",
    "القذف",
    "السب",
    "القتل",
    "الضرب",
)

LEGAL_PHRASE_LIBRARY: dict[str, dict[str, object]] = {
    "عقد العمل الفردي": {
        "expansions": ["عقد العمل الفردي", "عقد العمل"],
        "domains": ["labor_law"],
    },
    "عقد العمل": {
        "expansions": ["عقد العمل"],
        "domains": ["labor_law"],
    },
    "انهاء الخدمة": {
        "expansions": ["انهاء الخدمة", "انتهاء الخدمة", "انتهاء عقد العمل"],
        "domains": ["labor_law"],
    },
    "انتهاء الخدمة": {
        "expansions": ["انتهاء الخدمة", "انهاء الخدمة", "انتهاء عقد العمل"],
        "domains": ["labor_law"],
    },
    "الفصل التعسفي": {
        "expansions": ["الفصل التعسفي", "فصل العامل"],
        "domains": ["labor_law"],
    },
    "الجزاءات": {
        "expansions": ["الجزاءات", "الجزاءات التاديبيه", "العقوبات التاديبيه"],
        "domains": ["labor_law"],
    },
    "الاجور": {
        "expansions": ["الاجور", "الاجر", "استحقاق الاجر"],
        "domains": ["labor_law"],
    },
    "الاجر": {
        "expansions": ["الاجر", "الاجور"],
        "domains": ["labor_law"],
    },
    "عقد البيع": {
        "expansions": ["عقد البيع", "البيع"],
        "domains": ["civil_law"],
    },
    "المسؤولية المدنية": {
        "expansions": ["المسؤولية المدنية", "التعويض", "الخطا"],
        "domains": ["civil_law"],
    },
    "المسئولية المدنية": {
        "expansions": ["المسؤولية المدنية", "المسئولية المدنية", "التعويض", "الخطا"],
        "domains": ["civil_law"],
    },
    "الالتزامات": {
        "expansions": ["الالتزامات", "الالتزام"],
        "domains": ["civil_law"],
    },
    "الرشوة": {
        "expansions": ["الرشوة", "جريمة الرشوة", "عقوبة الرشوة"],
        "domains": ["criminal_law"],
    },
    "العقوبات": {
        "expansions": ["العقوبات", "العقوبه"],
        "domains": ["criminal_law"],
    },
    "الحقوق والحريات": {
        "expansions": [
            "الحقوق والحريات",
            "باب الحقوق والحريات والواجبات العامة",
            "الحقوق والحريات والواجبات العامة",
            "الحقوق الأساسية",
        ],
        "domains": ["constitutional_law"],
    },
    "حقوق وحريات": {
        "expansions": [
            "الحقوق والحريات",
            "باب الحقوق والحريات والواجبات العامة",
            "الحقوق والحريات والواجبات العامة",
            "الحقوق الأساسية",
        ],
        "domains": ["constitutional_law"],
    },
    "الديباجة": {
        "expansions": ["الديباجة", "مبادئ الدستور", "فلسفة الدستور"],
        "domains": ["constitutional_law"],
    },
    "الديباجه": {
        "expansions": ["الديباجة", "مبادئ الدستور", "فلسفة الدستور"],
        "domains": ["constitutional_law"],
    },
}


@dataclass(slots=True)
class QueryAnalysis:
    original_query: str
    normalized_query: str
    rewritten_query: str
    tokens: list[str] = field(default_factory=list)
    legal_keywords: list[str] = field(default_factory=list)
    key_phrases: list[str] = field(default_factory=list)
    expansion_terms: list[str] = field(default_factory=list)
    domain_hints: list[str] = field(default_factory=list)
    intent: str = "substantive"
    prefer_specific_articles: bool = False
    constitutional_rights_query: bool = False
    preamble_related: bool = False
    domain_scores: dict[str, int] = field(default_factory=dict)
    suggested_domain: str | None = None
    out_of_domain: bool = False
    out_of_domain_reason: str | None = None
    criminal_offense_query: bool = False
    criminal_offense_terms: list[str] = field(default_factory=list)
    multi_offense_query: bool = False


def analyze_legal_query(query: str) -> QueryAnalysis:
    normalized_query = normalize_legal_arabic(query or "")
    tokens = extract_search_terms(normalized_query)
    legal_keywords = _filter_keywords(tokens)
    key_phrases, expansion_terms, domain_hints = _detect_legal_phrases(normalized_query)
    intent = _classify_query_intent(normalized_query, legal_keywords, key_phrases)
    constitutional_rights_query = any(cue in normalized_query for cue in CONSTITUTIONAL_RIGHTS_CUES)
    preamble_related = any(cue in normalized_query for cue in PREAMBLE_QUERY_CUES)
    domain_scores = _score_domains(normalized_query, legal_keywords, domain_hints)
    criminal_offense_terms = _detect_criminal_offense_terms(normalized_query)
    if constitutional_rights_query or preamble_related:
        domain_scores["constitutional_law"] = domain_scores.get("constitutional_law", 0) + 2
    if criminal_offense_terms:
        domain_scores["criminal_law"] = domain_scores.get("criminal_law", 0) + 2 + min(1, len(criminal_offense_terms))
    suggested_domain = _suggest_domain(domain_scores)
    out_of_domain, out_of_domain_reason = _detect_out_of_domain(normalized_query, domain_scores)
    criminal_offense_query = bool(criminal_offense_terms) and (
        suggested_domain == "criminal_law" or "criminal_law" in domain_hints or "جريمة" in normalized_query or "عقوبة" in normalized_query
    )
    multi_offense_query = len(criminal_offense_terms) > 1
    prefer_specific_articles = intent != "definition" and bool(
        key_phrases or len(legal_keywords) >= 2 or SPECIFIC_INTENT_TERMS.intersection(legal_keywords)
    )
    if constitutional_rights_query:
        prefer_specific_articles = True
    rewritten_query = _rewrite_query(normalized_query, key_phrases, expansion_terms, legal_keywords)
    return QueryAnalysis(
        original_query=query,
        normalized_query=normalized_query,
        rewritten_query=rewritten_query,
        tokens=tokens,
        legal_keywords=legal_keywords,
        key_phrases=key_phrases,
        expansion_terms=expansion_terms,
        domain_hints=domain_hints,
        intent=intent,
        prefer_specific_articles=prefer_specific_articles,
        constitutional_rights_query=constitutional_rights_query,
        preamble_related=preamble_related,
        domain_scores=domain_scores,
        suggested_domain=suggested_domain,
        out_of_domain=out_of_domain,
        out_of_domain_reason=out_of_domain_reason,
        criminal_offense_query=criminal_offense_query,
        criminal_offense_terms=criminal_offense_terms,
        multi_offense_query=multi_offense_query,
    )


def _filter_keywords(tokens: list[str]) -> list[str]:
    keywords: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if len(token) <= 1 or token in QUESTION_STOPWORDS:
            continue
        if token in seen:
            continue
        seen.add(token)
        keywords.append(token)
    return keywords


def _detect_legal_phrases(normalized_query: str) -> tuple[list[str], list[str], list[str]]:
    phrases: list[str] = []
    expansions: list[str] = []
    domains: list[str] = []

    for phrase in sorted(LEGAL_PHRASE_LIBRARY, key=len, reverse=True):
        if phrase not in normalized_query:
            continue
        phrases.append(phrase)
        payload = LEGAL_PHRASE_LIBRARY[phrase]
        expansions.extend(payload.get("expansions", []))
        domains.extend(payload.get("domains", []))

    return _unique_preserve_order(phrases), _unique_preserve_order(expansions), _unique_preserve_order(domains)


def _classify_query_intent(normalized_query: str, legal_keywords: list[str], key_phrases: list[str]) -> str:
    if any(cue in normalized_query for cue in PREAMBLE_QUERY_CUES):
        return "constitutional_preamble"
    if any(cue in normalized_query for cue in CONSTITUTIONAL_RIGHTS_CUES):
        return "constitutional_rights"
    if any(cue in normalized_query for cue in DEFINITION_CUES):
        return "definition"
    if any(term in normalized_query for term in ("عقوبه", "عقوبات", "جزاء", "جزاءات", "جريمه", "الرشوة")):
        return "penalty"
    if any(term in normalized_query for term in ("انهاء", "انتهاء", "فصل", "خدمة")):
        return "termination"
    if any(term in normalized_query for term in ("بيع", "التزامات", "مسؤوليه", "مسئوليه")):
        return "substantive"
    if key_phrases or SPECIFIC_INTENT_TERMS.intersection(legal_keywords):
        return "substantive"
    return "general"


def _rewrite_query(
    normalized_query: str,
    key_phrases: list[str],
    expansion_terms: list[str],
    legal_keywords: list[str],
) -> str:
    parts = [normalized_query]
    parts.extend(key_phrases)
    parts.extend(expansion_terms)
    parts.extend(legal_keywords[:4])
    return " ".join(_unique_preserve_order(part for part in parts if part)).strip()


def _score_domains(normalized_query: str, legal_keywords: list[str], domain_hints: list[str]) -> dict[str, int]:
    scores = {domain: 0 for domain in DOMAIN_TERM_HINTS}
    for domain in domain_hints:
        if domain in scores:
            scores[domain] += 3

    for domain, cues in DOMAIN_TERM_HINTS.items():
        for cue in cues:
            cue_norm = normalize_legal_arabic(cue)
            if " " in cue_norm:
                if cue_norm in normalized_query:
                    scores[domain] += 2
            elif cue_norm in legal_keywords:
                scores[domain] += 1
    return scores


def _suggest_domain(domain_scores: dict[str, int]) -> str | None:
    ranked = sorted(domain_scores.items(), key=lambda item: item[1], reverse=True)
    if not ranked or ranked[0][1] <= 0:
        return None
    if len(ranked) == 1:
        return ranked[0][0]
    top_domain, top_score = ranked[0]
    second_score = ranked[1][1]
    if top_score >= 2 and top_score >= second_score + 1:
        return top_domain
    return None


def _detect_out_of_domain(normalized_query: str, domain_scores: dict[str, int]) -> tuple[bool, str | None]:
    family_hits = [cue for cue in OUT_OF_DOMAIN_FAMILY_CUES if normalize_legal_arabic(cue) in normalized_query]
    if not family_hits:
        return False, None
    in_domain_strength = max(domain_scores.values()) if domain_scores else 0
    if in_domain_strength >= 3 and domain_scores.get("constitutional_law", 0) > 0:
        return False, None
    return True, f"query appears to target personal-status/family-law topics not covered by the corpus: {family_hits[:3]}"


def _detect_criminal_offense_terms(normalized_query: str) -> list[str]:
    return _unique_preserve_order(
        normalize_legal_arabic(term)
        for term in CRIMINAL_OFFENSE_TERMS
        if normalize_legal_arabic(term) in normalized_query
    )


def _unique_preserve_order(values) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique
