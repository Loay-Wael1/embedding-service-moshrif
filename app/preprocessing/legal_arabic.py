from __future__ import annotations

import re
from dataclasses import dataclass


ARABIC_DIACRITICS_RE = re.compile(r"[\u064B-\u0652\u0670\u06D6-\u06ED]")
ALEF_VARIANTS_RE = re.compile(r"[إأآٱ]")
TATWEEL_RE = re.compile(r"\u0640+")
SPACED_DOTS_RE = re.compile(r"(?:\.\s+){2,}\.")
BROKEN_PUNCT_RE = re.compile(r"([،؛:,.!?])\s*([،؛:,.!?])+")
MULTI_SPACE_RE = re.compile(r"\s+")
SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([،؛:,.!?])")
SPACE_AFTER_OPEN_RE = re.compile(r"([(\[{])\s+")
SPACE_BEFORE_CLOSE_RE = re.compile(r"\s+([)\]}])")
LATIN_NOISE_RE = re.compile(r"[A-Za-z]{3,}")
SPACED_ARABIC_LETTERS_RE = re.compile(r"(?:\b[\u0621-\u064A]\s+){4,}[\u0621-\u064A]\b")
BROKEN_OCR_PATTERN_RE = re.compile(r"\)\s*\d+\s*\(|[:؛،]\s*[.،؛:]")
TOKEN_RE = re.compile(r"[\u0621-\u064A0-9]+")

ARABIC_INDIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")

REPEALED_TERMS = (
    "ملغاة",
    "ملغى",
    "ملغي",
    "الغيت",
    "ألغيت",
    "منسوخ",
    "نسخت",
    "موقوفة",
    "موقوف",
    "أوقفت",
)

NORMALIZATION_RULES = [
    "remove_tashkeel",
    "normalize_alef_variants",
    "normalize_alef_maqsura_to_yaa",
    "normalize_digits_to_ascii",
    "normalize_safe_punctuation_spacing",
    "collapse_repeated_whitespace",
]


@dataclass(frozen=True)
class TextQualityAssessment:
    noise_score: float
    warnings: list[str]


def remove_diacritics(text: str) -> str:
    return ARABIC_DIACRITICS_RE.sub("", text)


def normalize_alef_variants(text: str) -> str:
    return ALEF_VARIANTS_RE.sub("ا", text)


def normalize_yaa_variants(text: str) -> str:
    return text.replace("ى", "ي")


def normalize_digits(text: str) -> str:
    return text.translate(ARABIC_INDIC_DIGITS)


def normalize_safe_punctuation(text: str) -> str:
    text = TATWEEL_RE.sub("", text)
    text = SPACED_DOTS_RE.sub("...", text)
    text = BROKEN_PUNCT_RE.sub(r"\1", text)
    text = re.sub(r"\s*([،؛:,.!?])\s*", r"\1 ", text)
    text = SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)
    text = SPACE_AFTER_OPEN_RE.sub(r"\1", text)
    text = SPACE_BEFORE_CLOSE_RE.sub(r"\1", text)
    text = re.sub(r"\s*-\s*", " - ", text)
    text = re.sub(r"\s*/\s*", " / ", text)
    return text


def collapse_whitespace(text: str) -> str:
    return MULTI_SPACE_RE.sub(" ", text).strip()


def normalize_legal_arabic(text: str) -> str:
    if not text or not isinstance(text, str):
        return text

    text = remove_diacritics(text)
    text = normalize_alef_variants(text)
    text = normalize_yaa_variants(text)
    text = normalize_digits(text)
    text = normalize_safe_punctuation(text)
    return collapse_whitespace(text)


def normalize_legal_reference_text(text: str) -> str:
    normalized = normalize_legal_arabic(text)
    return normalized.replace("مادة", "المادة")


def extract_search_terms(text: str) -> list[str]:
    normalized = normalize_legal_arabic(text)
    return TOKEN_RE.findall(normalized)


def is_repealed_text(text: str) -> bool:
    normalized = normalize_legal_arabic(text or "")
    return any(term in normalized for term in REPEALED_TERMS)


def assess_text_quality(text: str) -> TextQualityAssessment:
    if not text:
        return TextQualityAssessment(noise_score=0.0, warnings=["empty_text"])

    warnings: list[str] = []
    normalized = collapse_whitespace(text)

    if LATIN_NOISE_RE.search(normalized):
        warnings.append("latin_noise")
    if SPACED_ARABIC_LETTERS_RE.search(normalized):
        warnings.append("spaced_letters")
    if BROKEN_OCR_PATTERN_RE.search(normalized):
        warnings.append("broken_punctuation")

    tokens = TOKEN_RE.findall(normalized)
    if tokens:
        single_char_ratio = sum(1 for token in tokens if len(token) == 1) / len(tokens)
        if len(tokens) >= 10 and single_char_ratio > 0.32:
            warnings.append("high_single_char_ratio")

    noise_score = min(0.95, 0.18 * len(warnings))
    return TextQualityAssessment(noise_score=noise_score, warnings=warnings)
