"""Utilities for parsing, normalizing, and comparing liability-cap clauses."""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Literal, Optional


LiabilityCapKind = Literal[
    "nullish",
    "uncapped",
    "none_specified",
    "months_fees",
    "money_cap",
    "other",
]

_NULL_STRINGS = {"", "null", "none", "n/a", "na", "unknown"}
_STOPWORDS = {
    "the",
    "a",
    "an",
    "to",
    "of",
    "and",
    "or",
    "for",
    "by",
    "in",
    "on",
    "at",
    "is",
    "are",
    "be",
    "this",
    "that",
    "will",
    "shall",
    "not",
    "no",
    "any",
    "all",
}
_CURRENCY_SYMBOL_MAP = {"$": "usd", "\u20ac": "eur", "\u00a3": "gbp"}
_MULTIPLIER_MAP = {
    "k": 1_000.0,
    "thousand": 1_000.0,
    "m": 1_000_000.0,
    "mm": 1_000_000.0,
    "million": 1_000_000.0,
    "b": 1_000_000_000.0,
    "bn": 1_000_000_000.0,
    "billion": 1_000_000_000.0,
}
_NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
_LIABILITY_ANCHORS = (
    "limitation of liability",
    "liability",
    "liable",
    "damages",
    "maximum aggregate",
    "total aggregate",
)
_UNCAPPED_TERMS = (
    "uncapped",
    "unlimited",
    "without limit",
    "not limited",
    "no limitation",
    "no cap",
)
_NONE_SPECIFIED_TERMS = (
    "none specified",
    "not specified",
    "not stated",
    "unspecified",
    "no explicit cap",
    "no monetary cap",
    "no stated cap",
)
_CAP_PHRASES = (
    "limitation of liability",
    "liability shall not exceed",
    "shall not exceed",
    "aggregate liability",
    "cap on liability",
    "liability is limited",
    "liability will be limited",
    "liability shall be limited",
    "maximum liability",
    "total liability",
    "limited to",
    "up to",
    "greater of",
)


@dataclass(frozen=True)
class LiabilityCapSignal:
    raw_text: str
    normalized_text: str
    kind: LiabilityCapKind
    is_nullish: bool
    is_uncapped: bool
    months: Optional[int]
    amount: Optional[float]
    currency: Optional[str]
    canonical: Optional[str] = None
    clause_span: Optional[str] = None


def parse_liability_cap(value: Any) -> LiabilityCapSignal:
    raw_text = "" if value is None else str(value)
    normalized_text = _normalize_text(raw_text)
    if _is_nullish(raw_text):
        return LiabilityCapSignal(
            raw_text=raw_text,
            normalized_text=normalized_text,
            kind="nullish",
            is_nullish=True,
            is_uncapped=False,
            months=None,
            amount=None,
            currency=None,
            canonical=None,
            clause_span=None,
        )

    lowered = raw_text.strip().lower()
    span = _extract_liability_span(lowered)
    target = span or lowered

    uncapped = _contains_any(target, _UNCAPPED_TERMS)
    none_specified = _contains_any(target, _NONE_SPECIFIED_TERMS)
    months = _extract_cap_months(target)
    amount, currency = _extract_cap_amount(target)

    kind: LiabilityCapKind
    if uncapped and none_specified:
        kind = "none_specified"
    elif none_specified:
        kind = "none_specified"
    elif uncapped:
        kind = "uncapped"
    elif months is not None:
        kind = "months_fees"
    elif amount is not None:
        kind = "money_cap"
    elif _mentions_liability(target) and not _has_cap_phrase(target):
        kind = "none_specified"
    else:
        kind = "other"

    canonical = _canonical_from_parts(
        kind=kind,
        normalized_text=normalized_text,
        months=months,
        amount=amount,
        currency=currency,
    )
    is_uncapped = kind in {"uncapped", "none_specified"}
    return LiabilityCapSignal(
        raw_text=raw_text,
        normalized_text=normalized_text,
        kind=kind,
        is_nullish=False,
        is_uncapped=is_uncapped,
        months=months,
        amount=amount,
        currency=currency,
        canonical=canonical,
        clause_span=span,
    )


def canonicalize_liability_cap(value: Any) -> Optional[str]:
    return parse_liability_cap(value).canonical


def liability_cap_similarity(gold: Any, pred: Any) -> float:
    left = parse_liability_cap(gold)
    right = parse_liability_cap(pred)
    if left.is_nullish and right.is_nullish:
        return 1.0
    if left.is_nullish or right.is_nullish:
        return 0.0

    candidates: list[float] = []
    if left.kind == right.kind:
        candidates.append(0.92)

    if left.is_uncapped and right.is_uncapped:
        uncapped_score = 1.0 if left.kind == right.kind else 0.95
        candidates.append(uncapped_score)
    elif left.is_uncapped != right.is_uncapped:
        candidates.append(0.1)

    if left.months is not None and right.months is not None:
        candidates.append(_months_similarity(left.months, right.months))

    if left.amount is not None and right.amount is not None:
        candidates.append(_amount_similarity(left.amount, right.amount, left.currency, right.currency))

    if left.canonical and right.canonical:
        candidates.append(_text_similarity(left.canonical, right.canonical))
    else:
        candidates.append(_text_similarity(left.normalized_text, right.normalized_text))

    return max(0.0, min(1.0, max(candidates)))


def _canonical_from_parts(
    *,
    kind: LiabilityCapKind,
    normalized_text: str,
    months: Optional[int],
    amount: Optional[float],
    currency: Optional[str],
) -> Optional[str]:
    if kind == "nullish":
        return None
    if kind == "none_specified":
        return "none specified"
    if kind == "uncapped":
        return "uncapped"
    if kind == "months_fees" and months is not None:
        return f"{months} months fees"
    if kind == "money_cap" and amount is not None:
        amount_text = _format_amount(amount)
        if currency:
            return f"{currency} {amount_text}"
        return f"amount {amount_text}"
    if normalized_text:
        return normalized_text
    return None


def _extract_liability_span(text: str) -> Optional[str]:
    if not text:
        return None
    sentences = [part.strip() for part in re.split(r"[.;\n]+", text) if part.strip()]
    if not sentences:
        return None

    for idx, sentence in enumerate(sentences):
        if _mentions_liability(sentence):
            span_parts = [sentence]
            if idx + 1 < len(sentences) and _mentions_cap_context(sentences[idx + 1]):
                span_parts.append(sentences[idx + 1])
            return " ; ".join(span_parts)
    return None


def _is_nullish(value: str) -> bool:
    return value.strip().lower() in _NULL_STRINGS


def _mentions_liability(text: str) -> bool:
    lowered = text.lower()
    return any(anchor in lowered for anchor in _LIABILITY_ANCHORS)


def _mentions_cap_context(text: str) -> bool:
    lowered = text.lower()
    return _has_cap_phrase(lowered) or _has_fee_basis(lowered) or _contains_any(lowered, _UNCAPPED_TERMS)


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in terms)


def _extract_cap_months(text: str) -> Optional[int]:
    cap_phrase = _has_cap_phrase(text)
    fee_basis = _has_fee_basis(text)
    if not cap_phrase and not fee_basis:
        return None

    paren_match = re.search(
        r"(?:\b[a-z]+)?\s*\(\s*(\d{1,3})\s*\)\s*(month|months|mo|mos|year|years|yr|yrs)\b",
        text,
    )
    if paren_match:
        num = int(paren_match.group(1))
        unit = paren_match.group(2)
        return _to_months(num, unit)

    numeric_match = re.search(r"(\d+(?:\.\d+)?)\s*(month|months|mo|mos|year|years|yr|yrs)\b", text)
    if numeric_match:
        num = float(numeric_match.group(1))
        unit = numeric_match.group(2)
        return _to_months(num, unit)

    word_months = _extract_word_number_with_unit(text, unit="months")
    if word_months is not None:
        return word_months

    word_years = _extract_word_number_with_unit(text, unit="years")
    if word_years is not None:
        return word_years * 12

    if re.search(r"\bannual\s+fees?\b", text):
        mult = re.search(r"(\d+)\s*x\s*annual", text)
        if mult:
            return int(mult.group(1)) * 12
        return 12
    return None


def _to_months(value: float, unit: str) -> int:
    if unit in {"year", "years", "yr", "yrs"}:
        return max(0, int(round(value * 12)))
    return max(0, int(round(value)))


def _extract_word_number_with_unit(text: str, *, unit: str) -> Optional[int]:
    tokens = re.findall(r"[a-z]+", text.lower())
    if len(tokens) < 2:
        return None

    target_units = {"year", "years", "yr", "yrs"}
    if unit == "months":
        target_units = {"month", "months", "mo", "mos"}

    for idx, token in enumerate(tokens):
        if token not in target_units or idx == 0:
            continue
        num = _word_number_at(tokens, idx - 1)
        if num is not None:
            return num
    return None


def _word_number_at(tokens: list[str], idx: int) -> Optional[int]:
    token = tokens[idx]
    if token in _NUMBER_WORDS:
        value = _NUMBER_WORDS[token]
        if value >= 20 and value % 10 == 0 and idx + 1 < len(tokens):
            nxt = tokens[idx + 1]
            if nxt in _NUMBER_WORDS and 0 < _NUMBER_WORDS[nxt] < 10:
                return value + _NUMBER_WORDS[nxt]
        return value
    return None


def _extract_cap_amount(text: str) -> tuple[Optional[float], Optional[str]]:
    cap_phrase = _has_cap_phrase(text)
    fee_basis = _has_fee_basis(text)
    amount_only_signal = _looks_like_money_only(text)
    if not cap_phrase and not fee_basis and not amount_only_signal:
        return None, None
    if "insurance" in text and not cap_phrase:
        return None, None

    symbol_match = re.search(
        r"([$\u20ac\u00a3])\s*(\d[\d,]*(?:\.\d+)?)\s*(k|m|mm|bn|b|thousand|million|billion)?\b",
        text,
    )
    if symbol_match:
        currency = _CURRENCY_SYMBOL_MAP.get(symbol_match.group(1))
        amount = _parse_amount(symbol_match.group(2), symbol_match.group(3))
        return amount, currency

    code_match = re.search(
        r"\b(usd|eur|gbp|cad|aud|inr)\s*(\d[\d,]*(?:\.\d+)?)\s*(k|m|mm|bn|b|thousand|million|billion)?\b",
        text,
    )
    if code_match:
        amount = _parse_amount(code_match.group(2), code_match.group(3))
        return amount, code_match.group(1)

    if amount_only_signal:
        plain_match = re.search(
            r"(\d[\d,]*(?:\.\d+)?)\s*(k|m|mm|bn|b|thousand|million|billion)\b",
            text,
        )
        if plain_match:
            amount = _parse_amount(plain_match.group(1), plain_match.group(2))
            return amount, None

    return None, None


def _looks_like_money_only(text: str) -> bool:
    cleaned = " ".join(text.strip().lower().split())
    if not cleaned:
        return False
    # Common extractor outputs can be short canonical values like "$1,000,000" or "usd 2m".
    currency_prefixed = re.fullmatch(
        r"(?:about|around|approximately)?\s*(?:[$\u20ac\u00a3]|usd|eur|gbp|cad|aud|inr)\s*"
        r"\d[\d,]*(?:\.\d+)?\s*(?:k|m|mm|bn|b|thousand|million|billion)?"
        r"(?:\s*(?:in\s+aggregate|aggregate|per\s+claim))?",
        cleaned,
    )
    if currency_prefixed:
        return True

    # No currency is accepted only when a magnitude marker is present (e.g., "2 million").
    magnitude_only = re.fullmatch(
        r"(?:about|around|approximately)?\s*\d[\d,]*(?:\.\d+)?\s*"
        r"(?:k|m|mm|bn|b|thousand|million|billion)"
        r"(?:\s*(?:in\s+aggregate|aggregate|per\s+claim))?",
        cleaned,
    )
    return bool(magnitude_only)


def _parse_amount(text: str, multiplier: Optional[str]) -> Optional[float]:
    try:
        base = float(text.replace(",", ""))
    except Exception:
        return None
    mult = 1.0
    if isinstance(multiplier, str):
        mult = _MULTIPLIER_MAP.get(multiplier.lower(), 1.0)
    return base * mult


def _format_amount(value: float) -> str:
    if abs(value - round(value)) <= 1e-9:
        return str(int(round(value)))
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _months_similarity(left: int, right: int) -> float:
    diff = abs(int(left) - int(right))
    if diff == 0:
        return 1.0
    if diff <= 1:
        return 0.95
    if diff <= 3:
        return 0.88
    if diff <= 6:
        return 0.78
    if diff <= 12:
        return 0.6
    return 0.35


def _amount_similarity(left: float, right: float, left_cur: Optional[str], right_cur: Optional[str]) -> float:
    hi = max(abs(left), abs(right), 1e-9)
    rel_diff = abs(left - right) / hi
    score = max(0.0, 1.0 - (1.4 * rel_diff))
    if left_cur and right_cur and left_cur != right_cur:
        score *= 0.85
    return score


def _text_similarity(left: str, right: str) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    seq = SequenceMatcher(None, left, right).ratio()
    jacc = _token_jaccard(left, right)
    return 0.55 * seq + 0.45 * jacc


def _normalize_text(text: str) -> str:
    lowered = text.lower()
    lowered = re.sub(r"[^a-z0-9]+", " ", lowered)
    return " ".join(lowered.split())


def _token_jaccard(left: str, right: str) -> float:
    left_tokens = {tok for tok in left.split() if tok and tok not in _STOPWORDS}
    right_tokens = {tok for tok in right.split() if tok and tok not in _STOPWORDS}
    if not left_tokens and not right_tokens:
        return 1.0
    if not left_tokens or not right_tokens:
        return 0.0
    inter = len(left_tokens & right_tokens)
    union = len(left_tokens | right_tokens)
    return inter / union if union else 0.0


def _has_cap_phrase(text: str) -> bool:
    lowered = text.lower()
    return any(phrase in lowered for phrase in _CAP_PHRASES)


def _has_fee_basis(text: str) -> bool:
    return bool(
        re.search(
            r"\b(fee|fees|charge|charges|payment|payments|paid|payable|revenue|consideration)\b",
            text.lower(),
        )
    )

