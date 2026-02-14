"""Utilities for parsing and comparing liability-cap clause values."""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Optional


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
_UNCAPPED_TERMS = (
    "uncapped",
    "unlimited",
    "no cap",
    "without limit",
    "not limited",
    "no limitation",
    "none specified",
    "not specified",
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
)


@dataclass(frozen=True)
class LiabilityCapSignal:
    raw_text: str
    normalized_text: str
    is_nullish: bool
    is_uncapped: bool
    months: Optional[int]
    amount: Optional[float]
    currency: Optional[str]


def parse_liability_cap(value: Any) -> LiabilityCapSignal:
    raw_text = "" if value is None else str(value)
    normalized_text = _normalize_text(raw_text)
    is_nullish = _is_nullish(raw_text)
    if is_nullish:
        return LiabilityCapSignal(
            raw_text=raw_text,
            normalized_text=normalized_text,
            is_nullish=True,
            is_uncapped=False,
            months=None,
            amount=None,
            currency=None,
        )

    lowered = raw_text.strip().lower()
    is_uncapped = any(term in lowered for term in _UNCAPPED_TERMS)
    months = _extract_cap_months(lowered)
    amount, currency = _extract_cap_amount(lowered)
    return LiabilityCapSignal(
        raw_text=raw_text,
        normalized_text=normalized_text,
        is_nullish=False,
        is_uncapped=is_uncapped,
        months=months,
        amount=amount,
        currency=currency,
    )


def canonicalize_liability_cap(value: Any) -> Optional[str]:
    signal = parse_liability_cap(value)
    if signal.is_nullish:
        return None
    if signal.is_uncapped:
        lowered = signal.raw_text.strip().lower()
        if any(
            term in lowered
            for term in ("not specified", "none specified", "unspecified", "not stated", "no stated")
        ):
            return "none specified"
        return "uncapped"
    if signal.months is not None:
        return f"{signal.months} months fees"
    if signal.amount is not None:
        amount_text = _format_amount(signal.amount)
        if signal.currency:
            return f"{signal.currency} {amount_text}"
        return f"amount {amount_text}"
    if signal.normalized_text:
        lowered = signal.raw_text.strip().lower()
        if ("liability" in lowered or "damages" in lowered) and not _has_cap_phrase(lowered):
            return "none specified"
        return signal.normalized_text
    return None


def liability_cap_similarity(gold: Any, pred: Any) -> float:
    left = parse_liability_cap(gold)
    right = parse_liability_cap(pred)
    if left.is_nullish and right.is_nullish:
        return 1.0
    if left.is_nullish or right.is_nullish:
        return 0.0

    candidates: list[float] = []
    if left.is_uncapped and right.is_uncapped:
        candidates.append(1.0)
    elif left.is_uncapped != right.is_uncapped:
        candidates.append(0.1)

    if left.months is not None and right.months is not None:
        candidates.append(_months_similarity(left.months, right.months))

    if left.amount is not None and right.amount is not None:
        candidates.append(_amount_similarity(left.amount, right.amount, left.currency, right.currency))

    candidates.append(_text_similarity(left.normalized_text, right.normalized_text))
    return max(0.0, min(1.0, max(candidates)))


def _is_nullish(value: str) -> bool:
    cleaned = value.strip().lower()
    return cleaned in _NULL_STRINGS


def _extract_cap_months(text: str) -> Optional[int]:
    cap_phrase = _has_cap_phrase(text)
    fee_basis = _has_fee_basis(text)
    if not cap_phrase and not fee_basis:
        return None

    month_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:month|months|mo|mos)\b", text)
    if month_match:
        return max(0, int(round(float(month_match.group(1)))))

    year_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:year|years|yr|yrs)\b", text)
    if year_match:
        years = float(year_match.group(1))
        return max(0, int(round(years * 12)))

    word_months = _extract_word_number_with_unit(text, unit="months")
    if word_months is not None:
        return word_months

    word_years = _extract_word_number_with_unit(text, unit="years")
    if word_years is not None:
        return word_years * 12

    # Common legal phrasing: "annual fees" implies a 12-month fee base.
    if re.search(r"\bannual\s+fees?\b", text):
        mult = re.search(r"(\d+)\s*x\s*annual", text)
        if mult:
            return int(mult.group(1)) * 12
        return 12
    return None


def _extract_word_number_with_unit(text: str, *, unit: str) -> Optional[int]:
    tokens = re.findall(r"[a-z]+", text.lower())
    if len(tokens) < 2:
        return None

    target_units = {"months"} if unit == "months" else {"year", "years", "yr", "yrs"}
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
    if not cap_phrase and not fee_basis:
        return None, None
    if "insurance" in text and not cap_phrase:
        return None, None

    symbol_match = re.search(r"([$\u20ac\u00a3])\s*(\d[\d,]*(?:\.\d+)?)", text)
    if symbol_match:
        currency = _CURRENCY_SYMBOL_MAP.get(symbol_match.group(1))
        amount = _parse_amount(symbol_match.group(2))
        return amount, currency

    code_match = re.search(r"\b(usd|eur|gbp|cad|aud|inr)\s*(\d[\d,]*(?:\.\d+)?)\b", text)
    if code_match:
        amount = _parse_amount(code_match.group(2))
        return amount, code_match.group(1)

    return None, None


def _parse_amount(text: str) -> Optional[float]:
    try:
        return float(text.replace(",", ""))
    except Exception:
        return None


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

