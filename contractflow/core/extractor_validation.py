"""Schema validation and field normalization helpers for extraction."""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any, Dict, Optional

from contractflow.core.liability import canonicalize_liability_cap, parse_liability_cap

_MISSING = object()
_NULL_STRINGS = {"", "null", "none", "n/a", "na", "unknown"}
_US_STATE_NAMES: tuple[str, ...] = (
    "Alabama",
    "Alaska",
    "Arizona",
    "Arkansas",
    "California",
    "Colorado",
    "Connecticut",
    "Delaware",
    "Florida",
    "Georgia",
    "Hawaii",
    "Idaho",
    "Illinois",
    "Indiana",
    "Iowa",
    "Kansas",
    "Kentucky",
    "Louisiana",
    "Maine",
    "Maryland",
    "Massachusetts",
    "Michigan",
    "Minnesota",
    "Mississippi",
    "Missouri",
    "Montana",
    "Nebraska",
    "Nevada",
    "New Hampshire",
    "New Jersey",
    "New Mexico",
    "New York",
    "North Carolina",
    "North Dakota",
    "Ohio",
    "Oklahoma",
    "Oregon",
    "Pennsylvania",
    "Rhode Island",
    "South Carolina",
    "South Dakota",
    "Tennessee",
    "Texas",
    "Utah",
    "Vermont",
    "Virginia",
    "Washington",
    "West Virginia",
    "Wisconsin",
    "Wyoming",
)


def _combine_evidence_text(evidence: list[Dict[str, Any]]) -> str:
    snippets = []
    for item in evidence:
        snippet = str(item.get("snippet", "")).strip()
        if snippet:
            snippets.append(snippet)
    return " ".join(snippets)


def _extract_int_from_text(text: str) -> Optional[int]:
    cleaned = text.strip().lower()
    match = re.search(r"-?\d+", cleaned.replace(",", ""))
    if match:
        return int(match.group(0))

    word_map = {
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
    }
    for word, value in word_map.items():
        if re.search(rf"\b{word}\b", cleaned):
            return value
    return None


def _term_unit_hint(text: str) -> Optional[str]:
    lowered = text.lower()
    has_year = bool(re.search(r"\b(year|years|yr|yrs)\b", lowered))
    has_month = bool(re.search(r"\b(month|months|mo|mos)\b", lowered))
    if has_year and not has_month:
        return "years"
    if has_month and not has_year:
        return "months"
    return None


def _normalize_effective_date(value: Any) -> tuple[Any, Optional[str], bool]:
    if value is None:
        return None, None, False
    if not isinstance(value, str):
        return value, "effective_date is not a string", True

    cleaned = value.strip()
    if not cleaned:
        return None, "effective_date is empty", True

    formats = [
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%Y.%m.%d",
        "%B %d, %Y",
        "%b %d, %Y",
        "%d %B %Y",
        "%d %b %Y",
    ]
    for fmt in formats:
        try:
            parsed = datetime.strptime(cleaned, fmt).date()
            iso = parsed.isoformat()
            if iso != cleaned:
                return iso, "normalized effective_date to ISO", False
            return cleaned, None, False
        except ValueError:
            continue

    numeric_match = re.search(r"(\d{1,4})[/-](\d{1,2})[/-](\d{2,4})", cleaned)
    if numeric_match:
        a, b, c = (int(n) for n in numeric_match.groups())
        if a >= 1000:
            year, month, day = a, b, c
        else:
            year = c if c >= 100 else (2000 + c if c < 50 else 1900 + c)
            if a > 12 and b <= 12:
                day, month = a, b
            else:
                month, day = a, b
        try:
            parsed = date(year, month, day)
            iso = parsed.isoformat()
            if iso != cleaned:
                return iso, "normalized effective_date to ISO", False
            return cleaned, None, False
        except ValueError:
            pass

    return value, "unable to normalize effective_date to ISO", True


def _normalize_term_length(value: Any, evidence_text: str) -> tuple[Any, Optional[str], bool]:
    if value is None:
        return None, None, False

    number: Optional[int] = None
    unit_hint: Optional[str] = None
    if isinstance(value, int):
        number = value
        unit_hint = "months"
    elif isinstance(value, str):
        value_text = value.strip().lower()
        number = _extract_int_from_text(value_text)
        unit_hint = _term_unit_hint(value_text)

    if number is None:
        number = _extract_int_from_text(evidence_text)
        if unit_hint is None:
            unit_hint = _term_unit_hint(evidence_text)

    if number is None:
        return None, "unable to parse term_length", True

    if unit_hint == "years":
        return number * 12, "normalized term_length from years to months", False
    if unit_hint == "months":
        return number, None, False

    evidence_unit = _term_unit_hint(evidence_text)
    if evidence_unit == "years" and number <= 15:
        return number * 12, "normalized term_length from years to months (inferred)", False
    return number, None, False


def _match_us_state_name(text: str) -> Optional[str]:
    lowered = text.lower()
    for state in sorted(_US_STATE_NAMES, key=len, reverse=True):
        if re.search(rf"\b{re.escape(state.lower())}\b", lowered):
            return state
    return None


def _title_case_jurisdiction(text: str) -> str:
    tokens = re.findall(r"[A-Za-z']+|[^A-Za-z']+", text)
    small_words = {"of", "and", "the"}
    out: list[str] = []
    word_index = 0
    for token in tokens:
        if re.fullmatch(r"[A-Za-z']+", token):
            lowered = token.lower()
            if word_index > 0 and lowered in small_words:
                out.append(lowered)
            else:
                out.append(lowered.capitalize())
            word_index += 1
        else:
            out.append(token)
    return "".join(out).strip()


def _trim_jurisdiction_fragment(text: str) -> str:
    fragment = text.strip().strip(" ,;:.()[]{}")
    if not fragment:
        return ""
    fragment = re.split(
        r"\b(without|excluding|except|venue|jurisdiction|forum|court|courts|arbitration)\b",
        fragment,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    fragment = re.split(r"[,;:\.]", fragment, maxsplit=1)[0]
    return " ".join(fragment.split()).strip(" ,;:.()[]{}")


def _canonical_governing_law_from_text(text: str) -> Optional[str]:
    if not text:
        return None
    cleaned = " ".join(text.split()).strip()
    if not cleaned:
        return None
    lowered = (
        cleaned.lower()
        .replace("\u2019", "'")
        .replace("\u2018", "'")
        .replace("â€™", "'")
    )

    if "people's republic of china" in lowered or "peoples republic of china" in lowered or re.search(r"\bprc\b", lowered):
        return "People's Republic of China"
    if "england and wales" in lowered:
        return "England and Wales"

    patterns = (
        r"laws?\s+of\s+the\s+state\s+of\s+([a-z][a-z\s'\-]+)",
        r"state\s+of\s+([a-z][a-z\s'\-]+)",
        r"laws?\s+of\s+([a-z][a-z\s'\-]+)",
    )
    for pattern in patterns:
        match = re.search(pattern, lowered)
        if not match:
            continue
        fragment = _trim_jurisdiction_fragment(match.group(1))
        if not fragment:
            continue
        state = _match_us_state_name(fragment)
        if state:
            return f"State of {state}"
        if "people's republic of china" in fragment or "peoples republic of china" in fragment:
            return "People's Republic of China"
        if "england and wales" in fragment:
            return "England and Wales"
        return _title_case_jurisdiction(fragment)

    state = _match_us_state_name(lowered)
    if state and (len(cleaned.split()) <= 4 or any(token in lowered for token in ("govern", "law", "venue", "jurisdiction"))):
        return f"State of {state}"
    return None


def _normalize_governing_law(value: Any, evidence_text: str) -> tuple[Any, Optional[str], bool]:
    if isinstance(value, str) and value.strip().lower() in _NULL_STRINGS:
        value = None

    inferred_from_evidence = _canonical_governing_law_from_text(evidence_text)
    if value is None:
        if inferred_from_evidence:
            return inferred_from_evidence, "inferred governing_law from evidence snippets", False
        return None, "governing_law missing", True

    if not isinstance(value, str):
        value = str(value)
    cleaned = " ".join(value.split()).strip()
    if not cleaned:
        if inferred_from_evidence:
            return inferred_from_evidence, "inferred governing_law from evidence snippets", False
        return None, "governing_law is empty", True

    canonical = _canonical_governing_law_from_text(cleaned)
    if canonical is None and inferred_from_evidence:
        return inferred_from_evidence, "normalized governing_law from evidence snippets", False
    if canonical is None:
        return cleaned, None, False
    if canonical.lower() != cleaned.lower():
        return canonical, "normalized governing_law to canonical jurisdiction", False
    return canonical, None, False


def _has_liability_cap_context(text: str) -> bool:
    lowered = text.lower()
    has_cap_keywords = bool(
        re.search(
            r"\b(cap|capped|maximum|max(?:imum)?|limit(?:ed|ation)?|exceed|aggregate|not\s+exceed)\b",
            lowered,
        )
    )
    has_liability_term = bool(re.search(r"\b(liability|liable|damages?)\b", lowered))
    has_cap_relation = bool(
        re.search(
            r"\b(not\s+exceed|limited\s+to|up\s+to|at\s+most|no\s+more\s+than|equal\s+to)\b",
            lowered,
        )
    )
    has_amount_or_window = bool(
        re.search(
            r"\b(month|months|year|years|annual|fee|fees|paid|payable|amount|usd|eur|gbp|million|thousand)\b",
            lowered,
        )
        or re.search(r"[$\u20ac\u00a3]\s*\d", lowered)
    )
    return has_cap_keywords or (has_liability_term and has_amount_or_window and has_cap_relation)


def _normalize_liability_cap(value: Any, evidence_text: str) -> tuple[Any, Optional[str], bool]:
    if isinstance(value, str) and value.strip().lower() in _NULL_STRINGS:
        value = None

    evidence_lower = evidence_text.strip().lower()
    has_liability_signal = any(
        term in evidence_lower
        for term in ("liability", "damages", "indirect", "consequential", "cap")
    )
    evidence_signal = parse_liability_cap(evidence_text) if (has_liability_signal and evidence_lower) else None
    inferred_from_evidence = evidence_signal.canonical if evidence_signal is not None else None

    def _liability_strength(kind: str) -> int:
        if kind in {"months_fees", "money_cap"}:
            return 4
        if kind == "uncapped":
            return 3
        if kind == "none_specified":
            return 2
        if kind == "other":
            return 1
        return 0

    if value is None:
        if inferred_from_evidence and evidence_signal and _liability_strength(evidence_signal.kind) >= 2:
            return inferred_from_evidence, "inferred liability_cap from evidence snippets", False
        return "none specified", "defaulted liability_cap to none specified", False
    if not isinstance(value, str):
        value = str(value)

    value_signal = parse_liability_cap(value)
    canonical = value_signal.canonical
    if canonical is None:
        canonical = inferred_from_evidence

    if canonical is None:
        cleaned = " ".join(value.split()).strip()
        if not cleaned:
            return None, "liability_cap is empty", True
        return cleaned, None, False

    value_sig = parse_liability_cap(canonical)
    if (
        evidence_lower
        and has_liability_signal
        and (value_sig.months is not None or value_sig.amount is not None)
        and not _has_liability_cap_context(evidence_lower)
    ):
        if inferred_from_evidence:
            return (
                inferred_from_evidence,
                "downgraded liability_cap to evidence-supported none specified",
                False,
            )
        return "none specified", "downgraded liability_cap due weak cap context", False

    if inferred_from_evidence and evidence_signal is not None:
        evidence_sig = parse_liability_cap(inferred_from_evidence)
        if evidence_sig.months is not None and (
            value_sig.months is None or abs(evidence_sig.months - value_sig.months) >= 6
        ):
            return (
                inferred_from_evidence,
                "normalized liability_cap from evidence-derived fee window",
                False,
            )
        if evidence_sig.is_uncapped and not value_sig.is_uncapped and value_sig.months is None:
            return (
                inferred_from_evidence,
                "normalized liability_cap from evidence-derived uncapped posture",
                False,
            )
        value_strength = _liability_strength(value_sig.kind)
        evidence_strength = _liability_strength(evidence_sig.kind)
        if evidence_strength > value_strength and _has_liability_cap_context(evidence_lower):
            return (
                inferred_from_evidence,
                "normalized liability_cap from stronger evidence-derived clause parse",
                False,
            )

    original_clean = " ".join(str(value).split()).strip().lower()
    if canonical != original_clean:
        return canonical, "normalized liability_cap to canonical representation", False
    return canonical, None, False


def _normalize_enum_value(enum_vals: Any, value: str) -> Optional[str]:
    if not isinstance(enum_vals, list):
        return None
    lookup = {str(v).strip().lower(): str(v) for v in enum_vals}
    return lookup.get(value.strip().lower())


def _coerce_and_validate_value(field: str, meta: Dict[str, Any], value: Any, *, coerce: bool) -> Any:
    nullable = bool(meta.get("nullable"))
    expected_type = meta.get("type")
    enum_vals = meta.get("enum")

    if value is _MISSING:
        if nullable:
            return None
        raise ValueError(f"Missing required field '{field}'")

    if value is None:
        if nullable:
            return None
        raise ValueError(f"Field '{field}' is null but not nullable")

    if expected_type == "string":
        out: Any
        if isinstance(value, str):
            out = value.strip()
        elif coerce and isinstance(value, (int, float)) and not isinstance(value, bool):
            out = str(value)
        else:
            raise ValueError(f"Field '{field}' expected string, got {type(value).__name__}")

        if not nullable and out.strip() == "":
            raise ValueError(f"Field '{field}' must be a non-empty string")

        if out.strip().lower() == "unknown" and field != "data_transfer_outside_uk_eu":
            raise ValueError(
                f"Field '{field}' must not be 'unknown' (reserved for data_transfer_outside_uk_eu)"
            )

        if enum_vals:
            if out is None:
                if nullable:
                    return None
                raise ValueError(f"Field '{field}' is null but not nullable")
            if not isinstance(out, str):
                raise ValueError(
                    f"Field '{field}' must be one of {enum_vals}, got {type(out).__name__}"
                )
            normalized_enum = _normalize_enum_value(enum_vals, out)
            if normalized_enum is None:
                raise ValueError(f"Field '{field}' must be one of {enum_vals}, got {out!r}")
            out = normalized_enum

        return out

    if expected_type == "integer":
        out_int: Optional[int] = None

        if isinstance(value, bool):
            raise ValueError(f"Field '{field}' expected integer, got boolean")
        if isinstance(value, int):
            out_int = value
        elif coerce and isinstance(value, float) and value.is_integer():
            out_int = int(value)
        elif coerce and isinstance(value, str):
            cleaned = value.strip()
            if cleaned.lower() in _NULL_STRINGS:
                if nullable:
                    return None
                raise ValueError(f"Field '{field}' expected integer, got {value!r}")
            match = re.search(r"-?\d+", cleaned.replace(",", ""))
            if match:
                out_int = int(match.group(0))
            elif nullable:
                return None
            else:
                raise ValueError(f"Field '{field}' expected integer, got {value!r}")
        else:
            raise ValueError(f"Field '{field}' expected integer, got {type(value).__name__}")

        if enum_vals:
            if out_int not in enum_vals:
                raise ValueError(f"Field '{field}' must be one of {enum_vals}, got {out_int!r}")

        return out_int

    if expected_type == "boolean":
        out_bool: Optional[bool] = None

        if isinstance(value, bool):
            out_bool = value
        elif coerce and isinstance(value, int) and value in (0, 1):
            out_bool = bool(value)
        elif coerce and isinstance(value, str):
            cleaned = value.strip().lower()
            if cleaned in _NULL_STRINGS:
                if nullable:
                    return None
                raise ValueError(f"Field '{field}' expected boolean, got {value!r}")
            if cleaned in {"true", "t", "yes", "y", "1"}:
                out_bool = True
            elif cleaned in {"false", "f", "no", "n", "0"}:
                out_bool = False
            else:
                raise ValueError(f"Field '{field}' expected boolean, got {value!r}")
        else:
            raise ValueError(f"Field '{field}' expected boolean, got {type(value).__name__}")

        if enum_vals:
            if out_bool not in enum_vals:
                raise ValueError(f"Field '{field}' must be one of {enum_vals}, got {out_bool!r}")

        return out_bool

    raise ValueError(f"Field '{field}' has unsupported schema type: {expected_type!r}")


def _validate_and_normalize_field(
    field: str,
    meta: Dict[str, Any],
    value: Any,
    evidence: list[Dict[str, Any]],
    *,
    coerce: bool,
) -> tuple[Any, list[str], bool]:
    issues: list[str] = []
    conflict = False
    try:
        normalized = _coerce_and_validate_value(field, meta, value, coerce=coerce)
    except ValueError as e:
        issues.append(str(e))
        normalized = None
        conflict = True

    evidence_text = _combine_evidence_text(evidence)
    if field == "effective_date":
        normalized, issue, date_conflict = _normalize_effective_date(normalized)
        if issue:
            issues.append(issue)
        conflict = conflict or date_conflict
    elif field == "term_length":
        normalized, issue, term_conflict = _normalize_term_length(normalized, evidence_text)
        if issue:
            issues.append(issue)
        conflict = conflict or term_conflict
    elif field == "governing_law":
        normalized, issue, law_conflict = _normalize_governing_law(normalized, evidence_text)
        if issue:
            issues.append(issue)
        conflict = conflict or law_conflict
    elif field == "liability_cap":
        normalized, issue, liab_conflict = _normalize_liability_cap(normalized, evidence_text)
        if issue:
            issues.append(issue)
        conflict = conflict or liab_conflict

    return normalized, issues, conflict


def _validate_and_normalize_to_schema(
    schema: Dict[str, Any],
    data: Any,
    *,
    coerce: bool,
) -> tuple[Dict[str, Any], list[str]]:
    issues: list[str] = []
    if not isinstance(data, dict):
        issues.append(f"Expected JSON object at top level, got {type(data).__name__}")
        data = {}

    extra_keys = sorted(set(data.keys()) - set(schema.keys()))
    if extra_keys:
        issues.append(f"Unexpected keys not in schema: {extra_keys}")

    normalized: Dict[str, Any] = {}
    for field, meta in schema.items():
        value = data.get(field, _MISSING)
        try:
            field_value = _coerce_and_validate_value(field, meta, value, coerce=coerce)
            issue: Optional[str] = None
            if field == "effective_date":
                field_value, issue, _conflict = _normalize_effective_date(field_value)
            elif field == "term_length":
                field_value, issue, _conflict = _normalize_term_length(field_value, "")
            elif field == "governing_law":
                field_value, issue, _conflict = _normalize_governing_law(field_value, "")
            elif field == "liability_cap":
                field_value, issue, _conflict = _normalize_liability_cap(field_value, "")
            if issue:
                issues.append(f"field '{field}': {issue}")
            normalized[field] = field_value
        except ValueError as e:
            issues.append(str(e))
            normalized[field] = None

    return normalized, issues
