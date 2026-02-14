"""Policy-driven risk scoring and optional judge arbitration."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Dict, Literal, Optional

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field

from contractflow.core.liability import parse_liability_cap


RiskLevel = Literal["low", "medium", "high"]


@dataclass
class RiskFactor:
    factor_id: str
    label: str
    value: Any
    severity: RiskLevel
    contribution: float
    confidence: float
    evidence_count: int
    pages: list[int]
    notes: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "factor_id": self.factor_id,
            "label": self.label,
            "value": self.value,
            "severity": self.severity,
            "contribution": round(self.contribution, 4),
            "confidence": round(self.confidence, 4),
            "evidence_count": self.evidence_count,
            "pages": self.pages,
            "notes": self.notes,
        }


@dataclass
class RiskAssessment:
    risk_level: RiskLevel
    risk_explanation: str
    confidence: float
    score: float
    rule_level: RiskLevel
    rule_score: float
    judge_level: Optional[RiskLevel]
    judge_confidence: Optional[float]
    arbitration: str
    hard_triggers: list[str]
    uncertainty: Dict[str, Any]
    factors: list[RiskFactor]
    policy_version: str
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    judge_raw_text: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "version": self.policy_version,
            "risk_level": self.risk_level,
            "risk_explanation": self.risk_explanation,
            "confidence": round(self.confidence, 4),
            "score": round(self.score, 4),
            "rule_level": self.rule_level,
            "rule_score": round(self.rule_score, 4),
            "judge_level": self.judge_level,
            "judge_confidence": round(self.judge_confidence, 4)
            if isinstance(self.judge_confidence, (int, float))
            else None,
            "arbitration": self.arbitration,
            "hard_triggers": self.hard_triggers,
            "uncertainty": self.uncertainty,
            "factors": [factor.as_dict() for factor in self.factors],
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "judge_raw_text": self.judge_raw_text,
        }


class RiskJudgeOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    risk_level: RiskLevel
    confidence: Annotated[float, Field(ge=0.0, le=1.0)]
    rationale: str


_NULL_STRINGS = {"", "null", "none", "n/a", "na", "unknown"}
_POLICY_PATH = Path(__file__).resolve().parents[2] / "docs" / "risk_policy.json"
_DEFAULT_POLICY = {
    "version": "risk_v2",
    "thresholds": {"low_max": 33.0, "medium_max": 66.0},
    "weights": {
        "liability_uncapped": 35.0,
        "liability_unknown": 14.0,
        "liability_cap_12_or_less": -12.0,
        "liability_cap_13_to_24": -6.0,
        "liability_cap_25_plus": 7.0,
        "liability_money_cap_known": -4.0,
        "governing_law_outside_uk_eu": 20.0,
        "governing_law_unknown": 8.0,
        "governing_law_uk_eu": -8.0,
        "data_transfer_yes": 17.0,
        "data_transfer_unknown": 6.0,
        "data_transfer_no": -5.0,
        "term_12_or_less": -4.0,
        "term_13_to_36": 2.0,
        "term_37_plus": 8.0,
        "term_unknown": 2.0,
        "termination_notice_short": 7.0,
        "termination_notice_medium": 2.0,
        "termination_notice_long": -2.0,
        "termination_notice_unknown": 1.0,
        "non_solicit_present": -3.0,
        "non_solicit_absent": 2.0,
        "uncertainty_bonus": 6.0,
        "high_coverage_credit": -3.0,
    },
    "hard_triggers": {
        "uncapped_and_outside_law": 80.0,
        "uncapped_and_cross_border_transfer": 84.0,
        "outside_law_and_cross_border_transfer": 78.0,
    },
    "judge": {
        "enable": True,
        "high_confidence": 0.82,
        "medium_confidence": 0.72,
    },
}


def assess_contract_risk(
    values: Dict[str, Any],
    *,
    field_meta: Optional[Dict[str, Any]] = None,
    model: str,
    client: Optional[OpenAI] = None,
    structured_outputs: bool = True,
    enable_judge: bool = True,
    judge_model: Optional[str] = None,
    policy_path: Optional[str | Path] = None,
) -> RiskAssessment:
    policy = _load_policy(policy_path)
    weights = policy.get("weights", _DEFAULT_POLICY["weights"])

    factors: list[RiskFactor] = []
    hard_triggers: list[str] = []

    liability_cap = values.get("liability_cap")
    governing_law = values.get("governing_law")
    term_length = _coerce_int(values.get("term_length"))
    termination_notice_days = _coerce_int(values.get("termination_notice_days"))
    data_transfer = _normalize_data_transfer(values.get("data_transfer_outside_uk_eu"))
    non_solicit = _coerce_bool(values.get("non_solicit_clause_present"))

    liability_signal = _field_signal(field_meta, "liability_cap")
    governing_signal = _field_signal(field_meta, "governing_law")
    data_signal = _field_signal(field_meta, "data_transfer_outside_uk_eu")
    term_signal = _field_signal(field_meta, "term_length")
    termination_signal = _field_signal(field_meta, "termination_notice_days")
    non_solicit_signal = _field_signal(field_meta, "non_solicit_clause_present")

    liability_signal_parsed = parse_liability_cap(liability_cap)
    liability_missing = _is_nullish(liability_cap)
    liability_uncapped = liability_signal_parsed.is_uncapped
    liability_months = liability_signal_parsed.months
    liability_amount = liability_signal_parsed.amount
    liability_currency = liability_signal_parsed.currency
    liability_kind = liability_signal_parsed.kind
    if liability_missing:
        factors.append(
            _factor(
                "liability_cap",
                "Liability cap",
                liability_cap,
                "medium",
                weights.get("liability_unknown", 14.0),
                liability_signal,
                "Liability cap is not explicitly specified.",
            )
        )
    elif liability_kind in {"uncapped", "none_specified"} or liability_uncapped:
        factors.append(
            _factor(
                "liability_cap",
                "Liability cap",
                liability_cap,
                "high",
                weights.get("liability_uncapped", 35.0),
                liability_signal,
                "Liability appears uncapped or unspecified.",
            )
        )
    elif liability_kind == "months_fees" and liability_months is not None:
        if liability_months <= 12:
            factors.append(
                _factor(
                    "liability_cap",
                    "Liability cap",
                    f"{liability_months} months",
                    "low",
                    weights.get("liability_cap_12_or_less", -12.0),
                    liability_signal,
                    "Liability cap is at or below 12 months.",
                )
            )
        elif liability_months <= 24:
            factors.append(
                _factor(
                    "liability_cap",
                    "Liability cap",
                    f"{liability_months} months",
                    "low",
                    weights.get("liability_cap_13_to_24", -6.0),
                    liability_signal,
                    "Liability cap is between 13 and 24 months.",
                )
            )
        else:
            factors.append(
                _factor(
                    "liability_cap",
                    "Liability cap",
                    f"{liability_months} months",
                    "medium",
                    weights.get("liability_cap_25_plus", 7.0),
                    liability_signal,
                    "Liability cap exceeds 24 months.",
                )
            )
    elif liability_kind == "money_cap" and liability_amount is not None:
        amount_text = (
            f"{liability_currency.upper()} {_format_liability_amount(liability_amount)}"
            if liability_currency
            else _format_liability_amount(liability_amount)
        )
        factors.append(
            _factor(
                "liability_cap",
                "Liability cap",
                amount_text,
                "low",
                weights.get("liability_money_cap_known", -4.0),
                liability_signal,
                "Liability cap is a fixed monetary amount.",
            )
        )
    elif liability_kind == "other":
        factors.append(
            _factor(
                "liability_cap",
                "Liability cap",
                liability_cap,
                "medium",
                weights.get("liability_unknown", 14.0),
                liability_signal,
                "Liability clause exists but cap type could not be normalized.",
            )
        )
    else:
        factors.append(
            _factor(
                "liability_cap",
                "Liability cap",
                liability_cap,
                "medium",
                weights.get("liability_unknown", 14.0),
                liability_signal,
                "Liability cap text could not be normalized.",
            )
        )

    law_region = _governing_law_region(governing_law)
    if law_region == "outside":
        factors.append(
            _factor(
                "governing_law",
                "Governing law",
                governing_law,
                "high",
                weights.get("governing_law_outside_uk_eu", 20.0),
                governing_signal,
                "Governing law appears outside UK/EU.",
            )
        )
    elif law_region == "unknown":
        factors.append(
            _factor(
                "governing_law",
                "Governing law",
                governing_law,
                "medium",
                weights.get("governing_law_unknown", 8.0),
                governing_signal,
                "Governing law could not be confidently mapped to a region.",
            )
        )
    else:
        factors.append(
            _factor(
                "governing_law",
                "Governing law",
                governing_law,
                "low",
                weights.get("governing_law_uk_eu", -8.0),
                governing_signal,
                "Governing law appears inside UK/EU.",
            )
        )

    if data_transfer == "yes":
        factors.append(
            _factor(
                "data_transfer_outside_uk_eu",
                "Cross-border transfer",
                data_transfer,
                "high",
                weights.get("data_transfer_yes", 17.0),
                data_signal,
                "Cross-border transfers are allowed.",
            )
        )
    elif data_transfer == "unknown":
        factors.append(
            _factor(
                "data_transfer_outside_uk_eu",
                "Cross-border transfer",
                data_transfer,
                "medium",
                weights.get("data_transfer_unknown", 6.0),
                data_signal,
                "Cross-border transfer handling is uncertain.",
            )
        )
    else:
        factors.append(
            _factor(
                "data_transfer_outside_uk_eu",
                "Cross-border transfer",
                data_transfer,
                "low",
                weights.get("data_transfer_no", -5.0),
                data_signal,
                "Cross-border transfers are restricted or absent.",
            )
        )

    if term_length is None:
        factors.append(
            _factor(
                "term_length",
                "Contract term length",
                term_length,
                "medium",
                weights.get("term_unknown", 2.0),
                term_signal,
                "Term length is unknown.",
            )
        )
    elif term_length <= 12:
        factors.append(
            _factor(
                "term_length",
                "Contract term length",
                term_length,
                "low",
                weights.get("term_12_or_less", -4.0),
                term_signal,
                "Initial term is at most 12 months.",
            )
        )
    elif term_length <= 36:
        factors.append(
            _factor(
                "term_length",
                "Contract term length",
                term_length,
                "medium",
                weights.get("term_13_to_36", 2.0),
                term_signal,
                "Initial term is between 13 and 36 months.",
            )
        )
    else:
        factors.append(
            _factor(
                "term_length",
                "Contract term length",
                term_length,
                "high",
                weights.get("term_37_plus", 8.0),
                term_signal,
                "Initial term exceeds 36 months.",
            )
        )

    if termination_notice_days is None:
        factors.append(
            _factor(
                "termination_notice_days",
                "Termination notice",
                termination_notice_days,
                "medium",
                weights.get("termination_notice_unknown", 1.0),
                termination_signal,
                "Termination notice period is unspecified.",
            )
        )
    elif termination_notice_days <= 15:
        factors.append(
            _factor(
                "termination_notice_days",
                "Termination notice",
                termination_notice_days,
                "high",
                weights.get("termination_notice_short", 7.0),
                termination_signal,
                "Termination notice period is short (<=15 days).",
            )
        )
    elif termination_notice_days <= 30:
        factors.append(
            _factor(
                "termination_notice_days",
                "Termination notice",
                termination_notice_days,
                "medium",
                weights.get("termination_notice_medium", 2.0),
                termination_signal,
                "Termination notice period is moderate (16-30 days).",
            )
        )
    else:
        factors.append(
            _factor(
                "termination_notice_days",
                "Termination notice",
                termination_notice_days,
                "low",
                weights.get("termination_notice_long", -2.0),
                termination_signal,
                "Termination notice period is relatively long (>30 days).",
            )
        )

    if non_solicit is True:
        factors.append(
            _factor(
                "non_solicit_clause_present",
                "Non-solicit clause",
                non_solicit,
                "low",
                weights.get("non_solicit_present", -3.0),
                non_solicit_signal,
                "Non-solicit obligations are present.",
            )
        )
    else:
        factors.append(
            _factor(
                "non_solicit_clause_present",
                "Non-solicit clause",
                non_solicit,
                "medium",
                weights.get("non_solicit_absent", 2.0),
                non_solicit_signal,
                "No non-solicit protection was found.",
            )
        )

    if liability_uncapped and law_region == "outside":
        hard_triggers.append("uncapped_and_outside_law")
    if liability_uncapped and data_transfer == "yes":
        hard_triggers.append("uncapped_and_cross_border_transfer")
    if law_region == "outside" and data_transfer == "yes":
        hard_triggers.append("outside_law_and_cross_border_transfer")

    rule_score = 50.0 + sum(f.contribution for f in factors)
    uncertainty = _compute_uncertainty(values, field_meta)
    if uncertainty["high_uncertainty"]:
        rule_score += float(weights.get("uncertainty_bonus", 6.0))
    elif uncertainty["high_coverage"]:
        rule_score += float(weights.get("high_coverage_credit", -3.0))

    hard_trigger_floor = 0.0
    for trigger in hard_triggers:
        trigger_floor = float(policy.get("hard_triggers", {}).get(trigger, 0.0))
        hard_trigger_floor = max(hard_trigger_floor, trigger_floor)
    if hard_trigger_floor > 0.0:
        rule_score = max(rule_score, hard_trigger_floor)

    rule_score = max(0.0, min(100.0, rule_score))
    rule_level = _score_to_level(rule_score, policy)
    rule_confidence = _rule_confidence(uncertainty)

    judge_level: Optional[RiskLevel] = None
    judge_confidence: Optional[float] = None
    judge_rationale: Optional[str] = None
    judge_raw_text = ""
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None

    judge_enabled = bool(enable_judge and policy.get("judge", {}).get("enable", True))
    if judge_enabled:
        try:
            judge_output, judge_raw_text, prompt_tokens, completion_tokens = _call_risk_judge(
                values=values,
                rule_level=rule_level,
                rule_score=rule_score,
                factors=factors,
                hard_triggers=hard_triggers,
                uncertainty=uncertainty,
                model=judge_model or model,
                client=client or OpenAI(),
                structured_outputs=structured_outputs,
            )
            judge_level = judge_output.risk_level
            judge_confidence = float(judge_output.confidence)
            judge_rationale = judge_output.rationale.strip()
        except Exception:
            judge_level = None
            judge_confidence = None
            judge_rationale = None
            judge_raw_text = ""

    final_level, arbitration = _arbitrate_level(
        rule_level=rule_level,
        judge_level=judge_level,
        judge_confidence=judge_confidence,
        uncertainty=uncertainty,
        policy=policy,
    )
    final_score = _level_midpoint(final_level)
    final_confidence = _final_confidence(
        rule_confidence=rule_confidence,
        judge_confidence=judge_confidence,
        arbitration=arbitration,
    )
    explanation = _compose_explanation(
        level=final_level,
        factors=factors,
        hard_triggers=hard_triggers,
        uncertainty=uncertainty,
        judge_rationale=judge_rationale,
        arbitration=arbitration,
    )

    return RiskAssessment(
        risk_level=final_level,
        risk_explanation=explanation,
        confidence=final_confidence,
        score=final_score,
        rule_level=rule_level,
        rule_score=rule_score,
        judge_level=judge_level,
        judge_confidence=judge_confidence,
        arbitration=arbitration,
        hard_triggers=hard_triggers,
        uncertainty=uncertainty,
        factors=factors,
        policy_version=str(policy.get("version", "risk_v2")),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        judge_raw_text=judge_raw_text,
    )


def _load_policy(policy_path: Optional[str | Path]) -> Dict[str, Any]:
    path = Path(policy_path) if policy_path else _POLICY_PATH
    if not path.exists():
        return _DEFAULT_POLICY
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return _DEFAULT_POLICY
    if not isinstance(data, dict):
        return _DEFAULT_POLICY
    merged = dict(_DEFAULT_POLICY)
    merged.update(data)
    if not isinstance(merged.get("weights"), dict):
        merged["weights"] = dict(_DEFAULT_POLICY["weights"])
    if not isinstance(merged.get("thresholds"), dict):
        merged["thresholds"] = dict(_DEFAULT_POLICY["thresholds"])
    if not isinstance(merged.get("hard_triggers"), dict):
        merged["hard_triggers"] = dict(_DEFAULT_POLICY["hard_triggers"])
    if not isinstance(merged.get("judge"), dict):
        merged["judge"] = dict(_DEFAULT_POLICY["judge"])
    return merged


def _field_signal(field_meta: Optional[Dict[str, Any]], field: str) -> Dict[str, Any]:
    if not isinstance(field_meta, dict):
        return {"confidence": 0.5, "evidence_count": 0, "pages": []}
    meta = field_meta.get(field)
    if not isinstance(meta, dict):
        return {"confidence": 0.5, "evidence_count": 0, "pages": []}
    confidence = meta.get("confidence")
    if not isinstance(confidence, (int, float)):
        confidence = 0.5
    evidence = meta.get("evidence")
    evidence_count = len(evidence) if isinstance(evidence, list) else 0
    pages: list[int] = []
    if isinstance(evidence, list):
        for item in evidence:
            if not isinstance(item, dict):
                continue
            page = item.get("page_num")
            if isinstance(page, int):
                pages.append(page)
    return {
        "confidence": max(0.0, min(1.0, float(confidence))),
        "evidence_count": evidence_count,
        "pages": sorted(set(pages)),
    }


def _factor(
    factor_id: str,
    label: str,
    value: Any,
    severity: RiskLevel,
    contribution: float,
    signal: Dict[str, Any],
    notes: str,
) -> RiskFactor:
    return RiskFactor(
        factor_id=factor_id,
        label=label,
        value=value,
        severity=severity,
        contribution=float(contribution),
        confidence=float(signal.get("confidence", 0.5)),
        evidence_count=int(signal.get("evidence_count", 0)),
        pages=list(signal.get("pages", [])),
        notes=notes,
    )


def _coerce_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        match = re.search(r"-?\d+", value.replace(",", ""))
        if match:
            return int(match.group(0))
    return None


def _coerce_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if cleaned in {"true", "t", "yes", "y", "1"}:
            return True
        if cleaned in {"false", "f", "no", "n", "0"}:
            return False
    return None


def _is_nullish(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.strip().lower() in _NULL_STRINGS:
        return True
    return False


def _normalize_data_transfer(value: Any) -> Literal["yes", "no", "unknown"]:
    if value is None:
        return "unknown"
    cleaned = str(value).strip().lower()
    if cleaned in {"yes", "true"}:
        return "yes"
    if cleaned in {"no", "false"}:
        return "no"
    return "unknown"


def _liability_uncapped(liability_cap: Any) -> bool:
    return parse_liability_cap(liability_cap).is_uncapped


def _liability_cap_months(liability_cap: Any) -> Optional[int]:
    return parse_liability_cap(liability_cap).months


def _governing_law_region(governing_law: Any) -> Literal["uk_eu", "outside", "unknown"]:
    if governing_law is None:
        return "unknown"
    text = str(governing_law).strip().lower()
    if not text:
        return "unknown"
    uk_terms = ["england", "wales", "scotland", "northern ireland", "uk", "united kingdom"]
    eu_terms = [
        "eu",
        "european union",
        "austria",
        "belgium",
        "bulgaria",
        "croatia",
        "cyprus",
        "czech",
        "denmark",
        "estonia",
        "finland",
        "france",
        "germany",
        "greece",
        "hungary",
        "ireland",
        "italy",
        "latvia",
        "lithuania",
        "luxembourg",
        "malta",
        "netherlands",
        "poland",
        "portugal",
        "romania",
        "slovakia",
        "slovenia",
        "spain",
        "sweden",
    ]
    if any(term in text for term in uk_terms + eu_terms):
        return "uk_eu"
    outside_terms = [
        "new york",
        "delaware",
        "california",
        "texas",
        "usa",
        "united states",
        "india",
        "singapore",
        "hong kong",
        "australia",
        "canada",
    ]
    if any(term in text for term in outside_terms):
        return "outside"
    return "unknown"


def _compute_uncertainty(values: Dict[str, Any], field_meta: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    critical_fields = ["liability_cap", "governing_law", "data_transfer_outside_uk_eu"]
    confidences: list[float] = []
    with_evidence = 0
    unknown_critical = 0

    for field in critical_fields:
        signal = _field_signal(field_meta, field)
        confidences.append(float(signal["confidence"]))
        if int(signal["evidence_count"]) > 0:
            with_evidence += 1
        value = values.get(field)
        if value is None:
            unknown_critical += 1
        elif isinstance(value, str) and value.strip().lower() in _NULL_STRINGS:
            unknown_critical += 1

    coverage = with_evidence / max(1, len(critical_fields))
    avg_confidence = sum(confidences) / max(1, len(confidences))
    high_uncertainty = coverage < 0.5 or avg_confidence < 0.5 or unknown_critical >= 2
    high_coverage = coverage >= 0.8 and avg_confidence >= 0.7 and unknown_critical == 0
    return {
        "critical_fields": critical_fields,
        "critical_coverage": round(coverage, 4),
        "critical_avg_confidence": round(avg_confidence, 4),
        "critical_unknown_count": unknown_critical,
        "high_uncertainty": high_uncertainty,
        "high_coverage": high_coverage,
    }


def _score_to_level(score: float, policy: Dict[str, Any]) -> RiskLevel:
    thresholds = policy.get("thresholds", {})
    low_max = float(thresholds.get("low_max", 33.0))
    medium_max = float(thresholds.get("medium_max", 66.0))
    if score <= low_max:
        return "low"
    if score <= medium_max:
        return "medium"
    return "high"


def _rule_confidence(uncertainty: Dict[str, Any]) -> float:
    coverage = float(uncertainty.get("critical_coverage", 0.0))
    avg_conf = float(uncertainty.get("critical_avg_confidence", 0.0))
    unknown_count = int(uncertainty.get("critical_unknown_count", 0))
    confidence = 0.45 + 0.3 * coverage + 0.3 * avg_conf - 0.08 * unknown_count
    return max(0.2, min(0.95, confidence))


def _call_risk_judge(
    *,
    values: Dict[str, Any],
    rule_level: RiskLevel,
    rule_score: float,
    factors: list[RiskFactor],
    hard_triggers: list[str],
    uncertainty: Dict[str, Any],
    model: str,
    client: OpenAI,
    structured_outputs: bool,
) -> tuple[RiskJudgeOutput, str, Optional[int], Optional[int]]:
    factor_payload = [factor.as_dict() for factor in factors]
    system_prompt = (
        "You are a contract risk judge. Use only the provided structured factors.\n"
        "Return JSON with keys: risk_level, confidence, rationale.\n"
        "risk_level must be one of: low, medium, high."
    )
    user_prompt = (
        "Policy context:\n"
        f"- rule_level={rule_level}\n"
        f"- rule_score={round(rule_score, 2)}\n"
        f"- hard_triggers={hard_triggers}\n"
        f"- uncertainty={json.dumps(uncertainty, ensure_ascii=False)}\n\n"
        "Extracted values:\n"
        f"{json.dumps(values, ensure_ascii=False)}\n\n"
        "Factor table:\n"
        f"{json.dumps(factor_payload, ensure_ascii=False)}\n\n"
        "Output constraints:\n"
        "- Keep rationale concise and evidence-grounded.\n"
        "- If uncertainty is high, avoid low risk unless factors are clearly protective."
    )
    input_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    response: Any
    raw_output: str
    parsed_obj: RiskJudgeOutput

    if structured_outputs and hasattr(client.responses, "parse"):
        try:
            response = client.responses.parse(
                model=model,
                input=input_messages,
                text_format=RiskJudgeOutput,
                reasoning={"effort": "none"},
                temperature=0,
                max_output_tokens=320,
            )
            raw_output = _extract_response_text(response)
            parsed = getattr(response, "output_parsed", None)
            if parsed is None:
                parsed_obj = RiskJudgeOutput.model_validate(_safe_parse_json(raw_output))
            else:
                parsed_obj = parsed
        except Exception:
            response = client.responses.create(
                model=model,
                input=input_messages,
                reasoning={"effort": "none"},
                temperature=0,
                max_output_tokens=320,
            )
            raw_output = _extract_response_text(response)
            parsed_obj = RiskJudgeOutput.model_validate(_safe_parse_json(raw_output))
    else:
        response = client.responses.create(
            model=model,
            input=input_messages,
            reasoning={"effort": "none"},
            temperature=0,
            max_output_tokens=320,
        )
        raw_output = _extract_response_text(response)
        parsed_obj = RiskJudgeOutput.model_validate(_safe_parse_json(raw_output))

    usage = getattr(response, "usage", None)
    prompt_tokens = getattr(usage, "input_tokens", None)
    completion_tokens = getattr(usage, "output_tokens", None)
    return parsed_obj, raw_output, prompt_tokens, completion_tokens


def _arbitrate_level(
    *,
    rule_level: RiskLevel,
    judge_level: Optional[RiskLevel],
    judge_confidence: Optional[float],
    uncertainty: Dict[str, Any],
    policy: Dict[str, Any],
) -> tuple[RiskLevel, str]:
    if judge_level is None or judge_confidence is None:
        return rule_level, "rules_only"

    rule_rank = _level_rank(rule_level)
    judge_rank = _level_rank(judge_level)
    diff = abs(rule_rank - judge_rank)
    judge_cfg = policy.get("judge", {})
    high_conf = float(judge_cfg.get("high_confidence", 0.82))
    med_conf = float(judge_cfg.get("medium_confidence", 0.72))
    high_uncertainty = bool(uncertainty.get("high_uncertainty", False))

    if judge_confidence >= high_conf and diff >= 1:
        return judge_level, "judge_override_high_confidence"
    if judge_confidence >= med_conf and diff >= 1 and high_uncertainty:
        return judge_level, "judge_override_uncertainty"
    return rule_level, "rules_with_judge_check"


def _final_confidence(
    *,
    rule_confidence: float,
    judge_confidence: Optional[float],
    arbitration: str,
) -> float:
    if judge_confidence is None:
        return round(rule_confidence, 4)
    if arbitration.startswith("judge_override"):
        return round(max(0.45, min(0.95, 0.35 * rule_confidence + 0.65 * judge_confidence)), 4)
    return round(max(0.35, min(0.95, 0.75 * rule_confidence + 0.25 * judge_confidence)), 4)


def _compose_explanation(
    *,
    level: RiskLevel,
    factors: list[RiskFactor],
    hard_triggers: list[str],
    uncertainty: Dict[str, Any],
    judge_rationale: Optional[str],
    arbitration: str,
) -> str:
    risk_drivers = sorted((f for f in factors if f.contribution > 0), key=lambda f: f.contribution, reverse=True)
    protectors = sorted((f for f in factors if f.contribution < 0), key=lambda f: f.contribution)

    parts: list[str] = []
    if risk_drivers:
        top = risk_drivers[0]
        parts.append(f"Primary risk driver: {top.label.lower()} ({top.notes.lower()})")
    if len(risk_drivers) > 1:
        second = risk_drivers[1]
        parts.append(f"Secondary driver: {second.label.lower()} ({second.notes.lower()})")
    if protectors:
        protect = protectors[0]
        parts.append(f"Mitigating factor: {protect.label.lower()} ({protect.notes.lower()})")
    if hard_triggers:
        parts.append(f"Hard trigger(s): {', '.join(hard_triggers)}")
    if uncertainty.get("high_uncertainty"):
        parts.append("Some critical factors have limited evidence or confidence")
    if arbitration.startswith("judge_override") and judge_rationale:
        parts.append(f"Judge adjustment: {judge_rationale}")

    if not parts:
        parts.append("Risk is assessed from the available contract factors")
    tail = f"Final risk level: {level}."
    return "; ".join(parts) + ". " + tail


def _level_rank(level: RiskLevel) -> int:
    if level == "low":
        return 0
    if level == "medium":
        return 1
    return 2


def _level_midpoint(level: RiskLevel) -> float:
    if level == "low":
        return 20.0
    if level == "medium":
        return 50.0
    return 82.0


def _format_liability_amount(value: float) -> str:
    if abs(value - round(value)) <= 1e-9:
        return f"{int(round(value)):,}"
    return f"{value:,.2f}".rstrip("0").rstrip(".")


def _extract_response_text(response: Any) -> str:
    if hasattr(response, "output_text"):
        return response.output_text
    if hasattr(response, "model_dump"):
        data = response.model_dump()
        text = _dig_for_text(data)
        if text:
            return text
    return str(response)


def _dig_for_text(data: Any) -> Optional[str]:
    if isinstance(data, dict):
        for key in ("output_text", "text"):
            if key in data and isinstance(data[key], str):
                return data[key]
        for value in data.values():
            found = _dig_for_text(value)
            if found:
                return found
    elif isinstance(data, list):
        for item in data:
            found = _dig_for_text(item)
            if found:
                return found
    return None


def _safe_parse_json(raw: str) -> Dict[str, Any]:
    raw = raw.strip()
    try:
        return json.loads(raw)
    except Exception:
        pass

    start = raw.find("{")
    if start == -1:
        raise ValueError(f"No JSON object found in model output: {raw!r}")
    decoder = json.JSONDecoder()
    obj, _end = decoder.raw_decode(raw[start:])
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object, got {type(obj).__name__}")
    return obj
