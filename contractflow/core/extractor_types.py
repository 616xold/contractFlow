"""Typed payloads used by the extraction pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, Dict, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from contractflow.core.risk_engine import RiskAssessment


@dataclass
class ExtractionResult:
    raw_text: str
    json_result: Dict[str, Any]
    issues: list[str] | None = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    retrieval: Optional[Dict[str, Any]] = None


class EvidenceSnippet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    page_num: int
    heading: Optional[str] = None
    snippet: str


class FieldExtractionBase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence: list[EvidenceSnippet] = Field(default_factory=list)
    confidence: Annotated[float, Field(ge=0.0, le=1.0)]


class PartyRolesExtractionOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    party_a_name: Optional[str] = None
    party_b_name: Optional[str] = None
    party_a_evidence: list[EvidenceSnippet] = Field(default_factory=list)
    party_b_evidence: list[EvidenceSnippet] = Field(default_factory=list)
    party_a_confidence: Annotated[float, Field(ge=0.0, le=1.0)]
    party_b_confidence: Annotated[float, Field(ge=0.0, le=1.0)]


class FieldVerifierOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: Literal["accept", "revise", "unknown"]
    reason: str
    confidence: Annotated[float, Field(ge=0.0, le=1.0)]
    revised_query: Optional[str] = None


class RiskReviewOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    liability_cap: Optional[str] = None
    governing_law: Optional[str] = None
    data_transfer_outside_uk_eu: Optional[Literal["yes", "no", "unknown"]] = None
    term_length: Optional[int] = None
    termination_notice_days: Optional[int] = None
    non_solicit_clause_present: Optional[bool] = None
    confidence: Annotated[float, Field(ge=0.0, le=1.0)]
    rationale: str


@dataclass
class FieldExtractionResult:
    field: str
    value: Any
    evidence: list[Dict[str, Any]]
    confidence: float
    raw_text: str
    issues: list[str] | None = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    attempts: int = 1


@dataclass
class FieldCandidate:
    source: str
    value: Any
    confidence: float
    evidence: list[Dict[str, Any]]
    issues: list[str]
    attempts: int = 1
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class FieldVerifierResult:
    decision: str
    reason: str
    confidence: float
    revised_query: Optional[str]
    raw_text: str
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None


@dataclass
class JointPartyExtractionResult:
    values: Dict[str, Any]
    field_results: Dict[str, FieldExtractionResult]
    field_issues: Dict[str, list[str]]
    raw_text: str
    prompt_tokens: int
    completion_tokens: int


@dataclass
class RiskPipelineResult:
    assessment: RiskAssessment
    orchestration: Dict[str, Any]
    raw_outputs: list[str]
    prompt_tokens: int = 0
    completion_tokens: int = 0

