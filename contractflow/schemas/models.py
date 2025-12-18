"""Pydantic models for structured contract extraction outputs."""

from __future__ import annotations

from datetime import date
from typing import Annotated, Literal, Optional

from pydantic import BaseModel, ConfigDict, StringConstraints

NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class ContractExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    doc_type: Literal["nda", "msa", "other"]
    party_a_name: NonEmptyStr
    party_b_name: NonEmptyStr
    effective_date: date
    term_length: Optional[int] = None
    governing_law: NonEmptyStr
    termination_notice_days: Optional[int] = None
    liability_cap: str
    non_solicit_clause_present: bool
    data_transfer_outside_uk_eu: Literal["yes", "no", "unknown"]
    risk_level: Literal["low", "medium", "high"]
    risk_explanation: str
