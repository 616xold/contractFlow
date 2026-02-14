from contractflow.core.risk_engine import assess_contract_risk


def _base_values(liability_cap: str) -> dict[str, object]:
    return {
        "liability_cap": liability_cap,
        "governing_law": "England and Wales",
        "data_transfer_outside_uk_eu": "no",
        "term_length": 12,
        "termination_notice_days": 45,
        "non_solicit_clause_present": True,
    }


def test_money_cap_is_scored_as_known_monetary_cap() -> None:
    out = assess_contract_risk(_base_values("USD 1,000,000"), model="gpt-5.2", enable_judge=False)
    liability = next(f for f in out.factors if f.factor_id == "liability_cap")
    assert liability.contribution == -4.0
    assert "fixed monetary amount" in liability.notes.lower()


def test_mixed_clause_prefers_month_window_scoring() -> None:
    out = assess_contract_risk(
        _base_values("liability shall not exceed 12 months of fees or USD 1,000,000"),
        model="gpt-5.2",
        enable_judge=False,
    )
    liability = next(f for f in out.factors if f.factor_id == "liability_cap")
    assert liability.contribution == -12.0
    assert str(liability.value).startswith("12 months")
