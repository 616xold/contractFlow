from contractflow.core.liability import (
    canonicalize_liability_cap,
    liability_cap_similarity,
    parse_liability_cap,
)


def test_parse_months_based_cap() -> None:
    signal = parse_liability_cap("liability shall not exceed 12 months of fees")
    assert signal.kind == "months_fees"
    assert signal.months == 12
    assert signal.canonical == "12 months fees"


def test_parse_money_cap_with_short_output() -> None:
    signal = parse_liability_cap("$1,000,000")
    assert signal.kind == "money_cap"
    assert signal.amount == 1_000_000.0
    assert signal.currency == "usd"
    assert signal.canonical == "usd 1000000"


def test_uncapped_and_none_specified_are_detected() -> None:
    uncapped = parse_liability_cap("uncapped")
    none_specified = parse_liability_cap("none specified")
    assert uncapped.is_uncapped is True
    assert none_specified.is_uncapped is True
    assert uncapped.kind == "uncapped"
    assert none_specified.kind == "none_specified"


def test_canonicalize_liability_cap_uses_parser() -> None:
    assert canonicalize_liability_cap("USD 2m") == "usd 2000000"


def test_similarity_handles_normalized_money_forms() -> None:
    score = liability_cap_similarity("USD 1,000,000", "$1000000")
    assert score >= 0.9
