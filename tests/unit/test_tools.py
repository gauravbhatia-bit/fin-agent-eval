from decimal import Decimal

import pytest
from pydantic import ValidationError

from app.tools.escalate import EscalateInput, escalate_to_human
from app.tools.fees import FeeInput, calculate_fees
from app.tools.fx import FxInput, convert_currency
from app.tools.growth import GrowthInput, project_growth
from app.tools.instruments import InstrumentInput, lookup_instrument
from app.tools.registry import TOOLS, dispatch_tool, tool_schemas
from app.tools.risk import PositionSizeInput, position_size
from app.tools.search import SearchInput, search_knowledge


def test_registry_exposes_all_seven_typed_tools() -> None:
    expected = {
        "calculate_fees",
        "project_growth",
        "convert_currency",
        "lookup_instrument",
        "position_size",
        "search_knowledge",
        "escalate_to_human",
    }

    assert set(TOOLS) == expected
    assert {schema["name"] for schema in tool_schemas()} == expected
    assert all(schema["parameters"]["type"] == "object" for schema in tool_schemas())


def test_dispatch_is_deterministic_for_all_seven_tools() -> None:
    requests = [
        (
            "calculate_fees",
            {"instrument_type": "etf", "side": "buy", "order_value": 20},
        ),
        (
            "project_growth",
            {
                "monthly_amount": 50,
                "years": 1,
                "annual_return_pct": 5,
                "annual_fee_pct": 0.2,
            },
        ),
        ("convert_currency", {"amount": 20, "from_ccy": "EUR", "to_ccy": "USD"}),
        ("lookup_instrument", {"symbol": "DEMO-WORLD-ETF"}),
        (
            "position_size",
            {
                "account_value": 1000,
                "risk_pct": 1,
                "entry_price": 20,
                "stop_price": 18,
            },
        ),
        ("search_knowledge", {"query": "ETF basics", "top_k": 2}),
        ("escalate_to_human", {"reason": "demo request"}),
    ]

    for name, arguments in requests:
        first = dispatch_tool(name, arguments)
        assert first == dispatch_tool(name, arguments)
        assert first.get("error") is None


def test_fees_follow_the_checked_in_formula_and_caps() -> None:
    normal = calculate_fees(
        FeeInput(instrument_type="etf", side="buy", order_value=Decimal("2000"))
    )
    maximum = calculate_fees(
        FeeInput(
            instrument_type="etf",
            side="sell",
            order_value=Decimal("1000000"),
        )
    )
    minimum = calculate_fees(
        FeeInput(
            instrument_type="savings_plan",
            side="buy",
            order_value=Decimal("0.01"),
        )
    )

    # ETF fee = fixed EUR 1.00 + 0.10% * EUR 2,000 = EUR 3.00.
    assert normal.fee == 3.0
    assert normal.fee_pct == 0.15
    assert [part.amount for part in normal.breakdown] == [1.0, 2.0, 0.0]
    # The uncapped ETF fee is EUR 1 + EUR 1,000; the schedule caps it at EUR 25.
    assert maximum.fee == 25.0
    assert sum(part.amount for part in maximum.breakdown) == maximum.fee
    # EUR 0.01 * 0.25% = EUR 0.000025, rounded to the cent for output.
    assert minimum.fee == 0.0


def test_fee_input_rejects_out_of_range_and_extra_fields() -> None:
    with pytest.raises(ValidationError):
        FeeInput(instrument_type="etf", side="buy", order_value=-1)
    with pytest.raises(ValidationError):
        FeeInput(instrument_type="etf", side="buy", order_value=1_000_001)
    with pytest.raises(ValidationError):
        FeeInput(
            instrument_type="crypto",
            side="buy",
            order_value=10,  # type: ignore[arg-type]
        )
    with pytest.raises(ValidationError):
        FeeInput(instrument_type="etf", side="buy", order_value=10, extra=True)


def test_growth_zero_rate_and_fee_adjusted_formula() -> None:
    no_growth = project_growth(
        GrowthInput(
            monthly_amount=100,
            years=1,
            annual_return_pct=0,
            annual_fee_pct=0,
        )
    )
    projected = project_growth(
        GrowthInput(
            monthly_amount=100,
            years=1,
            annual_return_pct=12,
            annual_fee_pct=1,
        )
    )

    # 12 end-of-month deposits of EUR 100, with no return or fee, total EUR 1,200.
    assert no_growth.final_value == 1200.0
    assert no_growth.total_contributed == 1200.0
    assert no_growth.total_fees == 0.0
    # Net annual factor is (1 + 12%) * (1 - 1%); monthly contributions compound monthly.
    net_annual_factor = (1 + 12 / 100) * (1 - 1 / 100)
    monthly_factor = net_annual_factor ** (1 / 12)
    expected_net = 100 * (monthly_factor**12 - 1) / (monthly_factor - 1)
    gross_monthly_factor = (1 + 12 / 100) ** (1 / 12)
    expected_gross = 100 * (gross_monthly_factor**12 - 1) / (gross_monthly_factor - 1)
    assert projected.final_value == pytest.approx(round(expected_net, 2))
    assert projected.total_fees == pytest.approx(
        round(expected_gross - expected_net, 2)
    )
    assert (
        projected.note
        == "Hypothetical, not a forecast. Fees are modeled, not actual charges."
    )


def test_growth_bounds_and_invalid_values() -> None:
    zero_growth_factor = project_growth(
        GrowthInput(
            monthly_amount=25,
            years=1,
            annual_return_pct=-100,
            annual_fee_pct=0,
        )
    )
    assert zero_growth_factor.final_value == 25.0
    with pytest.raises(ValidationError):
        GrowthInput(
            monthly_amount=0,
            years=1,
            annual_return_pct=2,
            annual_fee_pct=0,
        )
    with pytest.raises(ValidationError):
        GrowthInput(
            monthly_amount=100,
            years=81,
            annual_return_pct=2,
            annual_fee_pct=0,
        )


def test_currency_conversion_uses_static_dated_rates_and_inverse() -> None:
    usd = convert_currency(FxInput(amount=100, from_ccy="EUR", to_ccy="USD"))
    eur = convert_currency(FxInput(amount=100, from_ccy="USD", to_ccy="EUR"))
    same = convert_currency(FxInput(amount=12.34, from_ccy="EUR", to_ccy="EUR"))

    # EUR 100 * fictional USD-per-EUR rate 1.08 = USD 108.
    assert usd.result == 108.0
    assert usd.rate == 1.08
    assert usd.rate_date == "2025-01-15"
    # USD 100 / 1.08 = EUR 92.592..., rounded to two decimal places.
    assert eur.result == 92.59
    assert eur.rate == pytest.approx(1 / 1.08, abs=0.000001)
    assert eur.rate_date == "2025-01-15"
    assert (same.result, same.rate, same.rate_date) == (12.34, 1.0, "not_applicable")


def test_fx_input_rejects_invalid_amount_and_currency() -> None:
    assert convert_currency(FxInput(amount=0, from_ccy="EUR", to_ccy="USD")).result == 0
    with pytest.raises(ValidationError):
        FxInput(amount=-1, from_ccy="EUR", to_ccy="USD")
    with pytest.raises(ValidationError):
        FxInput(amount=10, from_ccy="INR", to_ccy="USD")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        FxInput(amount=10, from_ccy="USD", to_ccy="GBP")


def test_instrument_lookup_normalizes_symbol_and_returns_not_found() -> None:
    result = lookup_instrument(InstrumentInput(symbol=" demo-world-etf "))
    missing = lookup_instrument(InstrumentInput(symbol="DEMO-NOT-REAL"))

    assert result.found is True
    assert result.symbol == "DEMO-WORLD-ETF"
    assert result.name == "Fictional Global Basket ETF"
    assert result.currency == "EUR"
    assert result.expense_ratio_pct == 0.2
    assert missing.found is False
    assert missing.error == "No matching fictional demo instrument."


def test_instrument_input_rejects_blank_symbol() -> None:
    with pytest.raises(ValidationError):
        InstrumentInput(symbol="   ")


def test_position_size_matches_risk_budget_formula_and_boundaries() -> None:
    result = position_size(
        PositionSizeInput(
            account_value=10_000,
            risk_pct=1,
            entry_price=100,
            stop_price=95,
        )
    )
    zero_risk = position_size(
        PositionSizeInput(
            account_value=10_000,
            risk_pct=0,
            entry_price=100,
            stop_price=95,
        )
    )

    # Risk budget EUR 10,000 * 1% = EUR 100; loss per unit is EUR 5; 100 // 5 = 20.
    assert result.units == 20
    assert result.max_loss == 100.0
    assert zero_risk.units == 0
    assert zero_risk.max_loss == 0.0
    with pytest.raises(ValidationError):
        PositionSizeInput(
            account_value=10_000,
            risk_pct=1,
            entry_price=95,
            stop_price=100,
        )
    with pytest.raises(ValidationError):
        PositionSizeInput(
            account_value=10_000,
            risk_pct=5.01,
            entry_price=100,
            stop_price=95,
        )


def test_search_returns_deterministic_relevant_hits_and_validates_bounds() -> None:
    query = SearchInput(query="fictional DemoBroker fee schedule", top_k=4)
    first = search_knowledge(query)
    second = search_knowledge(query)

    assert first == second
    assert first.results
    assert first.results[0].doc_id == "demobroker-fees"
    assert len(first.results) <= 4
    assert search_knowledge(SearchInput(query="!!!")).results == []
    with pytest.raises(ValidationError):
        SearchInput(query="   ")
    with pytest.raises(ValidationError):
        SearchInput(query="fees", top_k=11)


def test_escalation_is_deterministic_and_does_not_contact_a_person() -> None:
    first = escalate_to_human(EscalateInput(reason="Account-specific request"))
    repeated = escalate_to_human(EscalateInput(reason=" account-specific request "))

    assert first == repeated
    assert first.ticket_id.startswith("DEMO-")
    assert first.message == (
        "This is a fictional local escalation stub; no human was contacted."
    )
    assert len(escalate_to_human(EscalateInput(reason="x" * 500)).ticket_id) == 17
    with pytest.raises(ValidationError):
        EscalateInput(reason="   ")
    with pytest.raises(ValidationError):
        EscalateInput(reason="x" * 501)


def test_dispatch_returns_explicit_errors_for_invalid_requests() -> None:
    invalid = dispatch_tool(
        "calculate_fees",
        {"instrument_type": "etf", "side": "buy", "order_value": -1},
    )
    unknown = dispatch_tool("not_a_tool", {})

    assert "error" in invalid
    assert "order_value" in invalid["error"]
    assert unknown == {"error": "Unknown tool: not_a_tool"}
