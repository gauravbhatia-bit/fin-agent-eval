"""Fee calculator using the fictional DemoBroker fee schedule."""

from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.tools.common import load_yaml_model


class FeeInput(BaseModel):
    """Inputs for the fictional DemoBroker fee calculator."""

    model_config = ConfigDict(extra="forbid")

    instrument_type: Literal["etf", "stock", "savings_plan"]
    side: Literal["buy", "sell"]
    order_value: Decimal = Field(
        gt=0, le=1_000_000, decimal_places=2, allow_inf_nan=False
    )


class FeeBreakdown(BaseModel):
    """One component of the fictional fee estimate."""

    label: str
    amount: float


class FeeOutput(BaseModel):
    """Fictional fee estimate and its component breakdown."""

    fee: float
    fee_pct: float
    breakdown: list[FeeBreakdown]


class FeeRule(BaseModel):
    """Validated fee rule loaded from the fictional pack."""

    fixed_fee: Decimal = Field(ge=0)
    rate_pct: Decimal = Field(ge=0)
    min_fee: Decimal = Field(ge=0)
    max_fee: Decimal = Field(gt=0)


class FeeSchedule(BaseModel):
    """Validated fee schedule loaded from the fictional pack."""

    model_config = ConfigDict(extra="forbid")

    fictional_data: Literal[True]
    broker_name: Literal["DemoBroker"]
    currency: Literal["EUR"]
    schedule_version: int = Field(ge=1)
    rules: dict[Literal["etf", "stock", "savings_plan"], FeeRule]


def calculate_fees(data: FeeInput) -> FeeOutput:
    """Calculate the fictional fee for one order; this is not broker pricing."""
    schedule = load_yaml_model("fees.yaml", FeeSchedule)
    rule = schedule.rules[data.instrument_type]
    variable = data.order_value * rule.rate_pct / Decimal(100)
    total = min(
        max(rule.fixed_fee + variable, rule.min_fee),
        rule.max_fee,
    )
    fee = total.quantize(Decimal("0.01"))
    fee_pct = (fee / data.order_value * Decimal(100)).quantize(Decimal("0.0001"))
    fixed_component = rule.fixed_fee.quantize(Decimal("0.01"))
    variable_component = variable.quantize(Decimal("0.01"))
    adjustment = fee - fixed_component - variable_component
    return FeeOutput(
        fee=float(fee),
        fee_pct=float(fee_pct),
        breakdown=[
            FeeBreakdown(label="fixed", amount=float(fixed_component)),
            FeeBreakdown(label="variable", amount=float(variable_component)),
            FeeBreakdown(
                label="minimum_or_maximum_adjustment", amount=float(adjustment)
            ),
        ],
    )
