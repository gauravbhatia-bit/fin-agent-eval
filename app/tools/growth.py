"""Hypothetical compound-growth calculator."""

import math
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field


class GrowthInput(BaseModel):
    """Bounded inputs for a hypothetical monthly-contribution calculation."""

    model_config = ConfigDict(extra="forbid")

    monthly_amount: Decimal = Field(
        gt=0, le=100_000, decimal_places=2, allow_inf_nan=False
    )
    years: int = Field(ge=1, le=80)
    annual_return_pct: Decimal = Field(ge=-100, le=100, allow_inf_nan=False)
    annual_fee_pct: Decimal = Field(ge=0, le=100, allow_inf_nan=False)


class GrowthOutput(BaseModel):
    """Projected values; the calculation is hypothetical, not a forecast."""

    final_value: float
    total_contributed: float
    total_fees: float
    note: str


class GrowthProjection(BaseModel):
    """Internal result before the public response is rounded."""

    ending_value: Decimal
    contributions: Decimal


def _future_value(
    monthly_amount: Decimal, months: int, annual_factor: Decimal
) -> Decimal:
    """Compute end-of-month contributions using an effective annual factor."""
    factor = float(annual_factor)
    if factor == 1:
        return monthly_amount * months
    if factor == 0:
        return monthly_amount
    monthly_factor = math.pow(factor, 1 / 12)
    monthly_rate = monthly_factor - 1
    value = float(monthly_amount) * (monthly_factor**months - 1) / monthly_rate
    return Decimal(str(value))


def project_growth(data: GrowthInput) -> GrowthOutput:
    """Estimate hypothetical growth; never present this as a forecast."""
    months = data.years * 12
    gross_factor = Decimal(1) + data.annual_return_pct / Decimal(100)
    net_factor = gross_factor * (Decimal(1) - data.annual_fee_pct / Decimal(100))
    contributed = data.monthly_amount * months
    gross_value = _future_value(data.monthly_amount, months, gross_factor)
    net_value = _future_value(data.monthly_amount, months, net_factor)
    cents = Decimal("0.01")
    return GrowthOutput(
        final_value=float(net_value.quantize(cents)),
        total_contributed=float(contributed.quantize(cents)),
        total_fees=float(max(gross_value - net_value, Decimal(0)).quantize(cents)),
        note="Hypothetical, not a forecast. Fees are modeled, not actual charges.",
    )
