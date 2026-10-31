"""Currency conversion using dated, fictional static rates."""

import csv
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.tools.common import pack_directory


class FxInput(BaseModel):
    """Inputs for conversion at a fictional, static demo rate."""

    model_config = ConfigDict(extra="forbid")

    amount: Decimal = Field(
        ge=0, le=1_000_000_000, decimal_places=2, allow_inf_nan=False
    )
    from_ccy: Literal["EUR", "USD", "GBP", "CHF"]
    to_ccy: Literal["EUR", "USD", "GBP", "CHF"]

    @model_validator(mode="after")
    def validate_supported_pair(self) -> "FxInput":
        """Only allow same-currency or pairs backed by a static EUR rate."""
        if (
            self.from_ccy != self.to_ccy
            and self.from_ccy != "EUR"
            and self.to_ccy != "EUR"
        ):
            raise ValueError("only EUR-based fictional demo rates are available")
        return self


class FxRate(BaseModel):
    """One fictional exchange-rate row from the pack."""

    base: Literal["EUR", "USD", "GBP", "CHF"]
    quote: Literal["EUR", "USD", "GBP", "CHF"]
    rate: Decimal = Field(gt=0, allow_inf_nan=False)
    rate_date: str


class FxOutput(BaseModel):
    """Converted amount, fictional rate, and the rate's date."""

    result: float
    rate: float
    rate_date: str


def _read_rates() -> list[FxRate]:
    path = pack_directory() / "fx_rates.csv"
    with path.open(encoding="utf-8", newline="") as file:
        return [FxRate.model_validate(row) for row in csv.DictReader(file)]


def convert_currency(data: FxInput) -> FxOutput:
    """Convert using a static fictional rate; it is not a live quote."""
    if data.from_ccy == data.to_ccy:
        return FxOutput(result=float(data.amount), rate=1.0, rate_date="not_applicable")

    for row in _read_rates():
        if (row.base, row.quote) == (data.from_ccy, data.to_ccy):
            rate = row.rate
            rate_date = row.rate_date
            break
        if (row.base, row.quote) == (data.to_ccy, data.from_ccy):
            rate = Decimal(1) / row.rate
            rate_date = row.rate_date
            break
    else:
        raise ValueError(f"No fictional demo rate for {data.from_ccy}/{data.to_ccy}")

    result = (data.amount * rate).quantize(Decimal("0.01"))
    return FxOutput(
        result=float(result),
        rate=float(rate.quantize(Decimal("0.000001"))),
        rate_date=rate_date,
    )
