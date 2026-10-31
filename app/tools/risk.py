"""Education-only position-size calculator."""

from decimal import ROUND_FLOOR, Decimal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class PositionSizeInput(BaseModel):
    """Inputs for a bounded, long-position risk illustration."""

    model_config = ConfigDict(extra="forbid")

    account_value: Decimal = Field(
        gt=0, le=100_000_000, decimal_places=2, allow_inf_nan=False
    )
    risk_pct: Decimal = Field(ge=0, le=5, allow_inf_nan=False)
    entry_price: Decimal = Field(
        gt=0, le=10_000_000, decimal_places=6, allow_inf_nan=False
    )
    stop_price: Decimal = Field(
        gt=0, le=10_000_000, decimal_places=6, allow_inf_nan=False
    )

    @model_validator(mode="after")
    def validate_prices(self) -> "PositionSizeInput":
        """Only calculate a long position with a lower stop price."""
        if self.stop_price >= self.entry_price:
            raise ValueError("stop_price must be below entry_price for this calculator")
        return self


class PositionSizeOutput(BaseModel):
    """Maximum whole units and modeled loss at the stop price."""

    units: int
    max_loss: float
    note: str


def position_size(data: PositionSizeInput) -> PositionSizeOutput:
    """Illustrate a risk-based long position size; not investment advice."""
    risk_budget = data.account_value * data.risk_pct / Decimal(100)
    per_unit_loss = data.entry_price - data.stop_price
    units = int((risk_budget / per_unit_loss).to_integral_value(rounding=ROUND_FLOOR))
    loss = (Decimal(units) * per_unit_loss).quantize(Decimal("0.01"))
    return PositionSizeOutput(
        units=units,
        max_loss=float(loss),
        note="Education-only estimate; excludes fees, slippage, and price gaps.",
    )
