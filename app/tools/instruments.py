"""Lookup of fictional instruments in the local domain pack."""

import csv
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.tools.common import pack_directory


class InstrumentInput(BaseModel):
    """Symbol to find in the fictional instrument table."""

    model_config = ConfigDict(extra="forbid")

    symbol: str = Field(min_length=1, max_length=40)

    @field_validator("symbol")
    @classmethod
    def normalize_symbol(cls, value: str) -> str:
        """Normalize symbols so lookups are case-insensitive."""
        normalized = value.strip().upper()
        if not normalized:
            raise ValueError("symbol must not be blank")
        return normalized


class InstrumentRow(BaseModel):
    """A fictional instrument record."""

    symbol: str
    name: str
    type: Literal["etf"]
    currency: Literal["EUR", "USD"]
    expense_ratio_pct: float = Field(ge=0, le=10)
    domicile: str
    distribution: Literal["accumulating", "distributing"]


class InstrumentOutput(BaseModel):
    """Instrument lookup result, including a not-found result."""

    found: bool
    symbol: str
    name: str | None = None
    type: str | None = None
    currency: str | None = None
    expense_ratio_pct: float | None = None
    domicile: str | None = None
    distribution: str | None = None
    error: str | None = None


def _read_instruments() -> list[InstrumentRow]:
    path = pack_directory() / "instruments.csv"
    with path.open(encoding="utf-8", newline="") as file:
        return [InstrumentRow.model_validate(row) for row in csv.DictReader(file)]


def lookup_instrument(data: InstrumentInput) -> InstrumentOutput:
    """Look up a fictional instrument; never returns real-market data."""
    for row in _read_instruments():
        if row.symbol == data.symbol:
            return InstrumentOutput(found=True, **row.model_dump())
    return InstrumentOutput(
        found=False,
        symbol=data.symbol,
        error="No matching fictional demo instrument.",
    )
