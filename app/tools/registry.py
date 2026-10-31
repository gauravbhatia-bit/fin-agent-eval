"""Pydantic-backed registry and safe dispatcher for deterministic tools."""

from collections.abc import Callable
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from app.retrieval.base import Retriever
from app.tools import escalate, fees, fx, growth, instruments, risk, search

InputModel = TypeVar("InputModel", bound=BaseModel)
ToolFunction = Callable[[BaseModel], BaseModel]


class ToolSpec(BaseModel):
    """Schema and callable for one tool exposed to the agent."""

    name: str
    description: str
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    function: ToolFunction

    model_config = {"arbitrary_types_allowed": True}


TOOLS: dict[str, ToolSpec] = {
    "calculate_fees": ToolSpec(
        name="calculate_fees",
        description=fees.FeeInput.__doc__ or "Calculate fictional demo fees.",
        input_model=fees.FeeInput,
        output_model=fees.FeeOutput,
        function=fees.calculate_fees,
    ),
    "project_growth": ToolSpec(
        name="project_growth",
        description=growth.GrowthInput.__doc__ or "Calculate hypothetical growth.",
        input_model=growth.GrowthInput,
        output_model=growth.GrowthOutput,
        function=growth.project_growth,
    ),
    "convert_currency": ToolSpec(
        name="convert_currency",
        description=fx.FxInput.__doc__ or "Convert using fictional static rates.",
        input_model=fx.FxInput,
        output_model=fx.FxOutput,
        function=fx.convert_currency,
    ),
    "lookup_instrument": ToolSpec(
        name="lookup_instrument",
        description=instruments.InstrumentInput.__doc__
        or "Look up a fictional symbol.",
        input_model=instruments.InstrumentInput,
        output_model=instruments.InstrumentOutput,
        function=instruments.lookup_instrument,
    ),
    "position_size": ToolSpec(
        name="position_size",
        description=risk.PositionSizeInput.__doc__ or "Estimate education-only sizing.",
        input_model=risk.PositionSizeInput,
        output_model=risk.PositionSizeOutput,
        function=risk.position_size,
    ),
    "search_knowledge": ToolSpec(
        name="search_knowledge",
        description=search.SearchInput.__doc__ or "Search the local knowledge pack.",
        input_model=search.SearchInput,
        output_model=search.SearchOutput,
        function=search.search_knowledge,
    ),
    "escalate_to_human": ToolSpec(
        name="escalate_to_human",
        description=escalate.EscalateInput.__doc__ or "Create a fake local ticket.",
        input_model=escalate.EscalateInput,
        output_model=escalate.EscalateOutput,
        function=escalate.escalate_to_human,
    ),
}


class ToolError(BaseModel):
    """Explicit, user-input error returned instead of raising to the agent."""

    error: str


def dispatch_tool(
    name: str,
    arguments: dict[str, object],
    retriever: Retriever | None = None,
) -> dict[str, object]:
    """Validate and invoke a registered tool; report invalid requests explicitly."""
    spec = TOOLS.get(name)
    if spec is None:
        return ToolError(error=f"Unknown tool: {name}").model_dump()
    try:
        validated = spec.input_model.model_validate(arguments)
    except ValidationError as error:
        details = "; ".join(
            f"{'.'.join(str(part) for part in item['loc'])}: {item['msg']}"
            for item in error.errors()
        )
        return ToolError(error=details).model_dump()
    if name == "search_knowledge" and retriever is not None:
        search_input = search.SearchInput.model_validate(validated.model_dump())
        return search.SearchOutput(
            results=retriever.search(search_input.query, search_input.top_k)
        ).model_dump(mode="json")
    return spec.function(validated).model_dump(mode="json")


def tool_schemas() -> list[dict[str, object]]:
    """Return deterministic JSON-schema definitions for agent tool registration."""
    return [
        {
            "name": spec.name,
            "description": spec.description,
            "parameters": spec.input_model.model_json_schema(),
        }
        for spec in TOOLS.values()
    ]
