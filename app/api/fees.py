"""Direct API for the fictional, deterministic DemoBroker fee calculator."""

from fastapi import APIRouter

from app.tools.fees import FeeInput, FeeOutput, calculate_fees

router = APIRouter()


@router.post("/fees", response_model=FeeOutput)
def calculate_demo_fee(payload: FeeInput) -> FeeOutput:
    """Calculate a fictional DemoBroker fee without calling an LLM."""
    return calculate_fees(payload)
