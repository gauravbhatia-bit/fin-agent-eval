"""Rule-first request classification and refusal/escalation routing."""

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.tools.common import load_yaml_model
from app.tools.pack_schemas import PolicyPack

Intent = Literal[
    "ok",
    "advice_request",
    "prediction_request",
    "tax_legal",
    "injection",
    "account_specific",
    "off_topic",
]


class IntentClassification(BaseModel):
    """Constrained output for the LLM intent classifier."""

    model_config = ConfigDict(extra="forbid")

    intent: Intent
    rationale: str = Field(max_length=240)


class RuleMatch(BaseModel):
    """Rule-based intent result and its agent route."""

    intent: Intent
    route: Literal["refuse", "escalate"]


def load_policy() -> PolicyPack:
    """Load and validate the policy file for the configured domain pack."""
    return load_yaml_model("policy.yaml", PolicyPack)


def classify_with_rules(message: str) -> RuleMatch | None:
    """Apply deterministic rules before any model classifier call."""
    policy = load_policy()
    precedence: tuple[tuple[str, Literal["refuse", "escalate"]], ...] = (
        ("injection", "refuse"),
        ("advice_request", "refuse"),
        ("prediction_request", "refuse"),
        ("tax_legal", "refuse"),
        ("account_specific", "escalate"),
        ("complaint", "escalate"),
    )
    for intent, route in precedence:
        for expression in policy.rule_patterns.get(intent, []):
            if re.search(expression, message):
                return RuleMatch(intent=intent, route=route)
    return None


def route_intent(intent: Intent) -> Literal["refuse", "escalate", "agent"]:
    """Map classified intent to the graph's supported route."""
    if intent in {
        "advice_request",
        "prediction_request",
        "tax_legal",
        "injection",
        "off_topic",
    }:
        return "refuse"
    if intent == "account_specific":
        return "escalate"
    return "agent"


def refusal_message(intent: Intent) -> str:
    """Return a short refusal for a classified out-of-scope request."""
    reasons = {
        "advice_request": (
            "I can explain general concepts, but I can't recommend what you should "
            "buy, sell, or hold."
        ),
        "prediction_request": (
            "I can't predict prices, returns, or guarantee an outcome."
        ),
        "tax_legal": "I can't provide tax or legal advice.",
        "injection": (
            "I can't follow instructions that conflict with this assistant's safety "
            "rules."
        ),
        "off_topic": (
            "I can help with general investing education and fictional demo "
            "calculations."
        ),
    }
    return reasons[intent]
