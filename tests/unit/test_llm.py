from collections import deque
from pathlib import Path
from typing import Any

import pytest

from app.config import Settings
from app.llm.cassette import CassetteMiss, CassetteStore
from app.llm.client import BudgetExceeded, DailyCallBudget, LLMClient
from app.llm.gemini import GeminiClient
from app.llm.types import EmbeddingResult, GenerationRequest, LLMResult, ToolCall


class FakeBackend:
    def __init__(
        self,
        results: list[LLMResult] | None = None,
        failures: list[Exception] | None = None,
    ) -> None:
        self.results = deque(results or [])
        self.failures = deque(failures or [])
        self.generate_calls: list[dict[str, Any]] = []
        self.embed_calls: list[dict[str, Any]] = []

    def generate(self, **kwargs: Any) -> LLMResult:
        self.generate_calls.append(kwargs)
        if self.failures:
            raise self.failures.popleft()
        return self.results.popleft()

    def embed(self, **kwargs: Any) -> EmbeddingResult:
        self.embed_calls.append(kwargs)
        return EmbeddingResult(embeddings=[[0.1, 0.2] for _ in kwargs["texts"]])


def settings(mode: str = "live") -> Settings:
    return Settings(
        _env_file=None,
        llm_mode=mode,
        gemini_model="gemini-test",
        gemini_judge_model="gemini-judge",
        gemini_embed_model="gemini-embed",
    )


def test_recorded_generation_replays_without_calling_backend(tmp_path: Path) -> None:
    backend = FakeBackend(results=[LLMResult(text="recorded")])
    cassettes = CassetteStore(tmp_path)
    record_client = LLMClient(settings("record"), backend, cassettes)

    assert (
        record_client.generate([{"role": "user", "content": "hello"}]).text
        == "recorded"
    )

    replay_backend = FakeBackend()
    replay_client = LLMClient(settings("replay"), replay_backend, cassettes)
    assert (
        replay_client.generate([{"role": "user", "content": "hello"}]).text
        == "recorded"
    )
    assert replay_backend.generate_calls == []


def test_replay_miss_is_explicit(tmp_path: Path) -> None:
    client = LLMClient(settings("replay"), FakeBackend(), CassetteStore(tmp_path))

    with pytest.raises(CassetteMiss, match="eval-record"):
        client.generate([{"role": "user", "content": "unrecorded"}])


def test_retry_counts_each_attempt_against_daily_budget(tmp_path: Path) -> None:
    class RateLimitError(Exception):
        status_code = 429

    backend = FakeBackend(
        results=[LLMResult(text="ok")],
        failures=[RateLimitError("retry"), RateLimitError("retry")],
    )
    budget = DailyCallBudget(3)
    client = LLMClient(
        settings(),
        backend,
        CassetteStore(tmp_path),
        budget=budget,
        sleeper=lambda _: None,
    )

    assert client.generate([{"role": "user", "content": "hello"}]).text == "ok"
    assert budget.calls == 3
    assert len(backend.generate_calls) == 3


def test_budget_blocks_call_before_backend_request(tmp_path: Path) -> None:
    backend = FakeBackend(results=[LLMResult(text="first")])
    budget = DailyCallBudget(1)
    client = LLMClient(settings(), backend, CassetteStore(tmp_path), budget=budget)
    client.generate([{"role": "user", "content": "first"}])

    with pytest.raises(BudgetExceeded):
        client.generate([{"role": "user", "content": "second"}])
    assert len(backend.generate_calls) == 1


def test_purpose_routes_models_and_embeddings(tmp_path: Path) -> None:
    backend = FakeBackend(
        results=[
            LLMResult(text="guard"),
            LLMResult(text="judge"),
            LLMResult(tool_calls=[ToolCall(name="test", arguments={})]),
        ]
    )
    client = LLMClient(settings(), backend, CassetteStore(tmp_path))
    client.generate([], purpose="guard")
    client.generate([], purpose="judge")
    client.generate([], purpose="agent")
    client.embed(["text"], task_type="RETRIEVAL_QUERY")

    assert [call["model"] for call in backend.generate_calls] == [
        "gemini-test",
        "gemini-judge",
        "gemini-test",
    ]
    assert backend.embed_calls[0]["model"] == "gemini-embed"


def test_cassette_hash_includes_model_and_generation_options() -> None:
    store = CassetteStore(Path("."))
    request = GenerationRequest(
        model="model-a",
        messages=[{"role": "user", "content": "hello"}],
        tools=None,
        temperature=0.2,
        purpose="agent",
        prompt_version="v1",
    )
    alternate = request.model_copy(update={"model": "model-b"})
    changed_temperature = request.model_copy(update={"temperature": 0.8})

    first_hash = store.request_hash(request)
    second_hash = store.request_hash(alternate)
    assert first_hash != second_hash
    assert first_hash != store.request_hash(changed_temperature)


def test_gemini_adapter_parses_text_and_function_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from google.genai import types

    class Models:
        def generate_content(self, **kwargs: Any) -> Any:
            self.kwargs = kwargs
            return type(
                "Response",
                (),
                {
                    "candidates": [
                        type(
                            "Candidate",
                            (),
                            {
                                "content": types.Content(
                                    role="model",
                                    parts=[
                                        types.Part.from_text(text="checking"),
                                        types.Part.from_function_call(
                                            name="calculate_fees",
                                            args={"order_value": 100},
                                        ),
                                    ],
                                )
                            },
                        )()
                    ],
                    "usage_metadata": type(
                        "Usage",
                        (),
                        {"prompt_token_count": 3, "candidates_token_count": 2},
                    )(),
                },
            )()

    models = Models()
    backend = GeminiClient(settings())
    monkeypatch.setattr(
        GeminiClient,
        "client",
        property(lambda _: type("Client", (), {"models": models})()),
    )

    result = backend.generate(
        model="gemini-test",
        messages=[{"role": "user", "content": "fees?"}],
        tools=[{"name": "calculate_fees", "description": "fees", "parameters": {}}],
        temperature=0,
        max_output_tokens=100,
        response_schema=None,
    )

    assert result.text == "checking"
    assert result.tool_calls == [
        ToolCall(name="calculate_fees", arguments={"order_value": 100})
    ]
    assert result.input_tokens == 3
    assert models.kwargs["config"].automatic_function_calling.disable is True


def test_gemini_message_conversion_includes_function_responses() -> None:
    _, contents = GeminiClient._convert_messages(
        [
            {"role": "assistant", "tool_calls": [{"name": "tool", "arguments": {}}]},
            {"role": "tool", "name": "tool", "content": {"value": 2}},
        ]
    )

    assert len(contents) == 2
    assert contents[0].parts[0].function_call.name == "tool"
    assert contents[1].parts[0].function_response.response == {"value": 2}
