"""Record/replay/live LLM facade with a shared daily API-call budget."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from time import sleep
from typing import Any, Protocol

from tenacity import Retrying, retry_if_exception, stop_after_attempt, wait_exponential

from app.agent.prompts import PROMPT_VERSION
from app.config import Settings
from app.llm.cassette import CassetteMiss, CassetteStore
from app.llm.gemini import GeminiClient
from app.llm.types import (
    EmbeddingRequest,
    EmbeddingResult,
    GenerationRequest,
    LLMResult,
)

CASSETTE_DIRECTORY = Path(__file__).resolve().parents[2] / "evals" / "cassettes"


@dataclass(frozen=True)
class CallStat:
    """Successful facade operation metadata used by evaluation traces."""

    kind: str
    purpose: str
    input_tokens: int
    output_tokens: int
    latency_ms: float


class LLMBackend(Protocol):
    """Provider interface used by the facade and deterministic tests."""

    def generate(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        temperature: float,
        max_output_tokens: int,
        response_schema: dict[str, Any] | None,
    ) -> LLMResult: ...

    def embed(
        self,
        *,
        model: str,
        texts: Sequence[str],
        task_type: str,
    ) -> EmbeddingResult: ...


class BudgetExceeded(RuntimeError):
    """Raised before an API request exceeds the configured daily call cap."""


class DailyCallBudget:
    """Thread-safe in-process request counter reset on UTC date boundaries."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self._date = datetime.now(UTC).date()
        self._calls = 0
        self._lock = Lock()

    @property
    def calls(self) -> int:
        with self._lock:
            self._refresh_date()
            return self._calls

    def consume(self) -> None:
        with self._lock:
            self._refresh_date()
            if self._calls >= self.limit:
                raise BudgetExceeded("The configured daily Gemini call cap is reached.")
            self._calls += 1

    def _refresh_date(self) -> None:
        today = datetime.now(UTC).date()
        if today != self._date:
            self._date = today
            self._calls = 0


def _is_retryable(error: BaseException) -> bool:
    status = getattr(error, "status_code", None)
    if status is None:
        status = getattr(error, "code", None)
    response = getattr(error, "response", None)
    if status is None and response is not None:
        status = getattr(response, "status_code", None)
    try:
        status_code = int(status)
    except (TypeError, ValueError):
        return False
    return status_code == 429 or 500 <= status_code <= 599


class LLMClient:
    """The only application-facing route for Gemini generation and embeddings."""

    def __init__(
        self,
        settings: Settings,
        backend: LLMBackend | None = None,
        cassette_store: CassetteStore | None = None,
        budget: DailyCallBudget | None = None,
        sleeper: Callable[[float], None] = sleep,
    ) -> None:
        self.settings = settings
        self.backend = backend or GeminiClient(settings)
        self.cassettes = cassette_store or CassetteStore(CASSETTE_DIRECTORY)
        self.budget = budget or DailyCallBudget(settings.daily_call_cap)
        self._sleeper = sleeper
        self._call_stats: list[CallStat] = []
        self._stats_lock = Lock()

    @property
    def call_stats(self) -> list[CallStat]:
        """Return a snapshot of successfully completed client operations."""
        with self._stats_lock:
            return self._call_stats.copy()

    def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        response_schema: dict[str, Any] | None = None,
        purpose: str = "agent",
        prompt_version: str = PROMPT_VERSION,
    ) -> LLMResult:
        """Generate text/function calls with live, record, or replay behavior."""
        model = self._model_for(purpose)
        request = GenerationRequest(
            model=model,
            messages=messages,
            tools=tools,
            temperature=(
                self.settings.llm_temperature if temperature is None else temperature
            ),
            response_schema=response_schema,
            purpose=purpose,
            prompt_version=prompt_version,
        )
        request_hash = self.cassettes.request_hash(
            request.model_copy(update={"messages": self._normalize_messages(messages)})
        )
        if self.settings.llm_mode == "replay":
            result = self.cassettes.get_generation(request_hash)
            self._record_stats(
                CallStat(
                    kind="generation",
                    purpose=purpose,
                    input_tokens=result.input_tokens,
                    output_tokens=result.output_tokens,
                    latency_ms=result.latency_ms,
                )
            )
            return result
        result = self._with_retries(
            lambda: self.backend.generate(
                model=model,
                messages=messages,
                tools=tools,
                temperature=request.temperature or 0,
                max_output_tokens=self.settings.llm_max_output_tokens,
                response_schema=response_schema,
            )
        )
        if self.settings.llm_mode == "record":
            self.cassettes.put_generation(request_hash, result)
        self._record_stats(
            CallStat(
                kind="generation",
                purpose=purpose,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                latency_ms=result.latency_ms,
            )
        )
        return result

    def embed(
        self,
        texts: Sequence[str],
        *,
        task_type: str,
        purpose: str = "embed",
        prompt_version: str = PROMPT_VERSION,
    ) -> EmbeddingResult:
        """Embed text through this facade, supporting cassette replay and quotas."""
        model = self._model_for(purpose)
        request = EmbeddingRequest(
            model=model,
            texts=list(texts),
            purpose=purpose,
            task_type=task_type,
            prompt_version=prompt_version,
        )
        request_hash = self.cassettes.request_hash(request)
        if self.settings.llm_mode == "replay":
            result = self.cassettes.get_embedding(request_hash)
            self._record_stats(
                CallStat(
                    kind="embedding",
                    purpose=purpose,
                    input_tokens=result.input_tokens,
                    output_tokens=0,
                    latency_ms=result.latency_ms,
                )
            )
            return result
        result = self._with_retries(
            lambda: self.backend.embed(
                model=model,
                texts=texts,
                task_type=task_type,
            )
        )
        if self.settings.llm_mode == "record":
            self.cassettes.put_embedding(request_hash, result)
        self._record_stats(
            CallStat(
                kind="embedding",
                purpose=purpose,
                input_tokens=result.input_tokens,
                output_tokens=0,
                latency_ms=result.latency_ms,
            )
        )
        return result

    def _model_for(self, purpose: str) -> str:
        if purpose in {"agent", "guard"}:
            model = self.settings.gemini_model
        elif purpose == "judge":
            model = self.settings.gemini_judge_model or self.settings.gemini_model
        elif purpose == "embed":
            model = self.settings.gemini_embed_model
        else:
            raise ValueError(f"Unknown LLM purpose: {purpose}")
        if not model:
            raise ValueError(
                f"Configure a model in the environment for purpose {purpose!r}."
            )
        return model

    def _with_retries(self, operation: Any) -> Any:
        retrying = Retrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=8),
            retry=retry_if_exception(_is_retryable),
            reraise=True,
            sleep=self._sleeper,
        )
        for attempt in retrying:
            with attempt:
                self.budget.consume()
                return operation()
        raise RuntimeError("LLM retry loop ended unexpectedly.")

    def _record_stats(self, stats: CallStat) -> None:
        with self._stats_lock:
            self._call_stats.append(stats)

    @staticmethod
    def _normalize_messages(
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return [
            {
                key: value
                for key, value in message.items()
                if key in {"role", "content", "name", "tool_calls"}
            }
            for message in messages
        ]


__all__ = ["BudgetExceeded", "CassetteMiss", "DailyCallBudget", "LLMClient"]
