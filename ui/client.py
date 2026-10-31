"""HTTP client for the backend used by the Streamlit UI."""

import time
from collections.abc import Callable
from typing import Any

import requests


class BackendError(RuntimeError):
    """A backend request failed without exposing a traceback in the UI."""

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


class BackendClient:
    """Small API client; the UI has no imports from the application package."""

    def __init__(
        self,
        base_url: str,
        *,
        session: requests.Session | None = None,
        timeout_seconds: float = 10,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.session = session or requests.Session()
        self.timeout_seconds = timeout_seconds

    def health(self) -> dict[str, Any]:
        """Return the backend health payload."""
        return self._get_json("/health")

    def chat(self, session_id: str, message: str) -> dict[str, Any]:
        """Submit one chat turn to the backend."""
        try:
            response = self.session.post(
                f"{self.base_url}/chat",
                json={"session_id": session_id, "message": message},
                timeout=self.timeout_seconds,
            )
        except requests.RequestException as error:
            raise BackendError(
                "The chat service could not be reached.", retryable=True
            ) from error
        return self._decode(response)

    def latest_evaluation(self) -> dict[str, Any] | None:
        """Return the latest or sample dashboard report, if available."""
        try:
            response = self.session.get(
                f"{self.base_url}/eval/latest",
                timeout=self.timeout_seconds,
            )
        except requests.RequestException as error:
            raise BackendError(
                "The evaluation service could not be reached.", retryable=True
            ) from error
        if response.status_code == 404:
            return None
        return self._decode(response)

    def wait_for_health(
        self,
        *,
        timeout_seconds: float = 90,
        interval_seconds: float = 3,
        on_retry: Callable[[float], None] | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> dict[str, Any]:
        """Retry health checks during backend cold starts, up to the deadline."""
        if timeout_seconds <= 0 or interval_seconds <= 0:
            raise ValueError("Health-check timeout and interval must be positive.")
        deadline = monotonic() + timeout_seconds
        while True:
            try:
                return self.health()
            except BackendError as error:
                if not error.retryable:
                    raise
                remaining = deadline - monotonic()
                if remaining <= 0:
                    raise BackendError(
                        "The server did not become ready within "
                        f"{timeout_seconds:g} seconds."
                    ) from error
                if on_retry is not None:
                    on_retry(timeout_seconds - remaining)
                sleep(min(interval_seconds, remaining))

    def _get_json(self, path: str) -> dict[str, Any]:
        try:
            response = self.session.get(
                f"{self.base_url}{path}",
                timeout=self.timeout_seconds,
            )
        except requests.RequestException as error:
            raise BackendError(
                "The backend could not be reached.", retryable=True
            ) from error
        return self._decode(response)

    @staticmethod
    def _decode(response: requests.Response) -> dict[str, Any]:
        if not response.ok:
            try:
                detail = response.json().get("detail")
            except (requests.exceptions.JSONDecodeError, AttributeError):
                detail = None
            message = (
                detail
                if isinstance(detail, str)
                else f"The backend returned HTTP {response.status_code}."
            )
            raise BackendError(
                message,
                retryable=response.status_code >= 500 or response.status_code == 429,
            )
        try:
            payload = response.json()
        except requests.exceptions.JSONDecodeError as error:
            raise BackendError("The backend returned invalid JSON.") from error
        if not isinstance(payload, dict):
            raise BackendError("The backend returned an unexpected response.")
        return payload
