"""In-process chat state with bounded session memory and request throttling."""

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from threading import Lock
from time import monotonic

from app.config import Settings
from app.llm.client import LLMClient
from app.retrieval.base import Retriever


class SessionMemory:
    """Keep up to six recent turns per session for at most 30 minutes."""

    def __init__(
        self,
        ttl_seconds: int = 30 * 60,
        max_turns: int = 6,
        max_chars: int = 12_000,
        max_sessions: int = 1_000,
    ) -> None:
        if min(ttl_seconds, max_turns, max_chars, max_sessions) < 1:
            raise ValueError("session memory limits must be positive")
        self.ttl_seconds = ttl_seconds
        self.max_turns = max_turns
        self.max_chars = max_chars
        self.max_sessions = max_sessions
        self._sessions: dict[str, tuple[float, deque[list[dict[str, str]]]]] = {}
        self._lock = Lock()

    def history(self, session_id: str) -> list[dict[str, str]]:
        """Return a copy of recent history, expiring stale session state."""
        now = monotonic()
        with self._lock:
            self._expire(now)
            session = self._sessions.get(session_id)
            if session is None:
                return []
            turns = session[1]
            self._sessions[session_id] = (now, turns)
            return [message.copy() for turn in turns for message in turn]

    def append(self, session_id: str, user_message: str, answer: str) -> None:
        """Store a completed turn and evict oldest turns to satisfy limits."""
        now = monotonic()
        turn = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": answer},
        ]
        with self._lock:
            self._expire(now)
            existing = self._sessions.get(session_id)
            turns = existing[1] if existing is not None else deque()
            turns.append(turn)
            while (
                len(turns) > self.max_turns or self._char_count(turns) > self.max_chars
            ):
                turns.popleft()
            if turns:
                self._sessions[session_id] = (now, turns)
                if len(self._sessions) > self.max_sessions:
                    oldest = min(
                        self._sessions,
                        key=lambda key: self._sessions[key][0],
                    )
                    del self._sessions[oldest]

    def _expire(self, now: float) -> None:
        expired = [
            session_id
            for session_id, (updated_at, _) in self._sessions.items()
            if now - updated_at >= self.ttl_seconds
        ]
        for session_id in expired:
            del self._sessions[session_id]

    @staticmethod
    def _char_count(turns: deque[list[dict[str, str]]]) -> int:
        return sum(len(message["content"]) for turn in turns for message in turn)


class IpRateLimiter:
    """Simple in-process sliding-window limiter keyed by client IP."""

    def __init__(self, max_tracked_ips: int = 10_000) -> None:
        if max_tracked_ips < 1:
            raise ValueError("max_tracked_ips must be positive")
        self.max_tracked_ips = max_tracked_ips
        self._requests: dict[str, deque[float]] = {}
        self._lock = Lock()

    def allow(self, client_ip: str, limit: int) -> bool:
        """Record one request if the IP has capacity in the last minute."""
        now = monotonic()
        with self._lock:
            if (
                client_ip not in self._requests
                and len(self._requests) >= self.max_tracked_ips
            ):
                self._expire(now)
                if len(self._requests) >= self.max_tracked_ips:
                    return False
            requests = self._requests.setdefault(client_ip, deque())
            while requests and now - requests[0] >= 60:
                requests.popleft()
            if len(requests) >= limit:
                return False
            requests.append(now)
            return True

    def _expire(self, now: float) -> None:
        expired = [
            client_ip
            for client_ip, requests in self._requests.items()
            if not requests or now - requests[-1] >= 60
        ]
        for client_ip in expired:
            del self._requests[client_ip]


@dataclass
class ChatRuntime:
    """Application-scoped state needed to serve chat requests."""

    settings: Settings
    llm_client: LLMClient
    session_memory: SessionMemory
    rate_limiter: IpRateLimiter
    retriever: Retriever | None = None
    _retriever_lock: Lock = field(default_factory=Lock)

    def get_retriever(self) -> Retriever:
        """Construct the hybrid retriever lazily, after guard routing allows it."""
        if self.retriever is not None:
            return self.retriever
        with self._retriever_lock:
            if self.retriever is None:
                from app.retrieval.hybrid import HybridRetriever

                model = self.settings.gemini_embed_model
                if not model:
                    raise ValueError(
                        "GEMINI_EMBED_MODEL must be configured for knowledge retrieval."
                    )

                def embed(
                    texts: Sequence[str],
                    task_type: str,
                ) -> list[list[float]]:
                    return self.llm_client.embed(
                        texts,
                        task_type=task_type,
                        purpose="embed",
                    ).embeddings

                self.retriever = HybridRetriever(
                    embedder=embed,
                    embedding_model=model,
                )
        return self.retriever
