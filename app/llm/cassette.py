"""Deterministic request hashing and atomic cassette persistence."""

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from app.llm.types import EmbeddingResult, LLMResult


class CassetteMiss(LookupError):
    """Raised when replay mode has no recorded response for a request."""


class CassetteStore:
    """Store one JSON cassette per request hash."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    @staticmethod
    def request_hash(request: BaseModel) -> str:
        payload = json.dumps(
            request.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get_generation(self, request_hash: str) -> LLMResult:
        path = self._path(request_hash)
        if not path.exists():
            raise CassetteMiss(
                f"No cassette for request {request_hash[:12]}. "
                "Run `make eval-record` to record it before replay."
            )
        return LLMResult.model_validate_json(path.read_text(encoding="utf-8"))

    def put_generation(self, request_hash: str, result: LLMResult) -> None:
        self._write(request_hash, result.model_dump_json())

    def get_embedding(self, request_hash: str) -> EmbeddingResult:
        path = self._path(request_hash)
        if not path.exists():
            raise CassetteMiss(
                f"No embedding cassette for request {request_hash[:12]}. "
                "Run `make eval-record` to record it before replay."
            )
        return EmbeddingResult.model_validate_json(path.read_text(encoding="utf-8"))

    def put_embedding(self, request_hash: str, result: EmbeddingResult) -> None:
        self._write(request_hash, result.model_dump_json())

    def _path(self, request_hash: str) -> Path:
        if len(request_hash) != 64 or any(
            character not in "0123456789abcdef" for character in request_hash
        ):
            raise ValueError("Invalid cassette request hash")
        return self.directory / f"{request_hash}.json"

    def _write(self, request_hash: str, payload: str) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        target = self._path(request_hash)
        temporary_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.directory,
                delete=False,
            ) as file:
                file.write(payload)
                file.flush()
                os.fsync(file.fileno())
                temporary_path = file.name
            os.replace(temporary_path, target)
        finally:
            if temporary_path and os.path.exists(temporary_path):
                os.unlink(temporary_path)


def canonical_hash(payload: dict[str, Any]) -> str:
    """Hash a normalized JSON-compatible request payload."""
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
