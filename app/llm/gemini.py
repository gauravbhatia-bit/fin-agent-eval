"""Google Gemini SDK adapter using explicit, manual function calling."""

import json
from collections.abc import Sequence
from time import perf_counter
from typing import Any

from google import genai
from google.genai import types

from app.config import Settings
from app.llm.types import EmbeddingResult, LLMResult, ToolCall


class GeminiClient:
    """Thin adapter around the Google GenAI SDK; it does not persist requests."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client: genai.Client | None = None

    @property
    def client(self) -> genai.Client:
        """Create the SDK client only when a live call is actually needed."""
        if self._client is None:
            api_key = self.settings.gemini_api_key
            if api_key is None or not api_key.get_secret_value():
                raise ValueError(
                    "GEMINI_API_KEY is required for live or record mode. "
                    "Set it in the local .env file."
                )
            self._client = genai.Client(api_key=api_key.get_secret_value())
        return self._client

    def generate(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        temperature: float,
        max_output_tokens: int,
        response_schema: dict[str, Any] | None,
    ) -> LLMResult:
        """Generate one model response and return any requested tool calls."""
        system_instruction, contents = self._convert_messages(messages)
        sdk_tools: list[types.Tool] | None = None
        if tools:
            sdk_tools = [
                types.Tool(
                    function_declarations=[
                        types.FunctionDeclaration(
                            name=tool["name"],
                            description=tool["description"],
                            parameters_json_schema=tool["parameters"],
                        )
                        for tool in tools
                    ]
                )
            ]
        config = types.GenerateContentConfig(
            system_instruction=system_instruction or None,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            tools=sdk_tools,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True
            ),
            response_mime_type="application/json" if response_schema else None,
            response_json_schema=response_schema,
        )
        started = perf_counter()
        response = self.client.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )
        elapsed_ms = (perf_counter() - started) * 1000
        candidate = response.candidates[0] if response.candidates else None
        parts = candidate.content.parts if candidate and candidate.content else []
        text_chunks: list[str] = []
        calls: list[ToolCall] = []
        for part in parts:
            if part.text:
                text_chunks.append(part.text)
            if part.function_call:
                calls.append(
                    ToolCall(
                        name=part.function_call.name or "",
                        arguments=dict(part.function_call.args or {}),
                    )
                )
        usage = response.usage_metadata
        return LLMResult(
            text="".join(text_chunks),
            tool_calls=calls,
            input_tokens=usage.prompt_token_count if usage else 0,
            output_tokens=usage.candidates_token_count if usage else 0,
            latency_ms=elapsed_ms,
        )

    def embed(
        self,
        *,
        model: str,
        texts: Sequence[str],
        task_type: str,
    ) -> EmbeddingResult:
        """Embed documents or queries through Gemini's documented SDK method."""
        started = perf_counter()
        response = self.client.models.embed_content(
            model=model,
            contents=list(texts),
            config=types.EmbedContentConfig(task_type=task_type),
        )
        embeddings = [list(item.values or []) for item in (response.embeddings or [])]
        return EmbeddingResult(
            embeddings=embeddings,
            input_tokens=0,
            latency_ms=(perf_counter() - started) * 1000,
        )

    @staticmethod
    def _convert_messages(
        messages: list[dict[str, Any]],
    ) -> tuple[str, list[types.Content]]:
        system_parts: list[str] = []
        contents: list[types.Content] = []
        for message in messages:
            role = message.get("role")
            if role == "system":
                system_parts.append(str(message.get("content", "")))
                continue
            if role == "tool":
                name = str(message.get("name", ""))
                raw_response = message.get("content", "{}")
                try:
                    response = (
                        raw_response
                        if isinstance(raw_response, dict)
                        else json.loads(str(raw_response))
                    )
                except json.JSONDecodeError:
                    response = {"result": str(raw_response)}
                contents.append(
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_function_response(
                                name=name,
                                response=response,
                            )
                        ],
                    )
                )
                continue
            role_name = "model" if role == "assistant" else "user"
            parts: list[types.Part] = []
            text = message.get("content")
            if text:
                parts.append(types.Part.from_text(text=str(text)))
            for call in message.get("tool_calls", []):
                parts.append(
                    types.Part.from_function_call(
                        name=str(call["name"]),
                        args=dict(call.get("arguments", {})),
                    )
                )
            if parts:
                contents.append(types.Content(role=role_name, parts=parts))
        return "\n".join(system_parts), contents
