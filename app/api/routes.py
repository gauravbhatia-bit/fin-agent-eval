"""Chat HTTP endpoint."""

import logging
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request, status

from app.agent.graph import agent_graph
from app.agent.nodes import InputTooLong, initialize_state
from app.api.runtime import ChatRuntime
from app.api.schemas import ChatRequest, ChatResponse
from app.llm.cassette import CassetteMiss
from app.llm.client import BudgetExceeded

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
def chat(
    payload: ChatRequest,
    request: Request,
) -> ChatResponse:
    """Run one guarded conversation turn without persisting user data."""
    runtime: ChatRuntime = request.app.state.runtime
    client_ip = request.client.host if request.client else "unknown"
    if not runtime.rate_limiter.allow(
        client_ip,
        runtime.settings.rate_limit_per_min,
    ):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many chat requests. Please wait a minute and try again.",
        )

    trace_id = uuid4().hex
    state = initialize_state(
        message=payload.message,
        llm_client=runtime.llm_client,
        retriever_factory=runtime.get_retriever,
        settings=runtime.settings,
        history=runtime.session_memory.history(payload.session_id),
        trace_id=trace_id,
    )
    try:
        result = agent_graph.invoke(state)
    except InputTooLong as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(error),
        ) from error
    except BudgetExceeded as error:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="The daily AI request limit has been reached. Please try tomorrow.",
        ) from error
    except CassetteMiss as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No recorded response is available for this request in replay mode.",
        ) from error
    except ValueError as error:
        logger.error(
            "Chat configuration or retrieval failed (%s)", type(error).__name__
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Chat is not configured for this request.",
        ) from error
    except Exception as error:
        provider_code = getattr(error, "code", None)
        if not isinstance(provider_code, int):
            provider_code = None
        provider_status = getattr(error, "status", None)
        if not isinstance(provider_status, str):
            provider_status = None
        logger.error(
            "Chat processing failed (%s; provider_code=%s; provider_status=%s)",
            type(error).__name__,
            provider_code,
            provider_status,
        )
        detail = "The chat request could not be completed. Please try again."
        if provider_code is not None and provider_status:
            if provider_code == 429 or provider_status == "RESOURCE_EXHAUSTED":
                detail = (
                    "Gemini quota or rate limit reached. AI chat is temporarily "
                    "unavailable; the local fictional Fee calculator remains "
                    "available and does not use Gemini."
                )
            else:
                detail = (
                    f"Gemini returned HTTP {provider_code} ({provider_status}). "
                    "Check the model ID, API key access, quota, and request settings."
                )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=detail,
        ) from error

    response = ChatResponse(
        decision=result["decision"],
        answer=result["answer"],
        tool_calls=result.get("tool_calls", []),
        citations=result.get("citations", []),
        disclaimer=result["disclaimer"],
        trace_id=result["trace_id"],
    )
    runtime.session_memory.append(
        payload.session_id,
        payload.message,
        response.answer,
    )
    return response
