"""NVIDIA-only transport, safe API errors, and streamed tool-call assembly."""
import json
import time
import uuid

import httpx
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI

from config import (
    NVIDIA_NIM_API_KEY,
    NVIDIA_NIM_BASE_URL,
    NVIDIA_NIM_MAX_TOKENS,
    NVIDIA_NIM_MODEL,
    NVIDIA_NIM_STREAM,
    NVIDIA_NIM_TEMPERATURE,
    NVIDIA_NIM_TIMEOUT_SECONDS,
    NVIDIA_NIM_TOP_P,
    nim_configuration_error,
)
from diagnostics import record, record_exception, sanitize


class NIMError(RuntimeError):
    """A user-facing error that never includes provider bodies or credentials."""


def _request_log(request):
    request.extensions["diagnostic_started"] = time.monotonic()
    record("nim_http_attempt", method=request.method, endpoint=str(request.url.copy_with(query=None)),
           retry_number=request.headers.get("x-stainless-retry-count", "0"))


def _response_log(response):
    request_id = next((response.headers.get(k) for k in ("x-request-id", "request-id", "nvcf-reqid", "nvcf-request-id")
                       if response.headers.get(k)), None)
    details = {}
    if response.status_code >= 400:
        response.read()
        try:
            body = response.json()
            error = body.get("error", body) if isinstance(body, dict) else {}
            if isinstance(error, dict):
                details = {k: error[k] for k in ("type", "code", "message", "detail") if k in error}
            elif isinstance(error, str):
                details = {"message": error}
            if not details and isinstance(body, dict):
                details = {k: body[k] for k in ("detail", "message", "title") if k in body}
        except ValueError:
            details = {"message": response.text[:2000]}
    record("nim_http_response", status=response.status_code, request_id=request_id,
           retry_after=response.headers.get("retry-after"), error=details,
           duration_seconds=round(time.monotonic() - response.request.extensions.get("diagnostic_started", time.monotonic()), 3))


def create_client():
    if error := nim_configuration_error():
        raise NIMError(error)
    return OpenAI(
        base_url=NVIDIA_NIM_BASE_URL, api_key=NVIDIA_NIM_API_KEY,
        timeout=NVIDIA_NIM_TIMEOUT_SECONDS, max_retries=2,
        http_client=httpx.Client(event_hooks={"request": [_request_log], "response": [_response_log]}),
    )


def chat_completion(client, messages, *, tools=None, tool_choice="auto", stream=None,
                    on_text=lambda text: None, usage=None):
    """Return a normalized assistant message after fully receiving the response.

    Tool arguments are never dispatched until the stream has ended successfully.
    Text callbacks receive whole lines, preserving the CLI's existing narration.
    """
    streaming = NVIDIA_NIM_STREAM if stream is None else stream
    options = {"model": NVIDIA_NIM_MODEL, "messages": messages,
               "temperature": NVIDIA_NIM_TEMPERATURE, "top_p": NVIDIA_NIM_TOP_P,
               "max_tokens": NVIDIA_NIM_MAX_TOKENS, "stream": streaming}
    if streaming:
        options["stream_options"] = {"include_usage": True}
    if tools:
        options.update(tools=tools, tool_choice=tool_choice)
    completion_id = uuid.uuid4().hex[:12]
    started = time.monotonic()
    record("nim_completion_started", completion_id=completion_id, model=NVIDIA_NIM_MODEL,
           streaming=streaming, max_tokens=NVIDIA_NIM_MAX_TOKENS, message_count=len(messages),
           message_roles=[m["role"] for m in messages], tool_count=len(tools or []),
           history_bytes=len(json.dumps(messages).encode()))
    try:
        response = client.chat.completions.create(**options)
        if not streaming:
            if not response.choices:
                raise NIMError("NVIDIA NIM returned no completion choices.")
            choice = response.choices[0]
            message = choice.message
            content = message.content or ""
            calls = [call.model_dump(exclude_none=True) for call in message.tool_calls or []]
            refused = bool(getattr(message, "refusal", None))
            finish = choice.finish_reason
            reasoning = getattr(message, "reasoning_content", None) or getattr(message, "reasoning", None)
            response_usage = getattr(response, "usage", None)
        else:
            content, reasoning, pending, calls_by_index, refused, finish = "", "", "", {}, False, None
            response_usage = None
            try:
                for chunk in response:
                    if getattr(chunk, "usage", None):
                        response_usage = chunk.usage
                    if not chunk.choices:
                        continue  # e.g. usage-only events
                    choice = chunk.choices[0]
                    delta = choice.delta
                    finish = choice.finish_reason or finish
                    refused = refused or bool(getattr(delta, "refusal", None))
                    if delta.content:
                        content += delta.content
                        pending += delta.content
                        while "\n" in pending:
                            line, pending = pending.split("\n", 1)
                            if line.strip():
                                on_text(line)
                    reasoning += (getattr(delta, "reasoning_content", None)
                                  or getattr(delta, "reasoning", None) or "")
                    for part in delta.tool_calls or []:
                        call = calls_by_index.setdefault(part.index, {
                            "id": "", "type": "function", "function": {"name": "", "arguments": ""},
                        })
                        if part.id:
                            call["id"] += part.id
                        if part.function:
                            call["function"]["name"] += part.function.name or ""
                            call["function"]["arguments"] += part.function.arguments or ""
                if pending.strip():
                    on_text(pending)
            finally:
                response.close()
            calls = [calls_by_index[index] for index in sorted(calls_by_index)]
            if finish is None:
                raise NIMError("NVIDIA NIM stream ended before completion. No tool calls were executed.")
        record("nim_completion_received", completion_id=completion_id, finish_reason=finish,
               duration_seconds=round(time.monotonic() - started, 3),
               tool_names=[call["function"].get("name") for call in calls], text_chars=len(content),
               usage=response_usage.model_dump() if response_usage else None)
        if usage is not None and response_usage:
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                usage[key] = usage.get(key, 0) + (getattr(response_usage, key, 0) or 0)
        if finish == "length":
            raise NIMError("NVIDIA NIM response was truncated. Adjust NVIDIA_NIM_MAX_TOKENS within the model's limit.")
        if refused or finish == "content_filter":
            return {"role": "assistant", "content": content or None}, True
        if calls and (any(not call.get("id") or not call["function"].get("name") for call in calls)
                      or len({call["id"] for call in calls}) != len(calls)):
            raise NIMError("NVIDIA NIM returned incomplete or duplicate tool calls. No tools were executed.")
        if not streaming and content.strip():
            on_text(content.strip())
        message = {"role": "assistant", "content": content or None}
        if reasoning:
            message["reasoning_content"] = reasoning
        if calls:
            message["tool_calls"] = calls
        return message, False
    except NIMError as exc:
        record_exception("nim_completion_error", exc)
        raise
    except (APITimeoutError, httpx.TimeoutException) as exc:
        record("nim_completion_error", completion_id=completion_id, category="timeout", exception_type=type(exc).__name__)
        raise NIMError("NVIDIA NIM request timed out. Retry or adjust NVIDIA_NIM_TIMEOUT_SECONDS.") from None
    except (APIConnectionError, httpx.TransportError) as exc:
        record("nim_completion_error", completion_id=completion_id, category="network", exception_type=type(exc).__name__)
        raise NIMError("Cannot connect to NVIDIA NIM. Check your network and retry.") from None
    except APIStatusError as exc:
        status = exc.status_code
        request_id = getattr(exc, "request_id", None) or exc.response.headers.get("nvcf-reqid")
        # Logs retain only allowlisted provider error fields, never request headers or full payloads.
        body = exc.body if isinstance(exc.body, dict) else {}
        error = body.get("error", body)
        details = {k: error[k] for k in ("type", "code", "message", "detail") if k in error} if isinstance(error, dict) else {}
        record("nim_completion_error", completion_id=completion_id, status=status, request_id=request_id, error=details)
        if status in (401, 403):
            message = "NVIDIA NIM authentication failed. Check NVIDIA_NIM_API_KEY and model access."
        elif status == 429:
            message = "NVIDIA NIM rate limit or quota reached. Wait and retry or check your NVIDIA quota."
        elif status in (400, 404, 422):
            message = ("NVIDIA NIM rejected the model or request. Check NVIDIA_NIM_MODEL, token limits, "
                       "and model support for tool calling, streaming, and vision.")
        else:
            message = f"NVIDIA NIM request failed (HTTP {status}). Retry or check NVIDIA service status."
        if request_id:
            message += f" Request ID: {sanitize(request_id)}."
        raise NIMError(message) from None
