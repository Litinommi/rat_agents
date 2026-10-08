"""NVIDIA-only transport, safe API errors, and streamed tool-call assembly."""
import httpx
from openai import OpenAI, APIConnectionError, APIStatusError, APITimeoutError

from config import (
    NVIDIA_NIM_API_KEY, NVIDIA_NIM_BASE_URL, NVIDIA_NIM_MODEL,
    NVIDIA_NIM_MAX_TOKENS, NVIDIA_NIM_STREAM, NVIDIA_NIM_TIMEOUT_SECONDS,
    nim_configuration_error,
)


class NIMError(RuntimeError):
    """A user-facing error that never includes provider bodies or credentials."""


def create_client():
    if error := nim_configuration_error():
        raise NIMError(error)
    return OpenAI(
        base_url=NVIDIA_NIM_BASE_URL, api_key=NVIDIA_NIM_API_KEY,
        timeout=NVIDIA_NIM_TIMEOUT_SECONDS, max_retries=2,
    )


def chat_completion(client, messages, *, tools=None, stream=None, on_text=lambda text: None):
    """Return a normalized assistant message after fully receiving the response.

    Tool arguments are never dispatched until the stream has ended successfully.
    Text callbacks receive whole lines, preserving the CLI's existing narration.
    """
    streaming = NVIDIA_NIM_STREAM if stream is None else stream
    options = dict(model=NVIDIA_NIM_MODEL, messages=messages,
                   max_tokens=NVIDIA_NIM_MAX_TOKENS, stream=streaming)
    if tools:
        options.update(tools=tools, tool_choice="auto")
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
        else:
            content, pending, calls_by_index, refused, finish = "", "", {}, False, None
            try:
                for chunk in response:
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
        if calls:
            message["tool_calls"] = calls
        return message, False
    except (APITimeoutError, httpx.TimeoutException):
        raise NIMError("NVIDIA NIM request timed out. Retry or adjust NVIDIA_NIM_TIMEOUT_SECONDS.") from None
    except (APIConnectionError, httpx.TransportError):
        raise NIMError("Cannot connect to NVIDIA NIM. Check your network and retry.") from None
    except APIStatusError as exc:
        status = exc.status_code
        if status in (401, 403):
            message = "NVIDIA NIM authentication failed. Check NVIDIA_NIM_API_KEY and model access."
        elif status == 429:
            message = "NVIDIA NIM rate limit or quota reached. Wait and retry or check your NVIDIA quota."
        elif status in (400, 404, 422):
            message = ("NVIDIA NIM rejected the model or request. Check NVIDIA_NIM_MODEL, token limits, "
                       "and model support for tool calling, streaming, and vision.")
        else:
            message = f"NVIDIA NIM request failed (HTTP {status}). Retry or check NVIDIA service status."
        raise NIMError(message) from None
