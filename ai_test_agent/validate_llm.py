#!/usr/bin/env python3
"""Live OpenAI-compatible LLM smoke checks. No device access or secret output."""
import json

from config import LLM_BASE_URL, LLM_MODEL, LLM_PROVIDER
from diagnostics import diagnostic_run
from nim_client import LLMError, chat_completion, create_client


def main():
    with diagnostic_run("provider-check", "validation") as log_path:
        print(f"Provider: {LLM_PROVIDER}")
        print(f"Endpoint: {LLM_BASE_URL}")
        print(f"Model: {LLM_MODEL}")
        print(f"Diagnostic log: {log_path}")
        return _main()


def _main():
    try:
        with create_client() as client:
            message, refused = chat_completion(
                client, [{"role": "user", "content": "Reply with Hello."}], stream=False)
            if refused or not message.get("content"):
                raise LLMError("Chat smoke check returned no text or a refusal.")
            print("PASS: normal chat")

            tool = {"type": "function", "function": {
                "name": "echo", "description": "Return supplied text.",
                "parameters": {"type": "object", "properties": {"text": {"type": "string"}},
                               "required": ["text"]},
            }}
            history = [{"role": "user", "content":
                        "Call the echo tool with text Hello. Do not answer without calling it."}]
            message, refused = chat_completion(
                client, history, tools=[tool], tool_choice="required", stream=False)
            calls = message.get("tool_calls", [])
            if refused or not calls:
                raise LLMError("Model did not call echo. Choose a model supporting OpenAI tool calling.")
            history.append(message)
            for call in calls:
                try:
                    arguments = json.loads(call["function"]["arguments"])
                except (KeyError, TypeError, ValueError):
                    raise LLMError("Tool smoke check returned malformed arguments.") from None
                if (call["function"]["name"] != "echo" or not isinstance(arguments, dict)
                        or not isinstance(arguments.get("text"), str)):
                    raise LLMError("Tool smoke check returned an unexpected tool or arguments.")
                history.append({"role": "tool", "tool_call_id": call["id"],
                                "content": json.dumps(arguments)})
            _, refused = chat_completion(client, history, tools=[tool], stream=False)
            if refused:
                raise LLMError("Model refused the tool-result follow-up.")
            print("PASS: tool calling and tool-result history")
        return 0
    except LLMError as exc:
        print(f"LLM validation failed: {exc}")
        return 1
    except Exception as exc:
        print(f"LLM validation failed ({type(exc).__name__}). No provider body or credentials printed.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
