#!/usr/bin/env python3
"""Live NVIDIA NIM smoke checks. No device access and no secret output."""
import json
from nim_client import create_client, chat_completion, NIMError
from diagnostics import diagnostic_run


def main():
    with diagnostic_run("provider-check", "validation") as log_path:
        print(f"Diagnostic log: {log_path}")
        return _main()


def _main():
    try:
        with create_client() as client:
            for streaming in (False, True):
                message, refused = chat_completion(client, [{"role": "user", "content": "Reply with Hello."}], stream=streaming)
                if refused or not message.get("content"):
                    raise NIMError("Chat smoke check returned no text or a refusal.")
                print(f"PASS: NVIDIA NIM {'streaming' if streaming else 'normal'} chat")
            tool = {"type": "function", "function": {
                "name": "echo", "description": "Return supplied text.",
                "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
            }}
            for streaming in (False, True):
                history = [{"role": "user", "content": "Call the echo tool with text Hello. Do not answer without calling it."}]
                message, refused = chat_completion(client, history, tools=[tool], stream=streaming)
                calls = message.get("tool_calls", [])
                if refused or not calls:
                    raise NIMError("Model did not call echo. Choose a model supporting automatic tool calling.")
                history.append(message)
                for call in calls:
                    try:
                        arguments = json.loads(call["function"]["arguments"])
                    except (ValueError, TypeError):
                        raise NIMError("Tool smoke check returned malformed arguments.") from None
                    if call["function"]["name"] != "echo" or not isinstance(arguments, dict) or not isinstance(arguments.get("text"), str):
                        raise NIMError("Tool smoke check returned an unexpected tool or arguments.")
                    history.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(arguments)})
                _, refused = chat_completion(client, history, tools=[tool], stream=streaming)
                if refused:
                    raise NIMError("Model refused the tool-result follow-up.")
                print(f"PASS: NVIDIA NIM {'streaming' if streaming else 'normal'} tool calling and history")
        return 0
    except NIMError as exc:
        print(f"NIM validation failed: {exc}")
        return 1
    except Exception as exc:
        print(f"NIM validation failed ({type(exc).__name__}). No provider body or credentials printed.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
