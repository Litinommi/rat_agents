"""Offline NVIDIA transport and step-repair protocol tests."""

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx
from openai import OpenAI

import config
import llm_client
import nim_client
from test_runner import TestState, ToolBudgetExceeded


def completion(content="Hello", calls=None, finish="stop", usage=None, reasoning=None):
    message = {"role": "assistant", "content": content, "tool_calls": calls}
    if reasoning:
        message["reasoning_content"] = reasoning
    body = {"id": "test", "object": "chat.completion", "created": 0, "model": "test-model",
            "choices": [{"index": 0, "finish_reason": finish, "message": message}]}
    if usage:
        body["usage"] = usage
    return body


def tool(name, arguments, ident="call_1"):
    return {"id": ident, "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)}}


def event(delta, finish=None):
    body = {"id": "test", "object": "chat.completion.chunk", "created": 0, "model": "test-model",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
    return "data: " + json.dumps(body) + "\n\n"


class NIMTests(unittest.TestCase):
    def client(self, handler):
        client = OpenAI(base_url=config.NVIDIA_NIM_BASE_URL, api_key="test-secret", max_retries=0,
                        http_client=httpx.Client(transport=httpx.MockTransport(handler)))
        self.addCleanup(client.close)
        return client

    def test_client_is_nvidia_only_and_configuration_is_required(self):
        with patch.object(nim_client, "NVIDIA_NIM_API_KEY", "test-secret"), \
             patch.object(nim_client, "nim_configuration_error", return_value=None), \
             nim_client.create_client() as client:
            self.assertEqual(str(client.base_url), "https://integrate.api.nvidia.com/v1/")
        with patch.object(config, "NVIDIA_NIM_API_KEY", ""):
            self.assertIn("NVIDIA_NIM_API_KEY", config.nim_configuration_error())

    def test_chat_request_uses_sampling_required_tools_usage_and_reasoning(self):
        captured = {}

        def handler(request):
            captured.update(json.loads(request.content))
            return httpx.Response(200, json=completion(
                calls=[tool("step_done", {"fixed": True, "summary": "done"})], finish="tool_calls",
                usage={"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
                reasoning="kept reasoning"))

        usage = {}
        message, refused = nim_client.chat_completion(
            self.client(handler), [{"role": "user", "content": "Hi"}],
            tools=llm_client.schemas_for(["step_done"]), tool_choice="required", stream=False, usage=usage)
        self.assertFalse(refused)
        self.assertEqual(captured["tool_choice"], "required")
        self.assertEqual(captured["temperature"], config.NVIDIA_NIM_TEMPERATURE)
        self.assertEqual(captured["top_p"], config.NVIDIA_NIM_TOP_P)
        self.assertEqual(message["reasoning_content"], "kept reasoning")
        self.assertEqual(usage["total_tokens"], 13)

    def test_stream_fragments_and_interleaved_tools(self):
        chunks = [
            event({"content": "Hel", "tool_calls": [{"index": 0, "id": "call_0",
                   "function": {"name": "check_", "arguments": "{"}}]}),
            event({"content": "lo\n", "tool_calls": [{"index": 1, "id": "call_1",
                   "function": {"name": "get_wifi_status", "arguments": "{}"}}]}),
            event({"tool_calls": [{"index": 0, "function": {"name": "device", "arguments": "}"}}]}),
            event({}, "tool_calls"), "data: [DONE]\n\n",
        ]
        client = self.client(lambda request: httpx.Response(
            200, text="".join(chunks), headers={"content-type": "text/event-stream"}))
        texts = []
        message, _ = nim_client.chat_completion(client, [], stream=True, on_text=texts.append)
        self.assertEqual(texts, ["Hello"])
        self.assertEqual([call["function"]["name"] for call in message["tool_calls"]],
                         ["check_device", "get_wifi_status"])

    def test_incomplete_stream_and_truncation_are_rejected(self):
        client = self.client(lambda request: httpx.Response(
            200, text=event({"content": "partial"}) + "data: [DONE]\n\n",
            headers={"content-type": "text/event-stream"}))
        with self.assertRaises(nim_client.NIMError):
            nim_client.chat_completion(client, [], stream=True)
        client = self.client(lambda request: httpx.Response(
            200, json=completion(calls=[tool("check_device", {})], finish="length")))
        with self.assertRaises(nim_client.NIMError):
            nim_client.chat_completion(client, [], stream=False)

    def test_safe_api_and_network_errors(self):
        for status, phrase in [(401, "authentication"), (429, "rate limit"), (422, "model"), (500, "HTTP 500")]:
            with self.subTest(status=status):
                client = self.client(lambda request, status=status: httpx.Response(
                    status, json={"error": {"message": "leaked test-secret"}}))
                with self.assertRaises(nim_client.NIMError) as raised:
                    nim_client.chat_completion(client, [], stream=False)
                self.assertIn(phrase, str(raised.exception))
                self.assertNotIn("test-secret", str(raised.exception))
        for error in (httpx.ConnectError("secret"), httpx.ReadTimeout("secret")):
            def handler(request, error=error):
                raise error
            with self.assertRaises(nim_client.NIMError):
                nim_client.chat_completion(self.client(handler), [], stream=False)

    def test_network_failure_mid_stream_never_dispatches(self):
        class BrokenStream(httpx.SyncByteStream):
            def __iter__(self):
                yield event({"tool_calls": [{"index": 0, "id": "call_1", "function": {
                    "name": "check_device", "arguments": "{}"}}]}).encode()
                raise httpx.ReadError("secret-provider-body")

        client = self.client(lambda request: httpx.Response(
            200, stream=BrokenStream(), headers={"content-type": "text/event-stream"}))
        state = TestState("fake", "test", "url", 5)
        dispatched = []
        with patch.object(nim_client, "NVIDIA_NIM_STREAM", True), self.assertRaises(nim_client.NIMError):
            llm_client._run_agent_loop(
                client, ["check_device", "step_done"], "system", "start", state,
                lambda name, args: dispatched.append(name), lambda text: None,
                lambda name, args: None, lambda name, result: None)
        self.assertEqual(dispatched, [])

    def test_fix_loop_sends_screenshots_and_stops_at_step_done(self):
        requests = []
        replies = [
            completion(calls=[tool("get_screen", {})], finish="tool_calls"),
            completion(calls=[tool("step_done", {"fixed": True, "summary": "done"}, "call_2")],
                       finish="tool_calls"),
        ]

        def handler(request):
            requests.append(json.loads(request.content))
            return httpx.Response(200, json=replies.pop(0))

        state = TestState("fake", "test", "url", 5)

        def dispatch(name, arguments):
            if name == "get_screen":
                return {"elements": [], "_image_jpeg_b64": "aW1hZ2U="}
            state.fix_result = arguments
            return arguments

        with patch.object(llm_client, "NVIDIA_NIM_VISION", True), \
             patch.object(nim_client, "NVIDIA_NIM_STREAM", False):
            outcome = llm_client._run_agent_loop(
                self.client(handler), ["get_screen", "step_done"], "system", "start", state, dispatch,
                lambda text: None, lambda name, args: None, lambda name, result: None)
        self.assertEqual(outcome, "finished")
        self.assertEqual(requests[1]["messages"][-1]["content"][1]["image_url"]["url"],
                         "data:image/jpeg;base64,aW1hZ2U=")

    def test_budget_stops_the_fix(self):
        replies = [completion(calls=[tool("get_screen", {})], finish="tool_calls")]
        client = self.client(lambda request: httpx.Response(200, json=replies.pop(0)))
        state = TestState("fake", "test", "url", 5)
        outcome = llm_client._run_agent_loop(
            client, ["get_screen"], "system", "start", state,
            lambda name, args: (_ for _ in ()).throw(ToolBudgetExceeded("limit")),
            lambda text: None, lambda name, args: None, lambda name, result: None)
        self.assertEqual(outcome, "budget_exceeded")


if __name__ == "__main__":
    unittest.main()
