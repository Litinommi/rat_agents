"""Offline protocol/regression tests using the real SDK with a mock HTTP server."""
import copy
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import httpx
from openai import OpenAI
import config
import nim_client
import llm_client
from test_runner import TestState, ToolBudgetExceeded, build_tool_registry
from tools.simulate import SimulatedState


def completion(content="Hello", calls=None, finish="stop"):
    return {"id": "test", "object": "chat.completion", "created": 0, "model": "test-model",
            "choices": [{"index": 0, "finish_reason": finish,
                         "message": {"role": "assistant", "content": content, "tool_calls": calls}}]}


def tool(name, arguments, ident="call_1"):
    return {"id": ident, "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}


def event(delta, finish=None):
    data = {"id": "test", "object": "chat.completion.chunk", "created": 0, "model": "test-model",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
    return "data: " + json.dumps(data) + "\n\n"


class NIMTests(unittest.TestCase):
    def client(self, handler):
        client = OpenAI(base_url=config.NVIDIA_NIM_BASE_URL, api_key="test-secret", max_retries=0,
                        http_client=httpx.Client(transport=httpx.MockTransport(handler)))
        self.addCleanup(client.close)
        return client

    def run_loop(self, replies, dispatch=None, state=None, vision=False):
        requests = []
        def handler(request):
            requests.append(json.loads(request.content))
            return httpx.Response(200, json=replies.pop(0))
        state = state or SimpleNamespace(device_id="fake", duration_seconds=5, video_url="https://youtu.be/test",
                                        fullscreen=True, stats_for_nerds=True, finished=False)
        dispatch = dispatch or (lambda name, args: {"success": True})
        with patch.object(llm_client, "NVIDIA_NIM_VISION", vision), patch.object(nim_client, "NVIDIA_NIM_STREAM", False):
            outcome = llm_client._run_agent_loop(self.client(handler), "test", state, dispatch,
                                                lambda text: None, lambda name, args: None, lambda name, result: None)
        return outcome, requests

    def test_client_is_nvidia_only(self):
        with patch.object(nim_client, "NVIDIA_NIM_API_KEY", "test-secret"), patch.object(nim_client, "nim_configuration_error", return_value=None):
            with nim_client.create_client() as client:
                self.assertEqual(str(client.base_url), "https://integrate.api.nvidia.com/v1/")
                self.assertEqual(client.api_key, "test-secret")

    def test_configuration_required(self):
        with patch.object(config, "NVIDIA_NIM_API_KEY", ""):
            self.assertIn("NVIDIA_NIM_API_KEY", config.nim_configuration_error())
        with patch.object(config, "NVIDIA_NIM_API_KEY", "test"), patch.object(config, "NVIDIA_NIM_MODEL", ""):
            self.assertIn("NVIDIA_NIM_MODEL", config.nim_configuration_error())

    def test_chat_url_auth_and_text(self):
        def handler(request):
            self.assertEqual(str(request.url), "https://integrate.api.nvidia.com/v1/chat/completions")
            self.assertEqual(request.headers["authorization"], "Bearer test-secret")
            self.assertNotIn("x-api-key", request.headers)
            self.assertNotIn("cache_control", json.loads(request.content))
            return httpx.Response(200, json=completion())
        texts = []
        msg, refused = nim_client.chat_completion(self.client(handler), [{"role": "user", "content": "Hi"}], stream=False, on_text=texts.append)
        self.assertEqual(msg["content"], "Hello")
        self.assertEqual(texts, ["Hello"])
        self.assertFalse(refused)

    def test_stream_fragments_and_interleaved_tools(self):
        chunks = [event({"content": "Hel", "tool_calls": [{"index": 0, "id": "call_0", "function": {"name": "check_", "arguments": "{"}}]}),
                  event({"content": "lo\n", "tool_calls": [{"index": 1, "id": "call_1", "function": {"name": "get_wifi_status", "arguments": "{}"}}]}),
                  event({"tool_calls": [{"index": 0, "function": {"name": "device", "arguments": "}"}}]}),
                  event({}, "tool_calls"), "data: [DONE]\n\n"]
        client = self.client(lambda r: httpx.Response(200, text="".join(chunks), headers={"content-type": "text/event-stream"}))
        texts = []
        msg, _ = nim_client.chat_completion(client, [], stream=True, on_text=texts.append)
        self.assertEqual(texts, ["Hello"])
        self.assertEqual(msg["content"], "Hello\n")
        self.assertEqual([c["function"]["name"] for c in msg["tool_calls"]], ["check_device", "get_wifi_status"])
        self.assertEqual(msg["tool_calls"][0]["function"]["arguments"], "{}")

    def test_incomplete_stream_and_truncation(self):
        client = self.client(lambda r: httpx.Response(200, text=event({"content": "partial"}) + "data: [DONE]\n\n", headers={"content-type": "text/event-stream"}))
        with self.assertRaises(nim_client.NIMError):
            nim_client.chat_completion(client, [], stream=True)
        client = self.client(lambda r: httpx.Response(200, json=completion(calls=[tool("check_device", {})], finish="length")))
        with self.assertRaises(nim_client.NIMError):
            nim_client.chat_completion(client, [], stream=False)

    def test_safe_api_errors(self):
        for status, phrase in [(401, "authentication"), (403, "authentication"), (429, "rate limit"), (404, "model"), (422, "model"), (500, "HTTP 500")]:
            with self.subTest(status=status):
                client = self.client(lambda r: httpx.Response(status, json={"error": {"message": "leaked test-secret"}}))
                with self.assertRaises(nim_client.NIMError) as raised:
                    nim_client.chat_completion(client, [], stream=False)
                self.assertIn(phrase, str(raised.exception))
                self.assertNotIn("test-secret", str(raised.exception))

    def test_network_and_timeout_errors(self):
        for error in [httpx.ConnectError("test-secret"), httpx.ReadTimeout("test-secret")]:
            def handler(request):
                raise error
            with self.assertRaises(nim_client.NIMError) as raised:
                nim_client.chat_completion(self.client(handler), [], stream=False)
            self.assertNotIn("test-secret", str(raised.exception))

    def test_network_failure_mid_stream_never_dispatches(self):
        class BrokenStream(httpx.SyncByteStream):
            def __iter__(self):
                yield event({"tool_calls": [{"index": 0, "id": "call_1", "function": {
                    "name": "toggle_wifi", "arguments": '{"enabled": false}'}}]}).encode()
                raise httpx.ReadError("secret-provider-body")
        client = self.client(lambda r: httpx.Response(200, stream=BrokenStream(),
                                                     headers={"content-type": "text/event-stream"}))
        dispatched = []
        state = SimpleNamespace(device_id="fake", duration_seconds=5, video_url="test", fullscreen=False,
                                stats_for_nerds=False, finished=False)
        with patch.object(nim_client, "NVIDIA_NIM_STREAM", True):
            with self.assertRaises(nim_client.NIMError) as raised:
                llm_client._run_agent_loop(client, "test", state, lambda n, a: dispatched.append(n),
                                          lambda t: None, lambda n, a: None, lambda n, r: None)
        self.assertEqual(dispatched, [])
        self.assertNotIn("secret-provider-body", str(raised.exception))

    def test_api_error_creates_needs_human_report(self):
        import agent
        args = SimpleNamespace(goal="test", video_url="https://youtu.be/test", stats_for_nerds=True,
                               fullscreen=True, simulate=True, inject_failure=False, sim_unfamiliar_ui=False)
        results = {}
        with patch.object(agent, "run_agent_loop", side_effect=nim_client.NIMError("NVIDIA NIM authentication failed.")), \
             patch.object(agent.report_generator, "save_report", return_value="mock-report.json"), \
             patch.object(agent.report_generator, "print_report") as printed, patch("builtins.print"):
            agent.run_device("fake", args, 5, False, results)
        self.assertEqual(results["fake"], "NEEDS_HUMAN")
        self.assertEqual(printed.call_args.args[0]["status"], "NEEDS_HUMAN")

    def test_history_tools_and_vision(self):
        result = {"elements": [], "_image_jpeg_b64": "aW1hZ2U="}
        original = copy.deepcopy(result)
        _, requests = self.run_loop([completion(calls=[tool("get_screen", {})], finish="tool_calls"), completion()],
                                    dispatch=lambda n, a: result, vision=True)
        history = requests[1]["messages"]
        self.assertEqual([m["role"] for m in history], ["system", "user", "assistant", "tool", "user"])
        self.assertEqual(history[3]["tool_call_id"], "call_1")
        self.assertEqual(history[4]["content"][1]["image_url"]["url"], "data:image/jpeg;base64,aW1hZ2U=")
        self.assertEqual(result, original)
        self.assertNotIn("device_id", json.dumps(requests[0]["tools"]))

    def test_text_only_screenshot(self):
        _, requests = self.run_loop([completion(calls=[tool("get_screen", {})], finish="tool_calls"), completion()],
                                    dispatch=lambda n, a: {"elements": ["Button"], "_image_jpeg_b64": "SECRETIMAGE"})
        history = requests[1]["messages"]
        self.assertNotIn("SECRETIMAGE", json.dumps(history))
        self.assertIn("Screenshot omitted", history[-1]["content"])

    def test_invalid_arguments_and_unknown_tool_not_dispatched(self):
        calls = [tool("check_device", []), tool("arbitrary_shell", {})]
        calls[1]["id"] = "call_2"
        dispatched = []
        _, requests = self.run_loop([completion(calls=calls, finish="tool_calls"), completion()], dispatch=lambda n, a: dispatched.append(n))
        self.assertEqual(dispatched, [])
        self.assertEqual([m["role"] for m in requests[1]["messages"][-2:]], ["tool", "tool"])

    def test_budget_and_finished_stop_later_tools(self):
        calls = [tool("check_device", {}), tool("toggle_wifi", {"enabled": False}, "call_2")]
        dispatched = []
        def dispatch(name, args):
            dispatched.append(name)
            raise ToolBudgetExceeded("limit")
        outcome, _ = self.run_loop([completion(calls=calls, finish="tool_calls")], dispatch=dispatch)
        self.assertEqual(outcome, "budget_exceeded")
        self.assertEqual(dispatched, ["check_device"])

    def test_report_finishes_before_later_actions(self):
        state = SimpleNamespace(device_id="fake", duration_seconds=5, video_url="https://youtu.be/test",
                                fullscreen=True, stats_for_nerds=True, finished=False)
        dispatched = []
        def dispatch(name, args):
            dispatched.append(name)
            state.finished = True
            return {"status": "PASS"}
        calls = [tool("generate_report", {"status": "PASS", "summary": "done"}),
                 tool("toggle_wifi", {"enabled": False}, "call_2")]
        outcome, _ = self.run_loop([completion(calls=calls, finish="tool_calls")], dispatch=dispatch, state=state)
        self.assertEqual(outcome, "finished")
        self.assertEqual(dispatched, ["generate_report"])

    def test_streamed_loop_dispatches_complete_arguments(self):
        calls_seen, requests = [], []
        responses = [event({"tool_calls": [{"index": 0, "id": "call_1", "function": {
                         "name": "toggle_wifi", "arguments": '{"enabled":'}}]}),
                     event({"tool_calls": [{"index": 0, "function": {"arguments": "true}"}}]}),
                     event({}, "tool_calls"), "data: [DONE]\n\n"]
        def handler(request):
            requests.append(json.loads(request.content))
            text = "".join(responses) if len(requests) == 1 else event({"content": "Done"}) + event({}, "stop") + "data: [DONE]\n\n"
            return httpx.Response(200, text=text, headers={"content-type": "text/event-stream"})
        state = SimpleNamespace(device_id="fake", duration_seconds=5, video_url="test", fullscreen=False,
                                stats_for_nerds=False, finished=False)
        def dispatch(name, args):
            calls_seen.append((name, args))
            return {"success": True}
        with patch.object(nim_client, "NVIDIA_NIM_STREAM", True):
            llm_client._run_agent_loop(self.client(handler), "test", state, dispatch,
                                      lambda t: None, lambda n, a: None, lambda n, r: None)
        self.assertEqual(calls_seen, [("toggle_wifi", {"enabled": True})])
        self.assertEqual(requests[1]["messages"][-1]["role"], "tool")

    def test_refusal(self):
        result = completion(finish="content_filter")
        outcome, _ = self.run_loop([result])
        self.assertEqual(outcome, "refused")

    def test_simulated_device_workflow(self):
        state = TestState(device_id="fake", goal="test", video_url="https://youtu.be/test", duration_seconds=5)
        sim = SimulatedState(device_id="fake")
        dispatch = build_tool_registry(state, sim)
        sequence = [("check_device", {}), ("get_device_info", {}), ("get_wifi_status", {}),
                    ("launch_youtube", {}), ("open_video_url", {}), ("wait_for_ads", {}),
                    ("enable_stats_for_nerds", {}), ("enter_fullscreen", {}),
                    ("get_playback_status", {"duration_seconds": 5}),
                    ("generate_report", {"status": "PASS", "summary": "Simulated playback verified"})]
        replies = [completion(calls=[tool(n, a, f"call_{i}")], finish="tool_calls") for i, (n, a) in enumerate(sequence)]
        with patch("tools.test_tools.report_generator.save_report", return_value="mock-report.json"):
            outcome, _ = self.run_loop(replies, dispatch=dispatch, state=state)
        self.assertEqual(outcome, "finished")
        self.assertEqual(state.final_report["status"], "PASS")
        json.dumps(state.final_report)  # terminal result must not create circular history
        self.assertEqual(state.tool_call_count, len(sequence))


if __name__ == "__main__":
    unittest.main()
