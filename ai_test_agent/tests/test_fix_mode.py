"""Focused regressions for step-scoped AI repair mode."""

import json
import sys
import tempfile
import threading
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import llm_client
import play_youtube
from test_runner import TestState, build_tool_registry
from tools import test_tools, ui_tools
from tools.simulate import SimulatedState, sim_ui_tools


def _call(name, arguments=None, ident="call_1"):
    return {"id": ident, "type": "function", "function": {
        "name": name, "arguments": json.dumps(arguments or {})}}


class FixModeTests(unittest.TestCase):
    def state(self):
        return TestState("fake", "test", "https://youtu.be/test", 5)

    def options(self, **overrides):
        values = {"ai_fallback": False, "stats_for_nerds": True, "fullscreen": True,
                  "keep_playing": True, "simulate": True, "inject_failure": False,
                  "sim_unfamiliar_ui": False, "sim_stats_setting_off": False}
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_screen_shows_unlabelled_switch_state(self):
        xml = ('<hierarchy><node class="android.widget.Switch" checkable="true" checked="false" '
               'clickable="true" bounds="[900,100][1050,180]" package="com.google.android.youtube"/></hierarchy>')
        driver = SimpleNamespace(
            dump_hierarchy=lambda compressed=False: xml,
            window_size=lambda: (1080, 2400),
            app_current=lambda: {"activity": ".Settings"},
        )
        with patch.object(ui_tools, "_driver", return_value=driver):
            result = ui_tools.get_screen("fake", include_screenshot=False)
        self.assertEqual(result["elements"][0]["switch"], "off")

    def test_already_on_switch_row_is_not_tapped(self):
        sim = SimulatedState("fake", stats_setting_off=True)
        sim.screen = "general"
        sim.stats_setting_enabled = True
        state = self.state()
        dispatch = build_tool_registry(state, sim)
        screen = dispatch("get_screen", {"include_screenshot": False})
        result = dispatch("tap_element", {"index": 0})
        self.assertTrue(result["already_on"])
        self.assertTrue(sim.stats_setting_enabled)
        self.assertEqual(result["recipe_step"]["action"], "switch_on")
        self.assertEqual(screen["elements"][0]["switch"], "on")

    def test_raw_simulator_switch_can_reproduce_double_tap_bug(self):
        sim = SimulatedState("fake", stats_setting_off=True, screen="general")
        tap = sim_ui_tools(sim)["tap_element"]
        tap(0)
        tap(0)
        self.assertFalse(sim.stats_setting_enabled)

    def test_actions_return_new_screen(self):
        sim = SimulatedState("fake")
        state = self.state()
        dispatch = build_tool_registry(state, sim)
        dispatch("get_screen", {"include_screenshot": False})
        result = dispatch("tap_element", {"index": 1})
        self.assertTrue(result["screen_changed"])
        self.assertIn("elements", result)

    def test_registry_rejects_tool_outside_allowed_list(self):
        dispatch = build_tool_registry(self.state(), SimulatedState("fake"), allowed={"get_screen"}, budget=2)
        result = dispatch("launch_youtube", {})
        self.assertIn("not available", result["error"])

    def test_only_first_action_runs_and_old_screens_shrink(self):
        replies = iter([
            ({"role": "assistant", "content": None, "tool_calls": [
                _call("get_screen"), _call("press_key", {"key": "back"}, "call_2")]}, False),
            ({"role": "assistant", "content": None, "tool_calls": [_call("step_done", {"fixed": True, "summary": "done"}, "call_3")]}, False),
        ])
        calls = []
        state = self.state()

        def dispatch(name, arguments):
            calls.append(name)
            if name == "get_screen":
                return {"success": True, "elements": [{"i": 0}], "recipe_step": {"action": "tap"}}
            state.fix_result = arguments
            return arguments

        histories = []

        def fake_completion(client, messages, **kwargs):
            histories.append(json.loads(json.dumps(messages)))
            return next(replies)

        with patch.object(llm_client, "chat_completion", side_effect=fake_completion):
            llm_client._run_agent_loop(None, ["get_screen", "press_key", "step_done"], "system", "start",
                                       state, dispatch, lambda text: None, lambda name, args: None,
                                       lambda name, result: None)
        self.assertEqual(calls, ["get_screen", "step_done"])
        self.assertIn("not executed: one action per turn", histories[1][-1]["content"])
        shrunk = llm_client._shrink_old_screens([
            {"role": "tool", "content": json.dumps({"success": True, "elements": [1],
                                                        "recipe_step": {"action": "tap"}})},
            {"role": "tool", "content": json.dumps({"success": True, "elements": [2]})},
        ])
        old_screen = json.loads(shrunk[0]["content"])
        self.assertNotIn("elements", old_screen)
        self.assertIn("recipe_step", old_screen)

    def test_recipe_failure_location_is_remembered(self):
        with patch.object(ui_tools, "candidate_recipes", return_value=[("profile", [{"action": "tap", "match": {"text": "x"}}])]), \
             patch.object(ui_tools, "device_profile_key", return_value="own"), \
             patch.object(ui_tools, "_run_steps", return_value={"success": False, "failed_step": 0, "error": "missing"}):
            ui_tools.try_recipes("fake", "dismiss_popup", lambda: False)
        failures = ui_tools.recipe_failures("fake", "dismiss_popup")
        self.assertEqual(failures["profile"]["failed_step"], 0)

    def test_recipe_is_skipped_after_two_consecutive_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            recipes = Path(directory) / "recipes.json"
            recipes.write_text(json.dumps({"profile": {"dismiss_popup": [
                {"action": "tap", "match": {"text": "Dismiss"}}]}}))
            with patch.object(ui_tools, "RECIPES_FILE", recipes), \
                 patch.object(ui_tools, "device_profile_key", return_value="profile"), \
                 patch.object(ui_tools, "_run_steps", return_value={"success": False, "failed_step": 0}):
                ui_tools.try_recipes("fake", "dismiss_popup", lambda: False)
                ui_tools.try_recipes("fake", "dismiss_popup", lambda: False)
                self.assertEqual(ui_tools.candidate_recipes("fake", "dismiss_popup"), [])
            stats = json.loads(recipes.read_text())["profile"]["_recipe_stats"]["dismiss_popup"]
            self.assertEqual(stats["failures"], 2)

    def test_recipe_notes_are_required(self):
        result = ui_tools.save_recipe("fake", "dismiss_popup", [], notes="")
        self.assertIn("notes is required", result["error"])

    def test_same_profile_group_takes_turns_and_second_skips_ai(self):
        learned = threading.Event()
        entered_ai = threading.Event()
        release_ai = threading.Event()
        contexts = []
        for device in ("one", "two"):
            state = TestState(device, "test", "https://youtu.be/test", 5)
            sim = SimulatedState(device)
            contexts.append(play_youtube.RunContext(
                state, sim, build_tool_registry(state, sim), self.options(), [], lambda message: None))
        step = play_youtube.Step("stats", "visible",
                                 lambda: {"success": learned.is_set()}, learned.is_set,
                                 {"enable_stats_for_nerds"})
        results = {}

        def agent(**kwargs):
            entered_ai.set()
            release_ai.wait(2)
            learned.set()
            kwargs["state"].fix_result = {"fixed": True, "summary": "learned", "needs_human": False}
            return "finished"

        with patch.object(play_youtube, "run_agent_loop", side_effect=agent) as agent_loop:
            first = threading.Thread(target=lambda: results.setdefault(
                "one", play_youtube.fix_step("one", step, {}, [], contexts[0])))
            second = threading.Thread(target=lambda: results.setdefault(
                "two", play_youtube.fix_step("two", step, {}, [], contexts[1])))
            first.start()
            entered_ai.wait(2)
            second.start()
            release_ai.set()
            first.join()
            second.join()
        self.assertEqual(agent_loop.call_count, 1)
        self.assertFalse(results["two"]["ai_called"])

    def test_fix_limit_and_ai_terminal_result(self):
        state = self.state()
        sim = SimulatedState("fake")
        ctx = play_youtube.RunContext(state, sim, build_tool_registry(state, sim), self.options(), [],
                                      lambda message: None, fix_count=play_youtube.MAX_FIXES_PER_DEVICE)
        step = play_youtube.Step("stats", "visible", lambda: {"success": False}, lambda: False,
                                 {"enable_stats_for_nerds"})
        self.assertIn("fix limit", play_youtube.fix_step("fake", step, {}, [], ctx)["summary"])

        ctx.fix_count = 0

        def finish(**kwargs):
            state.fix_result = {"fixed": True, "summary": "repaired", "needs_human": False}
            return "finished"

        with patch.object(play_youtube, "run_agent_loop", side_effect=finish):
            result = play_youtube.fix_step("fake", step, {"error": "missing"}, ["launch"], ctx)
        self.assertTrue(result["fixed"])
        self.assertTrue(result["ai_called"])

    def test_nim_error_becomes_needs_human(self):
        state = self.state()
        sim = SimulatedState("fake")
        ctx = play_youtube.RunContext(state, sim, build_tool_registry(state, sim), self.options(), [],
                                      lambda message: None)
        step = play_youtube.Step("stats", "visible", lambda: {"success": False}, lambda: False,
                                 {"enable_stats_for_nerds"})
        with patch.object(play_youtube, "run_agent_loop", side_effect=play_youtube.NIMError("provider down")):
            result = play_youtube.fix_step("fake", step, {}, [], ctx)
        self.assertTrue(result["needs_human"])

    def test_playback_repair_continues_remaining_seconds(self):
        def repair(device_id, step, failure, done, ctx):
            ctx.sim.wifi_connected = True
            ctx.sim.playback_active = True
            ctx.sim.stats_for_nerds = False
            ctx.sim.fullscreen = False
            return {"fixed": True, "summary": "wifi restored", "needs_human": False, "ai_called": True}

        with patch.object(play_youtube, "fix_step", side_effect=repair), \
             patch.object(play_youtube.report_generator, "save_report", return_value="report.json"):
            report = play_youtube.run_phone(
                "fake", "https://youtu.be/test", 10, self.options(ai_fallback=True, inject_failure=True))
        self.assertEqual(report["status"], "PASS")
        self.assertIn("playback", report["summary"])
        self.assertEqual(report["fixes"][0]["step"], "playback")
        tools = [entry["tool"] for entry in report["tool_history"]]
        self.assertEqual(tools.count("enable_stats_for_nerds"), 2)
        self.assertEqual(tools.count("enter_fullscreen"), 2)

    def test_plain_playback_failure_and_optional_setup(self):
        with patch.object(play_youtube.report_generator, "save_report", return_value="report.json"):
            failed = play_youtube.run_phone(
                "fake", "https://youtu.be/test", 5, self.options(inject_failure=True))
            passed = play_youtube.run_phone(
                "fake", "https://youtu.be/test", 5,
                self.options(stats_for_nerds=False, fullscreen=False, keep_playing=False))
        self.assertEqual(failed["status"], "FAIL")
        self.assertEqual(passed["status"], "PASS")
        self.assertEqual(passed["tool_history"][-1]["tool"], "stop_video")

    def test_simulated_setting_navigation_and_terminal_tool(self):
        sim = SimulatedState("fake", stats_setting_off=True)
        state = self.state()
        dispatch = build_tool_registry(state, sim)
        dispatch("press_key", {"key": "home"})
        for _ in range(3):
            dispatch("get_screen", {"include_screenshot": False})
            dispatch("tap_element", {"index": 0})
        screen = dispatch("get_screen", {"include_screenshot": False})
        self.assertEqual(screen["elements"][0]["switch"], "off")
        dispatch("tap_element", {"index": 0})
        result = dispatch("step_done", {"fixed": True, "summary": "enabled"})
        self.assertTrue(result["fixed"])

    def test_collect_logs_uses_simulator(self):
        result = test_tools.collect_logs("fake", "reason", sim=SimulatedState("fake"))
        self.assertTrue(result["success"])

    def test_cli_validation_list_and_simulated_success(self):
        cases = [
            (["play_youtube.py"], 2),
            (["play_youtube.py", "--devices", "D", "--video-url", "https://example.com/x"], 2),
            (["play_youtube.py", "--devices", "D", "--video-url", "https://youtu.be/x",
              "--duration", "0.01", "--simulate"], 2),
        ]
        with patch("builtins.print"):
            for argv, expected in cases:
                with self.subTest(argv=argv), patch.object(sys, "argv", argv):
                    self.assertEqual(play_youtube.main(), expected)
            with patch.object(sys, "argv", ["play_youtube.py", "--list-devices"]), \
                 patch.object(play_youtube.adb_tools, "list_devices", return_value=[]):
                self.assertEqual(play_youtube.main(), 0)

            report = {"status": "PASS", "fixes": []}
            argv = ["play_youtube.py", "--devices", "D", "--video-url", "https://youtu.be/x",
                    "--duration", "0.1", "--simulate"]
            with patch.object(sys, "argv", argv), \
                 patch.object(play_youtube, "diagnostic_run", return_value=nullcontext("log")), \
                 patch.object(play_youtube, "run_phone", return_value=report), \
                 patch.object(play_youtube.report_generator, "print_report"):
                self.assertEqual(play_youtube.main(), 0)

    def test_runner_resumes_after_stats_without_relaunching(self):
        options = self.options(ai_fallback=True, sim_stats_setting_off=True)

        def fake_model(**kwargs):
            dispatch = kwargs["dispatch"]
            recipe_steps = []
            for _ in range(4):
                result = dispatch("tap_element", {"index": 0})
                recipe_steps.append(result["recipe_step"])
            dispatch("get_screen", {"include_screenshot": False})
            dispatch("save_recipe", {"task": "enable_stats_in_settings", "steps": recipe_steps,
                                      "notes": "setting was off; simulated YouTube General screen differed"})
            dispatch("open_video_url", {})
            dispatch("wait_for_ads", {})
            dispatch("enable_stats_for_nerds", {})
            dispatch("get_player_state", {})
            dispatch("step_done", {"fixed": True, "summary": "enabled setting and restored overlay"})
            return "finished"

        with patch.object(play_youtube, "run_agent_loop", side_effect=fake_model), \
             patch.object(play_youtube.report_generator, "save_report", return_value="report.json"):
            report = play_youtube.run_phone("fake", "https://youtu.be/test", 5, options)
        tools = [entry["tool"] for entry in report["tool_history"]]
        self.assertEqual(tools.count("launch_youtube"), 1)
        self.assertIn("enter_fullscreen", tools)
        after_fix = tools[tools.index("step_done") + 1:]
        self.assertNotIn("open_video_url", after_fix)
        self.assertLess(after_fix.index("enter_fullscreen"), after_fix.index("get_playback_status"))
        self.assertEqual(report["status"], "PASS")


if __name__ == "__main__":
    unittest.main()
