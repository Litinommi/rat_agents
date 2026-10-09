"""Shared run state and the only dispatch boundary for device tools."""

import time
from dataclasses import dataclass, field
from functools import partial

from tools import adb_tools, test_tools, ui_tools, youtube_tools
from tools.simulate import SimulatedState, sim_ui_tools


@dataclass
class TestState:
    device_id: str
    goal: str
    video_url: str
    duration_seconds: int
    fullscreen: bool = True
    stats_for_nerds: bool = True
    attempts: int = 0
    tool_call_count: int = 0
    tool_history: list = field(default_factory=list)
    failure_observed: bool = False
    fix_result: dict | None = None
    fixes: list = field(default_factory=list)
    start_time: float = field(default_factory=time.time)

    def record(self, tool_name: str, tool_input: dict, result: dict) -> None:
        self.tool_call_count += 1
        self.tool_history.append({
            "index": self.tool_call_count, "tool": tool_name, "input": tool_input, "result": result,
            "elapsed_s": round(time.time() - self.start_time, 1),
        })


class ToolBudgetExceeded(Exception):
    pass


_SCREEN_ACTIONS = {"tap_element", "press_key", "swipe", "reveal_player_controls"}


def build_tool_registry(state: TestState, sim: SimulatedState | None, allowed=None, budget: int | None = None):
    """Bind one device and optionally constrain a repair to named tools/a local budget."""
    d = state.device_id
    raw = {
        "check_device": partial(adb_tools.check_device, d, sim=sim),
        "get_device_info": partial(adb_tools.get_device_info, d, sim=sim),
        "get_wifi_status": partial(adb_tools.get_wifi_status, d, sim=sim),
        "toggle_wifi": partial(adb_tools.toggle_wifi, d, sim=sim),
        "launch_youtube": partial(youtube_tools.launch_youtube, d, sim=sim),
        "stop_video": partial(youtube_tools.stop_video, d, sim=sim),
        "get_playback_status": partial(
            youtube_tools.get_playback_status, d, sim=sim, max_seconds=state.duration_seconds + 30),
        "collect_logs": partial(test_tools.collect_logs, d, sim=sim),
        "step_done": partial(test_tools.step_done, state=state),
    }
    raw |= ({
        "open_video_url": partial(youtube_tools.open_video_url, d, state.video_url),
        "wait_for_ads": partial(youtube_tools.wait_for_ads, d),
        "enter_fullscreen": partial(youtube_tools.enter_fullscreen, d),
        "enable_stats_for_nerds": partial(youtube_tools.enable_stats_for_nerds, d),
        "enable_stats_in_app_settings": partial(youtube_tools.enable_stats_in_app_settings, d),
        "reveal_player_controls": partial(youtube_tools.reveal_player_controls, d),
        "get_player_state": partial(youtube_tools.get_player_state, d),
        "get_screen": partial(ui_tools.get_screen, d),
        "tap_element": partial(ui_tools.tap_element, d),
        "press_key": partial(ui_tools.press_key, d),
        "swipe": partial(ui_tools.swipe, d),
        "save_recipe": partial(ui_tools.save_recipe, d),
    } if sim is None else sim_ui_tools(sim))

    allowed = set(raw) if allowed is None else set(allowed)
    local_calls = 0
    last_screen = None

    def dispatch(tool_name: str, tool_input: dict) -> dict:
        nonlocal local_calls, last_screen
        if tool_name not in raw or tool_name not in allowed:
            return {"success": False, "error": f"Tool '{tool_name}' is not available for this step."}
        if budget is not None and local_calls >= budget:
            raise ToolBudgetExceeded(f"fix tool-call budget ({budget}) reached")
        local_calls += 1
        if tool_name == "open_video_url":
            state.attempts += 1

        before = last_screen
        selected = None
        if tool_name == "tap_element" and before:
            index = tool_input.get("index")
            elements = before.get("elements", [])
            selected = elements[index] if isinstance(index, int) and 0 <= index < len(elements) else None
        if (selected and selected.get("switch") == "on"
                and (selected.get("text") or selected.get("desc"))):
            match = ({"text": selected["text"]} if selected.get("text") else
                     {"desc": selected["desc"]} if selected.get("desc") else
                     {"resource_id": selected.get("resource_id")})
            result = {"success": True, "already_on": True,
                      "recipe_step": {"action": "switch_on", "match": match}}
        else:
            result = raw[tool_name](**tool_input)
        if not isinstance(result, dict):
            result = {"success": True, "value": result}
        if tool_name == "get_screen":
            last_screen = {k: v for k, v in result.items() if k != "_image_jpeg_b64"}
        elif tool_name in _SCREEN_ACTIONS and result.get("success"):
            screen = raw["get_screen"](include_screenshot=True)
            public = {k: v for k, v in screen.items() if k != "_image_jpeg_b64"}
            result = {**result, **screen, "screen_changed": before != public}
            last_screen = public
        if result.get("success") is False or result.get("error_detected"):
            state.failure_observed = True
        recorded = {k: v for k, v in result.items() if k != "_image_jpeg_b64"}
        state.record(tool_name, tool_input, recorded)
        return result

    return dispatch
