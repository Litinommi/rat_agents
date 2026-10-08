"""Run state and the tool registry the agent loop dispatches into.

This is the safety boundary of the whole PoC: TOOL_SCHEMAS (in
llm_client.py) is the only thing the model ever sees, and build_tool_registry
here is the only thing that turns a tool name + LLM-supplied arguments
into an actual function call. device_id and the simulated-backend handle
are bound via functools.partial and never appear in a schema, so the
model has no path to picking a different device or a different backend.
"""

import time
from dataclasses import dataclass, field
from functools import partial

from config import MAX_RETRIES, MAX_TOOL_CALLS
from tools import adb_tools, youtube_tools, test_tools, ui_tools
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
    retries_used: int = 0
    tool_call_count: int = 0
    tool_history: list = field(default_factory=list)
    failure_observed: bool = False
    finished: bool = False
    final_report: dict | None = None
    start_time: float = field(default_factory=time.time)

    def record(self, tool_name: str, tool_input: dict, result: dict) -> None:
        self.tool_call_count += 1
        self.tool_history.append(
            {
                "index": self.tool_call_count,
                "tool": tool_name,
                "input": tool_input,
                "result": result,
                "elapsed_s": round(time.time() - self.start_time, 1),
            }
        )


class ToolBudgetExceeded(Exception):
    """Raised when the agent tries to exceed MAX_TOOL_CALLS - a hard stop
    independent of anything the LLM decides."""


def _marks_failure(tool_name: str, result: dict) -> bool:
    if not isinstance(result, dict):
        return False
    if result.get("error_detected"):
        return True
    if tool_name in {"launch_youtube", "open_video_url", "stop_video"} and result.get("success") is False:
        return True
    if tool_name == "retry_test":
        return any(step["result"].get("error_detected") or step["result"].get("success") is False for step in result.get("steps", []))
    return False


def _require_stats_for_nerds(monitor, player_state):
    """Playback monitoring only starts once Stats for nerds is verified on
    screen - enforced here, not left to the model's judgement."""
    def gated(**kwargs) -> dict:
        if not player_state().get("stats_for_nerds_visible"):
            return {
                "success": False,
                "monitoring_started": False,
                "error": "Stats for nerds is not visible, so playback monitoring was not started. Enable it "
                "(enable_stats_for_nerds, or manually via get_screen/tap_element), confirm with get_player_state, "
                "then call get_playback_status again.",
            }
        return monitor(**kwargs)

    return gated


def build_tool_registry(state: TestState, sim: SimulatedState | None):
    """Bind device_id/sim/state into each tool function so the LLM-facing
    call signature exactly matches TOOL_SCHEMAS in llm_client.py.
    """
    d = state.device_id
    raw = {
        "check_device": partial(adb_tools.check_device, d, sim=sim),
        "get_device_info": partial(adb_tools.get_device_info, d, sim=sim),
        "get_wifi_status": partial(adb_tools.get_wifi_status, d, sim=sim),
        "toggle_wifi": partial(adb_tools.toggle_wifi, d, sim=sim),
        "launch_youtube": partial(youtube_tools.launch_youtube, d, sim=sim),
        "stop_video": partial(youtube_tools.stop_video, d, sim=sim),
        # Monitoring may cover the whole requested test window in one call.
        "get_playback_status": partial(
            youtube_tools.get_playback_status, d, sim=sim, max_seconds=state.duration_seconds + 30
        ),
        "collect_logs": partial(test_tools.collect_logs, d, sim=sim),
        "retry_test": partial(test_tools.retry_test, d, state=state, sim=sim),
        "generate_report": partial(test_tools.generate_report, d, state=state),
    }
    if sim is None:
        raw |= {
            # The URL is fixed by the CLI, never chosen by the model.
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
        }
    else:
        raw |= sim_ui_tools(sim)

    if state.stats_for_nerds:
        raw["get_playback_status"] = _require_stats_for_nerds(raw["get_playback_status"], raw["get_player_state"])

    def dispatch(tool_name: str, tool_input: dict) -> dict:
        if tool_name not in raw:
            return {"error": f"Unknown tool '{tool_name}'. This should never happen - only "
                              f"predefined tools are exposed to the model."}

        if state.tool_call_count >= MAX_TOOL_CALLS:
            raise ToolBudgetExceeded(f"MAX_TOOL_CALLS ({MAX_TOOL_CALLS}) reached")

        # open_video_url is how we count the first attempt; retry_test bumps
        # state.attempts itself for subsequent attempts.
        if tool_name == "open_video_url" and state.attempts == 0:
            state.attempts = 1

        result = raw[tool_name](**tool_input)

        if _marks_failure(tool_name, result):
            state.failure_observed = True

        # Screenshots go to the model, not into the saved report/history.
        recorded = {k: v for k, v in result.items() if k != "_image_jpeg_b64"} if isinstance(result, dict) else result
        state.record(tool_name, tool_input, recorded)
        return result

    return dispatch
