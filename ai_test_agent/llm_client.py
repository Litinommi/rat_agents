"""Step-scoped NIM repair loop and its constrained tool schemas."""

import json
import time

from config import MAX_LLM_TURNS, NVIDIA_NIM_VISION
from diagnostics import record
from nim_client import chat_completion, create_client
from test_runner import ToolBudgetExceeded
from tools.ui_tools import RECIPE_TASKS

_NO_INPUT = {"type": "object", "properties": {}, "required": []}


def _schema(name, description, properties=None, required=None):
    return {"type": "function", "function": {"name": name, "description": description,
            "parameters": {"type": "object", "properties": properties or {}, "required": required or []}}}


TOOL_SCHEMAS = [
    _schema("check_device", "Return whether this Android device is connected."),
    _schema("get_device_info", "Return the phone, YouTube version, locale and screen size."),
    _schema("get_wifi_status", "Return current Wi-Fi state; association does not prove internet access."),
    _schema("toggle_wifi", "Enable or disable Wi-Fi.", {"enabled": {"type": "boolean"}}, ["enabled"]),
    _schema("launch_youtube", "Restart YouTube on its home screen."),
    _schema("open_video_url", "Open the configured test URL in YouTube."),
    _schema("wait_for_ads", "Wait for or skip player ads."),
    _schema("enable_stats_for_nerds", "Try recipes and built-in locators for the stats overlay."),
    _schema("enable_stats_in_app_settings", "Try a saved recipe for YouTube's General stats setting."),
    _schema("enter_fullscreen", "Try recipes and the built-in fullscreen locator."),
    _schema("stop_video", "Press Back once."),
    _schema("get_playback_status", "Monitor playback for a bounded duration.",
            {"duration_seconds": {"type": "integer"}}, ["duration_seconds"]),
    _schema("get_player_state", "Return player, fullscreen, stats, ad and media state."),
    _schema("get_screen", "Return the current indexed UI elements and optional screenshot.",
            {"include_screenshot": {"type": "boolean"}}),
    _schema("reveal_player_controls", "Reveal auto-hiding player controls."),
    _schema("tap_element", "Tap an element from the latest screen. Already-on switches and their rows are not tapped.",
            {"index": {"type": "integer"}, "reveal_player_controls_first": {"type": "boolean"}}, ["index"]),
    _schema("press_key", "Press a named Android key.",
            {"key": {"type": "string", "enum": ["back", "home", "enter", "space", "media_play_pause", "escape"]}},
            ["key"]),
    _schema("swipe", "Swipe the current screen.",
            {"direction": {"type": "string", "enum": ["up", "down", "left", "right"]}}, ["direction"]),
    _schema("save_recipe", "Save only the minimal verified actions. Notes must state the cause and phone difference.",
            {"task": {"type": "string", "enum": list(RECIPE_TASKS)}, "steps": {"type": "array", "items": {"type": "object"}},
             "notes": {"type": "string"}}, ["task", "steps", "notes"]),
    _schema("collect_logs", "Capture diagnostic logs.", {"reason": {"type": "string"}}),
    _schema("step_done", "End this repair attempt with its verified result.",
            {"fixed": {"type": "boolean"}, "summary": {"type": "string"},
             "needs_human": {"type": "boolean"}}, ["fixed", "summary"]),
]
_SCHEMAS_BY_NAME = {item["function"]["name"]: item for item in TOOL_SCHEMAS}


KNOWN_CAUSES = {
    "launch": "A connection/authorization problem or app-start failure can prevent YouTube reaching foreground.",
    "open_url": "An app chooser, browser handoff, popup, or missing player can block the deep link.",
    "ads": "Ad controls vary; a missing player is distinct from a visible ad.",
    "stats": "If Stats for nerds is absent from More, its YouTube setting may be off: You → Settings → General.",
    "fullscreen": "Player controls auto-hide and the fullscreen locator varies by YouTube version.",
    "playback": "Playback may be paused, buffering, disconnected, or showing an in-player error.",
}


def schemas_for(names) -> list[dict]:
    return [_SCHEMAS_BY_NAME[name] for name in names if name in _SCHEMAS_BY_NAME]


def build_fix_prompt(step: str, target: str) -> str:
    return f"""You repair one failed YouTube test step on one Android phone.
Target step: {step}
Target state: {target}
Do not continue to a later test step. Only the tools available for this repair can be called.
Take one action per turn. If screen_changed is false, that action did nothing; do not repeat it.
Stay inside YouTube, dismiss blocking popups, and never tap ads or links into another app.
Never tap a switch or its row when its current screen state is "on".
launch_youtube restarts YouTube; open_video_url replaces the current screen with the configured video.
Verify player targets with get_player_state. For an app-setting switch, verify get_screen shows switch "on".
After a verified repair, save_recipe with notes stating the cause and what differed on this phone, then call step_done.
If the safe fix requires a person, call step_done with fixed=false and needs_human=true.
Known cause: {KNOWN_CAUSES.get(step, "The observed state differs from the target.")}"""


def _tool_result_content(result: dict):
    payload = dict(result)
    image_b64 = payload.pop("_image_jpeg_b64", None)
    if image_b64 and not NVIDIA_NIM_VISION:
        payload["screenshot_note"] = "Screenshot omitted because NVIDIA_NIM_VISION is disabled."
    return json.dumps(payload, default=str, ensure_ascii=False), image_b64


def _shrink_old_screens(messages: list[dict]) -> list[dict]:
    """Only the newest tool result with elements keeps the full screen."""
    copied = [dict(message) for message in messages]
    newest_kept = False
    for message in reversed(copied):
        if message.get("role") != "tool" or not isinstance(message.get("content"), str):
            continue
        try:
            result = json.loads(message["content"])
        except (TypeError, json.JSONDecodeError):
            continue
        if not isinstance(result, dict) or "elements" not in result:
            continue
        if not newest_kept:
            newest_kept = True
            continue
        kept = {key: result[key] for key in ("success", "tapped", "recipe_step", "screen_changed", "error")
                if key in result}
        message["content"] = json.dumps(kept, ensure_ascii=False)
    return copied


def run_agent_loop(*, tools, system_prompt, starting_message, state, dispatch,
                   on_assistant_text=lambda text: None, on_tool_call=lambda name, args: None,
                   on_tool_result=lambda name, result: None, usage=None) -> str:
    with create_client() as client:
        return _run_agent_loop(client, tools, system_prompt, starting_message, state, dispatch,
                               on_assistant_text, on_tool_call, on_tool_result, usage)


def _run_agent_loop(client, tools, system_prompt, starting_message, state, dispatch,
                    on_assistant_text, on_tool_call, on_tool_result, usage=None):
    schemas = schemas_for(tools)
    allowed = {schema["function"]["name"] for schema in schemas}
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": starting_message}]
    for _ in range(MAX_LLM_TURNS):
        request_messages = _shrink_old_screens(messages)
        message, refused = chat_completion(
            client, request_messages, tools=schemas, tool_choice="required",
            on_text=on_assistant_text, usage=usage)
        if refused:
            return "refused"
        messages.append(message)
        calls = message.get("tool_calls", [])
        if not calls:
            continue

        budget_exceeded, images = False, []
        for index, call in enumerate(calls):
            name = call["function"]["name"]
            if index:
                result = {"success": False, "error": "not executed: one action per turn"}
            else:
                try:
                    arguments = json.loads(call["function"]["arguments"])
                    if not isinstance(arguments, dict) or name not in allowed:
                        raise ValueError
                    on_tool_call(name, arguments)
                    started = time.monotonic()
                    result = dispatch(name, arguments)
                    record("fix_tool_completed", tool=name, tool_call_id=call["id"],
                           duration_seconds=round(time.monotonic() - started, 3))
                except ToolBudgetExceeded as exc:
                    result = {"success": False, "error": str(exc)}
                    budget_exceeded = True
                except (ValueError, TypeError, json.JSONDecodeError):
                    result = {"success": False, "error": "Invalid tool name or arguments."}
            content, image_b64 = _tool_result_content(result)
            on_tool_result(name, {key: value for key, value in result.items() if key != "_image_jpeg_b64"})
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": content})
            if image_b64 and NVIDIA_NIM_VISION:
                images.extend([
                    {"type": "text", "text": f"Current screen after {name}:"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                ])
        if images:
            messages.append({"role": "user", "content": images})
        if budget_exceeded:
            return "budget_exceeded"
        if state.fix_result is not None:
            return "finished"
    return "turns_exhausted"
