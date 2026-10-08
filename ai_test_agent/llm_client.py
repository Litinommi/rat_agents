"""Claude tool-use wiring: the tool schemas Claude is allowed to call, the
system prompt that frames the loop, and the manual agentic loop itself.

This is a manual `while` loop (not the SDK's beta tool_runner) so agent.py
gets full control over per-call console narration, the hard MAX_TOOL_CALLS
/ MAX_RETRIES safety valves, and injecting device_id / video URL outside the
schema - none of which the model ever sees or controls.
"""

import json

import anthropic

from config import ANTHROPIC_API_KEY, CLAUDE_MAX_TOKENS, CLAUDE_MODEL, MAX_LLM_TURNS, MAX_RETRIES, MAX_TOOL_CALLS
from test_runner import ToolBudgetExceeded
from tools.ui_tools import RECIPE_TASKS

_NO_INPUT = {"type": "object", "properties": {}, "required": []}

TOOL_SCHEMAS = [
    # --- environment ---------------------------------------------------------
    {
        "name": "check_device",
        "description": "Check whether the target Android device is connected and responsive via ADB. Always call this first.",
        "input_schema": _NO_INPUT,
    },
    {
        "name": "get_device_info",
        "description": "Get manufacturer, model, Android version, YouTube app version, system locale and screen size. "
        "These decide which UI YouTube shows, so call this early - it tells you whether to expect a non-English UI.",
        "input_schema": _NO_INPUT,
    },
    {
        "name": "get_wifi_status",
        "description": "Check whether Wi-Fi is enabled and connected on the device via ADB.",
        "input_schema": _NO_INPUT,
    },
    {
        "name": "toggle_wifi",
        "description": "Enable or disable Wi-Fi on the device via ADB (`svc wifi`). Use this as a recovery action when you diagnose a connectivity problem.",
        "input_schema": {
            "type": "object",
            "properties": {"enabled": {"type": "boolean", "description": "True to enable Wi-Fi, false to disable it."}},
            "required": ["enabled"],
        },
    },
    # --- built-in (fast) playback steps --------------------------------------
    {
        "name": "launch_youtube",
        "description": "Connect to the phone and (re)launch the YouTube app fresh on its home screen.",
        "input_schema": _NO_INPUT,
    },
    {
        "name": "open_video_url",
        "description": "Open the test's video URL directly in the YouTube app (deep link). The URL is fixed by the test configuration.",
        "input_schema": _NO_INPUT,
    },
    {
        "name": "wait_for_ads",
        "description": "Tap 'Skip' on ads when offered, otherwise wait until pre-roll ads finish. Call after open_video_url and before player actions.",
        "input_schema": _NO_INPUT,
    },
    {
        "name": "enable_stats_for_nerds",
        "description": "Built-in step to turn on the 'Stats for nerds' overlay. Tries a learned recipe for this device first, "
        "then English-UI locators. If it fails, do the step yourself with the UI tools (see system prompt).",
        "input_schema": _NO_INPUT,
    },
    {
        "name": "enable_stats_in_app_settings",
        "description": "Replay a SAVED RECIPE that turns on YouTube's app setting 'Enable stats for nerds'. On some "
        "phones 'Stats for nerds' only appears in the player's More menu once this setting is on. There is no "
        "built-in path: if no recipe exists for this phone it fails, and you must do it yourself (get_screen / "
        "tap_element, typically You -> Settings -> General -> 'Enable stats for nerds'), verify the switch is on, "
        "and save_recipe('enable_stats_in_settings', steps) using a switch_on step for the toggle. Afterwards call "
        "open_video_url and wait_for_ads, then enable_stats_for_nerds again.",
        "input_schema": _NO_INPUT,
    },
    {
        "name": "enter_fullscreen",
        "description": "Built-in step to put the player in full screen. Tries a learned recipe for this device first, then the "
        "standard fullscreen button. If it fails, do the step yourself with the UI tools.",
        "input_schema": _NO_INPUT,
    },
    {
        "name": "stop_video",
        "description": "Press Back once (exits full screen, or leaves the player).",
        "input_schema": _NO_INPUT,
    },
    {
        "name": "get_playback_status",
        "description": "Monitor playback for the given number of seconds (up to the full test duration in one call). Returns early "
        "if an error is detected. Reports playing, error_detected, error_message and media_state.",
        "input_schema": {
            "type": "object",
            "properties": {
                "duration_seconds": {"type": "integer", "description": "How many seconds to monitor playback for."}
            },
            "required": ["duration_seconds"],
        },
    },
    # --- generic UI tools: how you adapt to unfamiliar devices ----------------
    {
        "name": "get_player_state",
        "description": "Language-independent verification: fullscreen (player geometry), stats_for_nerds_visible, ad_showing, "
        "orientation, media_state. Use it to verify any step - especially ones you performed manually.",
        "input_schema": _NO_INPUT,
    },
    {
        "name": "get_screen",
        "description": "See the current screen: a numbered list of UI elements (class, resource_id, text, desc, clickable, "
        "bounds) plus a screenshot. Labels may be in any language - use the screenshot and icons/positions to understand them.",
        "input_schema": {
            "type": "object",
            "properties": {
                "include_screenshot": {
                    "type": "boolean",
                    "description": "Attach a screenshot (default true). Set false for a cheaper element-list-only look.",
                }
            },
            "required": [],
        },
    },
    {
        "name": "reveal_player_controls",
        "description": "Tap the video player once so its auto-hiding controls (settings, fullscreen, play/pause) appear for ~3s.",
        "input_schema": _NO_INPUT,
    },
    {
        "name": "tap_element",
        "description": "Tap element number `index` from the most recent get_screen. Player controls auto-hide within ~3s, so "
        "for a player button set reveal_player_controls_first=true (taps the player, re-finds the button, taps it). "
        "Each successful tap returns a ready-made recipe_step.",
        "input_schema": {
            "type": "object",
            "properties": {
                "index": {"type": "integer", "description": "Element number from the last get_screen."},
                "reveal_player_controls_first": {"type": "boolean", "description": "Use for buttons inside the video player."},
            },
            "required": ["index"],
        },
    },
    {
        "name": "press_key",
        "description": "Press a hardware/system key.",
        "input_schema": {
            "type": "object",
            "properties": {"key": {"type": "string", "enum": ["back", "home", "enter", "space", "media_play_pause", "escape"]}},
            "required": ["key"],
        },
    },
    {
        "name": "swipe",
        "description": "Swipe the screen to scroll; 'up' reveals content further down (e.g. more menu items).",
        "input_schema": {
            "type": "object",
            "properties": {"direction": {"type": "string", "enum": ["up", "down", "left", "right"]}},
            "required": ["direction"],
        },
    },
    {
        "name": "save_recipe",
        "description": "After you completed a step manually AND verified it with get_player_state, save the steps so the built-in "
        "tool replays them automatically on this device profile (same model, YouTube version, locale) in future runs. "
        "Copy the recipe_step objects returned by tap_element / press_key / reveal_player_controls, in order, "
        "leaving out detours that didn't contribute.",
        "input_schema": {
            "type": "object",
            "properties": {
                "task": {"type": "string", "enum": list(RECIPE_TASKS)},
                "steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "string",
                                "enum": ["tap", "switch_on", "press_key", "wait", "reveal_player_controls"],
                                "description": "switch_on: make sure the on/off switch next to `match` is ON "
                                "(safe to repeat; use it instead of tap for toggles).",
                            },
                            "match": {
                                "type": "object",
                                "properties": {
                                    "resource_id": {"type": "string"},
                                    "desc": {"type": "string"},
                                    "text": {"type": "string"},
                                },
                            },
                            "reveal_player_controls": {"type": "boolean"},
                            "key": {"type": "string"},
                            "seconds": {"type": "number"},
                        },
                        "required": ["action"],
                    },
                },
                "notes": {"type": "string", "description": "What was different on this device (e.g. 'Hindi UI; menu item renamed')."},
            },
            "required": ["task", "steps"],
        },
    },
    # --- diagnosis / recovery / finish -----------------------------------------
    {
        "name": "collect_logs",
        "description": "Capture recent device logs (logcat) relevant to YouTube/Wi-Fi and save them for analysis.",
        "input_schema": {
            "type": "object",
            "properties": {"reason": {"type": "string", "description": "Why you are collecting logs right now."}},
            "required": [],
        },
    },
    {
        "name": "retry_test",
        "description": (
            f"Re-run the core playback steps (stop, relaunch, open the URL, clear ads). Bounded to {MAX_RETRIES} retries "
            f"total - check retries_remaining in prior results before calling again. Redo fullscreen/stats afterwards."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "reason": {"type": "string", "description": "Your diagnosis of what went wrong and why a retry is expected to help."}
            },
            "required": ["reason"],
        },
    },
    {
        "name": "generate_report",
        "description": (
            "Finish the test and produce the final report. This ALWAYS ends the run - call it exactly once, "
            "after you have a verified PASS/FAIL, or once you conclude the issue needs a human."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["PASS", "FAIL", "NEEDS_HUMAN"]},
                "summary": {"type": "string", "description": "One or two sentence summary of the outcome."},
                "root_cause": {"type": "string", "description": "If a failure occurred, your diagnosis of the root cause."},
                "recovery_action": {
                    "type": "string",
                    "description": "What you did to recover or adapt (including any recipes you learned). Omit if nothing.",
                },
            },
            "required": ["status", "summary"],
        },
    },
]


def build_system_prompt(*, device_id: str, duration_seconds: int, video_url: str, fullscreen: bool, stats_for_nerds: bool) -> str:
    extras = [name for name, on in (("enable Stats for nerds", stats_for_nerds), ("enter full screen", fullscreen)) if on]
    return f"""You are an AI test agent that controls a real Android phone to test YouTube playback, using a fixed set of \
tools. You never write or request shell commands - only the predefined tools are available, and they always \
operate on device {device_id}.

Test configuration (fixed - you cannot change it):
- Video URL: {video_url}
- Playback monitoring duration: {duration_seconds} seconds
- Player setup required: {", ".join(extras) if extras else "none"}

Your core job is to get this video playing correctly on THIS phone, whatever its manufacturer, Android version, \
YouTube version or language. The built-in steps were written for one phone with an English UI; on other phones \
they may fail. When they do, you adapt - you are the fallback, so a built-in failure is a problem to solve, not \
a reason to give up.

Flow:
1. check_device, get_device_info (note manufacturer, YouTube version, locale), get_wifi_status.
2. launch_youtube -> open_video_url -> wait_for_ads.
3. Player setup: enable_stats_for_nerds (if required) BEFORE enter_fullscreen (if required) - the menu is easier \
to reach in portrait.
4. get_playback_status with duration_seconds={duration_seconds}.{" Monitoring only starts once Stats for nerds is visible - the tool refuses otherwise, so make sure step 3 succeeded (and redo it after any retry_test)." if stats_for_nerds else ""}
5. generate_report exactly once.

When a built-in step fails (or returns a hint), fix it yourself:
a. get_screen to see what is actually on screen. Read labels in whatever language they are in; use the \
screenshot, icons (gear, three dots, expand arrows) and positions to identify controls.
b. Handle blockers first: popups, consent/permission dialogs, "open with" choosers, update prompts, \
sign-in nags - dismiss them (tap the dismiss/skip/not-now option, or press_key back).
   Stay inside YouTube: never tap ads ("Install", "Visit advertiser", "Learn more"), share targets, or \
anything that opens another app. If a tap result carries a warning that another app opened (or you see an \
app-lock/PIN screen, Play Store, browser, messaging app), press_key back immediately and never save that step.
c. Do the step manually: reveal_player_controls / tap_element (reveal_player_controls_first=true for player \
buttons) / swipe / press_key. Re-run get_screen after each change rather than assuming.
d. Verify with get_player_state (fullscreen / stats_for_nerds_visible / media_state). Never claim success \
without verification.
e. save_recipe with the recipe_step objects of the steps that worked, so future runs on this device profile \
are automatic. Only save verified, minimal steps.

Common device-specific issues: the player settings button may be a gear or three dots and may be labelled in \
another language; "Stats for nerds" may sit under an "Additional settings"/"More" submenu, or be missing from the \
player menu until it is switched on in YouTube's settings (enable_stats_in_app_settings); some phones open the \
URL in a browser or show an app chooser; ads may need waiting out; Xiaomi/Oppo/Vivo phones need "USB debugging \
(Security settings)" enabled for taps to work - if taps have no effect at all, report NEEDS_HUMAN with that \
instruction.

Failures during playback: collect_logs, re-check relevant state (get_wifi_status, check_device) to diagnose - \
verify causes with tools, don't assume. Take a safe recovery action if one exists (e.g. toggle_wifi), verify it, \
then retry_test (max {MAX_RETRIES} retries; don't call it after retries_remaining is 0) and redo the player setup.

Report honestly: PASS if the video played for the full duration (setup steps done, or clearly explained if one \
was impossible on this device), FAIL if playback did not work, NEEDS_HUMAN if you cannot proceed safely. In \
recovery_action, mention anything you adapted and any recipe you saved.

You have a hard limit of {MAX_TOOL_CALLS} tool calls - be efficient: prefer element lists over screenshots \
when the labels are readable, and don't re-check state without a reason. Keep your text brief; you are narrating \
your reasoning live to a human watching the console."""


def _tool_result_content(result: dict):
    """Tool results are JSON text; a get_screen screenshot is attached as an
    image block so the model can see non-English or icon-only UIs."""
    image_b64 = result.pop("_image_jpeg_b64", None) if isinstance(result, dict) else None
    text = json.dumps(result, default=str, ensure_ascii=False)
    if not image_b64:
        return text
    return [
        {"type": "text", "text": text},
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image_b64}},
    ]


def _model_specific_options(model: str) -> dict:
    """Haiku 4.5 (the low-cost option) runs without thinking - it doesn't
    support adaptive thinking - and without server-side fallbacks. Larger
    models get adaptive thinking plus a fallback model if they decline."""
    if model.startswith("claude-haiku"):
        return {}
    return {
        "thinking": {"type": "adaptive"},
        "betas": ["server-side-fallback-2026-07-01"],
        "fallbacks": "default",
    }


def run_agent_loop(*, goal: str, state, dispatch, on_assistant_text, on_tool_call, on_tool_result) -> str:
    """Drives the manual agentic loop. Returns 'finished', 'budget_exceeded',
    'refused' or 'turns_exhausted' - agent.py decides what to do if it's not 'finished'.
    """
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    system_prompt = build_system_prompt(
        device_id=state.device_id,
        duration_seconds=state.duration_seconds,
        video_url=state.video_url,
        fullscreen=state.fullscreen,
        stats_for_nerds=state.stats_for_nerds,
    )
    messages = [{"role": "user", "content": f"Test goal: {goal}"}]

    for _ in range(MAX_LLM_TURNS):
        response = client.beta.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=CLAUDE_MAX_TOKENS,
            system=system_prompt,
            tools=TOOL_SCHEMAS,
            messages=messages,
            # Caches tools + system + history prefix across turns of the loop.
            cache_control={"type": "ephemeral"},
            **_model_specific_options(CLAUDE_MODEL),
        )

        text = "\n".join(block.text for block in response.content if block.type == "text").strip()
        if text:
            on_assistant_text(text)

        if response.stop_reason == "refusal":
            return "refused"

        # Append the full content (thinking + tool_use blocks) - required for the next turn.
        messages.append({"role": "assistant", "content": response.content})

        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
        if not tool_use_blocks:
            break  # model stopped without calling generate_report; treat as done

        tool_results = []
        budget_exceeded = False
        for block in tool_use_blocks:
            on_tool_call(block.name, block.input)
            try:
                result = dispatch(block.name, block.input)
                is_error = False
            except ToolBudgetExceeded as exc:
                result = {"error": str(exc)}
                is_error = True
                budget_exceeded = True
            except TypeError as exc:  # model sent arguments the tool doesn't accept
                result = {"error": f"Invalid arguments for {block.name}: {exc}"}
                is_error = True

            content = _tool_result_content(result)
            on_tool_result(block.name, result)
            tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": content, "is_error": is_error})

        messages.append({"role": "user", "content": tool_results})

        if budget_exceeded:
            return "budget_exceeded"
        if state.finished:
            return "finished"

    return "finished" if state.finished else "turns_exhausted"
