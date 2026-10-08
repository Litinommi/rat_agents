#!/usr/bin/env python3
"""CLI entry point for the AI Test Agent.

    python agent.py --devices <ID> [<ID> ...] --video-url <URL> --duration <MINUTES>
    python agent.py --list-devices

Each device gets its own agent (run in parallel). The agent plays the video
with the built-in steps and, on phones/YouTube versions/languages those steps
don't handle, inspects the screen and works out the steps itself - saving
them as a recipe so the next run on that device profile is automatic.
"""

import argparse
import sys
import threading
from urllib.parse import urlparse

import report_generator
from config import nim_configuration_error
from llm_client import run_agent_loop
from nim_client import NIMError
from test_runner import TestState, build_tool_registry
from tools import adb_tools, youtube_tools, test_tools
from tools.simulate import SimulatedState

_YOUTUBE_HOSTS = ("youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be")

LABELS = {
    "check_device": "ADB",
    "get_device_info": "ADB",
    "get_wifi_status": "ADB",
    "toggle_wifi": "ADB",
    "collect_logs": "LOGS",
    "get_playback_status": "TEST",
    "retry_test": "TEST",
    "get_screen": "UI",
    "tap_element": "UI",
    "press_key": "UI",
    "swipe": "UI",
    "reveal_player_controls": "UI",
    "get_player_state": "UI",
    "save_recipe": "LEARN",
    "generate_report": "AGENT",
}

_NARRATION = {
    "check_device": lambda i: "Checking device...",
    "get_device_info": lambda i: "Getting device profile...",
    "get_wifi_status": lambda i: "Checking Wi-Fi status...",
    "toggle_wifi": lambda i: f"{'Enabling' if i.get('enabled') else 'Disabling'} Wi-Fi...",
    "launch_youtube": lambda i: "Launching YouTube...",
    "open_video_url": lambda i: "Opening video URL...",
    "wait_for_ads": lambda i: "Handling ads...",
    "enable_stats_for_nerds": lambda i: "Enabling Stats for nerds...",
    "enable_stats_in_app_settings": lambda i: "Turning on 'Enable stats for nerds' in YouTube settings...",
    "enter_fullscreen": lambda i: "Entering full screen...",
    "stop_video": lambda i: "Pressing Back...",
    "get_playback_status": lambda i: f"Monitoring playback for {i.get('duration_seconds', 3)}s...",
    "get_player_state": lambda i: "Verifying player state...",
    "get_screen": lambda i: "Looking at the screen...",
    "reveal_player_controls": lambda i: "Revealing player controls...",
    "tap_element": lambda i: f"Tapping element #{i.get('index')}...",
    "press_key": lambda i: f"Pressing {i.get('key')}...",
    "swipe": lambda i: f"Swiping {i.get('direction')}...",
    "save_recipe": lambda i: f"Saving learned steps for '{i.get('task')}' ({len(i.get('steps', []))} steps)...",
    "collect_logs": lambda i: "Collecting logs...",
    "retry_test": lambda i: "Retrying...",
    "generate_report": lambda i: "Finalizing report...",
}

_print_lock = threading.Lock()


def _ok(result: dict) -> bool:
    return bool(result.get("success")) and "error" not in result


def _summarize_result(tool_name: str, result: dict) -> str:
    if tool_name == "check_device":
        return "Device connected" if result.get("connected") else f"Device NOT connected ({result.get('error', 'unknown')})"
    if tool_name == "get_device_info":
        if "error" in result:
            return f"Could not read device info: {result['error']}"
        return (f"{result.get('manufacturer', '')} {result.get('model', '')} / Android {result.get('android_version', '?')} / "
                f"YouTube {result.get('youtube_version', '?')} / locale {result.get('locale', '?')}")
    if tool_name == "get_wifi_status":
        return "Wi-Fi connected" if result.get("connected") else "Wi-Fi disconnected"
    if tool_name == "toggle_wifi":
        if not result.get("success", True):
            return f"Failed to toggle Wi-Fi: {result.get('error')}"
        return f"Wi-Fi {'enabled' if result.get('wifi_enabled') else 'disabled'}"
    if tool_name == "launch_youtube":
        return "YouTube launched" if result.get("launched") else f"Launch failed: {result.get('error')}"
    if tool_name in ("open_video_url", "enable_stats_for_nerds", "enable_stats_in_app_settings", "enter_fullscreen",
                     "wait_for_ads", "stop_video",
                     "reveal_player_controls", "press_key", "swipe"):
        if not _ok(result):
            return f"FAILED: {result.get('error')}"
        via = result.get("via")
        return "OK" + (f" (via {via.replace('_', ' ')})" if via else "") + (
            f", skipped {result['ads_skipped']} ad(s)" if result.get("ads_skipped") else "")
    if tool_name == "tap_element":
        return f"Tapped {result.get('tapped')}" if _ok(result) else f"FAILED: {result.get('error')}"
    if tool_name == "get_screen":
        if "error" in result:
            return f"FAILED: {result['error']}"
        return f"{result.get('element_count', len(result.get('elements', [])))} elements, {result.get('orientation')}"
    if tool_name == "get_player_state":
        if "error" in result:
            return f"FAILED: {result['error']}"
        return (f"fullscreen={result.get('fullscreen')} stats={result.get('stats_for_nerds_visible')} "
                f"ad={result.get('ad_showing')} media={result.get('media_state')}")
    if tool_name == "save_recipe":
        return (f"Saved {result.get('steps_saved')} steps for {result.get('device_profile')}" if _ok(result)
                else f"FAILED: {result.get('error')}")
    if tool_name == "get_playback_status":
        secs = result.get("seconds_monitored", 0)
        if result.get("error_detected"):
            return f"Playback failure detected after {secs}s: {result.get('error_message')}"
        return f"Playback {'healthy' if result.get('playing') else 'NOT playing'} for {secs}s (media: {result.get('media_state')})"
    if tool_name == "collect_logs":
        if not result.get("success"):
            return f"Log collection failed: {result.get('error')}"
        return f"Collected {result.get('relevant_lines', 0)} relevant lines -> {result.get('log_file')}"
    return str(result)


class DeviceConsole:
    """Console narration for one device; lines are prefixed with the device
    ID when several devices run in parallel."""

    def __init__(self, device_id: str, prefixed: bool):
        self.prefix = f"[{device_id}] " if prefixed else ""

    def _print(self, line: str) -> None:
        with _print_lock:
            print(f"{self.prefix}{line}", flush=True)

    def on_assistant_text(self, text: str) -> None:
        for line in text.splitlines():
            if line.strip():
                self._print(f"[AGENT] {line.strip()}")

    def on_tool_call(self, name: str, tool_input: dict) -> None:
        self._print(f"[AGENT] {_NARRATION.get(name, lambda i: f'Calling {name}...')(tool_input)}")

    def on_tool_result(self, name: str, result: dict) -> None:
        if name == "generate_report":
            return  # final report is printed separately once the run is over
        if name == "retry_test":
            for step in result.get("steps", []):
                self._print(f"[YOUTUBE] {step['tool']}: {_summarize_result(step['tool'], step['result'])}")
            self._print(f"[TEST] Retry attempt {result.get('attempt_number')} complete "
                        f"(retries remaining: {result.get('retries_remaining')})")
            return
        self._print(f"[{LABELS.get(name, 'YOUTUBE')}] {_summarize_result(name, result)}")


def run_device(device_id: str, args: argparse.Namespace, duration_seconds: int, prefixed: bool, results: dict) -> None:
    console = DeviceConsole(device_id, prefixed)
    goal = args.goal or (
        f"Play {args.video_url} on this phone for {duration_seconds} seconds"
        + (" with Stats for nerds enabled" if args.stats_for_nerds else "")
        + (" in full screen" if args.fullscreen else "")
        + ", and verify playback is healthy throughout."
    )
    sim = (SimulatedState(device_id=device_id, inject_failure=args.inject_failure, unfamiliar_ui=args.sim_unfamiliar_ui)
           if args.simulate else None)
    state = TestState(
        device_id=device_id,
        goal=goal,
        video_url=args.video_url,
        duration_seconds=duration_seconds,
        fullscreen=args.fullscreen,
        stats_for_nerds=args.stats_for_nerds,
    )
    dispatch = build_tool_registry(state, sim)

    try:
        outcome = run_agent_loop(
            goal=goal,
            state=state,
            dispatch=dispatch,
            on_assistant_text=console.on_assistant_text,
            on_tool_call=console.on_tool_call,
            on_tool_result=console.on_tool_result,
        )

        if outcome != "finished" and not state.finished:
            reason = {
                "budget_exceeded": "tool call budget exhausted",
                "refused": "the model declined to continue",
            }.get(outcome, "reasoning turn limit reached")
            # Called directly, not via dispatch, so the report is written even past the tool budget.
            test_tools.generate_report(
                device_id,
                status="NEEDS_HUMAN",
                summary=f"Agent stopped without reaching a verified outcome ({reason}).",
                state=state,
                root_cause=reason,
            )
    except NIMError as exc:
        console._print(f"[AGENT] {exc}")
        test_tools.generate_report(
            device_id, status="NEEDS_HUMAN", summary=str(exc), state=state,
            root_cause=str(exc),
        )
    except Exception as exc:  # noqa: BLE001 - one device's crash must not take down the others
        console._print(f"[AGENT] Run crashed ({type(exc).__name__}).")
    finally:
        if not args.simulate:
            youtube_tools.quit_session(device_id)

    if state.final_report:
        with _print_lock:
            if prefixed:
                print(f"\n===== Report for {device_id} =====")
            report_generator.print_report(state.final_report)
    results[device_id] = state.final_report.get("status") if state.final_report else "NO_REPORT"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI-driven YouTube playback test agent")
    parser.add_argument("--devices", nargs="+", help="One or more device IDs as shown by 'adb devices'")
    parser.add_argument("--video-url", help="YouTube video URL to play")
    parser.add_argument("--duration", type=float, default=1.0, help="Playback monitoring duration in minutes (default 1)")
    parser.add_argument("--fullscreen", action=argparse.BooleanOptionalAction, default=True, help="Enter full screen (default: on)")
    parser.add_argument("--stats-for-nerds", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable the Stats for nerds overlay (default: on)")
    parser.add_argument("--goal", help="Optional extra natural-language instruction for the agent")
    parser.add_argument("--list-devices", action="store_true", help="List devices adb currently sees and exit")
    parser.add_argument("--simulate", action="store_true", help="Run against a fake device/app backend (no hardware required)")
    parser.add_argument("--inject-failure", action="store_true", help="With --simulate, fail the first playback attempt (network drop)")
    parser.add_argument("--sim-unfamiliar-ui", action="store_true",
                        help="With --simulate, make the built-in Stats-for-nerds step fail on a Spanish UI so the agent must adapt")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.list_devices:
        try:
            devices = adb_tools.list_devices()
        except adb_tools.AdbError as exc:
            print(f"Could not list devices: {exc}", file=sys.stderr)
            return 1
        if not devices:
            print("No devices found by adb.")
        for d in devices:
            print(f"{d['device_id']}\t{d['state']}")
        return 0

    if not args.devices or not args.video_url:
        print("Usage: python agent.py --devices <ID> [<ID> ...] --video-url <URL> --duration <MINUTES>  "
              "(or --list-devices)", file=sys.stderr)
        return 2
    if urlparse(args.video_url).hostname not in _YOUTUBE_HOSTS:
        print(f"Not a YouTube URL: {args.video_url}", file=sys.stderr)
        return 2
    duration_seconds = round(args.duration * 60)
    if duration_seconds < 5:
        print("--duration must be at least 5 seconds (0.1 minutes)", file=sys.stderr)
        return 2
    if error := nim_configuration_error():
        print(error + " "
              "Without a key, use play_youtube.py (no AI, uses learned recipes).", file=sys.stderr)
        return 2

    devices = list(dict.fromkeys(args.devices))
    if not args.simulate:
        missing = [d for d in devices if not adb_tools.device_exists(d)]
        if missing:
            print(f"Not found by adb (or not in 'device' state): {', '.join(missing)}", file=sys.stderr)
            print("Run 'python agent.py --list-devices', or pass --simulate to demo without hardware.", file=sys.stderr)
            return 1

    print("=" * 40)
    print(" AI TEST AGENT")
    print("=" * 40)
    print(f"Devices:  {', '.join(devices)}")
    print(f"Video:    {args.video_url}")
    print(f"Duration: {duration_seconds}s | fullscreen={args.fullscreen} | stats_for_nerds={args.stats_for_nerds}")
    print()

    results: dict[str, str] = {}
    prefixed = len(devices) > 1
    threads = [threading.Thread(target=run_device, args=(d, args, duration_seconds, prefixed, results)) for d in devices]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    if prefixed:
        print("\n===== Summary =====")
        for d in devices:
            print(f"{d}: {results.get(d, 'NO_REPORT')}")
    return 0 if all(results.get(d) == "PASS" for d in devices) else 1


if __name__ == "__main__":
    raise SystemExit(main())
