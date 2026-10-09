"""Deterministic YouTube test runner with step-scoped AI repair."""

import argparse
import json
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass
from urllib.parse import urlparse

import report_generator
from config import (
    FIX_MAX_TOOL_CALLS,
    MAX_FIXES_PER_DEVICE,
    PLAYBACK_POLL_INTERVAL_SECONDS,
)
from diagnostics import (
    current_log_path,
    diagnostic_run,
    record,
    record_exception,
    sanitize,
)
from llm_client import build_fix_prompt, run_agent_loop
from nim_client import NIMError
from test_runner import TestState, build_tool_registry
from tools import adb_tools, ui_tools, youtube_tools
from tools.simulate import SimulatedState

_MONITOR_CHUNK_SECONDS = 60
_YOUTUBE_HOSTS = ("youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be")
_print_lock = threading.Lock()
_group_locks: dict[tuple[str, str], threading.Lock] = {}
_group_locks_guard = threading.Lock()


@dataclass
class Step:
    name: str
    target: str
    run: Callable[[], dict]
    check: Callable[[], bool]
    tools: set[str]


@dataclass
class RunContext:
    state: TestState
    sim: SimulatedState | None
    registry: Callable[[str, dict], dict]
    options: argparse.Namespace
    fixes: list[dict]
    console: Callable[[str], None]
    fix_count: int = 0


def _log(device_id: str, message: str) -> None:
    record("console", message=message)
    with _print_lock:
        print(sanitize(f"[{device_id}] {message}"), flush=True)


def _profile_group(device_id: str, sim: SimulatedState | None) -> tuple[str, str]:
    if sim is not None:
        return f"yt{sim.youtube_version}", sim.locale.split("-")[0].lower()
    return ui_tools._yt_and_locale(ui_tools.device_profile_key(device_id))


def _group_lock(group: tuple[str, str]) -> threading.Lock:
    with _group_locks_guard:
        return _group_locks.setdefault(group, threading.Lock())


def _screen_ok(result: dict) -> bool:
    return bool(result.get("success", True)) and "error" not in result


def _steps(ctx: RunContext) -> list[Step]:
    call = ctx.registry

    def player() -> dict:
        return call("get_player_state", {})

    def run_stats() -> dict:
        result = call("enable_stats_for_nerds", {})
        if _screen_ok(result):
            return result
        ctx.console(f"...  stats unavailable ({result.get('error')}); checking the saved app-setting recipe")
        setting = call("enable_stats_in_app_settings", {})
        if _screen_ok(setting):
            opened = call("open_video_url", {})
            if not _screen_ok(opened):
                return opened
            ads = call("wait_for_ads", {})
            if not _screen_ok(ads):
                return ads
            return call("enable_stats_for_nerds", {})
        return result

    def ads_cleared() -> bool:
        state = player()
        return bool(state.get("player_on_screen")) and not state.get("ad_showing")

    return [
        Step("launch", "YouTube is in the foreground", lambda: call("launch_youtube", {}),
             lambda: (ctx.sim.youtube_launched if ctx.sim else
                      youtube_tools._current_package(youtube_tools._get_driver(ctx.state.device_id))
                      == youtube_tools.YOUTUBE_PACKAGE), {"launch_youtube"}),
        Step("open_url", "the configured video player is on screen", lambda: call("open_video_url", {}),
             lambda: bool(player().get("player_on_screen")), {"launch_youtube", "open_video_url"}),
        Step("ads", "the player is on screen with no ad showing", lambda: call("wait_for_ads", {}),
             ads_cleared,
             {"launch_youtube", "open_video_url", "wait_for_ads"}),
        Step("stats", "the Stats for nerds overlay is visible", run_stats,
             lambda: not ctx.state.stats_for_nerds or bool(player().get("stats_for_nerds_visible")),
             {"launch_youtube", "open_video_url", "wait_for_ads", "enable_stats_for_nerds",
              "enable_stats_in_app_settings"}),
        Step("fullscreen", "the player is fullscreen", lambda: call("enter_fullscreen", {}),
             lambda: not ctx.state.fullscreen or bool(player().get("fullscreen")), {"enter_fullscreen"}),
    ]


_COMMON_FIX_TOOLS = {
    "check_device", "get_screen", "tap_element", "press_key", "swipe", "reveal_player_controls",
    "get_player_state", "save_recipe", "collect_logs", "step_done",
}


def _dismiss_popups(device_id: str, sim: SimulatedState | None) -> None:
    if sim is not None:
        return
    ui_tools.try_recipes(
        device_id, "dismiss_popup",
        lambda: youtube_tools._current_package(youtube_tools._get_driver(device_id)) == youtube_tools.YOUTUBE_PACKAGE)


def fix_step(device_id: str, step: Step, failure: dict, done: list[str], ctx: RunContext) -> dict:
    """Repair exactly one step, serialized by YouTube version and language."""
    if ctx.fix_count >= MAX_FIXES_PER_DEVICE:
        return {"fixed": False, "summary": f"fix limit ({MAX_FIXES_PER_DEVICE}) reached", "needs_human": False}

    with _group_lock(_profile_group(device_id, ctx.sim)):
        rerun = step.run()
        if _screen_ok(rerun) and step.check():
            return {"fixed": True, "summary": "step passed when rechecked after waiting for a similar phone",
                    "needs_human": False, "ai_called": False}
        _dismiss_popups(device_id, ctx.sim)
        rerun = step.run()
        if _screen_ok(rerun) and step.check():
            return {"fixed": True, "summary": "saved popup recipe cleared the step",
                    "needs_human": False, "ai_called": False}

        ctx.fix_count += 1
        ctx.state.fix_result = None
        allowed = _COMMON_FIX_TOOLS | step.tools
        dispatch = build_tool_registry(ctx.state, ctx.sim, allowed=allowed, budget=FIX_MAX_TOOL_CALLS)
        screen = dispatch("get_screen", {"include_screenshot": False})
        recipes = [] if ctx.sim else ui_tools.recipe_context(device_id, {
            "stats": "enable_stats_for_nerds", "fullscreen": "enter_fullscreen",
        }.get(step.name, "dismiss_popup"))
        start = json.dumps({
            "failed_step": step.name, "target": step.target, "done": [f"{name} ✓" for name in done],
            "failure": failure, "known_recipes": recipes, "current_screen": screen,
        }, default=str, ensure_ascii=False)
        usage = {}
        try:
            with diagnostic_run(device_id, f"fix_{step.name}") as fix_log:
                record("fix_started", step=step.name, failure=failure)
                outcome = run_agent_loop(
                    tools=allowed, system_prompt=build_fix_prompt(step.name, step.target),
                    starting_message=start, state=ctx.state, dispatch=dispatch, usage=usage)
                record("fix_finished", step=step.name, outcome=outcome, usage=usage)
        except NIMError as exc:
            return {"fixed": False, "summary": str(exc), "needs_human": True, "ai_called": True, "usage": usage}
        result = ctx.state.fix_result or {
            "fixed": False, "summary": f"repair ended without step_done ({outcome})", "needs_human": False}
        return {**result, "ai_called": True, "usage": usage, "diagnostic_log": fix_log}


def _build_report(ctx: RunContext, status: str, summary: str, root_cause: str | None) -> dict:
    report = report_generator.build_report(
        device_id=ctx.state.device_id, goal=ctx.state.goal, status=status, summary=summary,
        attempts=max(ctx.state.attempts, 1), duration_seconds=ctx.state.duration_seconds,
        failure_observed=ctx.state.failure_observed, root_cause=root_cause,
        recovery_action="; ".join(fix["summary"] for fix in ctx.fixes) or None,
        tool_call_count=ctx.state.tool_call_count, tool_history=list(ctx.state.tool_history))
    report["fixes"] = ctx.fixes
    report["diagnostic_log"] = current_log_path()
    report["saved_to"] = report_generator.save_report(report)
    return report


def _run_setup(ctx: RunContext, steps: list[Step], start: int = 0) -> dict | None:
    index = start
    done = [step.name for step in steps[:start]]
    while index < len(steps):
        step = steps[index]
        if ((step.name == "stats" and not ctx.options.stats_for_nerds)
                or (step.name == "fullscreen" and not ctx.options.fullscreen)):
            done.append(step.name)
            index += 1
            continue
        result = step.run()
        if _screen_ok(result) and step.check():
            ctx.console(f"OK   {step.name}")
            done.append(step.name)
            index += 1
            continue

        ctx.state.failure_observed = True
        reason = f"{step.name}: {result.get('error', 'target state not reached')}"
        if not ctx.options.ai_fallback and step.name in {"ads", "stats", "fullscreen"}:
            ctx.console(f"WARN {reason}")
            done.append(step.name)
            index += 1
            continue
        if not ctx.options.ai_fallback:
            return {"status": "FAIL", "summary": reason, "root_cause": reason}

        ctx.console(f"FAIL {reason}; attempting scoped repair")
        fix = fix_step(ctx.state.device_id, step, result, done, ctx)
        ctx.fixes.append({"step": step.name, **fix})
        if not fix.get("fixed"):
            return {"status": "NEEDS_HUMAN" if fix.get("needs_human") else "FAIL",
                    "summary": fix["summary"], "root_cause": reason}
        broken = next((i for i, prior in enumerate(steps[:index + 1]) if not prior.check()), None)
        if broken is not None:
            done = [item.name for item in steps[:broken]]
            index = broken
        else:
            done.append(step.name)
            index += 1
    return None


def run_phone(device_id: str, url: str, duration_seconds: int, options: argparse.Namespace) -> dict:
    sim = SimulatedState(
        device_id, inject_failure=options.inject_failure, unfamiliar_ui=options.sim_unfamiliar_ui,
        stats_setting_off=options.sim_stats_setting_off) if options.simulate else None
    state = TestState(device_id, f"Play {url} for {duration_seconds} seconds", url, duration_seconds,
                      fullscreen=options.fullscreen, stats_for_nerds=options.stats_for_nerds)
    ctx = RunContext(state, sim, build_tool_registry(state, sim), options, [], lambda msg: _log(device_id, msg))
    steps = _steps(ctx)
    failure_reason = None

    try:
        if setup_failure := _run_setup(ctx, steps):
            return _build_report(ctx, **setup_failure)

        if options.stats_for_nerds and not ctx.registry("get_player_state", {}).get("stats_for_nerds_visible"):
            failure_reason = "stats: overlay not visible, so playback monitoring was not started"
            return _build_report(ctx, "FAIL", failure_reason, failure_reason)

        remaining = duration_seconds
        monitored = 0
        playback_step = Step(
            "playback", "media_state is playing",
            lambda: ctx.registry("get_player_state", {}),
            lambda: ctx.registry("get_player_state", {}).get("media_state") == "playing",
            {"launch_youtube", "open_video_url", "wait_for_ads", "enable_stats_for_nerds",
             "enter_fullscreen", "get_wifi_status", "toggle_wifi", "get_playback_status"})
        while remaining > 0:
            status = ctx.registry("get_playback_status", {"duration_seconds": min(_MONITOR_CHUNK_SECONDS, remaining)})
            elapsed = status.get("seconds_monitored", 0)
            monitored += elapsed
            remaining -= elapsed
            healthy = not status.get("error_detected") and status.get("playing") and status.get("media_state", "playing") == "playing"
            if healthy:
                _log(device_id, f"...  {monitored}/{duration_seconds}s monitored, playing")
                if elapsed == 0:
                    break
                continue
            state.failure_observed = True
            failure_reason = f"playback after {monitored}s: {status.get('error_message') or status.get('media_state')}"
            if not options.ai_fallback:
                return _build_report(ctx, "FAIL", failure_reason, failure_reason)
            fix = fix_step(device_id, playback_step, status, [step.name for step in steps], ctx)
            ctx.fixes.append({"step": "playback", **fix})
            if not fix.get("fixed"):
                result_status = "NEEDS_HUMAN" if fix.get("needs_human") else "FAIL"
                return _build_report(ctx, result_status, fix["summary"], failure_reason)
            broken = next((i for i, step in enumerate(steps)
                           if not ((step.name == "stats" and not options.stats_for_nerds)
                                  or (step.name == "fullscreen" and not options.fullscreen))
                           and not step.check()), None)
            if broken is not None and (setup_failure := _run_setup(ctx, steps, broken)):
                return _build_report(ctx, **setup_failure)
            if elapsed == 0 and remaining == duration_seconds:
                remaining = duration_seconds

        if remaining > 0:
            failure_reason = f"playback monitor stopped after {monitored}s"
            return _build_report(ctx, "FAIL", failure_reason, failure_reason)
        if not options.keep_playing:
            ctx.registry("stop_video", {})
        fixed_names = {"stats": "enable_stats_for_nerds", "fullscreen": "enter_fullscreen"}
        ai_fixes = [fixed_names.get(fix["step"], fix["step"])
                    for fix in ctx.fixes if fix.get("ai_called")]
        suffix = f" (AI fixed: {', '.join(ai_fixes)})" if ai_fixes else ""
        return _build_report(ctx, "PASS", f"played for {monitored}s{suffix}", failure_reason)
    except Exception as exc:  # noqa: BLE001 - one phone must not stop the others
        record_exception("phone_run_crash", exc)
        return _build_report(ctx, "NEEDS_HUMAN", f"run crashed ({type(exc).__name__})", str(exc))
    finally:
        if sim is None:
            youtube_tools.quit_session(device_id)


def _parse_args(ai_fallback_default: bool) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play and verify a YouTube video on Android device(s)")
    parser.add_argument("--devices", nargs="+", help="Device IDs shown by adb devices")
    parser.add_argument("--video-url", help="YouTube video URL")
    parser.add_argument("--duration", type=float, default=1.0, help="Playback duration in minutes (default: 1)")
    parser.add_argument("--keep-playing", action="store_true")
    parser.add_argument("--fullscreen", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stats-for-nerds", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ai-fallback", action="store_true", default=ai_fallback_default)
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--simulate", action="store_true")
    parser.add_argument("--inject-failure", action="store_true")
    parser.add_argument("--sim-unfamiliar-ui", action="store_true")
    parser.add_argument("--sim-stats-setting-off", action="store_true")
    return parser.parse_args()


def main(ai_fallback_default: bool = False) -> int:
    args = _parse_args(ai_fallback_default)
    if args.list_devices:
        try:
            devices = adb_tools.list_devices()
        except adb_tools.AdbError as exc:
            print(f"Could not list devices: {exc}", file=sys.stderr)
            return 1
        print("\n".join(f"{item['device_id']}\t{item['state']}" for item in devices) or "No devices found by adb.")
        return 0
    if not args.devices or not args.video_url:
        print("Usage: python play_youtube.py --devices ID [ID ...] --video-url URL [--duration MINUTES]",
              file=sys.stderr)
        return 2
    if urlparse(args.video_url).hostname not in _YOUTUBE_HOSTS:
        print(f"Not a YouTube URL: {args.video_url}", file=sys.stderr)
        return 2
    duration_seconds = round(args.duration * 60)
    if duration_seconds < PLAYBACK_POLL_INTERVAL_SECONDS:
        print(f"--duration must be at least {PLAYBACK_POLL_INTERVAL_SECONDS} seconds", file=sys.stderr)
        return 2
    devices = list(dict.fromkeys(args.devices))
    if not args.simulate:
        missing = [device for device in devices if not adb_tools.device_exists(device)]
        if missing:
            print(f"Not found by adb: {', '.join(missing)}", file=sys.stderr)
            return 1

    reports = {}

    def worker(device):
        with diagnostic_run(device, "play_youtube") as log_path:
            _log(device, f"Diagnostic log: {log_path}")
            reports[device] = run_phone(device, args.video_url, duration_seconds, args)
            report_generator.print_report(reports[device])

    threads = [threading.Thread(target=worker, args=(device,)) for device in devices]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    for device in devices:
        report = reports.get(device)
        if not report:
            print(f"{device}: NEEDS_HUMAN")
            continue
        fixes = ", ".join({"stats": "enable_stats_for_nerds", "fullscreen": "enter_fullscreen"}.get(
            fix["step"], fix["step"]) for fix in report.get("fixes", []) if fix.get("ai_called"))
        print(f"{device}: {report['status']}" + (f" (AI fixed: {fixes})" if fixes and report["status"] == "PASS" else ""))
    return 0 if all(reports.get(device, {}).get("status") == "PASS" for device in devices) else 1


if __name__ == "__main__":
    raise SystemExit(main())
