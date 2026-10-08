"""Play a given YouTube video on one or more real devices without the LLM.

For each device (in parallel): launch YouTube -> open the video URL ->
clear ads -> stats for nerds -> full screen -> monitor playback. Costs no
tokens. With --ai-fallback, devices where a step failed are then handed to
the AI agent, which works out the fix (so only failing phones use tokens).

Usage:
  python play_youtube.py --devices <ID> [<ID> ...] --video-url <URL> --duration <MINUTES> [--ai-fallback]
"""

import argparse
import sys
import threading
import time

import log_collector
from urllib.parse import urlparse

from config import PLAYBACK_POLL_INTERVAL_SECONDS
from diagnostics import diagnostic_run, record, record_exception, sanitize
from tools import adb_tools, youtube_tools, ui_tools

# get_playback_status caps a single call at 180s, so long runs are
# monitored in chunks and progress is reported after each one.
_MONITOR_CHUNK_SECONDS = 60
_YOUTUBE_HOSTS = ("youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be")

_print_lock = threading.Lock()


def _log(device_id: str, message: str) -> None:
    record("console", message=message)
    with _print_lock:
        print(sanitize(f"[{device_id}] {message}"), flush=True)


def _play_on_device(device_id: str, url: str, duration_seconds: int, options: argparse.Namespace, results: dict) -> None:
    with diagnostic_run(device_id, "scripted") as log_path:
        _log(device_id, f"Diagnostic log: {log_path}")
        _play_on_device_impl(device_id, url, duration_seconds, options, results)
        record("device_outcome", failure=results.get(device_id))


def _play_on_device_impl(device_id: str, url: str, duration_seconds: int, options: argparse.Namespace, results: dict) -> None:
    """results[device_id] = None on PASS, else a "step: reason" string
    describing the first failure (handed to the AI agent with --ai-fallback)."""
    failure = "did not finish"

    def call_tool(name, *args):
        started = time.monotonic()
        record("scripted_tool_started", tool=name)
        try:
            result = getattr(youtube_tools, name)(*args)
            record("scripted_tool_result", tool=name, result=result,
                   duration_seconds=round(time.monotonic() - started, 3))
            return result
        except Exception as exc:
            record_exception("scripted_tool_error", exc)
            raise

    def setup_problem(step: str, result: dict) -> bool:
        """Log a setup step. A failure is only a warning in plain mode, but
        with --ai-fallback we stop so the agent can take over from here."""
        nonlocal failure
        record("scripted_step", tool=step, result=result)
        if result.get("success"):
            return False
        failure = f"{step}: {result.get('error')}"
        _log(device_id, f"{'FAIL' if options.ai_fallback else 'WARN'} {failure}")
        return options.ai_fallback

    try:
        launch = call_tool("launch_youtube", device_id)
        if not launch["success"]:
            failure = f"launch_youtube: {launch.get('error')}"
            _log(device_id, f"FAIL {failure}")
            return
        _log(device_id, "OK   launched YouTube")

        opened = call_tool("open_video_url", device_id, url)
        if not opened["success"]:
            failure = f"open_video_url: {opened.get('error')}"
            _log(device_id, f"FAIL {failure}")
            return
        _log(device_id, f"OK   opened {url}")

        ads = call_tool("wait_for_ads", device_id)
        if setup_problem("wait_for_ads", ads):
            return
        if ads["success"]:
            _log(device_id, f"OK   ads cleared (skipped {ads['ads_skipped']})")

        # Stats first: the player menu is easier to reach before rotating.
        if options.stats_for_nerds:
            stats = call_tool("enable_stats_for_nerds", device_id)
            if not stats["success"]:
                # Some phones only list "Stats for nerds" in the More menu once it's switched
                # on in YouTube's settings. That fix comes from saved recipes only - with no
                # recipe this phone fails here (and --ai-fallback hands it to the AI agent).
                _log(device_id, f"...  stats for nerds not available ({stats.get('error')}); "
                                "checking saved recipes for the 'Enable stats for nerds' setting")
                setting = call_tool("enable_stats_in_app_settings", device_id)
                if setting["success"]:
                    _log(device_id, "OK   'Enable stats for nerds' setting turned on by saved recipe"
                                    f" (from {setting.get('recipe_from', 'this phone')})")
                    call_tool("open_video_url", device_id, url)
                    call_tool("wait_for_ads", device_id)
                    stats = call_tool("enable_stats_for_nerds", device_id)
                else:
                    _log(device_id, f"...  {setting.get('error')}")
            if setup_problem("enable_stats_for_nerds", stats):
                return
            if stats["success"]:
                _log(device_id, "OK   stats for nerds enabled")
        if options.fullscreen:
            fs = call_tool("enter_fullscreen", device_id)
            if setup_problem("enter_fullscreen", fs):
                return
            if fs["success"]:
                _log(device_id, "OK   entered full screen")

        # Monitoring only counts with Stats for nerds on screen - verify it right
        # before starting (an earlier step failing or a later one closing it).
        if options.stats_for_nerds and not call_tool("get_player_state", device_id).get("stats_for_nerds_visible"):
            failure = "enable_stats_for_nerds: overlay not visible, so playback monitoring was not started"
            _log(device_id, f"FAIL {failure}")
            return
        _log(device_id, "OK   starting playback monitoring" + (" (stats for nerds visible)" if options.stats_for_nerds else ""))

        monitored = 0
        while monitored < duration_seconds:
            chunk = min(_MONITOR_CHUNK_SECONDS, duration_seconds - monitored)
            status = call_tool("get_playback_status", device_id, chunk)
            monitored += status.get("seconds_monitored", 0)
            if status.get("error_detected"):
                failure = f"playback after {monitored}s: {status.get('error_message')}"
                _log(device_id, f"FAIL {failure}")
                return
            state = "playing" if status.get("playing") else "not playing"
            _log(device_id, f"...  {monitored}/{duration_seconds}s monitored, {state}")
            if status.get("seconds_monitored", 0) == 0:
                break  # defensive: avoid spinning if the monitor returns nothing

        failure = None
        _log(device_id, f"PASS played for {monitored}s")
        if not options.keep_playing:
            call_tool("stop_video", device_id)
    finally:
        if failure:
            evidence = log_collector.capture_failure(device_id, failure)
            if evidence.get("success"):
                _log(device_id, f"Device logcat: {evidence['log_file']}")
        call_tool("quit_session", device_id)
        results[device_id] = failure


def _run_ai_fallback(failed: dict[str, str], args: argparse.Namespace, duration_seconds: int) -> dict[str, str]:
    """Hand each failed device to the AI agent, in parallel. Returns the
    agent's final status per device (PASS / FAIL / NEEDS_HUMAN / ...)."""
    # Imported here so plain runs never need the OpenAI SDK or an API key.
    import agent
    from config import nim_configuration_error

    if error := nim_configuration_error():
        print(f"--ai-fallback: {error} Skipping the AI agent.", file=sys.stderr)
        return {}

    agent_results: dict[str, str] = {}
    prefixed = len(failed) > 1
    threads = []
    for device_id, failure in failed.items():
        agent_args = argparse.Namespace(
            video_url=args.video_url,
            fullscreen=args.fullscreen,
            stats_for_nerds=args.stats_for_nerds,
            goal=(
                f"The scripted (no-AI) run on this phone failed at {failure}. Find out why and fix it, "
                f"then play {args.video_url} for {duration_seconds} seconds"
                + (" with Stats for nerds enabled" if args.stats_for_nerds else "")
                + (" in full screen" if args.fullscreen else "")
                + ". Save a recipe for any step you had to work out yourself."
            ),
            initial_failure=failure,
            simulate=False,
            inject_failure=False,
            sim_unfamiliar_ui=False,
        )
        threads.append(threading.Thread(
            target=agent.run_device, args=(device_id, agent_args, duration_seconds, prefixed, agent_results)
        ))
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return agent_results


def main() -> int:
    parser = argparse.ArgumentParser(description="Play a YouTube video on device(s) via uiautomator2 (no LLM)")
    parser.add_argument("--devices", nargs="+", required=True, help="One or more device IDs as shown by 'adb devices'")
    parser.add_argument("--video-url", required=True, help="YouTube video URL to play")
    parser.add_argument("--duration", type=float, required=True, help="Playback monitoring duration in minutes (e.g. 0.5, 2, 10)")
    parser.add_argument("--keep-playing", action="store_true", help="Leave the video playing on exit")
    parser.add_argument("--fullscreen", action=argparse.BooleanOptionalAction, default=True, help="Enter full screen (default: on)")
    parser.add_argument("--stats-for-nerds", action=argparse.BooleanOptionalAction, default=True, help="Enable the Stats for nerds overlay (default: on)")
    parser.add_argument("--ai-fallback", action="store_true",
                        help="Call the AI agent only for devices where a step failed (needs NVIDIA_NIM_API_KEY and NVIDIA_NIM_MODEL)")
    args = parser.parse_args()

    if urlparse(args.video_url).hostname not in _YOUTUBE_HOSTS:
        print(f"Not a YouTube URL: {args.video_url}", file=sys.stderr)
        return 2

    duration_seconds = round(args.duration * 60)
    if duration_seconds < PLAYBACK_POLL_INTERVAL_SECONDS:
        print(f"--duration must be at least {PLAYBACK_POLL_INTERVAL_SECONDS} seconds", file=sys.stderr)
        return 2

    devices = list(dict.fromkeys(args.devices))  # de-dupe, keep order
    missing = [d for d in devices if not adb_tools.device_exists(d)]
    if missing:
        print(f"Not found by adb (or not in 'device' state): {', '.join(missing)}", file=sys.stderr)
        return 1

    results = _run_free(devices, args, duration_seconds)
    final = {d: "PASS" if results.get(d) is None else "FAIL" for d in devices}
    failed = {d: results[d] for d in devices if results.get(d) is not None}
    if failed and args.ai_fallback:
        final.update(_ai_fallback_grouped(failed, args, duration_seconds))

    print()
    for d in devices:
        print(f"{d}: {final[d]}")
    return 0 if all(s.startswith("PASS") for s in final.values()) else 1


def _run_free(devices: list[str], args: argparse.Namespace, duration_seconds: int) -> dict[str, str | None]:
    """The no-AI run on each device in parallel: {device: None on PASS, else "step: reason"}."""
    results: dict[str, str | None] = {}
    threads = [
        threading.Thread(target=_play_on_device, args=(d, args.video_url, duration_seconds, args, results))
        for d in devices
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results


def _banner(message: str) -> None:
    print(f"\n=== {message} ===\n", flush=True)


def _ai_fallback_grouped(failed: dict[str, str], args: argparse.Namespace, duration_seconds: int) -> dict[str, str]:
    """Pay for the AI once per *kind* of phone, not once per phone.

    Failed phones are grouped by YouTube version + language (what decides
    YouTube's screens; recipes are shared on that basis too). One phone per
    group goes to the AI agent, which fixes it and saves a recipe; the rest of
    the group is re-run with the free runner, which now replays that recipe.
    Phones that still fail (e.g. a different cause, like Wi-Fi) get the AI
    individually. Disconnected phones never reach the AI - a person has to
    fix those.
    """
    from config import nim_configuration_error

    statuses: dict[str, str] = {}
    if error := nim_configuration_error():
        print(f"\n--ai-fallback: {error} Skipping the AI agent.", file=sys.stderr, flush=True)
        return statuses

    disconnected = [d for d in failed if not adb_tools.device_exists(d)]
    for d in disconnected:
        statuses[d] = "FAIL (phone disconnected - no AI called)"
    remaining = {d: f for d, f in failed.items() if d not in disconnected}
    if not remaining:
        return statuses

    groups: dict[str, list[str]] = {}
    for d in remaining:
        groups.setdefault(ui_tools._yt_and_locale(ui_tools.device_profile_key(d)), []).append(d)

    representatives = {members[0]: remaining[members[0]] for members in groups.values()}
    followers = {members[0]: members[1:] for members in groups.values() if len(members) > 1}

    _banner(f"AI agent on {len(representatives)} phone(s), one per YouTube version + language: {', '.join(representatives)}"
            + (f" ({sum(len(f) for f in followers.values())} similar phone(s) wait to reuse the fix)" if followers else ""))
    for d, status in _run_ai_fallback(representatives, args, duration_seconds).items():
        statuses[d] = f"{status} (AI agent)"

    retry = [d for rep, members in followers.items() for d in members]
    if not retry:
        return statuses

    _banner(f"Re-running {len(retry)} similar phone(s) for free with what the AI learned: {', '.join(retry)}")
    rerun = _run_free(retry, args, duration_seconds)
    learned_from = {d: rep for rep, members in followers.items() for d in members}
    still_failed = {}
    for d in retry:
        if rerun.get(d) is None:
            statuses[d] = f"PASS (free re-run, reused fix from {learned_from[d]})"
        else:
            still_failed[d] = rerun[d]

    if still_failed:
        _banner(f"{len(still_failed)} phone(s) still failing for another reason - AI agent on each: {', '.join(still_failed)}")
        for d, status in _run_ai_fallback(still_failed, args, duration_seconds).items():
            statuses[d] = f"{status} (AI agent)"
        for d in still_failed:
            statuses.setdefault(d, "FAIL")
    return statuses


if __name__ == "__main__":
    sys.exit(main())
