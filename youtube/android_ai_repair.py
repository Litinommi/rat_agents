"""AI fallback for youtube_android_test.py, ported from play_youtube.py.

A step that misses its target is retried with saved recipes, then with saved
popup recipes, and only then handed to an OpenAI-compatible LLM (e.g. Ollama)
that may act on that one step. The LLM saves verified steps as a recipe in
learned_recipes.json, keyed by phone profile. Phones with the same YouTube
version and language take turns repairing, so the next phone reuses the
recipe instead of calling the LLM. All device access goes through the test's
own uiautomator2 session.

LLM settings come from the environment: LLM_BASE_URL, LLM_MODEL, LLM_API_KEY.
"""
import base64
import io
import json
import logging
import os
import re
import threading
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://192.168.200.182:11434/v1").rstrip("/")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3.8:latest")
LLM_API_KEY = os.getenv("LLM_API_KEY", "ollama")
# Send a screenshot with every screen; only for a model that accepts images and tools.
LLM_VISION = os.getenv("LLM_VISION", "false").lower() == "true"
MAX_FIXES_PER_DEVICE = 3
FIX_MAX_TOOL_CALLS = 40
MAX_LLM_TURNS = 60

YT = "com.google.android.youtube"
PLAYER = {"resourceId": f"{YT}:id/player_overlays"}
NERD_STATS = {"resourceId": f"{YT}:id/nerd_stats_layout"}
AD_ID = re.compile(r".*:id/(?:(?:modern_)?skip_ad_button|ad_badge|ad_text|ad_progress_text|ad_countdown"
                   r"|ad_duration|ad_duration_remaining|ad_overlay.*)$")
KEYCODES = {"back": 4, "home": 3, "enter": 66, "space": 62, "media_play_pause": 85, "escape": 111,
            "menu": 82, "tab": 61, "dpad_up": 19, "dpad_down": 20, "dpad_left": 21, "dpad_right": 22,
            "dpad_center": 23, "page_up": 92, "page_down": 93}
DIRECTIONS = ("up", "down", "left", "right")
PLAYBACK_STATES = {0: "none", 1: "stopped", 2: "paused", 3: "playing", 6: "buffering", 7: "error", 8: "connecting"}
RECIPES_FILE = Path(__file__).with_name("learned_recipes.json")
RECIPE_TASKS = ("enter_fullscreen", "enable_stats_for_nerds", "enable_stats_in_settings",
                "skip_ads", "dismiss_popup", "start_playback")
RECIPE_ACTIONS = {"tap", "tap_text", "tap_point", "long_press", "swipe", "switch_on", "press_key", "wait",
                  "reveal_player_controls"}
ROW_SLACK_PX = 60  # how far a switch may sit above/below its label's row
MAX_ELEMENTS = 150

# step -> (target state, recipe task, repair tools beyond the common ones)
STEPS = {
    "open_url": ("the configured video player is on screen", None,
                 ["launch_youtube", "open_video_url"]),
    "ads": ("the player is on screen with no ad showing", "skip_ads",
            ["launch_youtube", "open_video_url", "run_original_step"]),
    "resolution": ("the requested video quality is selected in the player", "set_resolution",
                   ["run_original_step"]),
    "fullscreen": ("the phone is in landscape with the video fullscreen", "enter_fullscreen",
                   ["run_original_step"]),
    "stats": ("the Stats for nerds overlay is visible", "enable_stats_for_nerds",
              ["launch_youtube", "open_video_url", "run_original_step", "enable_stats_in_app_settings"]),
    "playback": ("media_state is playing", "start_playback", ["launch_youtube", "open_video_url"]),
}
KNOWN_CAUSES = {
    "open_url": "An app chooser, browser handoff, popup, or missing player can block the deep link.",
    "ads": "Ad controls vary; a missing player is distinct from a visible ad.",
    "resolution": ("Path: More options -> Quality (its row shows the current quality, e.g. 'Quality Auto (1080p)') "
                   "-> Advanced -> the resolution. The Quality sheets are drawn without accessibility labels: "
                   "read their rows from screen_text / ocr_text and use tap_text. In landscape a sheet shows "
                   "only a few rows; swipe up inside it (swipe with its index) to see more. Confirm by reopening "
                   "More options and reading the Quality row, then call step_done."),
    "fullscreen": "Player controls auto-hide and the fullscreen locator varies by YouTube version.",
    "stats": "If Stats for nerds is absent from More, its YouTube setting may be off: You -> Settings -> General.",
    "playback": "Playback may be paused, buffering, disconnected, or showing an in-player error.",
}
COMMON_TOOLS = ["get_screen", "tap_element", "tap_text", "tap_point", "long_press", "swipe", "press_key", "wait",
                "reveal_player_controls", "get_player_state", "save_recipe", "step_done"]
SCREEN_ACTIONS = {"tap_element", "tap_text", "tap_point", "long_press", "swipe", "press_key", "wait",
                  "reveal_player_controls"}

_runs = {}
_screens = {}
_screen_text = {}
_profiles = {}
_recipe_failures = {}
_recipes_lock = threading.Lock()
_group_locks = {}
_group_guard = threading.Lock()


class LLMError(RuntimeError):
    """The LLM endpoint could not return a usable reply."""


# --- test-facing API --------------------------------------------------------

def start(d, video_url, resolution=None, should_stop=lambda: False):
    """Begin tracking one phone's test run; repairs end once `should_stop()` is true."""
    _runs[d.serial] = {"url": video_url, "resolution": resolution, "done": [], "runs": {},
                       "fixes": 0, "resolution_ok": False, "fix_result": None, "stalled": 0,
                       "should_stop": should_stop}


def ensure(d, step, run, result=None):
    """True when `step` reached its target, repairing it if it did not.

    `run` re-runs the test's original code for the step. `result` is what that
    code returned; it is the check for resolution, which has no on-screen state.
    """
    ctx = _runs[d.serial]
    ctx["runs"][step] = run
    if step == "resolution":
        ctx["resolution_ok"] = bool(result)
    if _wait_for(lambda: _check(d, step)):
        logger.info("[%s] %s: target reached (%s)", d.serial, step, STEPS[step][0])
    elif not _fix(d, step):
        return False
    if step != "playback" and step not in ctx["done"]:
        ctx["done"].append(step)
    return True


def check_playback(d):
    """Call periodically while playing; two stalled checks in a row start a repair."""
    ctx = _runs[d.serial]
    try:
        state = media_state(d)
    except Exception:
        logger.warning("[%s] Could not read the playback state", d.serial, exc_info=True)
        return
    healthy = state in (None, "playing", "buffering", "connecting")
    if not healthy:
        logger.warning("[%s] Playback check: media_state=%s", d.serial, state)
    ctx["stalled"] = 0 if healthy else ctx["stalled"] + 1
    if ctx["stalled"] >= 2:
        ctx["stalled"] = 0
        ensure(d, "playback", None)


# --- checks -----------------------------------------------------------------

def _sh(d, cmd):
    return d.shell(cmd).output


def _package(d):
    try:
        return d.app_current().get("package")
    except Exception:
        return None


def _ad_showing(d):
    """Ad controls inside the player, excluding sponsored cards below the video."""
    player = d(**PLAYER)
    pb = player.info["bounds"] if player.exists else None
    for node in ET.fromstring(d.dump_hierarchy()).iter("node"):
        if not AD_ID.fullmatch(node.get("resource-id", "")):
            continue
        b = [int(v) for v in re.findall(r"\d+", node.get("bounds", ""))]
        if len(b) != 4 or b[2] <= b[0] or b[3] <= b[1]:
            continue
        x, y = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        if not pb or (pb["left"] <= x <= pb["right"] and pb["top"] <= y <= pb["bottom"]):
            return True
    return False


def media_state(d):
    """YouTube's media session state, e.g. "playing"; None if it has no session."""
    package = None
    for line in _sh(d, "dumpsys media_session").splitlines():
        line = line.strip()
        if line.startswith("package="):
            package = line[len("package="):]
        elif package == YT:
            # Android <=14: "state=3"; Android 15+: "state=PLAYING(3)".
            match = re.match(r"state=PlaybackState \{state=(?:[A-Z_]+\()?(\d+)", line)
            if match:
                return PLAYBACK_STATES.get(int(match.group(1)), f"unknown({match.group(1)})")
    return None


def player_state(d):
    try:
        width, height = d.window_size()
        current = d.app_current()
        return {"player_on_screen": bool(d(**PLAYER).exists),
                "stats_for_nerds_visible": bool(d(**NERD_STATS).exists),
                "ad_showing": _ad_showing(d), "orientation": "LANDSCAPE" if width > height else "PORTRAIT",
                "package": current.get("package"), "activity": current.get("activity"),
                "media_state": media_state(d)}
    except Exception as exc:
        return {"error": str(exc)}


def _check(d, step):
    try:
        if step == "open_url":
            return d(**PLAYER).exists
        if step == "ads":
            return d(**PLAYER).exists and not _ad_showing(d)
        if step == "resolution":
            return _runs[d.serial]["resolution_ok"]
        if step == "fullscreen":
            width, height = d.window_size()
            return width > height
        if step == "stats":
            return d(**NERD_STATS).exists or d(textContains="CPN:").exists
        return media_state(d) == "playing"
    except Exception:
        logger.warning("[%s] Checking %s raised an error", d.serial, step, exc_info=True)
        return False


def _wait_for(check, timeout=4, interval=0.5):
    """Let asynchronous UI changes settle before declaring failure."""
    deadline = time.time() + timeout
    while not check():
        if time.time() >= deadline:
            return False
        time.sleep(interval)
    return True


# --- repair flow --------------------------------------------------------------

def profile_key(d):
    """maker|model|ytVERSION|locale: what decides which screens YouTube shows."""
    if d.serial not in _profiles:
        def prop(name):
            return _sh(d, f"getprop {name}").strip()
        version = re.search(r"versionName=(\S+)", _sh(d, f"dumpsys package {YT}"))
        _profiles[d.serial] = "|".join([
            prop("ro.product.manufacturer") or "?", prop("ro.product.model") or "?",
            f"yt{version.group(1) if version else 'not installed'}",
            prop("persist.sys.locale") or prop("ro.product.locale") or "?"])
    return _profiles[d.serial]


def _yt_and_language(key):
    """Only the language part of the locale counts: en-GB and en-IN show the same labels."""
    parts = key.split("|")
    return (parts[2] if len(parts) > 2 else ""), (parts[3].split("-")[0].lower() if len(parts) > 3 else "")


def _group_lock(d):
    with _group_guard:
        return _group_locks.setdefault(_yt_and_language(profile_key(d)), threading.Lock())


def _task(d, step):
    task = STEPS[step][1]
    return f"{task}_{_runs[d.serial]['resolution']}" if task == "set_resolution" else task


def _call(d, fn):
    if fn is None:
        return None
    try:
        return fn()
    except Exception:
        logger.exception("[%s] Original step raised an error", d.serial)
        return None


def _rerun(d, step):
    """Saved recipes for the step first, then the test's own code for it."""
    ctx = _runs[d.serial]
    task = _task(d, step)
    # Resolution has no on-screen check, so a fully replayed recipe counts.
    verify = (lambda: True) if step == "resolution" else (lambda: _check(d, step))
    if task and try_recipes(d, task, verify):
        ctx["resolution_ok"] |= step == "resolution"
        return True
    result = _call(d, ctx["runs"].get(step))
    if step == "resolution":
        ctx["resolution_ok"] = bool(result)
    if step == "stats" and not _wait_for(lambda: _check(d, step)) and \
            enable_stats_in_app_settings(d)["success"]:
        # Some phones list Stats for nerds only after this app setting is on.
        _call(d, ctx["runs"]["open_url"])
        _call(d, ctx["runs"]["stats"])
    return _wait_for(lambda: _check(d, step))


def _fix(d, step):
    """Repair one step; phones with the same YouTube version and language take turns."""
    serial, ctx = d.serial, _runs[d.serial]
    if ctx["should_stop"]():
        logger.warning("[%s] %s missed its target; test is stopping, so no repair", serial, step)
        return False
    if ctx["fixes"] >= MAX_FIXES_PER_DEVICE:
        logger.warning("[%s] %s missed its target; fix limit (%d) reached", serial, step, MAX_FIXES_PER_DEVICE)
        return False
    logger.warning("[%s] %s missed its target (%s); trying saved recipes", serial, step, STEPS[step][0])
    with _group_lock(d):
        how = "passed on retry (saved recipes, then the step's own code)"
        fixed = _rerun(d, step)
        if not fixed:
            try_recipes(d, "dismiss_popup", lambda: _package(d) == YT)
            how = "saved popup recipe cleared the step"
            fixed = _rerun(d, step)
        if not fixed:
            ctx["fixes"] += 1
            how = "AI repair"
            try:
                fixed = _ask_llm(d, step)
            except Exception:
                logger.exception("[%s] AI repair of %s failed", serial, step)
    if not fixed:
        logger.error("[%s] Could not repair %s; continuing the original flow", serial, step)
        return False
    logger.info("[%s] %s fixed: %s", serial, step, how)
    # A repair may reopen the video; redo earlier steps it undid.
    for prior in list(ctx["done"]):
        if not _check(d, prior):
            logger.warning("[%s] %s no longer holds after repairing %s", serial, prior, step)
            ensure(d, prior, ctx["runs"][prior])
    return True


# --- recipes ----------------------------------------------------------------

def _load_recipes():
    try:
        return json.loads(RECIPES_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _write_recipes(recipes):
    RECIPES_FILE.write_text(json.dumps(recipes, indent=2, ensure_ascii=False), encoding="utf-8")


def candidate_recipes(d, task, include_skipped=False):
    """Saved recipes for `task`, best match first: this exact profile, then same
    YouTube version + language, then same language, then everything else.
    Recipes that failed twice in a row are skipped."""
    own = profile_key(d)
    own_yt, own_lang = _yt_and_language(own)
    with _recipes_lock:
        recipes = _load_recipes()

    def rank(key):
        yt, lang = _yt_and_language(key)
        return 0 if key == own else 1 if (yt, lang) == (own_yt, own_lang) else 2 if lang == own_lang else 3

    found = [(key, tasks[task]) for key, tasks in recipes.items()
             if tasks.get(task) and (include_skipped or tasks.get("_recipe_stats", {}).get(task, {})
                                     .get("consecutive_failures", 0) < 2)]
    return sorted(found, key=lambda item: rank(item[0]))


def _record_recipe_result(source, task, passed):
    with _recipes_lock:
        recipes = _load_recipes()
        if source not in recipes:
            return
        stats = recipes[source].setdefault("_recipe_stats", {}).setdefault(
            task, {"passes": 0, "failures": 0, "consecutive_failures": 0})
        stats["passes" if passed else "failures"] = stats.get("passes" if passed else "failures", 0) + 1
        stats["consecutive_failures"] = 0 if passed else stats.get("consecutive_failures", 0) + 1
        _write_recipes(recipes)


def recipe_context(d, task, limit=3):
    with _recipes_lock:
        recipes = _load_recipes()
    failures = _recipe_failures.get((d.serial, task), {})
    return [{"profile": source, "steps": steps, "notes": recipes.get(source, {}).get("_notes", {}).get(task, ""),
             "last_failure": failures.get(source)}
            for source, steps in candidate_recipes(d, task, include_skipped=True)[:limit]]


def try_recipes(d, task, verify):
    """Replay candidate recipes until one passes `verify()`. A recipe borrowed
    from another profile that works is saved under this phone's profile."""
    own = profile_key(d)
    for source, steps in candidate_recipes(d, task):
        logger.info("[%s] Trying saved %s recipe from %s (%d steps)", d.serial, task, source, len(steps))
        try:
            result = _run_steps(d, steps)
        except Exception as exc:
            logger.exception("[%s] Saved %s recipe from %s raised an error", d.serial, task, source)
            result = {"success": False, "error": str(exc)}
        if result.get("success") and verify():
            logger.info("[%s] Saved %s recipe from %s worked", d.serial, task, source)
            _record_recipe_result(source, task, True)
            _recipe_failures.get((d.serial, task), {}).pop(source, None)
            if source != own:
                save_recipe(d, task, steps, notes=f"reused from {source}")
            return {**result, "recipe_from": source}
        logger.warning("[%s] Saved %s recipe from %s did not reach the target: %s",
                       d.serial, task, source, result.get("error", "verification failed"))
        _record_recipe_result(source, task, False)
        _recipe_failures.setdefault((d.serial, task), {})[source] = {
            "failed_step": result.get("failed_step"), "error": result.get("error", "verification failed")}
        if result.get("taps_done"):
            d.press("back")  # it got partway (e.g. opened a menu); close it
            time.sleep(1)
    return None


def save_recipe(d, task, steps, notes):
    """Persist verified steps for `task` under this phone's profile."""
    if task not in RECIPE_TASKS and not task.startswith("set_resolution_"):
        return {"success": False, "error": f"task must be one of: {', '.join(RECIPE_TASKS)}"}
    if not notes.strip():
        return {"success": False, "error": "notes is required (cause, and what differed on this phone)"}
    for i, step in enumerate(steps):
        if not isinstance(step, dict) or step.get("action") not in RECIPE_ACTIONS:
            return {"success": False, "error": f"Step {i}: action must be one of {sorted(RECIPE_ACTIONS)}"}
        if step["action"] in ("tap", "switch_on") and not _selectors_for(step.get("match", {})):
            return {"success": False,
                    "error": f"Step {i}: {step['action']} needs match.resource_id, match.desc or match.text"}
        if step["action"] == "press_key" and step.get("key") not in KEYCODES:
            return {"success": False, "error": f"Step {i}: key must be one of {sorted(KEYCODES)}"}
        if step["action"] == "tap_text" and not str(step.get("text", "")).strip():
            return {"success": False, "error": f"Step {i}: tap_text needs text"}
        if step["action"] == "swipe" and step.get("direction") not in DIRECTIONS:
            return {"success": False, "error": f"Step {i}: swipe needs direction {DIRECTIONS}"}
        if step["action"] in ("tap_point", "long_press") and not _selectors_for(step.get("match", {})) and not (
                isinstance(step.get("x"), (int, float)) and isinstance(step.get("y"), (int, float))
                and 0 <= step["x"] <= 1 and 0 <= step["y"] <= 1):
            return {"success": False, "error": f"Step {i}: {step['action']} needs x and y as screen fractions (0-1)"}
    key = profile_key(d)
    with _recipes_lock:
        recipes = _load_recipes()
        recipes.setdefault(key, {})[task] = steps
        recipes[key].setdefault("_notes", {})[task] = notes
        recipes[key].setdefault("_recipe_stats", {})[task] = {"passes": 0, "failures": 0, "consecutive_failures": 0}
        _write_recipes(recipes)
    logger.info("[%s] Saved %s recipe for %s", d.serial, task, key)
    return {"success": True, "device_profile": key, "task": task, "steps_saved": len(steps)}


def _run_steps(d, steps):
    """Replay one recipe; every tap is checked for leaving YouTube."""
    taps = 0
    for i, step in enumerate(steps):
        action, match = step.get("action"), step.get("match", {})
        point = None
        if action == "wait":
            time.sleep(min(float(step.get("seconds", 1)), 10))
        elif action == "press_key":
            d.press(KEYCODES[step["key"]])
        elif action == "reveal_player_controls":
            reveal_player_controls(d)
        elif action == "swipe":
            width, height = d.window_size()
            area = step.get("area")
            _swipe(d, step["direction"], area and (area[0] * width, area[1] * height,
                                                   area[2] * width, area[3] * height))
        elif action == "switch_on":
            _wait_for(lambda: _find_match(d, match) is not None, timeout=5)
            result = switch_on(d, match)
            if not result["success"]:
                return {**result, "taps_done": taps, "failed_step": i,
                        "error": f"Recipe step {i}: {result['error']}"}
            taps += int(result["changed"])
        elif action == "tap":
            if not _selectors_for(match):
                return {"success": False, "taps_done": taps, "failed_step": i,
                        "error": f"Recipe step {i} has no usable match"}
            if step.get("reveal_player_controls"):
                element = _reveal_and_find(d, match)
            else:
                _wait_for(lambda: _find_match(d, match) is not None, timeout=5)
                element = _find_match(d, match)
            if element is None or not _tap(element):
                return {"success": False, "taps_done": taps, "failed_step": i,
                        "error": f"Recipe step {i} target not found: {match}"}
        elif action == "tap_text":
            deadline = time.time() + 5
            while point is None and time.time() < deadline:
                point = _find_text(d, step["text"])
            if point is None:
                return {"success": False, "taps_done": taps, "failed_step": i,
                        "error": f"Recipe step {i}: text {step['text']!r} not on screen"}
            d.click(*point)
        elif action in ("tap_point", "long_press"):
            element = _find_match(d, match) if match else None
            if element is not None:
                b = element.info["bounds"]
                point = (b["left"] + b["right"]) // 2, (b["top"] + b["bottom"]) // 2
            else:
                width, height = d.window_size()
                point = int(step["x"] * width), int(step["y"] * height)
            d.long_click(*point, 1.0) if action == "long_press" else d.click(*point)
        else:
            return {"success": False, "taps_done": taps, "failed_step": i,
                    "error": f"Recipe step {i} has unknown action '{action}'"}
        if action in ("tap", "tap_text", "tap_point", "long_press"):
            taps += 1
            time.sleep(float(step.get("wait_after", 1)))
            other = _left_youtube(d)
            if other:
                d.press("back")
                return {"success": False, "taps_done": 0, "failed_step": i,
                        "error": f"Recipe step {i} opened another app ({other}); pressed back"}
        elif action == "swipe":
            time.sleep(1)
    return {"success": True, "via": "learned_recipe", "steps": len(steps)}


# --- UI helpers ---------------------------------------------------------------

def _parse_bounds(bounds):
    nums = [int(v) for v in re.findall(r"-?\d+", bounds or "")]
    return tuple(nums) if len(nums) == 4 else None


def _tap(obj):
    """Tap a UiObject's centre right away (auto-hiding player controls fade fast)."""
    try:
        b = obj.info["bounds"]
    except Exception:
        return False
    obj.session.click((b["left"] + b["right"]) // 2, (b["top"] + b["bottom"]) // 2)
    return True


def _selectors_for(match):
    """Exact selectors, most specific first. Exact matters: "More" must not hit
    "More options". YouTube pads some labels with a trailing space."""
    if match.get("resource_id"):
        return [{"resourceId": match["resource_id"]}]
    for key, field in (("desc", "description"), ("text", "text")):
        value = match.get(key)
        if value:
            return [{field: v} for v in dict.fromkeys([value, value.strip(), value.strip() + " "])]
    return []


def _find_match(d, match):
    """The element for `match`; `nth` picks among identical labels."""
    for selector in _selectors_for(match):
        found = d(**selector)
        if found.exists:
            return found[min(match.get("nth", 0), found.count - 1)]
    return None


def _reveal_and_find(d, match, attempts=5):
    """Tap the player to show its auto-hiding controls, then look for `match`."""
    for _ in range(attempts):
        found = _find_match(d, match)
        if found is not None:
            return found
        player = d(**PLAYER)
        if player.exists:
            _tap(player)
        time.sleep(0.3)
        found = _find_match(d, match)
        if found is not None:
            return found
        time.sleep(1)
    return None


def _left_youtube(d):
    package = _package(d)
    return package if package and package != YT else None


def switch_for(d, match):
    """The on/off switch in the same row as the label `match`, or None."""
    label = _find_match(d, match)
    if label is None:
        return None
    row = label.info["bounds"]
    switches = d(checkable=True)
    for i in range(switches.count):
        b = switches[i].info["bounds"]
        centre = (b["top"] + b["bottom"]) / 2
        if row["top"] - ROW_SLACK_PX <= centre <= row["bottom"] + ROW_SLACK_PX:
            return switches[i]
    return None


def switch_on(d, match):
    """Make sure the switch next to `match` is ON; an already-on switch is left alone."""
    switch = switch_for(d, match)
    if switch is None:
        return {"success": False, "error": f"No switch found next to {match}"}
    if switch.info.get("checked"):
        return {"success": True, "changed": False}
    _tap(switch)
    time.sleep(1)
    switch = switch_for(d, match)
    if switch is None or not switch.info.get("checked"):
        return {"success": False, "error": f"Tapped the switch next to {match} but it is still off"}
    return {"success": True, "changed": True}


def _stable_match(element, screen):
    """Re-findable descriptor: unique resource-id, else exact desc, else exact text
    (with `nth` when the label is not unique on screen)."""
    rid = element.get("resource_id")
    if rid and sum(1 for e in screen if e.get("resource_id") == rid) == 1:
        return {"resource_id": rid}
    for key in ("desc", "text"):
        value = element.get(f"_raw_{key}")
        if value:
            same = [e for e in screen if e.get(f"_raw_{key}") == value]
            match = {key: value}
            if len(same) > 1:
                match["nth"] = same.index(element)
            return match
    return {"resource_id": rid} if rid else {}


def _same_row(a, b):
    ba, bb = _parse_bounds(a["bounds"]), _parse_bounds(b["bounds"])
    return bool(ba and bb) and not (bb[3] < ba[1] - ROW_SLACK_PX or bb[1] > ba[3] + ROW_SLACK_PX)


def _row_switch(element, screen):
    """The switch a tap on `element` toggles: the switch itself, one inside a
    settings row, or one beside a plain text label. A button that merely shares
    a row with a switch (e.g. More options next to Autoplay) is not a switch tap."""
    if element.get("checkable"):
        return element
    x1, y1, x2, y2 = _parse_bounds(element["bounds"])
    for switch in (e for e in screen if e.get("checkable")):
        s1, t1, s2, t2 = _parse_bounds(switch["bounds"])
        in_row = (x1 <= (s1 + s2) / 2 <= x2 and y1 <= (t1 + t2) / 2 <= y2
                  and y1 >= t1 - 2 * ROW_SLACK_PX and y2 <= t2 + 2 * ROW_SLACK_PX)
        beside_label = not element["clickable"] and element["_raw_text"] and _same_row(element, switch)
        if in_row or beside_label:
            return switch
    return None


def _signature(serial):
    """What a screen shows, ignoring digits so a playing video's clock is not a change."""
    def plain(value):
        return re.sub(r"\d", "", value or "")
    return ([(e["class"], e["resource_id"], plain(e["_raw_text"]), plain(e["_raw_desc"]), e["switch"],
              plain(e.get("ocr_text"))) for e in _screens.get(serial, [])]
            + [plain(line["text"]) for line in _screen_text.get(serial, [])])


_ocr_warned = False


def _ocr(d):
    """Text lines drawn on screen, with their centres. YouTube draws some menus
    (e.g. Quality) without accessibility labels; the original script reads them with OCR too."""
    global _ocr_warned
    try:
        import cv2
        import pytesseract
        gray = cv2.cvtColor(d.screenshot(format="opencv"), cv2.COLOR_BGR2GRAY)
        image = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)[1]
        if image.mean() < 128:
            image = 255 - image  # dark theme: dark text on a light background reads best
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    except Exception:
        if not _ocr_warned:
            _ocr_warned = True
            logger.warning("OCR unavailable; text drawn without labels will not be readable", exc_info=True)
        return []
    lines = {}
    for i, word in enumerate(data["text"]):
        if not word.strip() or float(data["conf"][i]) < 50:
            continue
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        line = lines.setdefault((data["block_num"][i], data["par_num"][i], data["line_num"][i]),
                                {"words": [], "box": [x, y, x + w, y + h]})
        line["words"].append(word.strip())
        box = line["box"]
        line["box"] = [min(box[0], x), min(box[1], y), max(box[2], x + w), max(box[3], y + h)]
    return [{"text": " ".join(line["words"]), "x": (line["box"][0] + line["box"][2]) // 2,
             "y": (line["box"][1] + line["box"][3]) // 2}
            for line in lines.values() if len(re.sub(r"\W", "", " ".join(line["words"]))) >= 2]


def _find_text(d, text):
    """Centre of `text` on screen: an accessible label first, else OCR (exact line, then shortest containing it)."""
    for match in ({"text": text}, {"desc": text}):
        found = _find_match(d, match)
        if found is not None:
            b = found.info["bounds"]
            return (b["left"] + b["right"]) // 2, (b["top"] + b["bottom"]) // 2
    wanted = text.strip().lower()
    lines = [line for line in _ocr(d) if wanted in line["text"].lower()]
    if not lines:
        return None
    best = min(lines, key=lambda line: (line["text"].strip().lower() != wanted, len(line["text"])))
    return best["x"], best["y"]


def _screenshot_b64(d):
    image = d.screenshot().convert("RGB")
    if image.width > 720:
        image = image.resize((720, round(image.height * 720 / image.width)))
    out = io.BytesIO()
    image.save(out, format="JPEG", quality=70)
    return base64.b64encode(out.getvalue()).decode("ascii")


def get_screen(d):
    """Numbered elements on screen (tap_element takes these numbers) plus text read by OCR."""
    try:
        elements = []
        for node in ET.fromstring(d.dump_hierarchy(compressed=False).encode("utf-8")).iter("node"):
            a = node.attrib
            bounds = _parse_bounds(a.get("bounds"))
            if a.get("package") == "com.android.systemui" or not bounds or \
                    bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
                continue
            text, desc, rid = a.get("text", ""), a.get("content-desc", ""), a.get("resource-id", "")
            clickable, checkable = a.get("clickable") == "true", a.get("checkable") == "true"
            scrollable = a.get("scrollable") == "true"
            if not (text or desc or clickable or checkable or scrollable):
                continue
            elements.append({
                "i": len(elements), "class": a.get("class", "").rsplit(".", 1)[-1], "resource_id": rid,
                "text": text[:80], "desc": desc[:80], "_raw_text": text, "_raw_desc": desc,
                "clickable": clickable, "checkable": checkable, "scrollable": scrollable,
                "switch": ("on" if a.get("checked") == "true" else "off") if checkable else "",
                "bounds": a.get("bounds")})
            if len(elements) >= MAX_ELEMENTS:
                break
        width, height = d.window_size()
        # OCR text goes to the smallest unlabeled, not screen-sized element containing it;
        # the rest is listed as screen_text for tap_text.
        screen_text = []
        for line in _ocr(d):
            holders = [e for e in elements if not (e["_raw_text"] or e["_raw_desc"])
                       and (lambda b: b[0] <= line["x"] <= b[2] and b[1] <= line["y"] <= b[3]
                            and (b[2] - b[0]) * (b[3] - b[1]) < 0.25 * width * height)(_parse_bounds(e["bounds"]))]
            labeled = [e for e in elements if (e["_raw_text"] or e["_raw_desc"])
                       and (lambda b: b[0] <= line["x"] <= b[2] and b[1] <= line["y"] <= b[3])(_parse_bounds(e["bounds"]))
                       and line["text"].lower() in (e["_raw_text"] + " " + e["_raw_desc"]).lower()]
            if labeled:
                continue  # already known from its accessibility label
            if holders:
                holder = min(holders, key=lambda e: (lambda b: (b[2] - b[0]) * (b[3] - b[1]))(_parse_bounds(e["bounds"])))
                holder["ocr_text"] = (holder.get("ocr_text", "") + " " + line["text"]).strip()
            elif len(screen_text) < 40:
                screen_text.append(line)
        _screens[d.serial], _screen_text[d.serial] = elements, screen_text
        result = {"activity": d.app_current().get("activity"),
                  "orientation": "LANDSCAPE" if width > height else "PORTRAIT",
                  "screen_size": f"{width}x{height}", "truncated": len(elements) >= MAX_ELEMENTS,
                  "elements": [{k: (v.replace(f"{YT}:id/", "yt:") if k == "resource_id" else v)
                                for k, v in e.items() if not k.startswith("_") and v != "" and v is not False}
                               for e in elements],
                  "screen_text": screen_text}
        if LLM_VISION:
            result["_image"] = _screenshot_b64(d)
        return result
    except Exception as exc:
        logger.exception("[%s] Reading the screen failed", d.serial)
        return {"error": str(exc)}


def _after_tap(d, result):
    time.sleep(1)
    other = _left_youtube(d)
    if other:
        result["warning"] = f"This tap opened another app ({other}). Press back; don't save this step."
    return result


def _point_step(d, action, x, y):
    width, height = d.window_size()
    return {"action": action, "x": round(x / width, 4), "y": round(y / height, 4)}


def _element(d, index):
    screen = _screens.get(d.serial)
    if not screen:
        return None, {"success": False, "error": "Call get_screen first"}
    if not isinstance(index, int) or not 0 <= index < len(screen):
        return None, {"success": False, "error": f"No element #{index} on the last screen (0..{len(screen) - 1})"}
    return screen[index], None


def tap_element(d, index, reveal_player_controls_first=False):
    """Tap element #index from the latest get_screen. A switch that is already
    on (or its row) is never tapped, since that would turn it off."""
    element, error = _element(d, index)
    if error:
        return error
    screen = _screens[d.serial]
    switch = _row_switch(element, screen)
    x1, y1, x2, y2 = _parse_bounds(element["bounds"])
    match = _stable_match(element, screen)
    recipe_step = ({"action": "tap", "match": match, "reveal_player_controls": reveal_player_controls_first}
                   if match else {"action": "tap_text", "text": element["ocr_text"]} if element.get("ocr_text")
                   else _point_step(d, "tap_point", (x1 + x2) // 2, (y1 + y2) // 2))
    if switch is not None:
        label = element if element["_raw_text"] else next(
            (e for e in screen if e["_raw_text"] and not e.get("checkable") and _same_row(switch, e)), element)
        recipe_step = {"action": "switch_on", "match": _stable_match(label, screen)}
        if switch["switch"] == "on":
            return {"success": True, "already_on": True, "tapped": None, "recipe_step": recipe_step,
                    "note": "Not tapped: this switch is already on (tapping would turn it off)."}
    if reveal_player_controls_first and match:
        found = _reveal_and_find(d, match)
        if found is None or not _tap(found):
            return {"success": False, "error": f"Element {match} not found after revealing controls"}
    else:
        d.click((x1 + x2) // 2, (y1 + y2) // 2)
    tapped = {k: element[k] for k in ("class", "text", "desc", "ocr_text") if element.get(k)}
    return _after_tap(d, {"success": True, "tapped": tapped, "recipe_step": recipe_step})


def tap_text(d, text):
    """Tap text seen on screen, including text drawn without an accessibility label."""
    text = str(text)
    point = _find_text(d, text)
    if point is None:
        return {"success": False, "error": f"{text!r} is not on screen (checked labels and OCR)"}
    d.click(*point)
    return _after_tap(d, {"success": True, "tapped": {"text": text, "x": point[0], "y": point[1]},
                          "recipe_step": {"action": "tap_text", "text": text}})


def tap_point(d, x, y):
    """Tap exact screen coordinates; for controls with no element and no readable text."""
    width, height = d.window_size()
    if not (isinstance(x, int) and isinstance(y, int) and 0 <= x < width and 0 <= y < height):
        return {"success": False, "error": f"x and y must be integers inside the {width}x{height} screen"}
    d.click(x, y)
    return _after_tap(d, {"success": True, "tapped": {"x": x, "y": y}, "recipe_step": _point_step(d, "tap_point", x, y)})


def long_press(d, index):
    """Press and hold element #index for a second (opens context menus)."""
    element, error = _element(d, index)
    if error:
        return error
    x1, y1, x2, y2 = _parse_bounds(element["bounds"])
    d.long_click((x1 + x2) // 2, (y1 + y2) // 2, 1.0)
    match = _stable_match(element, _screens[d.serial])
    step = {"action": "long_press", "match": match} if match else \
        _point_step(d, "long_press", (x1 + x2) // 2, (y1 + y2) // 2)
    return _after_tap(d, {"success": True, "recipe_step": step})


def press_key(d, key):
    if key not in KEYCODES:
        return {"success": False, "error": f"Unknown key '{key}'. Allowed: {', '.join(KEYCODES)}"}
    d.press(KEYCODES[key])
    time.sleep(1)
    return {"success": True, "recipe_step": {"action": "press_key", "key": key}}


def _swipe(d, direction, box=None):
    """Swipe so content moves `direction`, inside `box` (x1, y1, x2, y2) or the screen's middle."""
    width, height = d.window_size()
    x1, y1, x2, y2 = box or (0.1 * width, 0.15 * height, 0.9 * width, 0.85 * height)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    dx, dy = 0.35 * (x2 - x1), 0.35 * (y2 - y1)
    start, end = {"up": ((cx, cy + dy), (cx, cy - dy)), "down": ((cx, cy - dy), (cx, cy + dy)),
                  "left": ((cx + dx, cy), (cx - dx, cy)), "right": ((cx - dx, cy), (cx + dx, cy))}[direction]
    d.swipe(int(start[0]), int(start[1]), int(end[0]), int(end[1]), 0.3)


def swipe(d, direction, index=None):
    """Swipe so content moves `direction` ('up' shows what is further down). With
    `index`, swipe inside that element (a bottom sheet, list or menu), else the whole screen."""
    if direction not in DIRECTIONS:
        return {"success": False, "error": f"direction must be one of {DIRECTIONS}"}
    step = {"action": "swipe", "direction": direction}
    box = None
    if index is not None:
        element, error = _element(d, index)
        if error:
            return error
        box = _parse_bounds(element["bounds"])
        width, height = d.window_size()
        step["area"] = [round(box[0] / width, 4), round(box[1] / height, 4),
                        round(box[2] / width, 4), round(box[3] / height, 4)]
    _swipe(d, direction, box)
    time.sleep(1)
    return {"success": True, "recipe_step": step}


def wait(d, seconds=2):
    """Let the screen finish loading."""
    seconds = min(max(float(seconds), 0.5), 10)
    time.sleep(seconds)
    return {"success": True, "recipe_step": {"action": "wait", "seconds": seconds}}


def reveal_player_controls(d):
    player = d(**PLAYER)
    if not player.exists:
        return {"success": False, "error": "Player not found on screen"}
    _tap(player)
    time.sleep(0.3)
    return {"success": True, "recipe_step": {"action": "reveal_player_controls"}}


def launch_youtube(d):
    """Restart YouTube on its home screen."""
    d.app_start(YT, stop=True, wait=True)
    time.sleep(2)
    return {"success": True}


def _stats_setting_on(d):
    switch = switch_for(d, {"text": "Enable stats for nerds"})
    return bool(switch is not None and switch.info.get("checked"))


def enable_stats_in_app_settings(d):
    """Recipe-only: turn on YouTube's "Enable stats for nerds" app setting.
    Restarts YouTube on its home screen; reopen the video afterwards."""
    if not candidate_recipes(d, "enable_stats_in_settings"):
        return {"success": False, "error": "No saved recipe for turning on 'Enable stats for nerds' on this phone"}
    launch_youtube(d)
    learned = try_recipes(d, "enable_stats_in_settings", lambda: _stats_setting_on(d))
    return learned or {"success": False, "error": "Saved recipe(s) for the 'Enable stats for nerds' setting failed"}


def _schema(name, description, properties=None, required=()):
    return {"type": "function", "function": {"name": name, "description": description, "parameters": {
        "type": "object", "properties": properties or {}, "required": list(required)}}}


def _schemas(d, names):
    tasks = list(RECIPE_TASKS) + ([_task(d, "resolution")] if _runs[d.serial]["resolution"] else [])
    all_schemas = {
        "get_screen": _schema("get_screen", "Return the indexed UI elements and screen_text (text read by OCR, "
                                            "with x/y) for menus drawn without labels."),
        "tap_element": _schema(
            "tap_element", "Tap an element from the latest screen. Already-on switches and their rows are not tapped.",
            {"index": {"type": "integer"}, "reveal_player_controls_first": {"type": "boolean"}}, ["index"]),
        "tap_text": _schema("tap_text", "Tap visible text: an element label, an element's ocr_text, or screen_text.",
                            {"text": {"type": "string"}}, ["text"]),
        "tap_point": _schema("tap_point", "Tap exact screen coordinates (last resort; see screen_size).",
                             {"x": {"type": "integer"}, "y": {"type": "integer"}}, ["x", "y"]),
        "long_press": _schema("long_press", "Press and hold an element for one second.",
                              {"index": {"type": "integer"}}, ["index"]),
        "swipe": _schema("swipe", "Swipe so content moves in `direction` ('up' reveals what is further down). Give "
                                  "`index` to swipe inside that element (e.g. a bottom sheet or list), else the screen.",
                         {"direction": {"type": "string", "enum": list(DIRECTIONS)}, "index": {"type": "integer"}},
                         ["direction"]),
        "press_key": _schema("press_key", "Press a named Android key.",
                             {"key": {"type": "string", "enum": list(KEYCODES)}}, ["key"]),
        "wait": _schema("wait", "Wait for the screen to finish loading, then see it again.",
                        {"seconds": {"type": "number"}}),
        "reveal_player_controls": _schema("reveal_player_controls", "Reveal auto-hiding player controls."),
        "get_player_state": _schema("get_player_state", "Return player, stats, ad, orientation and media state."),
        "save_recipe": _schema(
            "save_recipe", "Save only the minimal verified actions. Notes must state the cause and phone difference.",
            {"task": {"type": "string", "enum": tasks}, "steps": {"type": "array", "items": {"type": "object"}},
             "notes": {"type": "string"}}, ["task", "steps", "notes"]),
        "step_done": _schema("step_done", "End this repair attempt with its verified result.",
                             {"fixed": {"type": "boolean"}, "summary": {"type": "string"},
                              "needs_human": {"type": "boolean"}}, ["fixed", "summary"]),
        "launch_youtube": _schema("launch_youtube", "Restart YouTube on its home screen."),
        "open_video_url": _schema("open_video_url", "Open the configured test video and wait for its player."),
        "run_original_step": _schema("run_original_step", "Re-run the test's own code for this step, "
                                                          "e.g. after dismissing a popup."),
        "enable_stats_in_app_settings": _schema(
            "enable_stats_in_app_settings", "Replay a saved recipe for YouTube's General 'Enable stats for nerds' "
                                            "setting. Restarts YouTube; reopen the video afterwards."),
    }
    return [all_schemas[name] for name in names]


def _dispatch(d, step, name, args):
    ctx = _runs[d.serial]
    if name == "get_screen":
        before = _signature(d.serial)
        result = get_screen(d)
        return {**result, "screen_changed": before != _signature(d.serial)}
    if name == "get_player_state":
        return player_state(d)
    if name == "save_recipe":
        return save_recipe(d, str(args.get("task", "")), args.get("steps") or [], str(args.get("notes", "")))
    if name == "step_done":
        ctx["fix_result"] = {"fixed": bool(args.get("fixed")), "summary": str(args.get("summary", "")),
                             "needs_human": bool(args.get("needs_human"))}
        return ctx["fix_result"]
    if name == "launch_youtube":
        return launch_youtube(d)
    if name == "open_video_url":
        _call(d, ctx["runs"]["open_url"])
        return {"success": True}
    if name == "run_original_step":
        result = _call(d, ctx["runs"][step])
        if step == "resolution":
            ctx["resolution_ok"] = bool(result)
        return {"success": result is not False, "returned": result}
    if name == "enable_stats_in_app_settings":
        return enable_stats_in_app_settings(d)
    before = _signature(d.serial)
    result = {"tap_element": tap_element, "tap_text": tap_text, "tap_point": tap_point, "long_press": long_press,
              "swipe": swipe, "press_key": press_key, "wait": wait,
              "reveal_player_controls": reveal_player_controls}[name](d, **args)
    if name in SCREEN_ACTIONS and result.get("success") and not result.get("already_on"):
        screen = get_screen(d)
        result = {**result, **screen, "screen_changed": before != _signature(d.serial)}
    return result


def _chat(messages, tools):
    url = f"{LLM_BASE_URL}/chat/completions"
    started = time.time()
    try:
        response = requests.post(
            url, headers={"Authorization": f"Bearer {LLM_API_KEY}"},
            json={"model": LLM_MODEL, "messages": messages, "tools": tools, "tool_choice": "required",
                  "temperature": 0.2, "top_p": 0.95, "max_tokens": 8192}, timeout=300)
    except requests.RequestException as exc:
        raise LLMError(f"LLM request to {url} failed: {exc!r}") from exc
    if response.status_code != 200:
        raise LLMError(f"LLM {url} returned HTTP {response.status_code} for model {LLM_MODEL}: "
                       f"{response.text[:2000]}")
    try:
        message = response.json()["choices"][0]["message"]
    except (ValueError, KeyError, IndexError) as exc:
        raise LLMError(f"LLM {url} returned an unexpected reply: {response.text[:2000]}") from exc
    logger.info("LLM replied in %.1fs: %s", time.time() - started,
                json.dumps(message, ensure_ascii=False)[:2000])
    return message


def _shrink_old_screens(messages):
    """Only the newest tool result with elements keeps the full screen."""
    copied, newest_kept = [dict(m) for m in messages], False
    for message in reversed(copied):
        if message.get("role") != "tool":
            continue
        try:
            result = json.loads(message["content"])
        except (TypeError, ValueError):
            continue
        if not isinstance(result, dict) or "elements" not in result:
            continue
        if not newest_kept:
            newest_kept = True
            continue
        message["content"] = json.dumps({k: result[k] for k in ("success", "tapped", "recipe_step",
                                                                "screen_changed", "error") if k in result})
    images = [m for m in copied if m.get("role") == "user" and isinstance(m.get("content"), list)]
    for message in images[:-1]:
        message["content"] = "[older screenshot removed]"
    return copied


def _target(d, step):
    resolution = _runs[d.serial]["resolution"]
    return f"{STEPS[step][0]}: {resolution}" if step == "resolution" else STEPS[step][0]


def _system_prompt(d, step):
    return f"""You repair one failed YouTube test step on one Android phone.
Target step: {step}
Target state: {_target(d, step)}
Do not continue to a later test step. Only the tools available for this repair can be called.
Take one action per turn. If screen_changed is false, that action did nothing; do not repeat it.
Ways to act: tap_element (by index), tap_text (any text you see, including ocr_text and screen_text),
long_press, swipe (the whole screen, or inside one element such as a bottom sheet or list by its index),
press_key, wait (when a screen is still loading), reveal_player_controls, and tap_point as a last resort.
Some YouTube menus are drawn without accessibility labels: their text appears only as an element's
ocr_text or in screen_text (read by OCR from the screenshot). Use tap_text for those.
Stay inside YouTube, dismiss blocking popups, and never tap ads or links into another app.
Never tap a switch or its row when its current screen state is "on".
launch_youtube restarts YouTube; open_video_url replaces the current screen with the configured video.
Verify player targets with get_player_state. For an app-setting switch, verify get_screen shows switch "on".

If run_original_step or enable_stats_in_app_settings does not reach the target:
1. Navigate manually (e.g. for stats: YouTube -> You -> Settings -> General -> enable "Stats for nerds").
2. Verify the fix with get_player_state or get_screen.
3. Save the minimal manual steps with save_recipe (copy each action's recipe_step).
4. Call step_done.

If the safe fix requires a person, call step_done with fixed=false and needs_human=true.
Known cause: {KNOWN_CAUSES[step]}"""


def _ask_llm(d, step):
    """Constrained LLM repair of one step; True when it ends verified."""
    serial, ctx = d.serial, _runs[d.serial]
    ctx["fix_result"] = None
    names = COMMON_TOOLS + STEPS[step][2]
    recipes = recipe_context(d, _task(d, step) or "dismiss_popup")
    if step == "stats":
        recipes += recipe_context(d, "enable_stats_in_settings")
    start_message = {
        "failed_step": step, "target": _target(d, step), "done": [f"{name} ✓" for name in ctx["done"]],
        "failure": player_state(d), "known_recipes": recipes, "current_screen": get_screen(d)}
    screenshot = start_message["current_screen"].pop("_image", None)
    messages = [{"role": "system", "content": _system_prompt(d, step)},
                {"role": "user", "content": json.dumps(start_message, default=str, ensure_ascii=False)}]
    logger.warning("[%s] Asking the LLM (%s at %s) to repair %s; player state %s; %d known recipe(s); "
                   "%d element(s) on screen", serial, LLM_MODEL, LLM_BASE_URL, step, start_message["failure"],
                   len(recipes), len(start_message["current_screen"].get("elements", [])))
    calls = stuck = 0
    summary = None
    for _ in range(MAX_LLM_TURNS):
        if ctx["should_stop"]():
            summary = "test is stopping"
            break
        if screenshot:  # newest screen, sent after the tool results that produced it
            messages.append({"role": "user", "content": [
                {"type": "text", "text": "Current screen:"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{screenshot}"}}]})
            screenshot = None
        message = _chat(_shrink_old_screens(messages), _schemas(d, names))
        tool_calls = message.get("tool_calls") or []
        # Keep only what the model needs back; its reasoning text would grow every turn.
        messages.append({"role": "assistant", "content": message.get("content") or "",
                         **({"tool_calls": tool_calls} if tool_calls else {})})
        if not tool_calls:
            logger.warning("[%s] LLM replied without a tool call; asking again", serial)
            messages.append({"role": "user", "content": "Call exactly one of the supplied tools."})
            continue
        for i, call in enumerate(tool_calls):
            name = call["function"]["name"]
            if i:
                result = {"success": False, "error": "not executed: one action per turn"}
            elif calls >= FIX_MAX_TOOL_CALLS:
                result = {"success": False, "error": f"fix tool-call budget ({FIX_MAX_TOOL_CALLS}) reached"}
            else:
                try:
                    args = call["function"].get("arguments") or {}
                    args = json.loads(args) if isinstance(args, str) else args
                    if not isinstance(args, dict) or name not in names:
                        raise ValueError
                    calls += 1
                    logger.info("[%s] LLM action: %s %s", serial, name, args)
                    result = _dispatch(d, step, name, args)
                except (ValueError, TypeError):
                    logger.warning("[%s] LLM sent an invalid tool call: %s", serial, call.get("function"))
                    result = {"success": False, "error": "Invalid tool name or arguments."}
                except Exception as exc:
                    logger.exception("[%s] LLM tool %s raised an error", serial, name)
                    result = {"success": False, "error": str(exc)}
                no_progress = (result.get("success") is False or result.get("already_on")
                               or result.get("screen_changed") is False)
                stuck = stuck + 1 if no_progress else 0
                if stuck >= 2:
                    result["warning"] = (f"{stuck} actions in a row made no progress. Call get_screen and try a "
                                         "different element or approach, or call step_done.")
                logger.info("[%s] Tool result: %s", serial,
                            json.dumps({k: v for k, v in result.items() if k not in ("elements", "_image")},
                                       default=str, ensure_ascii=False)[:1000])
            screenshot = result.pop("_image", None) or screenshot
            messages.append({"role": "tool", "tool_call_id": call.get("id", ""),
                             "content": json.dumps(result, default=str, ensure_ascii=False)})
        if ctx["fix_result"] is not None:
            break
        if calls >= FIX_MAX_TOOL_CALLS:
            summary = f"tool-call budget ({FIX_MAX_TOOL_CALLS}) reached without step_done"
            break
        if stuck >= 4:
            summary = "4 actions in a row made no progress"
            break
    result = ctx["fix_result"] or {
        "fixed": False, "summary": summary or f"repair ended without step_done after {calls} tool call(s)"}
    logger.info("[%s] LLM repair of %s: %s", serial, step, result)
    if step == "resolution" and result["fixed"]:
        ctx["resolution_ok"] = True  # no on-screen check; the LLM verified it in the menu
    return result["fixed"] and _wait_for(lambda: _check(d, step))
