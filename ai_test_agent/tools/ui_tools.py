"""Generic screen tools + learned recipes: how the agent adapts to phones,
YouTube versions and languages the built-in locators don't cover.

When a built-in step (enter_fullscreen, enable_stats_for_nerds, ...) fails
on an unfamiliar device, the agent:
  1. get_screen          - sees the UI as an indexed element list (+ screenshot)
  2. tap_element / ...   - navigates by itself, like a human tester would
  3. get_player_state    - verifies the outcome with language-independent checks
  4. save_recipe         - stores the working steps for this device profile

Recipes are keyed by manufacturer/model/YouTube version/locale and replayed
automatically by the built-in steps next time, so a device only needs to be
"figured out" once - later runs (including the no-LLM play_youtube.py) are fast.

All actions are fixed, typed operations (tap a listed element, press a named
key, swipe a direction) - never raw shell text from the model.
"""

import base64
import io
import json
import threading
import time
import xml.etree.ElementTree as ET

from config import BASE_DIR, YOUTUBE_PACKAGE
from tools import adb_tools

RECIPES_FILE = BASE_DIR / "learned_recipes.json"
RECIPE_TASKS = (
    "enter_fullscreen",
    "enable_stats_for_nerds",
    # One-time app setting some phones/YouTube builds need before "Stats for
    # nerds" appears in the player menu (Settings -> General -> Enable stats for nerds).
    "enable_stats_in_settings",
    "skip_ads",
    "dismiss_popup",
    "start_playback",
)
_RECIPE_ACTIONS = {"tap", "switch_on", "press_key", "wait", "reveal_player_controls"}
_SWITCH_ROW_SLACK_PX = 60  # how far a switch may sit above/below its label's row

_MAX_ELEMENTS = 150
_MAX_LABEL_CHARS = 80
_SCREENSHOT_MAX_WIDTH = 720
_IGNORED_PACKAGES = ("com.android.systemui",)
_KEYCODES = {"back": 4, "home": 3, "enter": 66, "space": 62, "media_play_pause": 85, "escape": 111}
_RECIPE_STEP_TIMEOUT_SECONDS = 5

_LAST_SCREEN: dict[str, list[dict]] = {}
_RECIPE_FAILURES: dict[tuple[str, str], dict[str, dict]] = {}
_PROFILE_CACHE: dict[str, str] = {}
_recipes_lock = threading.Lock()


def _yt():
    from tools import youtube_tools  # lazy: youtube_tools imports this module

    return youtube_tools


def _driver(device_id: str):
    return _yt()._get_driver(device_id)


def _parse_bounds(bounds: str) -> tuple[int, int, int, int] | None:
    try:
        left_top, right_bottom = bounds.strip("[]").split("][")
        x1, y1 = map(int, left_top.split(","))
        x2, y2 = map(int, right_bottom.split(","))
        return x1, y1, x2, y2
    except ValueError:
        return None


def _short(value: str) -> str:
    value = (value or "").strip()
    return value if len(value) <= _MAX_LABEL_CHARS else value[: _MAX_LABEL_CHARS - 1] + "..."


def _screenshot_jpeg_b64(driver) -> str:
    """Downscaled JPEG keeps the image well under API size limits and
    cheap in tokens while staying readable."""
    img = driver.screenshot().convert("RGB")  # uiautomator2 returns a PIL image
    if img.width > _SCREENSHOT_MAX_WIDTH:
        img = img.resize((_SCREENSHOT_MAX_WIDTH, round(img.height * _SCREENSHOT_MAX_WIDTH / img.width)))
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=70)
    return base64.b64encode(out.getvalue()).decode("ascii")


def _selectors_for(match: dict) -> list[dict]:
    """Exact-match uiautomator2 selectors for a recipe/element match, most
    specific first. Exact (not prefix) matching matters: a prefix "More"
    would also hit "More options". YouTube pads some labels with a trailing
    space, so the stripped and space-padded forms are tried too."""
    if match.get("resource_id"):
        return [{"resourceId": match["resource_id"]}]
    for key, field in (("desc", "description"), ("text", "text")):
        value = match.get(key)
        if value:
            return [{field: v} for v in dict.fromkeys([value, value.strip(), value.strip() + " "])]
    return []


def _selector_for(match: dict) -> dict | None:
    selectors = _selectors_for(match)
    return selectors[0] if selectors else None


def _find_match(driver, match: dict):
    """Find the element for `match`; `nth` picks among identical labels."""
    for selector in _selectors_for(match):
        found = driver(**selector)
        if found.exists:
            return found[min(match.get("nth", 0), found.count - 1)]
    return None


def _stable_match(element: dict, screen: list[dict]) -> dict:
    """Best re-findable descriptor for an element: a unique resource-id,
    else its exact content-desc, else its exact text - with `nth` when the
    label isn't unique on screen."""
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


def switch_for(driver, match: dict):
    """The on/off switch in the same row as the label `match` (e.g. the
    text "Enable stats for nerds"), or None."""
    label = _find_match(driver, match)
    if label is None:
        return None
    row = label.info["bounds"]
    switches = driver(checkable=True)
    for i in range(switches.count):
        b = switches[i].info["bounds"]
        centre = (b["top"] + b["bottom"]) / 2
        if row["top"] - _SWITCH_ROW_SLACK_PX <= centre <= row["bottom"] + _SWITCH_ROW_SLACK_PX:
            return switches[i]
    return None


def switch_on(driver, match: dict) -> dict:
    """Make sure the switch next to `match` is ON. Unlike a plain tap this is
    safe to repeat: an already-on switch is left alone (a tap would turn it off)."""
    switch = switch_for(driver, match)
    if switch is None:
        return {"success": False, "error": f"No switch found next to {match}"}
    if switch.info.get("checked"):
        return {"success": True, "changed": False}
    _yt()._tap(switch)
    time.sleep(1)
    switch = switch_for(driver, match)
    if switch is None or not switch.info.get("checked"):
        return {"success": False, "error": f"Tapped the switch next to {match} but it is still off"}
    return {"success": True, "changed": True}


def _left_youtube(driver) -> str | None:
    """Return the foreground package if a tap took us out of YouTube."""
    package = _yt()._current_package(driver)
    return package if package and package != YOUTUBE_PACKAGE else None


# --- device profile / recipe store ---------------------------------------


def device_profile_key(device_id: str) -> str:
    if device_id not in _PROFILE_CACHE:
        p = adb_tools.get_device_profile(device_id)
        _PROFILE_CACHE[device_id] = "|".join(
            [p.get("manufacturer", "?"), p.get("model", "?"), f"yt{p.get('youtube_version', '?')}", p.get("locale", "?")]
        )
    return _PROFILE_CACHE[device_id]


def _load_recipes() -> dict:
    if not RECIPES_FILE.exists():
        return {}
    try:
        return json.loads(RECIPES_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _yt_and_language(profile_key: str) -> tuple[str, str]:
    """What decides YouTube's screens: app version + UI language. Only the
    language part of the locale counts - en-GB and en-IN show the same labels."""
    parts = profile_key.split("|")
    youtube = parts[2] if len(parts) > 2 else ""
    language = parts[3].split("-")[0].lower() if len(parts) > 3 else ""
    return youtube, language


def _yt_and_locale(profile_key: str) -> tuple[str, str]:
    """Grouping key for --ai-fallback (same as recipe sharing): YouTube version + language."""
    return _yt_and_language(profile_key)


def candidate_recipes(device_id: str, task: str, include_skipped: bool = False) -> list[tuple[str, list[dict]]]:
    """Every saved recipe for `task`, best match first. All are worth trying
    before failing / calling the AI - each replay is verified, and taps that
    leave YouTube are undone:
      1. this exact device profile
      2. same YouTube version + same language, another maker/model
      3. same language, another YouTube version
      4. everything else (other languages: labels may differ, so least likely)
    """
    own_key = device_profile_key(device_id)
    own_yt, own_lang = _yt_and_language(own_key)
    with _recipes_lock:
        recipes = _load_recipes()

    def rank(key: str) -> int:
        yt, lang = _yt_and_language(key)
        if key == own_key:
            return 0
        if yt == own_yt and lang == own_lang:
            return 1
        if lang == own_lang:
            return 2
        return 3

    found = [
        (key, tasks[task]) for key, tasks in recipes.items()
        if tasks.get(task) and (include_skipped
                                or tasks.get("_recipe_stats", {}).get(task, {}).get("consecutive_failures", 0) < 2)
    ]
    return sorted(found, key=lambda item: rank(item[0]))  # stable: keeps file order within a rank


def _record_recipe_result(source: str, task: str, passed: bool) -> None:
    with _recipes_lock:
        recipes = _load_recipes()
        if source not in recipes:
            return
        stats = recipes[source].setdefault("_recipe_stats", {}).setdefault(
            task, {"passes": 0, "failures": 0, "consecutive_failures": 0})
        key = "passes" if passed else "failures"
        stats[key] = stats.get(key, 0) + 1
        stats["consecutive_failures"] = 0 if passed else stats.get("consecutive_failures", 0) + 1
        RECIPES_FILE.write_text(json.dumps(recipes, indent=2, ensure_ascii=False), encoding="utf-8")


def recipe_failures(device_id: str, task: str) -> dict[str, dict]:
    return dict(_RECIPE_FAILURES.get((device_id, task), {}))


def recipe_context(device_id: str, task: str, limit: int = 3) -> list[dict]:
    with _recipes_lock:
        recipes = _load_recipes()
    failures = recipe_failures(device_id, task)
    result = []
    for source, steps in candidate_recipes(device_id, task, include_skipped=True)[:limit]:
        result.append({"profile": source, "steps": steps,
                       "notes": recipes.get(source, {}).get("_notes", {}).get(task, ""),
                       "last_failure": failures.get(source)})
    return result


def try_recipes(device_id: str, task: str, verify) -> dict | None:
    """Replay candidate recipes until one passes `verify()` (a no-argument
    check of the outcome, e.g. "is Stats for nerds visible?").

    A recipe borrowed from another profile that works is saved under this
    device's own profile, so next time it's an exact match. Returns the
    success dict, or None if no recipe exists/works (caller falls back to
    the built-in steps, then the AI)."""
    own = device_profile_key(device_id)
    for source, steps in candidate_recipes(device_id, task):
        try:
            result = _run_steps(device_id, steps)
        except Exception as exc:  # noqa: BLE001 - a broken recipe must never block the fallbacks
            result = {"success": False, "error": str(exc)}
        if result.get("success") and verify():
            _record_recipe_result(source, task, True)
            _RECIPE_FAILURES.get((device_id, task), {}).pop(source, None)
            if source != own:
                save_recipe(device_id, task, steps, notes=f"reused from {source}")
            return {**result, "recipe_from": source}
        _record_recipe_result(source, task, False)
        _RECIPE_FAILURES.setdefault((device_id, task), {})[source] = {
            "failed_step": result.get("failed_step"), "error": result.get("error", "verification failed")}
        if result.get("taps_done"):
            # It got partway (e.g. opened a menu) - close it before the next attempt.
            _driver(device_id).press("back")
            time.sleep(1)
    return None


def _run_steps(device_id: str, steps: list[dict]) -> dict:
    """Replay one recipe's steps. Never scrolls: scrolling a menu can land a
    tap on an item that opens another app."""
    yt = _yt()
    driver = _driver(device_id)
    taps = 0
    for i, step in enumerate(steps):
        action = step.get("action")
        if action == "wait":
            time.sleep(min(float(step.get("seconds", 1)), 10))
        elif action == "press_key":
            driver.press(_KEYCODES[step["key"]])
        elif action == "reveal_player_controls":
            yt.reveal_player_controls(device_id)
        elif action == "switch_on":
            deadline = time.time() + _RECIPE_STEP_TIMEOUT_SECONDS
            while _find_match(driver, step.get("match", {})) is None and time.time() < deadline:
                time.sleep(0.5)
            result = switch_on(driver, step.get("match", {}))
            if not result["success"]:
                return {**result, "taps_done": taps, "failed_step": i,
                        "error": f"Recipe step {i}: {result['error']}"}
            taps += int(result["changed"])
        elif action == "tap":
            match = step.get("match", {})
            selector = _selector_for(match)
            if selector is None:
                return {"success": False, "taps_done": taps, "failed_step": i,
                        "error": f"Recipe step {i} has no usable match"}
            element = None
            if step.get("reveal_player_controls"):
                yt._reveal_and_find(driver, selector)
                element = _find_match(driver, match)
            else:
                deadline = time.time() + _RECIPE_STEP_TIMEOUT_SECONDS
                while element is None and time.time() < deadline:
                    element = _find_match(driver, match)
                    if element is None:
                        time.sleep(0.5)
            if element is None or not yt._tap(element):
                return {"success": False, "taps_done": taps, "failed_step": i,
                        "error": f"Recipe step {i} target not found: {match}"}
            taps += 1
            time.sleep(float(step.get("wait_after", 1)))
            other = _left_youtube(driver)
            if other:
                driver.press("back")
                return {"success": False, "taps_done": 0, "failed_step": i,  # already pressed back
                        "error": f"Recipe step {i} opened another app ({other}); pressed back"}
        else:
            return {"success": False, "taps_done": taps, "failed_step": i,
                    "error": f"Recipe step {i} has unknown action '{action}'"}
    return {"success": True, "via": "learned_recipe", "steps": len(steps)}


# --- LLM-facing tools ---------------------------------------------------


def get_screen(device_id: str, include_screenshot: bool = True) -> dict:
    """Snapshot of the current UI as a numbered element list (+ optional
    screenshot). Element numbers are what tap_element takes."""
    try:
        driver = _driver(device_id)
        root = ET.fromstring(driver.dump_hierarchy(compressed=False).encode("utf-8"))
        elements = []
        for node in root.iter("node"):
            a = node.attrib
            if a.get("package", "") in _IGNORED_PACKAGES:
                continue
            bounds = _parse_bounds(a.get("bounds", ""))
            if not bounds or bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
                continue
            text, desc = a.get("text", ""), a.get("content-desc", "")
            rid = a.get("resource-id", "")
            clickable = a.get("clickable") == "true"
            if not (text or desc or (rid and clickable) or a.get("checkable") == "true"):
                continue
            elements.append(
                {
                    "i": len(elements),
                    "class": a.get("class", node.tag).rsplit(".", 1)[-1],
                    "resource_id": rid.replace(f"{YOUTUBE_PACKAGE}:id/", "yt:"),
                    "_resource_id": rid,
                    "text": _short(text),
                    "desc": _short(desc),
                    "_raw_text": text,
                    "_raw_desc": desc,
                    "clickable": clickable,
                    "checkable": a.get("checkable") == "true",
                    "checked": a.get("checked") == "true",
                    "switch": ("on" if a.get("checked") == "true" else "off")
                    if a.get("checkable") == "true" else "",
                    "bounds": a.get("bounds"),
                }
            )
            if len(elements) >= _MAX_ELEMENTS:
                break

        # Keep full resource-ids for tapping/recipes; show the model the short form.
        _LAST_SCREEN[device_id] = [{**e, "resource_id": e["_resource_id"]} for e in elements]
        # Drop empty fields to save tokens (`is not False`, not `!=`, so index 0 survives).
        shown = [{k: v for k, v in e.items() if not k.startswith("_") and v != "" and v is not False} for e in elements]
        width, height = driver.window_size()
        result = {
            "activity": driver.app_current().get("activity"),
            "orientation": "LANDSCAPE" if width > height else "PORTRAIT",
            "screen_size": f"{width}x{height}",
            "element_count": len(elements),
            "truncated": len(elements) >= _MAX_ELEMENTS,
            "elements": shown,
        }
        if include_screenshot:
            result["_image_jpeg_b64"] = _screenshot_jpeg_b64(driver)
        return result
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


def tap_element(device_id: str, index: int, reveal_player_controls_first: bool = False) -> dict:
    """Tap element #index from the most recent get_screen. With
    reveal_player_controls_first, the player is tapped first and the
    element re-found by its descriptor (for auto-hiding player buttons)."""
    screen = _LAST_SCREEN.get(device_id)
    if not screen:
        return {"success": False, "error": "Call get_screen first"}
    if not 0 <= index < len(screen):
        return {"success": False, "error": f"No element #{index} on the last screen (0..{len(screen) - 1})"}

    element = screen[index]
    match = _stable_match(element, screen)
    try:
        yt = _yt()
        driver = _driver(device_id)
        switch = None
        switch_match = None
        bounds = _parse_bounds(element["bounds"])
        for candidate in screen:
            candidate_bounds = _parse_bounds(candidate["bounds"])
            if not candidate.get("checkable") or not candidate_bounds:
                continue
            same_row = bounds and not (candidate_bounds[3] < bounds[1] - _SWITCH_ROW_SLACK_PX
                                       or candidate_bounds[1] > bounds[3] + _SWITCH_ROW_SLACK_PX)
            if candidate is element or same_row:
                switches = driver(checkable=True)
                switch = None
                for i in range(switches.count):
                    info = switches[i].info
                    if info.get("bounds") == {"left": candidate_bounds[0], "top": candidate_bounds[1],
                                              "right": candidate_bounds[2], "bottom": candidate_bounds[3]}:
                        switch = switches[i]
                        break
                if switch is None:
                    continue
                label = element if not element.get("checkable") else next(
                    (e for e in screen if e is not element and (e.get("_raw_text") or e.get("_raw_desc"))
                     and _parse_bounds(e["bounds"]) and not (_parse_bounds(e["bounds"])[3] < bounds[1] - _SWITCH_ROW_SLACK_PX
                     or _parse_bounds(e["bounds"])[1] > bounds[3] + _SWITCH_ROW_SLACK_PX)), element)
                switch_match = _stable_match(label, screen)
                if switch.info.get("checked"):
                    return {"success": True, "already_on": True,
                            "recipe_step": {"action": "switch_on", "match": switch_match}}
                break
        if reveal_player_controls_first:
            selector = _selector_for(match)
            if selector:
                yt._reveal_and_find(driver, selector)
            found = _find_match(driver, match)
            if found is None or not yt._tap(found):
                return {"success": False, "error": f"Element {match} not found after revealing player controls"}
        else:
            x1, y1, x2, y2 = _parse_bounds(element["bounds"])
            driver.click((x1 + x2) // 2, (y1 + y2) // 2)
        time.sleep(1)
        result = {
            "success": True,
            "tapped": {k: element[k] for k in ("class", "text", "desc") if element.get(k)},
            # Ready-to-save recipe step - copy into save_recipe once the task is verified.
            "recipe_step": {"action": "tap", "match": match, "reveal_player_controls": reveal_player_controls_first},
        }
        if switch is not None:
            result["recipe_step"] = {"action": "switch_on", "match": switch_match}
        other = _left_youtube(driver)
        if other:
            result["warning"] = f"This tap opened another app ({other}). Press back to return to YouTube; don't save this step."
        return result
    except Exception as exc:  # noqa: BLE001
        return {"success": False, "error": str(exc)}


def press_key(device_id: str, key: str) -> dict:
    if key not in _KEYCODES:
        return {"success": False, "error": f"Unknown key '{key}'. Allowed: {', '.join(_KEYCODES)}"}
    try:
        _driver(device_id).press(_KEYCODES[key])
        time.sleep(1)
        return {"success": True, "recipe_step": {"action": "press_key", "key": key}}
    except Exception as exc:  # noqa: BLE001
        return {"success": False, "error": str(exc)}


def swipe(device_id: str, direction: str) -> dict:
    """Scroll/swipe the whole screen. direction is where the content moves
    *to*: 'up' reveals content further down."""
    if direction not in ("up", "down", "left", "right"):
        return {"success": False, "error": "direction must be up/down/left/right"}
    try:
        _driver(device_id).swipe_ext(direction, scale=0.6)
        time.sleep(1)
        return {"success": True}
    except Exception as exc:  # noqa: BLE001
        return {"success": False, "error": str(exc)}


def save_recipe(device_id: str, task: str, steps: list[dict], notes: str) -> dict:
    """Persist verified steps for `task` on this device profile."""
    if task not in RECIPE_TASKS:
        return {"success": False, "error": f"task must be one of: {', '.join(RECIPE_TASKS)}"}
    if not notes.strip():
        return {"success": False, "error": "notes is required (cause, and what differed on this phone)"}
    for i, step in enumerate(steps):
        if step.get("action") not in _RECIPE_ACTIONS:
            return {"success": False, "error": f"Step {i}: action must be one of {sorted(_RECIPE_ACTIONS)}"}
        if step["action"] in ("tap", "switch_on") and not _selector_for(step.get("match", {})):
            return {"success": False,
                    "error": f"Step {i}: {step['action']} needs match.resource_id, match.desc or match.text"}
        if step["action"] == "press_key" and step.get("key") not in _KEYCODES:
            return {"success": False, "error": f"Step {i}: key must be one of {sorted(_KEYCODES)}"}

    key = device_profile_key(device_id)
    with _recipes_lock:
        recipes = _load_recipes()
        recipes.setdefault(key, {})[task] = steps
        recipes[key].setdefault("_notes", {})[task] = notes
        recipes[key].setdefault("_recipe_stats", {})[task] = {
            "passes": 0, "failures": 0, "consecutive_failures": 0}
        RECIPES_FILE.write_text(json.dumps(recipes, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"success": True, "device_profile": key, "task": task, "steps_saved": len(steps), "file": str(RECIPES_FILE)}
