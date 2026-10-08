"""Tools that drive the real YouTube Android app via uiautomator2.

uiautomator2 copies a small jar to /data/local/tmp and runs it as the adb
shell user - no helper APKs are installed (so no "Install this app?" prompts
on Realme/OPPO/Vivo) and no protected system settings are changed at startup
(which MIUI blocks). No Appium server is needed.

One uiautomator2 connection is kept per device_id in _SESSIONS. All UI
locators are best-effort - YouTube's UI changes across app versions/regions,
so every interaction is wrapped so a locator miss becomes a structured error
dict the agent can reason about, not an unhandled exception that kills the
whole run. Selectors are plain dicts of uiautomator2 selector fields.
"""

import shlex
import time

from config import MAX_WAIT_SECONDS, PLAYBACK_POLL_INTERVAL_SECONDS, YOUTUBE_APP_ACTIVITY, YOUTUBE_PACKAGE
from tools import adb_tools
from tools.simulate import SimulatedState

_SESSIONS: dict[str, object] = {}

_PLAY_PAUSE = {"resourceId": f"{YOUTUBE_PACKAGE}:id/player_control_play_pause_replay_button"}
_PLAYER_OVERLAYS = {"resourceId": f"{YOUTUBE_PACKAGE}:id/player_overlays"}
# Ad UI has had several generations (skip_ad_button, modern_skip_ad_button, ...);
# match by id pattern rather than English labels like "Sponsored", which also
# appear on non-ad promo cards below the video.
_SKIP_AD = {"resourceIdMatches": r".*:id/(modern_)?skip_ad_button", "clickable": True}
_AD = {"resourceIdMatches": r".*:id/(.*skip_ad.*|ad_.*)"}
_FULLSCREEN = {"resourceId": f"{YOUTUBE_PACKAGE}:id/fullscreen_button"}
# The player's settings entry: known id first; otherwise an id-less "More options"
# button inside the player bounds (the app toolbar has another one above it).
_PLAYER_OVERFLOW = {"resourceId": f"{YOUTUBE_PACKAGE}:id/player_overflow_button"}
_PLAYER_MENU = {"description": "More options"}
_MENU_MORE = {"descriptionMatches": "More ?", "clickable": True}
_STATS_FOR_NERDS = {"descriptionMatches": "Stats for nerds ?", "clickable": True}
_NERD_STATS = {"resourceId": f"{YOUTUBE_PACKAGE}:id/nerd_stats_layout"}
_ERROR_TEXT = {"textMatches": r"(?is).*(error|no internet|connection).*"}
_ADS_CLEAR_CHECKS = 3  # consecutive ad-free polls before declaring ads over
_MENU_OPEN_ATTEMPTS = 2  # the menu-button tap can land just as the controls fade out


def _get_driver(device_id: str):
    driver = _SESSIONS.get(device_id)
    if driver is None:
        raise RuntimeError("No active session for this device - call launch_youtube first")
    return driver


def _connect(device_id: str):
    # Imported lazily so a missing uiautomator2 dependency only breaks
    # real-device runs, never --simulate runs.
    import uiautomator2 as u2

    if device_id not in _SESSIONS:
        _SESSIONS[device_id] = u2.connect(device_id)
    return _SESSIONS[device_id]


def _shell_ok(driver, cmd: str) -> tuple[bool, str]:
    output = driver.shell(cmd).output
    return ("Error" not in output and "Exception" not in output), output.strip()


def _tap(obj) -> bool:
    """Tap a UiObject's centre as fast as possible: read its bounds, tap.
    (UiObject.click() adds a wait-for-exists round trip first, which is
    enough for YouTube's auto-hiding player controls to fade in between.)"""
    try:
        b = obj.info["bounds"]
    except Exception:  # noqa: BLE001 - it vanished between lookup and tap
        return False
    obj.session.click((b["left"] + b["right"]) // 2, (b["top"] + b["bottom"]) // 2)
    return True


def _bounds(obj) -> dict:
    return obj.info["bounds"]  # {"left", "top", "right", "bottom"}


def _current_package(driver) -> str | None:
    try:
        return driver.app_current().get("package")
    except Exception:  # noqa: BLE001
        return None


def _orientation(driver) -> str:
    width, height = driver.window_size()
    return "LANDSCAPE" if width > height else "PORTRAIT"


def _reveal_and_find(driver, selector: dict, attempts: int = 5):
    """Tap the player to show its (auto-hiding) controls, then look for
    `selector` right away. Retries because the tap can land while the
    controls are already fading out. Returns the UiObject or None."""
    for _ in range(attempts):
        found = driver(**selector)
        if found.exists:
            return found
        overlays = driver(**_PLAYER_OVERLAYS)
        if overlays.exists:
            _tap(overlays)
        time.sleep(0.3)
        if found.exists:
            return found
        time.sleep(1)
    return None


def _ad_showing(driver) -> bool:
    # bool(): uiautomator2's .exists is a truthy wrapper object, not JSON-serialisable.
    return bool(driver(**_AD).exists)


# --- LLM-facing tools ---------------------------------------------------


def launch_youtube(device_id: str, sim: SimulatedState | None = None) -> dict:
    """Fresh start: YouTube force-stopped, screen in portrait, app opened
    and *waited for* (am start -W) - opening a video link while the app is
    still starting can leave its launcher screen stuck on top of the player."""
    if sim is not None:
        sim.youtube_launched = True
        return {"success": True, "launched": True}

    try:
        driver = _connect(device_id)
        driver.app_stop(YOUTUBE_PACKAGE)
        if _orientation(driver) != "PORTRAIT":
            driver.set_orientation("n")
            time.sleep(1)
        ok, output = _shell_ok(driver, f"am start -W -n {YOUTUBE_PACKAGE}/{YOUTUBE_APP_ACTIVITY}")
        if not ok:
            # The configured activity can be missing/renamed on other YouTube versions.
            driver.app_start(YOUTUBE_PACKAGE)
            time.sleep(3)
        return {"success": True, "launched": True}
    except Exception as exc:  # noqa: BLE001 - surfaced to the agent as data, not a crash
        return {"success": False, "launched": False, "error": str(exc)}


def open_video_url(device_id: str, url: str) -> dict:
    """Open a specific video by URL via an Android deep link into the YouTube
    app (waiting for it to open) - no search or UI locators involved."""
    try:
        driver = _get_driver(device_id)
        ok, output = _shell_ok(
            driver, f"am start -W -a android.intent.action.VIEW -d {shlex.quote(url)} {YOUTUBE_PACKAGE}"
        )
        if not ok:
            return {"success": False, "error": f"Could not open the URL in YouTube: {output[-300:]}"}
        time.sleep(2)  # let the player attach
        return {"success": True, "url": url}
    except Exception as exc:  # noqa: BLE001
        return {"success": False, "error": str(exc)}


def wait_for_ads(device_id: str, timeout_seconds: int = 120) -> dict:
    """Skip (when possible) or wait out pre-roll ads. The player menu and
    fullscreen behave differently during ads, so call this first."""
    try:
        driver = _get_driver(device_id)
        skip = driver(**_SKIP_AD)
        player = driver(**_PLAYER_OVERLAYS)
        skipped = 0
        clear_checks = 0
        deadline = time.time() + timeout_seconds
        while time.time() < deadline and clear_checks < _ADS_CLEAR_CHECKS:
            if skip.exists and _tap(skip):
                skipped += 1
            # Only count "ad-free" once the real player is up - right after the
            # deep link the page is a loading skeleton with neither ad nor player.
            clear_checks = 0 if _ad_showing(driver) or not player.exists else clear_checks + 1
            time.sleep(1)
        if clear_checks < _ADS_CLEAR_CHECKS:
            return {"success": False, "error": f"Ads still showing after {timeout_seconds}s", "ads_skipped": skipped}
        return {"success": True, "ads_skipped": skipped}
    except Exception as exc:  # noqa: BLE001
        return {"success": False, "error": str(exc)}


def reveal_player_controls(device_id: str) -> dict:
    """Tap the player so its auto-hiding controls (fullscreen, settings,
    play/pause) show for a few seconds."""
    try:
        driver = _get_driver(device_id)
        overlays = driver(**_PLAYER_OVERLAYS)
        if not overlays.exists:
            return {"success": False, "error": "Player not found on screen"}
        _tap(overlays)
        time.sleep(0.3)
        return {"success": True, "recipe_step": {"action": "reveal_player_controls"}}
    except Exception as exc:  # noqa: BLE001
        return {"success": False, "error": str(exc)}


def get_player_state(device_id: str) -> dict:
    """Language-independent checks (resource-ids, geometry, media session)
    used to verify steps - including ones the agent performed by hand."""
    try:
        driver = _get_driver(device_id)
        width, height = driver.window_size()
        overlays = driver(**_PLAYER_OVERLAYS)
        on_screen = bool(overlays.exists)
        fullscreen = None
        if on_screen:
            b = _bounds(overlays)
            fullscreen = (b["right"] - b["left"]) >= 0.95 * width and (b["bottom"] - b["top"]) >= 0.85 * height
        state = {
            "player_on_screen": on_screen,
            "fullscreen": fullscreen,
            "stats_for_nerds_visible": bool(driver(**_NERD_STATS).exists),
            "ad_showing": _ad_showing(driver),
            "orientation": "LANDSCAPE" if width > height else "PORTRAIT",
            "activity": driver.app_current().get("activity"),
        }
        state["media_state"] = adb_tools.get_media_playback_state(device_id, YOUTUBE_PACKAGE)
        return state
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


def _learned_or(device_id: str, task: str, verify) -> dict | None:
    """Try saved recipes (this phone type first, then phones with the same
    YouTube version + language); return a success dict if one passes
    `verify()`, else None so the caller falls back to built-in steps."""
    from tools import ui_tools

    try:
        return ui_tools.try_recipes(device_id, task, verify)
    except Exception:  # noqa: BLE001 - a broken recipe must never block the built-in path
        return None


_STATS_SETTING_LABEL = {"text": "Enable stats for nerds"}


def _stats_setting_on(device_id: str) -> bool:
    from tools import ui_tools

    switch = ui_tools.switch_for(_get_driver(device_id), _STATS_SETTING_LABEL)
    return bool(switch is not None and switch.info.get("checked"))


def enable_stats_in_app_settings(device_id: str) -> dict:
    """Recipe-only: turn on YouTube's "Enable stats for nerds" app setting
    by replaying a saved 'enable_stats_in_settings' recipe. Some phones only
    list "Stats for nerds" in the player's More menu once this is on.

    There is deliberately no built-in path: when no recipe exists for this
    phone the step fails, and the AI agent works it out (get_screen /
    tap_element / switch_on) and saves the recipe for next time. Restarts
    YouTube on its home screen first - reopen the video afterwards."""
    from tools import ui_tools

    hint = _SELF_HEAL_HINT.format(task="enable_stats_in_settings")
    if not ui_tools.candidate_recipes(device_id, "enable_stats_in_settings"):
        return {"success": False, "error": "No saved recipe for turning on 'Enable stats for nerds' on this phone",
                "hint": hint}
    launched = launch_youtube(device_id)
    if not launched["success"]:
        return {**launched, "hint": hint}
    learned = _learned_or(device_id, "enable_stats_in_settings", lambda: _stats_setting_on(device_id))
    if learned:
        return learned
    return {"success": False, "error": "Saved recipe(s) for the 'Enable stats for nerds' setting did not work",
            "hint": hint}


_SELF_HEAL_HINT = (
    "Inspect the UI with get_screen, perform the step with tap_element/press_key, verify with "
    "get_player_state, then save_recipe('{task}', steps) so this device works automatically next time."
)
_STATS_SETTING_HINT = (
    " If 'Stats for nerds' is missing from the player's More menu, it may first need turning on in YouTube's "
    "settings (typically You -> Settings -> General -> 'Enable stats for nerds'): do that with get_screen/"
    "tap_element (use a switch_on recipe step for the toggle), save_recipe('enable_stats_in_settings', steps), "
    "then open_video_url, wait_for_ads and retry."
)


def enter_fullscreen(device_id: str) -> dict:
    learned = _learned_or(device_id, "enter_fullscreen", lambda: get_player_state(device_id).get("fullscreen"))
    if learned:
        return learned
    hint = _SELF_HEAL_HINT.format(task="enter_fullscreen")
    try:
        driver = _get_driver(device_id)
        if get_player_state(device_id).get("fullscreen"):
            return {"success": True, "already_fullscreen": True}
        button = _reveal_and_find(driver, _FULLSCREEN)
        if button is None or not _tap(button):
            return {"success": False, "error": "Could not locate the fullscreen button", "hint": hint}
        time.sleep(1.5)  # rotation/relayout
        if not get_player_state(device_id).get("fullscreen"):
            return {"success": False, "error": "Tapped fullscreen but the player is not fullscreen", "hint": hint}
        return {"success": True, "via": "built_in"}
    except Exception as exc:  # noqa: BLE001
        return {"success": False, "error": str(exc), "hint": hint}


def _open_player_menu(driver, player: dict) -> bool:
    """Reveal the controls and tap the player's settings (⋮) button."""
    # Known id first: in fullscreen, ads overlay their own "More options"
    # button inside the player bounds, which the label search could pick.
    menu_button = _reveal_and_find(driver, _PLAYER_OVERFLOW, attempts=3)
    for _ in range(0 if menu_button else 5):
        _reveal_and_find(driver, _PLAYER_MENU, attempts=1)
        found = driver(**_PLAYER_MENU)
        candidates = [found[i] for i in range(found.count)
                      if player["top"] <= _bounds(found[i])["top"] < player["bottom"]]
        if candidates:
            menu_button = candidates[-1]  # rightmost = player settings
            break
    return menu_button is not None and _tap(menu_button)


def enable_stats_for_nerds(device_id: str) -> dict:
    """Built-in path: player menu (More options) -> More -> Stats for nerds.
    These menu labels are English-only and move between YouTube versions,
    so saved recipes are tried first."""
    learned = _learned_or(device_id, "enable_stats_for_nerds",
                          lambda: get_player_state(device_id).get("stats_for_nerds_visible"))
    if learned:
        return learned
    hint = _SELF_HEAL_HINT.format(task="enable_stats_for_nerds") + _STATS_SETTING_HINT
    try:
        driver = _get_driver(device_id)
        if driver(**_NERD_STATS).exists:
            return {"success": True, "already_enabled": True}

        overlays = driver(**_PLAYER_OVERLAYS)
        if not overlays.exists:
            return {"success": False, "error": "Player not found on screen", "hint": hint}
        player = _bounds(overlays)
        for _ in range(_MENU_OPEN_ATTEMPTS):
            if not _open_player_menu(driver, player):
                return {"success": False, "error": "Could not locate the player settings button", "hint": hint}
            if driver(**_MENU_MORE).wait(timeout=2):
                break  # menu is open
        # (if it still didn't open, the "More" lookup below reports it)

        for step, selector in (("More", _MENU_MORE), ("Stats for nerds", _STATS_FOR_NERDS)):
            time.sleep(1)
            # No auto-scrolling here: a scroll gesture over the menu once landed a
            # tap on an item that opened another app. Off-screen items fail instead.
            items = driver(**selector)
            if not items.exists or not _tap(items[items.count - 1]):
                driver.press("back")  # close the half-open menu so playback isn't obscured
                return {"success": False, "error": f"Could not find '{step}' in the player menu", "hint": hint}
            if _current_package(driver) != YOUTUBE_PACKAGE:
                driver.press("back")
                return {"success": False, "error": f"Tapping '{step}' opened another app; pressed back", "hint": hint}

        time.sleep(1.5)
        if not driver(**_NERD_STATS).exists:
            return {"success": False, "error": "Stats for nerds overlay did not appear", "hint": hint}
        return {"success": True, "via": "built_in"}
    except Exception as exc:  # noqa: BLE001
        return {"success": False, "error": str(exc), "hint": hint}


def stop_video(device_id: str, sim: SimulatedState | None = None) -> dict:
    if sim is not None:
        sim.playback_active = False
        return {"success": True}

    try:
        _get_driver(device_id).press("back")  # exits fullscreen, or backs out of the player
        return {"success": True}
    except Exception as exc:  # noqa: BLE001
        return {"success": False, "error": str(exc)}


def get_playback_status(
    device_id: str, duration_seconds: int = 3, sim: SimulatedState | None = None, max_seconds: int = MAX_WAIT_SECONDS
) -> dict:
    """Poll the player for up to duration_seconds, sampling every
    PLAYBACK_POLL_INTERVAL_SECONDS. Returns as soon as a problem is
    detected, or once the full duration has elapsed cleanly.
    """
    duration_seconds = max(1, min(duration_seconds, max_seconds))

    if sim is not None:
        elapsed = 0
        while elapsed < duration_seconds:
            step = min(PLAYBACK_POLL_INTERVAL_SECONDS, duration_seconds - elapsed)
            time.sleep(0)  # keep the demo snappy - simulated backend doesn't need real sleeps
            elapsed += step
            sample = sim.sample_playback()
            if sample["error_detected"]:
                return {**sample, "seconds_monitored": elapsed}
        return {**sim.sample_playback(), "seconds_monitored": elapsed}

    try:
        driver = _get_driver(device_id)
        elapsed = 0
        playing = False
        media_state = None
        while elapsed < duration_seconds:
            step = min(PLAYBACK_POLL_INTERVAL_SECONDS, duration_seconds - elapsed)
            time.sleep(step)
            elapsed += step

            # The media session is the source of truth and language-independent;
            # YouTube auto-hides the on-screen play/pause button.
            media_state = adb_tools.get_media_playback_state(device_id, YOUTUBE_PACKAGE)
            if media_state == "error":
                return {
                    "playing": False,
                    "error_detected": True,
                    "error_message": "YouTube media session reports a playback error",
                    "media_state": media_state,
                    "seconds_monitored": elapsed,
                }
            if media_state in ("playing", "buffering"):
                playing = True
                continue

            # Not clearly healthy - look for an on-screen error message.
            error_el = driver(**_ERROR_TEXT)
            if error_el.exists:
                return {
                    "playing": False,
                    "error_detected": True,
                    "error_message": error_el.info.get("text") or "Playback error detected in UI",
                    "media_state": media_state,
                    "seconds_monitored": elapsed,
                }
            if media_state is not None:
                playing = False  # paused/stopped/none - reported, not treated as an error
                continue

            control = driver(**_PLAY_PAUSE)
            if not control.exists:
                return {
                    "playing": False,
                    "error_detected": True,
                    "error_message": "Player controls not found - playback may have stopped",
                    "seconds_monitored": elapsed,
                }
            playing = "pause" in (control.info.get("contentDescription") or "").lower()

        return {"playing": playing, "error_detected": False, "error_message": None,
                "media_state": media_state, "seconds_monitored": elapsed}
    except Exception as exc:  # noqa: BLE001
        return {"playing": False, "error_detected": True, "error_message": str(exc), "seconds_monitored": 0}


def quit_session(device_id: str) -> None:
    """Drop the connection and give rotation back to the phone's own setting
    (set_orientation in launch_youtube freezes it)."""
    driver = _SESSIONS.pop(device_id, None)
    if driver is not None:
        try:
            driver.freeze_rotation(False)
        except Exception:  # noqa: BLE001 - best-effort teardown
            pass
