"""Deterministic fake hardware/app backend used with --simulate.

Lets the agent's reasoning loop be rehearsed or demoed without a real
Android device. Real-device (uiautomator2) code paths never
import this module directly; test_runner wires it in only when the CLI
--simulate flag is set, so production ("real") tool functions stay
untouched.
"""

from dataclasses import dataclass, field


@dataclass
class SimulatedState:
    """Mutable fake device state, one instance per test run."""

    device_id: str
    inject_failure: bool = False

    connected: bool = True
    wifi_connected: bool = True
    youtube_launched: bool = False
    playback_active: bool = False
    last_query: str = ""
    attempt: int = 0
    failure_triggered: bool = False

    model: str = "Pixel 7 (Simulated)"
    android_version: str = "14"

    # Self-healing rehearsal: with unfamiliar_ui the built-in stats-for-nerds
    # step fails (as on a non-English phone) and the agent has to navigate a
    # small fake Spanish-language menu via get_screen/tap_element instead.
    unfamiliar_ui: bool = False
    fullscreen: bool = False
    stats_for_nerds: bool = False
    screen: str = "player"
    learned: dict = field(default_factory=dict)

    def note_video_started(self) -> None:
        self.attempt += 1
        self.playback_active = True

    def sample_playback(self) -> dict:
        """One monitoring sample. Injects a single wifi-drop failure on the
        first attempt when inject_failure is set, then behaves normally.
        """
        if self.inject_failure and self.attempt == 1 and not self.failure_triggered:
            self.failure_triggered = True
            self.wifi_connected = False
            self.playback_active = False
            return {
                "playing": False,
                "error_detected": True,
                "error_message": "Playback stalled: unable to load video (network error)",
            }

        if not self.wifi_connected:
            self.playback_active = False
            return {
                "playing": False,
                "error_detected": True,
                "error_message": "Playback stalled: unable to load video (network error)",
            }

        return {"playing": self.playback_active, "error_detected": False, "error_message": None}


# --- simulated UI tools -----------------------------------------------------

_SIM_SCREENS = {
    "player": [
        {"class": "ViewGroup", "resource_id": "yt:player_overlays", "clickable": True, "bounds": "[0,93][1080,701]"},
        {"class": "ImageView", "desc": "Más opciones", "clickable": True, "bounds": "[948,99][1080,231]", "_to": "menu"},
        {"class": "ImageView", "resource_id": "yt:fullscreen_button", "desc": "Pantalla completa", "clickable": True,
         "bounds": "[915,569][1080,701]", "_action": "fullscreen"},
    ],
    "menu": [
        {"class": "ViewGroup", "desc": "Calidad Automática (720p)", "clickable": True, "bounds": "[22,1478][1058,1610]"},
        {"class": "ViewGroup", "desc": "Velocidad de reproducción 1x", "clickable": True, "bounds": "[22,1610][1058,1742]"},
        {"class": "ViewGroup", "desc": "Más ", "clickable": True, "bounds": "[22,2180][1058,2312]", "_to": "more"},
    ],
    "more": [
        {"class": "ViewGroup", "desc": "Temporizador Desactivado", "clickable": True, "bounds": "[22,1627][1058,1759]"},
        {"class": "ViewGroup", "desc": "Ayuda y comentarios ", "clickable": True, "bounds": "[22,2048][1058,2180]"},
        {"class": "ViewGroup", "desc": "Estadísticas para nerds ", "clickable": True, "bounds": "[22,2180][1058,2312]",
         "_to": "player", "_action": "stats"},
    ],
}


def sim_ui_tools(sim: SimulatedState) -> dict:
    """Simulated counterparts of the real-device UI tools (same names and
    LLM-facing arguments) so --simulate exercises the full agent flow."""

    def visible() -> list[dict]:
        return [{"i": i, **{k: v for k, v in e.items() if not k.startswith("_")}} for i, e in enumerate(_SIM_SCREENS[sim.screen])]

    def player_state() -> dict:
        return {"player_on_screen": True, "fullscreen": sim.fullscreen, "stats_for_nerds_visible": sim.stats_for_nerds,
                "ad_showing": False, "orientation": "LANDSCAPE" if sim.fullscreen else "PORTRAIT",
                "media_state": "playing" if sim.playback_active else "paused"}

    def open_video_url() -> dict:
        sim.note_video_started()
        return {"success": True}

    def enable_stats_for_nerds() -> dict:
        if sim.learned.get("enable_stats_for_nerds"):
            sim.stats_for_nerds = True
            return {"success": True, "via": "learned_recipe"}
        if sim.unfamiliar_ui:
            return {"success": False, "error": "Could not locate the player settings button",
                    "hint": "Inspect the UI with get_screen, perform the step with tap_element/press_key, verify with "
                            "get_player_state, then save_recipe('enable_stats_for_nerds', steps)."}
        sim.stats_for_nerds = True
        return {"success": True, "via": "built_in"}

    def enter_fullscreen() -> dict:
        sim.fullscreen = True
        return {"success": True, "via": "built_in"}

    def tap_element(index: int, reveal_player_controls_first: bool = False) -> dict:
        screen = _SIM_SCREENS[sim.screen]
        if not 0 <= index < len(screen):
            return {"success": False, "error": f"No element #{index} on the last screen"}
        element = screen[index]
        if element.get("_action") == "stats":
            sim.stats_for_nerds = True
        elif element.get("_action") == "fullscreen":
            sim.fullscreen = True
        sim.screen = element.get("_to", sim.screen)
        match = {"desc": element["desc"]} if element.get("desc") else {"resource_id": element.get("resource_id")}
        return {"success": True, "recipe_step": {"action": "tap", "match": match,
                                                 "reveal_player_controls": reveal_player_controls_first}}

    def press_key(key: str) -> dict:
        if key == "back":
            sim.screen = "player"
        return {"success": True, "recipe_step": {"action": "press_key", "key": key}}

    def save_recipe(task: str, steps: list, notes: str = "") -> dict:
        sim.learned[task] = steps
        return {"success": True, "device_profile": f"Google|{sim.model}|ytSIM|es-ES", "task": task, "steps_saved": len(steps)}

    return {
        "open_video_url": open_video_url,
        "wait_for_ads": lambda: {"success": True, "ads_skipped": 0},
        "enter_fullscreen": enter_fullscreen,
        "enable_stats_for_nerds": enable_stats_for_nerds,
        "enable_stats_in_app_settings": lambda: {"success": True, "via": "built_in", "changed": False},
        "reveal_player_controls": lambda: {"success": True, "recipe_step": {"action": "reveal_player_controls"}},
        "get_player_state": player_state,
        "get_screen": lambda include_screenshot=True: {"activity": ".MainActivity", "orientation": "PORTRAIT",
                                                       "screen_size": "1080x2400", "elements": visible()},
        "tap_element": tap_element,
        "press_key": press_key,
        "swipe": lambda direction: {"success": True},
        "save_recipe": save_recipe,
    }
