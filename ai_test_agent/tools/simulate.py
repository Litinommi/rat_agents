"""Deterministic fake Android/YouTube backend for tests and demos."""

from dataclasses import dataclass, field


@dataclass
class SimulatedState:
    device_id: str
    inject_failure: bool = False
    unfamiliar_ui: bool = False
    stats_setting_off: bool = False
    connected: bool = True
    wifi_connected: bool = True
    youtube_launched: bool = False
    playback_active: bool = False
    attempt: int = 0
    failure_triggered: bool = False
    model: str = "Pixel 7 (Simulated)"
    android_version: str = "14"
    youtube_version: str = "SIM"
    locale: str = "es-ES"
    fullscreen: bool = False
    stats_for_nerds: bool = False
    stats_setting_enabled: bool = False
    screen: str = "player"
    learned: dict = field(default_factory=dict)

    def note_video_started(self) -> None:
        self.attempt += 1
        self.playback_active = True
        self.screen = "player"

    def sample_playback(self) -> dict:
        if self.inject_failure and self.attempt == 1 and not self.failure_triggered:
            self.failure_triggered = True
            self.wifi_connected = False
            self.playback_active = False
            return {"playing": False, "error_detected": True,
                    "error_message": "Playback stalled: unable to load video (network error)"}
        if not self.wifi_connected:
            self.playback_active = False
            return {"playing": False, "error_detected": True,
                    "error_message": "Playback stalled: unable to load video (network error)"}
        return {"playing": self.playback_active, "error_detected": False, "error_message": None}


def sim_ui_tools(sim: SimulatedState) -> dict:
    def elements() -> list[dict]:
        screens = {
            "player": [
                {"class": "ViewGroup", "resource_id": "yt:player_overlays", "clickable": True,
                 "bounds": "[0,93][1080,701]"},
                {"class": "ImageView", "desc": "Más opciones", "clickable": True,
                 "bounds": "[948,99][1080,231]", "to": "menu"},
                {"class": "ImageView", "resource_id": "yt:fullscreen_button", "desc": "Pantalla completa",
                 "clickable": True, "bounds": "[915,569][1080,701]", "action": "fullscreen"},
            ],
            "menu": [
                {"class": "ViewGroup", "desc": "Más ", "clickable": True,
                 "bounds": "[22,2180][1058,2312]", "to": "more"},
            ],
            "more": ([] if sim.stats_setting_off and not sim.stats_setting_enabled else [
                {"class": "ViewGroup", "desc": "Estadísticas para nerds ", "clickable": True,
                 "bounds": "[22,2180][1058,2312]", "to": "player", "action": "stats"},
            ]),
            "home": [
                {"class": "TextView", "text": "You", "clickable": True,
                 "bounds": "[800,2200][1080,2400]", "to": "you"},
            ],
            "you": [
                {"class": "TextView", "text": "Settings", "clickable": True,
                 "bounds": "[40,300][1040,430]", "to": "settings"},
            ],
            "settings": [
                {"class": "TextView", "text": "General", "clickable": True,
                 "bounds": "[40,300][1040,430]", "to": "general"},
            ],
            "general": [
                {"class": "TextView", "text": "Enable stats for nerds", "clickable": True,
                 "switch": "on" if sim.stats_setting_enabled else "off", "bounds": "[40,300][1040,430]",
                 "action": "stats_setting"},
            ],
        }
        return [{"i": i, **entry} for i, entry in enumerate(screens[sim.screen])]

    def get_screen(include_screenshot: bool = True) -> dict:
        return {"activity": ".MainActivity", "orientation": "LANDSCAPE" if sim.fullscreen else "PORTRAIT",
                "screen_size": "1080x2400", "element_count": len(elements()), "elements": elements()}

    def player_state() -> dict:
        return {"player_on_screen": sim.screen in {"player", "menu", "more"}, "fullscreen": sim.fullscreen,
                "stats_for_nerds_visible": sim.stats_for_nerds, "ad_showing": False,
                "orientation": "LANDSCAPE" if sim.fullscreen else "PORTRAIT",
                "media_state": "playing" if sim.playback_active else "paused",
                "package": "com.google.android.youtube", "activity": ".MainActivity"}

    def open_video_url() -> dict:
        sim.note_video_started()
        return {"success": True}

    def enable_stats_for_nerds() -> dict:
        if sim.learned.get("enable_stats_for_nerds"):
            sim.stats_for_nerds = True
            return {"success": True, "via": "learned_recipe"}
        if sim.stats_setting_off and not sim.stats_setting_enabled:
            return {"success": False, "error": "Stats for nerds is missing from the More menu"}
        if sim.unfamiliar_ui:
            return {"success": False, "error": "Could not locate the player settings button"}
        sim.stats_for_nerds = True
        return {"success": True, "via": "built_in"}

    def enable_stats_in_app_settings() -> dict:
        if sim.learned.get("enable_stats_in_settings"):
            sim.stats_setting_enabled = True
            return {"success": True, "via": "learned_recipe"}
        sim.screen = "home"
        return {"success": False, "error": "No saved recipe for turning on the setting"}

    def tap_element(index: int, reveal_player_controls_first: bool = False) -> dict:
        visible = elements()
        if not 0 <= index < len(visible):
            return {"success": False, "error": f"No element #{index} on the last screen"}
        element = visible[index]
        match = ({"text": element["text"]} if element.get("text") else
                 {"desc": element["desc"]} if element.get("desc") else
                 {"resource_id": element.get("resource_id")})
        recipe = {"action": "tap", "match": match, "reveal_player_controls": reveal_player_controls_first}
        if element.get("action") == "stats_setting":
            recipe = {"action": "switch_on", "match": {"text": "Enable stats for nerds"}}
            sim.stats_setting_enabled = not sim.stats_setting_enabled
        elif element.get("action") == "stats":
            sim.stats_for_nerds = True
        elif element.get("action") == "fullscreen":
            sim.fullscreen = True
        sim.screen = element.get("to", sim.screen)
        return {"success": True, "tapped": match, "recipe_step": recipe}

    def press_key(key: str) -> dict:
        back = {"menu": "player", "more": "menu", "general": "settings", "settings": "you", "you": "home"}
        if key == "back":
            sim.screen = back.get(sim.screen, "player")
        elif key == "home":
            sim.screen = "home"
        return {"success": True, "recipe_step": {"action": "press_key", "key": key}}

    def save_recipe(task: str, steps: list, notes: str) -> dict:
        if not notes.strip():
            return {"success": False, "error": "notes is required"}
        sim.learned[task] = steps
        return {"success": True, "device_profile": f"Google|{sim.model}|ytSIM|es-ES",
                "task": task, "steps_saved": len(steps)}

    return {
        "open_video_url": open_video_url,
        "wait_for_ads": lambda: {"success": True, "ads_skipped": 0},
        "enter_fullscreen": lambda: (setattr(sim, "fullscreen", True) or {"success": True, "via": "built_in"}),
        "enable_stats_for_nerds": enable_stats_for_nerds,
        "enable_stats_in_app_settings": enable_stats_in_app_settings,
        "reveal_player_controls": lambda: {"success": True, "recipe_step": {"action": "reveal_player_controls"}},
        "get_player_state": player_state,
        "get_screen": get_screen,
        "tap_element": tap_element,
        "press_key": press_key,
        "swipe": lambda direction: {"success": True},
        "save_recipe": save_recipe,
    }
