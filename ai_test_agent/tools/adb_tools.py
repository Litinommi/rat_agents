"""ADB-backed tools: device presence, device info, Wi-Fi state.

Every function shells out to a fixed, allowlisted `adb` argument vector
built entirely in Python - never a string the LLM could influence. The
model only ever supplies typed, schema-validated arguments (e.g. the
boolean `enabled` for toggle_wifi); it never supplies raw shell text.
"""

import re
import subprocess
import time

from config import ADB_TIMEOUT_SECONDS
from diagnostics import record
from tools.simulate import SimulatedState


class AdbError(Exception):
    pass


def _run_adb(args: list[str], timeout: int = ADB_TIMEOUT_SECONDS) -> subprocess.CompletedProcess:
    started = time.monotonic()
    try:
        result = subprocess.run(
            ["adb", *args],
            capture_output=True,
            text=True,
            # dumpsys output can contain non-ASCII (app titles, OEM strings);
            # Windows' default cp1252 decoding would crash the reader thread.
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
        record("adb_result", arguments=args, exit_code=result.returncode,
               duration_seconds=round(time.monotonic() - started, 3), stderr=result.stderr,
               stdout_chars=len(result.stdout))
        return result
    except FileNotFoundError as exc:
        record("adb_error", arguments=args, error="adb not found")
        raise AdbError("adb executable not found on PATH. Install Android platform-tools.") from exc
    except subprocess.TimeoutExpired as exc:
        record("adb_error", arguments=args, error="command timed out", timeout_seconds=timeout)
        raise AdbError(f"adb command timed out after {timeout}s: {' '.join(args)}") from exc


def list_devices() -> list[dict]:
    """Return every device adb currently sees, connected or not."""
    result = _run_adb(["devices", "-l"])
    devices = []
    for line in result.stdout.splitlines()[1:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        devices.append({"device_id": parts[0], "state": parts[1] if len(parts) > 1 else "unknown"})
    return devices


def device_exists(device_id: str) -> bool:
    """Fast pre-flight check used by the CLI before spending any LLM tokens."""
    try:
        return any(d["device_id"] == device_id and d["state"] == "device" for d in list_devices())
    except AdbError:
        return False


def _getprop(device_id: str, prop: str) -> str:
    return _run_adb(["-s", device_id, "shell", "getprop", prop]).stdout.strip()


def get_device_profile(device_id: str, package: str = "com.google.android.youtube") -> dict:
    """Everything that decides which UI the app shows: maker/model, Android
    version, app version, locale and screen size. Used to key learned recipes."""
    profile = {
        "manufacturer": _getprop(device_id, "ro.product.manufacturer"),
        "model": _getprop(device_id, "ro.product.model"),
        "android_version": _getprop(device_id, "ro.build.version.release"),
        "locale": _getprop(device_id, "persist.sys.locale") or _getprop(device_id, "ro.product.locale"),
    }
    dump = _run_adb(["-s", device_id, "shell", "dumpsys", "package", package]).stdout
    match = re.search(r"versionName=(\S+)", dump)
    profile["youtube_version"] = match.group(1) if match else "not installed"
    size = re.search(r"(\d+x\d+)", _run_adb(["-s", device_id, "shell", "wm", "size"]).stdout)
    profile["screen_size"] = size.group(1) if size else "?"
    return profile


# android.media.session.PlaybackState constants
_PLAYBACK_STATES = {0: "none", 1: "stopped", 2: "paused", 3: "playing", 6: "buffering", 7: "error", 8: "connecting"}


def get_media_playback_state(device_id: str, package: str) -> str | None:
    """Read `package`'s media session state from `dumpsys media_session`.

    Unlike the on-screen play/pause button (which YouTube auto-hides),
    the media session is always queryable. Returns e.g. "playing",
    "paused", "buffering", "error", or None if the app has no session.
    """
    result = _run_adb(["-s", device_id, "shell", "dumpsys", "media_session"])
    if result.returncode != 0:
        return None
    current_package = None
    for line in result.stdout.splitlines():
        line = line.strip()
        if line.startswith("package="):
            current_package = line.removeprefix("package=")
        elif current_package == package:
            # Android <=14: "state=3"; Android 15+: "state=PLAYING(3)".
            match = re.match(r"state=PlaybackState \{state=(?:[A-Z_]+\()?(\d+)", line)
            if match:
                return _PLAYBACK_STATES.get(int(match.group(1)), f"unknown({match.group(1)})")
    return None


# --- LLM-facing tools ---------------------------------------------------
# Each function's real signature is (device_id, ..., sim=None); the LLM
# never sees device_id/sim - test_runner binds them via functools.partial
# when building the tool registry, so the model cannot target another
# device or silently flip into simulation.


def check_device(device_id: str, sim: SimulatedState | None = None) -> dict:
    if sim is not None:
        return {"connected": sim.connected, "state": "device" if sim.connected else "offline"}

    try:
        devices = list_devices()
    except AdbError as exc:
        return {"connected": False, "error": str(exc)}

    match = next((d for d in devices if d["device_id"] == device_id), None)
    if match is None:
        return {"connected": False, "error": f"Device '{device_id}' not found by adb"}
    return {"connected": match["state"] == "device", "state": match["state"]}


def get_device_info(device_id: str, sim: SimulatedState | None = None) -> dict:
    if sim is not None:
        return {"model": sim.model, "android_version": sim.android_version, "manufacturer": "Google"}

    try:
        return get_device_profile(device_id)
    except AdbError as exc:
        return {"error": str(exc)}


def get_wifi_status(device_id: str, sim: SimulatedState | None = None) -> dict:
    if sim is not None:
        return {"wifi_enabled": True, "connected": sim.wifi_connected}

    try:
        enabled_result = _run_adb(["-s", device_id, "shell", "settings", "get", "global", "wifi_on"])
        enabled_value = enabled_result.stdout.strip()
        wifi_enabled = enabled_value == "1" if enabled_result.returncode == 0 and enabled_value in {"0", "1"} else None
        status = _run_adb(["-s", device_id, "shell", "cmd", "wifi", "status"])
        dump = _run_adb(["-s", device_id, "shell", "dumpsys", "wifi"], timeout=ADB_TIMEOUT_SECONDS)
    except AdbError as exc:
        return {"wifi_enabled": None, "connected": None, "error": str(exc)}
    record("wifi_diagnostics", settings_value=enabled_value, status_exit=status.returncode,
           status_output=status.stdout, dump_exit=dump.returncode,
           # Avoid dumping historical connection events/SSID lists into the model context.
           current_info=[line.strip() for line in dump.stdout.splitlines()
                         if re.match(r"\s*(?:mWifiInfo|WifiInfo|mNetworkInfo|NetworkInfo|Wi-Fi is|Wifi is)", line)])
    result = _parse_wifi_status(status.stdout if status.returncode == 0 else "", dump.stdout if dump.returncode == 0 else "", wifi_enabled)
    if result["connected"] is False and result["wifi_enabled"]:
        result["hint"] = "Wi-Fi is already enabled but not associated. Enabling it again will not select a network. Inspect playback/connectivity; mobile data may be in use."
    elif result["connected"] is None:
        result["hint"] = "Wi-Fi association could not be determined from this device's output. Do not assume loss of internet; verify playback or inspect the screen."
    return result


def _parse_wifi_status(status: str, dump: str, wifi_enabled: bool | None) -> dict:
    """Use current Android status, never historical CONNECTED events in dumpsys."""
    enabled_match = re.search(r"(?:Wi-?Fi|Wifi) is (enabled|disabled)", status, re.I)
    if enabled_match:
        wifi_enabled = enabled_match.group(1).lower() == "enabled"
    if re.search(r"(?:Wi-?Fi|Wifi) is (?:not connected|disconnected)", status, re.I):
        return {"wifi_enabled": wifi_enabled, "connected": False, "source": "cmd wifi status"}
    if re.search(r"(?:Wi-?Fi|Wifi) is connected(?:\s|$)", status, re.I):
        return {"wifi_enabled": wifi_enabled, "connected": True, "source": "cmd wifi status"}
    current = "\n".join(line for line in (status + "\n" + dump).splitlines()
                        if re.match(r"\s*(?:mWifiInfo|WifiInfo|mNetworkInfo|NetworkInfo|Wi-Fi is|Wifi is)", line))
    disconnected = re.search(r"(?:state|Supplicant state)\s*[:=]\s*(?:DISCONNECTED|INACTIVE|UNINITIALIZED)\b", current, re.I)
    connected = re.search(r"(?:state|detailed state)\s*[:=]\s*CONNECTED\b", current, re.I)
    completed = re.search(r"Supplicant state\s*[:=]\s*COMPLETED\b", current, re.I)
    valid_ip = re.search(r"IP(?: address)?\s*[:=]\s*/?(?!0\.0\.0\.0)(?:\d{1,3}\.){3}\d{1,3}", current, re.I)
    if disconnected:
        association = False
    elif connected or (completed and valid_ip) or re.search(r"Wi-Fi is connected", current, re.I):
        association = True
    elif wifi_enabled is False:
        association = False
    else:
        association = None
    return {"wifi_enabled": wifi_enabled, "connected": association,
            "source": "dumpsys wifi" if association is not None else "unknown"}


def toggle_wifi(device_id: str, enabled: bool, sim: SimulatedState | None = None) -> dict:
    """The agent's one concrete recovery action: enable/disable Wi-Fi via
    `svc wifi`. This is a fixed, boolean-parameterized command - not
    arbitrary shell text - so it stays within the "predefined safe tools
    only" constraint.
    """
    if sim is not None:
        sim.wifi_connected = enabled
        return {"success": True, "wifi_enabled": enabled, "connected": enabled}

    verb = "enable" if enabled else "disable"
    result = _run_adb(["-s", device_id, "shell", "svc", "wifi", verb])
    if result.returncode != 0:
        return {"success": False, "error": result.stderr.strip() or "svc wifi command failed"}
    return {"success": True, "command_accepted": True, "requested_enabled": enabled,
            "hint": "Enable/disable command accepted; this does not confirm network association. Re-check status."}
