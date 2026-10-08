"""Captures device logs (logcat) for a test run and saves them to disk.

Used by tools/test_tools.py:collect_logs - kept as its own module so the
"where do logs live / how are they filtered" concern is independent of
the tool-calling plumbing.
"""

import subprocess
from datetime import datetime, timezone

from config import ADB_TIMEOUT_SECONDS, LOG_DIR, LOGCAT_LINE_LIMIT, YOUTUBE_PACKAGE


def capture_logcat(device_id: str, reason: str = "") -> dict:
    """Dump the last LOGCAT_LINE_LIMIT logcat lines filtered to the
    YouTube package and save them under logs/. Returns a summary the LLM
    can read without needing the full file contents in context.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_device = "".join(c if c.isalnum() else "_" for c in device_id)
    out_path = LOG_DIR / f"{timestamp}_{safe_device}.log"

    try:
        result = subprocess.run(
            ["adb", "-s", device_id, "logcat", "-d", "-t", str(LOGCAT_LINE_LIMIT)],
            capture_output=True,
            text=True,
            timeout=ADB_TIMEOUT_SECONDS,
            check=False,
        )
    except FileNotFoundError:
        return {"success": False, "error": "adb executable not found on PATH"}
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "logcat capture timed out"}

    lines = result.stdout.splitlines()
    relevant = [ln for ln in lines if YOUTUBE_PACKAGE in ln or "wifi" in ln.lower()] or lines

    out_path.write_text(result.stdout, encoding="utf-8")

    return {
        "success": True,
        "reason": reason,
        "log_file": str(out_path),
        "total_lines": len(lines),
        "relevant_lines": len(relevant),
        "excerpt": "\n".join(relevant[-30:]),
    }


def capture_simulated(reason: str = "") -> dict:
    """Log summary used when running with --simulate (no real device)."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = LOG_DIR / f"{timestamp}_simulated.log"
    fake_log = (
        "[simulated] Player error: onPlayerError - ERROR_CODE_IO_NETWORK_CONNECTION_FAILED\n"
        "[simulated] ConnectivityService: NetworkAgentInfo Wifi disconnected\n"
    )
    out_path.write_text(fake_log, encoding="utf-8")
    return {
        "success": True,
        "reason": reason,
        "log_file": str(out_path),
        "total_lines": 2,
        "relevant_lines": 2,
        "excerpt": fake_log.strip(),
    }
