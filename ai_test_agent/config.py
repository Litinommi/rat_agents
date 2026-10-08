"""Central configuration for the AI Test Agent PoC.

All tunables live here so the rest of the codebase never hardcodes a
magic number. Values are read from the environment (via a .env file in
development) with sane defaults for a live demo.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs"
REPORT_DIR = BASE_DIR / "reports"
LOG_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

# --- NVIDIA NIM (OpenAI-compatible SDK; NVIDIA inference only) ---------
NVIDIA_NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_NIM_API_KEY = os.getenv("NVIDIA_NIM_API_KEY", "").strip()
NVIDIA_NIM_MODEL = os.getenv("NVIDIA_NIM_MODEL", "").strip()
NVIDIA_NIM_MAX_TOKENS = int(os.getenv("NVIDIA_NIM_MAX_TOKENS", "4096"))
NVIDIA_NIM_STREAM = os.getenv("NVIDIA_NIM_STREAM", "false").lower() == "true"
# Enable only for a model supporting images AND function calling.
NVIDIA_NIM_VISION = os.getenv("NVIDIA_NIM_VISION", "false").lower() == "true"
NVIDIA_NIM_TIMEOUT_SECONDS = float(os.getenv("NVIDIA_NIM_TIMEOUT_SECONDS", "120"))


def nim_configuration_error() -> str | None:
    if not NVIDIA_NIM_API_KEY or NVIDIA_NIM_API_KEY == "your_nvidia_nim_api_key":
        return "NVIDIA_NIM_API_KEY is not set. Add your NVIDIA key to .env (see .env.example)."
    if not NVIDIA_NIM_MODEL or NVIDIA_NIM_MODEL == "your_supported_model_id":
        return "NVIDIA_NIM_MODEL is not set. Choose a supported NVIDIA NIM model in .env."
    if NVIDIA_NIM_MAX_TOKENS <= 0 or NVIDIA_NIM_TIMEOUT_SECONDS <= 0:
        return "NVIDIA_NIM_MAX_TOKENS and NVIDIA_NIM_TIMEOUT_SECONDS must be positive."
    return None


# --- YouTube app (driven via uiautomator2, no Appium server) ---------------
YOUTUBE_PACKAGE = os.getenv("YOUTUBE_PACKAGE", "com.google.android.youtube")
YOUTUBE_APP_ACTIVITY = os.getenv(
    "YOUTUBE_APP_ACTIVITY", "com.google.android.apps.youtube.app.WatchWhileActivity"
)

# --- ADB ------------------------------------------------------------------
ADB_TIMEOUT_SECONDS = int(os.getenv("ADB_TIMEOUT_SECONDS", "15"))

# --- Test defaults ----------------------------------------------------
DEFAULT_TEST_DURATION_SECONDS = int(os.getenv("DEFAULT_TEST_DURATION_SECONDS", "60"))
DEFAULT_SEARCH_QUERY = os.getenv("DEFAULT_SEARCH_QUERY", "lofi hip hop radio")
PLAYBACK_POLL_INTERVAL_SECONDS = int(os.getenv("PLAYBACK_POLL_INTERVAL_SECONDS", "5"))

# --- Safety limits ------------------------------------------------------
# These are enforced in Python regardless of what the LLM decides to do -
# the model can *reason* about retries, but it cannot exceed these caps.
MAX_TOOL_CALLS = int(os.getenv("MAX_TOOL_CALLS", "60"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "2"))
MAX_WAIT_SECONDS = int(os.getenv("MAX_WAIT_SECONDS", "180"))
MAX_LLM_TURNS = int(os.getenv("MAX_LLM_TURNS", "60"))

# --- Logging ----------------------------------------------------------
LOGCAT_LINE_LIMIT = int(os.getenv("LOGCAT_LINE_LIMIT", "400"))
