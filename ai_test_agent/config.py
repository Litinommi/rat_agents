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

# --- OpenAI-compatible inference -----------------------------------------
# LLM_* is the preferred interface. NVIDIA_NIM_* remains a fallback so old
# .env files continue to work without changes.
def _llm_env(name: str, legacy: str, default: str = "") -> str:
    return os.getenv(name, os.getenv(legacy, default)).strip()


LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai-compatible").strip() or "openai-compatible"
LLM_BASE_URL = _llm_env("LLM_BASE_URL", "NVIDIA_NIM_BASE_URL", "https://integrate.api.nvidia.com/v1").rstrip("/")
LLM_API_KEY = _llm_env("LLM_API_KEY", "NVIDIA_NIM_API_KEY")
LLM_MODEL = _llm_env("LLM_MODEL", "NVIDIA_NIM_MODEL")
LLM_MAX_TOKENS = int(_llm_env("LLM_MAX_TOKENS", "NVIDIA_NIM_MAX_TOKENS", "32768"))
LLM_TEMPERATURE = float(_llm_env("LLM_TEMPERATURE", "NVIDIA_NIM_TEMPERATURE", "1.0"))
LLM_TOP_P = float(_llm_env("LLM_TOP_P", "NVIDIA_NIM_TOP_P", "0.95"))
LLM_STREAM = _llm_env("LLM_STREAM", "NVIDIA_NIM_STREAM", "false").lower() == "true"
# Enable only for a model supporting images AND function calling.
LLM_VISION = _llm_env("LLM_VISION", "NVIDIA_NIM_VISION", "false").lower() == "true"
LLM_TIMEOUT_SECONDS = float(_llm_env("LLM_TIMEOUT_SECONDS", "NVIDIA_NIM_TIMEOUT_SECONDS", "120"))

# Compatibility aliases for callers that imported the original names.
NVIDIA_NIM_BASE_URL = LLM_BASE_URL
NVIDIA_NIM_API_KEY = LLM_API_KEY
NVIDIA_NIM_MODEL = LLM_MODEL
NVIDIA_NIM_MAX_TOKENS = LLM_MAX_TOKENS
NVIDIA_NIM_TEMPERATURE = LLM_TEMPERATURE
NVIDIA_NIM_TOP_P = LLM_TOP_P
NVIDIA_NIM_STREAM = LLM_STREAM
NVIDIA_NIM_VISION = LLM_VISION
NVIDIA_NIM_TIMEOUT_SECONDS = LLM_TIMEOUT_SECONDS


def llm_configuration_error() -> str | None:
    if not LLM_API_KEY or LLM_API_KEY in {"your_api_key", "your_nvidia_nim_api_key"}:
        return "LLM_API_KEY is not set. Add it to .env (use any nonempty value for an unauthenticated local server)."
    if not LLM_MODEL or LLM_MODEL in {"your_model_id", "your_supported_model_id"}:
        return "LLM_MODEL is not set. Choose a tool-calling model in .env."
    if not LLM_BASE_URL.startswith(("http://", "https://")):
        return "LLM_BASE_URL must be an HTTP(S) OpenAI-compatible /v1 endpoint."
    if LLM_MAX_TOKENS <= 0 or LLM_TIMEOUT_SECONDS <= 0:
        return "LLM_MAX_TOKENS and LLM_TIMEOUT_SECONDS must be positive."
    if not 0 < LLM_TEMPERATURE <= 2 or not 0 < LLM_TOP_P <= 1:
        return "LLM_TEMPERATURE must be between 0 and 2 and LLM_TOP_P between 0 and 1."
    return None


def nim_configuration_error() -> str | None:
    """Backward-compatible name for the generic LLM configuration check."""
    return llm_configuration_error()


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
MAX_FIXES_PER_DEVICE = int(os.getenv("MAX_FIXES_PER_DEVICE", "3"))
FIX_MAX_TOOL_CALLS = int(os.getenv("FIX_MAX_TOOL_CALLS", "25"))
MAX_WAIT_SECONDS = int(os.getenv("MAX_WAIT_SECONDS", "180"))
MAX_LLM_TURNS = int(os.getenv("MAX_LLM_TURNS", "60"))

# --- Logging ----------------------------------------------------------
LOGCAT_LINE_LIMIT = int(os.getenv("LOGCAT_LINE_LIMIT", "400"))
