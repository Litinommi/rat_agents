"""Central configuration for the AI Test Agent PoC.

All tunables live here so the rest of the codebase never hardcodes a
magic number. Values are read from the environment (via a .env file in
development) with sane defaults for a live demo.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs"
REPORT_DIR = BASE_DIR / "reports"
LOG_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

# --- Claude API -------------------------------------------------------
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-opus-5")
CLAUDE_MAX_TOKENS = int(os.getenv("CLAUDE_MAX_TOKENS", "16000"))

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
