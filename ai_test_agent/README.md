# AI Test Agent — YouTube Playback PoC

A small, genuinely agentic proof of concept for **"AI Agents for Real-World
Application Test Automation."** the NVIDIA NIM model drives a real Android device over
ADB + Appium to run a YouTube playback test, diagnoses failures from real
tool output, decides on a recovery action, verifies it, and produces a
final PASS/FAIL/NEEDS_HUMAN report — all through explicit function/tool
calling, never free-form shell text.

## Architecture

```
CLI (agent.py) --device <ID> "<goal>"
        │
        ├─ adb devices          → fail fast if <ID> isn't connected
        │
        ▼
  TestState (test_runner.py)     — attempts, tool-call history, status
        │
        ▼
  Tool registry: {name: partial(fn, device_id=<ID>, ...)}
        │
        ▼
┌────────────── Agentic loop (llm_client.py) ───────────────────┐
│ send messages + tool schemas → NVIDIA_NIM_MODEL                   │
│  ├─ tool_calls? → print [AGENT] narration                        │
│  │    → dispatch ONE registry function (never arbitrary code)  │
│  │    → print [ADB]/[APPIUM]/[TEST]/[LOGS] result              │
│  │    → append role=tool results, loop                                │
│  └─ generate_report called → stop                              │
│ hard safety valves (MAX_TOOL_CALLS, MAX_RETRIES, MAX_LLM_TURNS) │
│ are enforced in Python, independent of what the model decides  │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
  report_generator.py → console block + JSON file under reports/
```

**Key safety property:** `device_id` is never a tool parameter the NVIDIA NIM model
fills in. It's bound into every tool function via `functools.partial`
when `test_runner.build_tool_registry()` runs, so the model has no path
to targeting a different device — it can only pick *which* predefined
tool to call and *what typed arguments* (a query string, a boolean, a
duration) to pass.

## Project structure

```
ai_test_agent/
├── agent.py            CLI entry point, console narration, top-level flow
├── llm_client.py        Tool schemas, system prompt, manual agentic loop
├── nim_client.py        NVIDIA transport, streaming assembly, safe API errors
├── validate_nim.py      Live chat/stream/tool smoke checks
├── test_runner.py       TestState, tool registry, safety-limit enforcement
├── tools/
│   ├── adb_tools.py      check_device, get_device_info, get_wifi_status, toggle_wifi
│   ├── appium_tools.py   launch_youtube, search_youtube, play_video, stop_video, get_playback_status
│   ├── test_tools.py     collect_logs, retry_test, generate_report
│   └── simulate.py       deterministic fake backend for --simulate
├── log_collector.py     logcat capture
├── report_generator.py  final report: build, print, save
├── config.py             all tunables (env-driven)
├── requirements.txt
├── .env.example
└── README.md
```

## Tools available to the agent

| Tool | Backend | Purpose |
|---|---|---|
| `check_device()` | ADB | Confirm the target device is connected |
| `get_device_info()` | ADB | Model, manufacturer, Android version |
| `get_wifi_status()` | ADB | Wi-Fi enabled / connected |
| `toggle_wifi(enabled)` | ADB (`svc wifi`) | The agent's one concrete recovery action |
| `launch_youtube()` | Appium | Launch/foreground the YouTube app |
| `search_youtube(query)` | Appium | Type + submit a search |
| `play_video()` | Appium | Tap the first result |
| `stop_video()` | Appium | Back out of the player |
| `get_playback_status(duration_seconds)` | Appium | Monitor the player, detect stalls/errors |
| `collect_logs(reason)` | logcat | Capture recent YouTube/Wi-Fi-relevant log lines |
| `retry_test(reason)` | compound | Re-run stop→relaunch→search→play; capped at `MAX_RETRIES` |
| `generate_report(status, summary, root_cause, recovery_action)` | — | Terminal tool; ends the loop |

Only these 12 functions are ever exposed to the NVIDIA NIM model (as JSON tool
schemas in `llm_client.py`). There is no "run shell command" tool, and
none of the Python tool implementations interpolate LLM-provided text
into a shell command — every `adb`/Appium call is built from a fixed
argument list plus typed, schema-validated parameters.

## Prerequisites

### Android / ADB
- Android SDK Platform Tools installed, `adb` on your `PATH`.
- A device (or emulator) with **USB debugging** enabled, connected and
  authorized (`adb devices` shows it as `device`, not `unauthorized`).
- The YouTube app installed and signed in (recommended, though not
  required — signed-out playback works too).

### Appium
- Node.js, then:
  ```bash
  npm install -g appium
  appium driver install uiautomator2
  appium   # starts the server on http://127.0.0.1:4723 by default
  ```
- Leave the Appium server running in its own terminal while you run the agent.

### NVIDIA NIM API
- Obtain a NVIDIA API key and choose a model at https://build.nvidia.com/.
- `pip install -r requirements.txt`
- Copy `.env.example` to `.env` and set both `NVIDIA_NIM_API_KEY` and `NVIDIA_NIM_MODEL`.
- The OpenAI SDK is only the client library. Requests always use
  `https://integrate.api.nvidia.com/v1`; no OpenAI key is used.
- The model must support OpenAI-compatible function calling (`tool_choice=auto`).
  Check its NVIDIA API Catalog documentation for availability and token limits.

## Setup

```bash
cd ai_test_agent
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
copy .env.example .env        # Windows: copy, macOS/Linux: cp
# edit .env and set NVIDIA_NIM_API_KEY and NVIDIA_NIM_MODEL
```

## Running

List connected devices:

```bash
python agent.py --list-devices
```

Run a real test against a specific device:

```bash
python agent.py --device 1A271FDF600B0E "Run a YouTube playback test for 60 seconds"
```

Optional flags:

```bash
python agent.py --device 1A271FDF600B0E "Run a YouTube playback test" \
  --duration 45 \
  --search-query "coding lofi mix"
```

### Demo mode (no hardware required)

For rehearsing the failure → diagnosis → recovery flow without fighting
venue Wi-Fi or a flaky device:

```bash
# Clean pass, fully simulated
python agent.py --device DEMO123 --simulate "Run a YouTube playback test for 30 seconds"

# Deterministically fail the first playback attempt (Wi-Fi drops mid-test),
# so you can show the agent diagnose and recover live
python agent.py --device DEMO123 --simulate --inject-failure "Run a YouTube playback test for 30 seconds"
```

`--simulate` swaps every ADB/Appium call for a fake in-memory backend
(`tools/simulate.py`) — same tool schemas, same agentic loop, same
safety limits — so the reasoning you see is genuinely the NVIDIA NIM model deciding
what to do next, not a scripted transcript.

## Example run

```
========================================
 AI TEST AGENT
========================================

Device: 1A271FDF600B0E
Goal: Run a YouTube playback test for 60 seconds

[AGENT] Checking device...
[ADB] Device connected

[AGENT] Checking Wi-Fi status...
[ADB] Wi-Fi connected

[AGENT] Launching YouTube...
[APPIUM] YouTube launched

[AGENT] Searching for "lofi hip hop radio"...
[APPIUM] Searched for "lofi hip hop radio"

[AGENT] Starting playback...
[APPIUM] Video started

[AGENT] Monitoring playback for 60s...
[TEST] Playback healthy for 60s

[AGENT] Finalizing report...

========================================
 FINAL RESULT
========================================
Device: 1A271FDF600B0E
Goal: Run a YouTube playback test for 60 seconds
Status: PASS
Attempts: 1
Duration: 60 seconds
Failure observed: No
========================================
```

With `--inject-failure`, playback stalls partway through, and the NVIDIA NIM model —
not a hardcoded branch — decides to collect logs, re-check Wi-Fi, toggle
it back on, verify, and retry, ending in a PASS with `Attempts: 2` and
`Recovery: Successful`.

Every run also writes a JSON report to `reports/` and captured logs to
`logs/`.

## Safety limits

All enforced in Python, not merely requested of the model (`config.py` / `.env`):

- `MAX_TOOL_CALLS` (default 25) — hard cap on tool calls per run.
- `MAX_RETRIES` (default 2) — `retry_test` refuses once exhausted.
- `MAX_LLM_TURNS` (default 30) — hard cap on the NVIDIA NIM model round-trips.
- `MAX_WAIT_SECONDS` (default 180) — clamps any single monitoring call.

If a limit is hit before the model calls `generate_report` itself,
`agent.py` force-generates a `NEEDS_HUMAN` report directly (bypassing
another LLM call) so the run always ends cleanly.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Device '<id>' was not found by adb` | Run `adb devices`; check USB debugging + authorization. Or use `--simulate`. |
| `adb executable not found on PATH` | Install Android platform-tools and add it to `PATH`. |
| Appium `ConnectionRefusedError` / session fails to start | Make sure `appium` is running (`appium` in a separate terminal) and `APPIUM_SERVER_URL` in `.env` matches. |
| `Could not locate the search icon` / other locator errors | YouTube's UI changes across app versions/regions — adjust the resource-ids/accessibility-ids in `tools/appium_tools.py`. The agent still handles this gracefully (reports it as a structured error) rather than crashing. |
| `NVIDIA_NIM_API_KEY is not set` | Copy `.env.example` to `.env` and fill in your key. |
| Model keeps re-checking the same state | Lower `MAX_TOOL_CALLS`/`MAX_LLM_TURNS` won't fix prompting issues — check `llm_client.build_system_prompt` if you customize the flow; the bundled prompt already tells it to be efficient. |
| Authentication failure (401/403) | Check `NVIDIA_NIM_API_KEY` and NVIDIA model access; keys from other providers do not work. |
| Unsupported model/request (400/404/422) | Verify `NVIDIA_NIM_MODEL`, its token limit, and support for tools, streaming, and vision. |
| Rate limit/quota (429) | Wait and retry or check your NVIDIA quota. Transient requests use bounded SDK retries. |
| Timeout / network error | Check connectivity; adjust `NVIDIA_NIM_TIMEOUT_SECONDS` if needed. |
| Want a different model | Set `NVIDIA_NIM_MODEL` to a supported ID from the NVIDIA API Catalog. |

## NVIDIA NIM configuration and validation

```ini
NVIDIA_NIM_API_KEY=your_nvidia_nim_api_key
NVIDIA_NIM_MODEL=your_supported_model_id
NVIDIA_NIM_MAX_TOKENS=4096
NVIDIA_NIM_STREAM=false
NVIDIA_NIM_VISION=false
NVIDIA_NIM_TIMEOUT_SECONDS=120
```

Environment variables override the `.env` next to `config.py`. Keep `.env`
private (it is ignored by Git). Never put the key in prompts or frontend code.
The client sends Bearer authentication rather than the previous provider's
`x-api-key`. Existing keys must be replaced with a NVIDIA-issued key; changing
the SDK cannot make an invalid credential valid.

Set `NVIDIA_NIM_STREAM=true` for live line-by-line console narration. Complete
streamed tool names and JSON argument fragments are assembled before dispatch;
a truncated or interrupted response never executes partial tool calls. Normal
responses remain the default. Tool history uses assistant `tool_calls` followed
by matching `role=tool` messages. Device binding, recipes, reports, retries, and
tool budgets remain enforced locally.

Vision defaults off because model capabilities vary. UI element lists and
player state remain available. With `NVIDIA_NIM_VISION=true`, JPEG screenshots
are sent as `image_url` data URLs in a user message after all tool results;
choose a model supporting both images and tools. Screenshot bytes stay out of
saved history, reports, and console tool results. An unsupported feature returns
an actionable configuration error and a `NEEDS_HUMAN` report; there is no
silent switch to a different model or inference provider.

Provider-specific thinking flags, ephemeral prompt-cache controls, beta headers,
and server-side model fallbacks were removed. They have no universal equivalent
in NIM's Chat Completions API. Reasoning and caching depend on the selected
model/service. See the [NVIDIA API Catalog](https://docs.api.nvidia.com/nim/reference/llm-apis)
for model-specific capabilities.

Run offline protocol and simulated-device regression checks:

```bash
python -m unittest discover -s tests -v
```

After configuring a real NVIDIA key and model, run the live smoke checks
(no Android device needed; these consume NVIDIA inference quota):

```bash
python validate_nim.py
```

This checks normal chat, streaming chat, and normal/streaming function calling
including tool-result history. Then run the existing `agent.py --simulate`
examples to exercise the full agent with live NVIDIA inference before using a
real device. Archived `.archify` exports describe the pre-migration architecture
and are historical artifacts.
