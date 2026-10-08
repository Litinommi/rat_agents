# AI Test Agent — YouTube Playback PoC

An AI agent tests YouTube playback on Android devices using ADB and
uiautomator2. A NVIDIA NIM model selects predefined tools, diagnoses failures,
verifies recovery, and produces a PASS, FAIL, or NEEDS_HUMAN report. The agent
can inspect unfamiliar interfaces and save verified UI steps as reusable recipes.

The OpenAI SDK is the compatible client library. All inference requests use
`https://integrate.api.nvidia.com/v1`; no OpenAI API key is required or used.

## Prerequisites

- Python 3.10 or newer; validation during this migration used Python 3.12.
- For real devices: Android SDK Platform Tools with `adb` on your `PATH`.
- An Android phone or emulator with USB debugging enabled and authorized.
  `adb devices` must show it in the `device` state.
- YouTube installed on the target device.
- For AI runs: a NVIDIA-issued API key and a NVIDIA NIM model supporting
  OpenAI-compatible function calling with `tool_choice=auto`. Choose a model
  and check its capabilities in the [NVIDIA API Catalog](https://build.nvidia.com/).

Device automation uses the Python `uiautomator2` package. An Appium server and
Node.js are not required. Simulated devices need neither ADB nor Android hardware,
but the agent still uses live NVIDIA inference unless running the offline tests.

## Setup

Run these commands from `ai_test_agent`:

```bash
python -m venv .venv
```

Activate the environment on macOS/Linux:

```bash
source .venv/bin/activate
```

Or on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` using `cp .env.example .env` on macOS/Linux or
`Copy-Item .env.example .env` in PowerShell. If `.env` already exists, edit it
instead of replacing your device settings.

Set both required values:

```ini
NVIDIA_NIM_API_KEY=your_nvidia_nim_api_key
NVIDIA_NIM_MODEL=your_supported_model_id
```

Replace the placeholders with your NVIDIA key and a supported model ID.
Environment variables override the `.env` file next to `config.py`. Keep `.env`
private; Git ignores it. Never put keys in prompts, frontend code, or logs.

For an existing environment, installing requirements replaces the declared SDK
dependency but does not uninstall previously installed packages. The application
no longer uses the Anthropic SDK or its environment settings. Remove its old
settings from your local `.env`; if no other project uses that environment, you
can remove the unused SDK with `python -m pip uninstall anthropic`.

## NVIDIA NIM configuration

| Variable | Default | Purpose |
|---|---|---|
| `NVIDIA_NIM_API_KEY` | Required | NVIDIA API key used for Bearer authentication |
| `NVIDIA_NIM_MODEL` | Required | Model ID; no automatic default or fallback model |
| `NVIDIA_NIM_MAX_TOKENS` | `4096` | Output token budget; stay within the selected model's limit |
| `NVIDIA_NIM_STREAM` | `false` | Enable live console narration as complete lines arrive |
| `NVIDIA_NIM_VISION` | `false` | Send screenshots to a model supporting images and tools |
| `NVIDIA_NIM_TIMEOUT_SECONDS` | `120` | SDK request timeout; must be positive |

The base URL is fixed in `config.py` to the NVIDIA endpoint. The SDK makes up to
two retries for eligible transient request failures. The application does not
switch models or inference providers after an error.

Function calling is required for the agent workflow. Streaming and vision depend
on the selected model. With vision disabled, the model receives UI element lists
and player state. With vision enabled, screenshots are JPEG `image_url` data URLs
in a user message following all tool results. Screenshot bytes are excluded from
saved tool history, reports, and console tool results.

The previous provider's thinking flags, ephemeral cache controls, beta headers,
and server-side fallbacks are no longer sent. Reasoning, caching, and feature
availability depend on the NVIDIA model/service. Consult the
[NVIDIA API reference](https://docs.api.nvidia.com/nim/reference/llm-apis)
for model-specific capabilities.

## Running

All examples below run from `ai_test_agent`. `--duration` is in **minutes**;
`agent.py` defaults to one minute and requires at least five seconds.

List devices:

```bash
python agent.py --list-devices
```

Run a one-minute test on a real device, replacing the device ID and video URL:

```bash
python agent.py --devices DEVICE_ID --video-url "https://www.youtube.com/watch?v=VIDEO_ID" --duration 1
```

Several devices run in parallel:

```bash
python agent.py --devices DEVICE_1 DEVICE_2 --video-url "https://www.youtube.com/watch?v=VIDEO_ID" --duration 1
```

Fullscreen and Stats for nerds are enabled by default. Disable them or add an
extra natural-language instruction using the supported flags:

```bash
python agent.py --devices DEVICE_ID --video-url "https://youtu.be/VIDEO_ID" --duration 0.5 --no-fullscreen --no-stats-for-nerds --goal "Verify playback and explain any recovery actions."
```

### Simulated devices

These examples use fake device/app state and live NVIDIA inference:

```bash
python agent.py --devices DEMO123 --video-url "https://youtu.be/VIDEO_ID" --duration 0.5 --simulate
python agent.py --devices DEMO123 --video-url "https://youtu.be/VIDEO_ID" --duration 0.5 --simulate --inject-failure
python agent.py --devices DEMO123 --video-url "https://youtu.be/VIDEO_ID" --duration 0.5 --simulate --sim-unfamiliar-ui
```

`--inject-failure` simulates a network failure on the first playback attempt.
`--sim-unfamiliar-ui` makes the built-in Stats for nerds step fail on a simulated
Spanish interface, allowing the agent to practice UI inspection and adaptation.
A successful recovery depends on the model's actions; it is not guaranteed.

### Scripted playback and optional AI fallback

The scripted runner needs no inference key unless `--ai-fallback` is enabled:

```bash
python play_youtube.py --devices DEVICE_ID --video-url "https://youtu.be/VIDEO_ID" --duration 1
python play_youtube.py --devices DEVICE_ID --video-url "https://youtu.be/VIDEO_ID" --duration 1 --ai-fallback
```

It tries built-in steps and saved recipes first. AI fallback groups failed
phones by YouTube version and language, attempts a repair on one representative,
and replays learned steps on similar phones. `--keep-playing` leaves playback
running after a successful scripted test. Use `--help` on either entry point
for its full CLI options.

## Architecture and tools

```text
agent.py / play_youtube.py --ai-fallback
  -> TestState + registry bound to one device
  -> llm_client.py: system prompt, tool schemas, conversation loop
  -> nim_client.py: OpenAI SDK -> NVIDIA NIM /chat/completions
  -> predefined Python tools -> ADB / uiautomator2 (or simulated device)
  -> matching tool results in conversation history
  -> final report
```

History consists of a system prompt, user goal, assistant `tool_calls`, and
matching `role=tool` results. Streamed tool names and JSON argument fragments are
assembled before dispatch. Interrupted or truncated responses never execute
partial tool calls. Multiple tools from a response execute sequentially; actions
stop once the report is finalized or the tool budget is exhausted.

The 22 tool schemas in `llm_client.py` expose these operations:

| Tools | Purpose |
|---|---|
| `check_device`, `get_device_info`, `get_wifi_status`, `toggle_wifi` | Inspect device/connectivity and recover Wi-Fi |
| `launch_youtube`, `open_video_url`, `wait_for_ads`, `stop_video` | Launch and control playback at the configured URL |
| `enable_stats_for_nerds`, `enable_stats_in_app_settings`, `enter_fullscreen` | Apply player setup and replay learned recipes |
| `get_playback_status`, `get_player_state` | Monitor playback and verify setup |
| `get_screen`, `reveal_player_controls`, `tap_element`, `press_key`, `swipe` | Inspect and navigate the device UI |
| `save_recipe` | Save verified UI steps for reuse |
| `collect_logs`, `retry_test`, `generate_report` | Diagnose, retry, and finish the run |

The model cannot choose a different device or video URL through tool arguments;
those values are bound locally. There is no arbitrary shell-execution tool.

## Project structure

```text
ai_test_agent/
├── agent.py              AI CLI and per-device console narration
├── play_youtube.py       Scripted runner and grouped AI fallback
├── llm_client.py         Tool schemas, prompt, conversation loop
├── nim_client.py         NVIDIA transport, streaming, safe API errors
├── validate_nim.py       Live chat/stream/tool smoke checks
├── test_runner.py        Run state, bound registry, safety limits
├── tools/
│   ├── adb_tools.py      ADB device and connectivity operations
│   ├── youtube_tools.py  uiautomator2 playback and player state
│   ├── ui_tools.py       UI inspection, navigation, recipe storage
│   ├── test_tools.py     Logs, retries, terminal report
│   └── simulate.py       Fake device/app backend
├── tests/test_nim.py     Offline protocol and simulation tests
├── log_collector.py      logcat capture
├── report_generator.py   Report creation, printing, persistence
├── learned_recipes.json  Saved device-profile recipes
├── config.py             Environment configuration
├── requirements.txt
├── .env.example
└── README.md
```

Reports are written to `reports/`, and captured device logs to `logs/`. Console
output includes tool narration and a final status per device. Exit status is zero
when all devices pass; failures or unfinished outcomes return a nonzero status.
Archived `.archify` exports describe the earlier architecture and are historical
artifacts.

## Safety limits

| Variable | Default | Enforcement |
|---|---|---|
| `MAX_TOOL_CALLS` | `60` | Hard cap on registry tool calls per run |
| `MAX_RETRIES` | `2` | Hard cap on playback retries |
| `MAX_LLM_TURNS` | `60` | Maximum inference turns in the agent loop |
| `MAX_WAIT_SECONDS` | `180` | Default cap for direct playback-monitor calls; the agent registry uses the configured test duration plus 30 seconds |

These limits are enforced in Python. Stats for nerds must be verified before
agent playback monitoring when that feature is requested. If a budget/turn limit
or model refusal prevents completion, the CLI generates a NEEDS_HUMAN report.
Handled inference failures also produce a NEEDS_HUMAN report with a sanitized
error instead of provider response bodies or credentials.

## Validation

Run offline checks without NVIDIA credentials or Android hardware:

```bash
python -m unittest discover -s tests -v
```

These use the real SDK with mocked HTTP transport to check authentication headers,
normal/streaming responses, tool-call assembly and history, screenshots, safe
errors, budget enforcement, and simulated playback/report generation.

After configuring a real NVIDIA key and model, run live checks:

```bash
python validate_nim.py
```

This consumes NVIDIA inference quota and checks normal chat, streaming chat, and
normal/streaming tool calling with tool-result history. It does not access an
Android device or validate vision. Test vision and real-device playback separately
with a suitable model and a connected phone.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Missing `NVIDIA_NIM_API_KEY` or `NVIDIA_NIM_MODEL` | Set both in the project `.env` or environment; replace example placeholders |
| Authentication failure (401/403) | Check your NVIDIA-issued key and access to the selected model |
| Old `invalid x-api-key` error | Ensure you are running the migrated checkout/environment; this client uses NVIDIA Bearer authentication |
| Unsupported model/request (400/404/422) | Verify the model ID, token budget, and support for function calling, streaming, or vision |
| Rate limit/quota (429) | Wait before retrying or check your NVIDIA quota |
| Timeout/network failure | Check connectivity; adjust `NVIDIA_NIM_TIMEOUT_SECONDS` if needed |
| Truncated response | Adjust `NVIDIA_NIM_MAX_TOKENS` within the model's supported limit |
| Model does not call tools | Select a model supporting automatic function calling; inspect the goal/prompt |
| `adb` missing or device not found | Install Android Platform Tools, check USB debugging/authorization, and run `adb devices` |
| UI step fails | Inspect tool results and current UI; the agent can adapt using `get_screen` and navigation tools |
| Screenshots omitted | Enable `NVIDIA_NIM_VISION=true` only for a model supporting images and tool calling |

## Diagnostic logs and this failure pattern

Each scripted or AI device run prints its diagnostic log path. Logs use JSONL
(one timestamped JSON event per line) under `logs/`; parallel devices and phases
have separate files. Final reports link to the AI diagnostic log. A fallback run
retains the initial scripted failure in its report's failure flag.

Events include tool inputs/results, tool durations, ADB command exit codes,
current Wi-Fi evidence, ad markers/player readiness, and the UI hierarchy at an
ad timeout. NVIDIA events include the model, history size/roles, HTTP attempts,
SDK retry numbers, response status, request IDs, retry-after headers, and
sanitized provider error fields. API keys, authorization values, and screenshot
bytes are redacted; full inference request payloads and headers are not logged.
Device logcat is captured automatically on scripted failures and AI inference
failures, and remains available through `collect_logs` as a separate `.log` file.
Capture is best-effort; failures to read device logs do not replace the original error. Diagnostic JSONL files are ignored by Git.

For `nvidia/nemotron-3-ultra-550b-a55b`, keep `NVIDIA_NIM_VISION=false`: the
[NVIDIA model page](https://build.nvidia.com/nvidia/nemotron-3-ultra-550b-a55b)
lists text input and tool calling. `NVIDIA_NIM_STREAM=true` provides live text
narration; it does not make a persistent provider HTTP 500 disappear. The SDK
already retries eligible failures twice. The logged provider error and request
ID distinguish service errors from model/request processing failures; the
console message alone cannot establish the exact cause.

An ad timeout now distinguishes visible player ad markers, missing player UI,
and unstable player state. Sponsored cards below the player do not count as
video ad evidence. Wi-Fi checks use current Android status and return unknown
when the association cannot be established. An accepted enable command does not
connect the phone to a saved network or prove internet availability. A phone can
also stream through mobile data; repeated Wi-Fi enabling is not a diagnosis.

After installing requirements and confirming your key/model, first run
`python validate_nim.py`, then repeat the original playback command. On failure,
inspect `nim_completion_error`, `nim_http_response`, `wifi_diagnostics`, and
`ad_timeout` in the printed files. Live provider checks require NVIDIA
credentials; real-device verification requires an ADB-visible phone.
