# AI Test Agent — YouTube Playback

This runner tests YouTube playback on one or more Android phones. Built-in
uiautomator2 steps do the normal work. When `--ai-fallback` is enabled, only a
failed step is handed to NVIDIA NIM; a successful repair is saved as a recipe
for later phones and runs.

Inference uses NVIDIA's OpenAI-compatible endpoint. There is no arbitrary shell
tool and the model cannot select another device, URL, or later test step.

## Setup

Requirements:

- Python 3.10+
- Android Platform Tools and an ADB-authorized phone for real runs
- YouTube installed on the phone
- A NVIDIA API key and tool-calling model for AI repair

From this directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Set `NVIDIA_NIM_API_KEY` and `NVIDIA_NIM_MODEL` in `.env`. Environment
variables override that file. Keep `.env` private.

The defaults match the NVIDIA catalog settings for
`nvidia/nemotron-3-ultra-550b-a55b`: temperature `1.0`, top-p `0.95`, and its
maximum output of `32768` tokens. Lower `NVIDIA_NIM_MAX_TOKENS` when using a
model with a smaller limit. Screenshots are sent only when
`NVIDIA_NIM_VISION=true` and the selected model supports vision plus tools.

## Running

List ADB devices:

```bash
python play_youtube.py --list-devices
```

Run the deterministic script without inference:

```bash
python play_youtube.py --devices DEVICE_ID --video-url "https://youtu.be/VIDEO_ID" --duration 1
```

Enable repair only for failed steps:

```bash
python play_youtube.py --devices DEVICE_ID --video-url "https://youtu.be/VIDEO_ID" --duration 1 --ai-fallback
```

Multiple device IDs run in parallel. Phones with the same YouTube version and
language take turns repairing: after one phone saves a recipe, the next retries
the built-in step before spending another model call.

`agent.py` remains as a compatibility entry point and simply runs
`play_youtube.py` with AI fallback enabled:

```bash
python agent.py --devices DEVICE_ID --video-url "https://youtu.be/VIDEO_ID" --duration 1
```

Fullscreen and Stats for nerds default to on. Use `--no-fullscreen`,
`--no-stats-for-nerds`, or `--keep-playing` as needed. The old free-form
`--goal` option was removed because repairs are now restricted to one failed
step.

### Simulation

Simulation still calls live NIM when a repair is needed:

```bash
python play_youtube.py --devices DEMO --video-url "https://youtu.be/VIDEO_ID" --duration 0.1 --simulate
python play_youtube.py --devices DEMO --video-url "https://youtu.be/VIDEO_ID" --duration 0.1 --simulate --inject-failure --ai-fallback
python play_youtube.py --devices DEMO --video-url "https://youtu.be/VIDEO_ID" --duration 0.1 --simulate --sim-unfamiliar-ui --ai-fallback
python play_youtube.py --devices DEMO --video-url "https://youtu.be/VIDEO_ID" --duration 0.1 --simulate --sim-stats-setting-off --ai-fallback
```

`--sim-stats-setting-off` reproduces the missing Stats-for-nerds case. The
simulated path is Home → You → Settings → General → switch; a plain second tap
would turn the switch off, so it also exercises the already-on guard.

## Execution model

```text
one thread per phone
  launch → open URL → clear ads → stats → fullscreen → monitor
    built-in step (saved recipes first)
    script verifies target state
    on failure with --ai-fallback:
      replay popup recipes → retry step
      lock by YouTube version + language → retry step
      constrained NIM repair → save recipe → step_done
      re-check this and every earlier step; resume at the first broken step
    playback error:
      constrained repair → verify playing → monitor remaining seconds
  write PASS / FAIL / NEEDS_HUMAN report
```

The script and each repair use the same `TestState`, so the report contains one
ordered tool history. The uiautomator2 session closes only after the phone's
report is written.

## Repair tools

Every repair receives screen inspection/actions, `get_player_state`,
`check_device`, `collect_logs`, `save_recipe`, and terminal `step_done`. It also
receives only the built-in tools relevant to its target:

| Failed step | Additional tools |
|---|---|
| launch | `launch_youtube` |
| open URL | `launch_youtube`, `open_video_url` |
| ads | launch/open plus `wait_for_ads` |
| stats | launch/open/ads plus both stats tools |
| fullscreen | `enter_fullscreen` |
| playback | connectivity, playback, and setup tools needed after a restart |

Only the first tool call in a model turn executes. Additional calls receive
`not executed: one action per turn`. Old screen results are compacted so only
the newest retains its full element list.

`get_screen` reports all switches—including unlabelled switches—as
`"switch": "on"` or `"off"`. `tap_element` never taps an already-on switch or
its row, and returns a reusable `switch_on` recipe step tied to the row label.
Action results include the updated screen and `screen_changed`.

Recipes require notes describing the cause and what differed on the phone.
Passes and failures are counted; a recipe is skipped after two consecutive
failures. Reports are saved under `reports/`, and JSONL diagnostics under
`logs/`.

## Safety limits

| Variable | Default | Purpose |
|---|---:|---|
| `MAX_FIXES_PER_DEVICE` | `3` | Maximum model repair attempts per phone |
| `FIX_MAX_TOOL_CALLS` | `25` | Tool calls available to one repair |
| `MAX_LLM_TURNS` | `60` | Inference turns available to one repair |
| `MAX_WAIT_SECONDS` | `180` | Direct playback monitor ceiling |

Without `--ai-fallback`, setup failures remain warnings, except a missing Stats
overlay still prevents monitoring and fails the run. `step_done(fixed=false)`
produces FAIL, or NEEDS_HUMAN when `needs_human=true`.

## Validation

Offline tests need no API key or phone:

```bash
python -m unittest discover -s tests -v
```

They cover NIM transport safety, the constrained repair loop, switch state and
guards, one-action turns, history compaction, recipe failures, tool filtering,
group serialization, and deterministic resume behavior.

Live NIM transport checks:

```bash
python validate_nim.py
```

The full simulated repair command is:

```bash
python play_youtube.py --devices DEMO --video-url "https://youtu.be/VIDEO_ID" --duration 0.1 --simulate --sim-stats-setting-off --ai-fallback
```

On a real phone, turn off YouTube's **Enable stats for nerds** setting and run
the same command without `--simulate`. The next run should use the saved recipe
without an AI call.

## Project layout

```text
agent.py              compatibility shim
play_youtube.py       per-phone step runner and repair orchestration
llm_client.py         repair prompt, schemas, and one-action model loop
nim_client.py         NVIDIA transport, streaming, usage logging
test_runner.py        shared state and filtered/budgeted registry
tools/youtube_tools.py
tools/ui_tools.py     screen actions and recipe store
tools/simulate.py
tools/test_tools.py   logs and step_done
report_generator.py
```
