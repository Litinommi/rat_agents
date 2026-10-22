# YouTube real-application test with AI fallback

`lf_interop_youtube.py` runs the normal LANforge YouTube test on Android phones
and laptops. Each device runs its original script; when a step misses its
target, the script falls back to saved recipes and then to an
OpenAI-compatible LLM (e.g. Ollama). A verified repair is saved as a recipe, so
the next device with the same issue replays it without the LLM.

| Devices | Files to deploy together | Recipes |
|---|---|---|
| Android (LANforge `py-scripts/real_application_tests/youtube/`) | `youtube_android_test.py`, `android_ai_repair.py`, `learned_recipes.json` | per phone profile (maker, model, YouTube version, locale); phones with the same YouTube version and language reuse each other's recipes |
| Laptops (`real_application_test/youtube/`) | `youtube.py`, `browser_repair.py` | `browser_recipes.json` (locators), `browser_stage_recipes.json` (stages) |

Android steps with a fallback: video open, ads, resolution, landscape,
Stats for nerds (including the YouTube "Enable stats for nerds" app setting),
and stalled playback. Order per failed step: saved recipe for the step → the
step's original code → saved popup recipes → LLM (limited to that step, at most
3 LLM repairs per phone and 25 actions per repair).

LLM settings come from the environment of the user running the script
(defaults in brackets):

```
LLM_BASE_URL  [http://192.168.200.182:11434/v1]
LLM_MODEL     [qwen3.8:latest]   # must support tool calling
LLM_API_KEY   [ollama]
```

If the AI file is missing or the LLM is unreachable, the original flow runs
unchanged.
