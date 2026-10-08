"""Per-device JSONL diagnostics with credential and screenshot redaction."""
import json
import os
import re
import time
import traceback
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone

from config import LOG_DIR, NVIDIA_NIM_API_KEY

_active = ContextVar("run_diagnostics", default=None)
_SENSITIVE = re.compile(r"api.?key|authorization|token|secret|password|cookie|_image_jpeg_b64", re.I)


def sanitize(value):
    if isinstance(value, dict):
        return {str(k): "[REDACTED]" if _SENSITIVE.search(str(k)) and str(k) != "max_tokens"
                else sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize(v) for v in value]
    if isinstance(value, str):
        secrets = {NVIDIA_NIM_API_KEY} | {v for k, v in os.environ.items() if _SENSITIVE.search(k) and len(v) >= 8}
        for secret in sorted((v for v in secrets if v), key=len, reverse=True):
            value = value.replace(secret, "[REDACTED]")
        value = re.sub(r"(?i)Bearer\s+\S+", "Bearer [REDACTED]", value)
        value = re.sub(r"\b(?:nvapi-|sk-)[A-Za-z0-9_-]+", "[REDACTED]", value)
        value = re.sub(r"data:image/[^;]+;base64,[A-Za-z0-9+/=]+", "[REDACTED IMAGE]", value)
        value = re.sub(r'(?i)((?:api[_-]?key|token|password|secret)\s*[=:]\s*)[^\s,;]+', r'\1[REDACTED]', value)
        return value[:16000]
    if value is None or isinstance(value, (int, float, bool)):
        return value
    return sanitize(str(value))


def current_log_path():
    active = _active.get()
    return str(active["path"]) if active else None


def record(event, **fields):
    active = _active.get()
    if active is None:
        return
    data = dict(timestamp=datetime.now(timezone.utc).isoformat(),
                elapsed_seconds=round(time.monotonic() - active["started"], 3),
                run_id=active["id"], device_id=active["device"], phase=active["phase"], event=event)
    data.update(fields)
    with active["path"].open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(sanitize(data), ensure_ascii=False) + "\n")


def record_exception(event, exc):
    # No locals, credentials, or arbitrary exception body in stack traces.
    record(event, exception_type=type(exc).__name__, message=str(exc),
           stack=traceback.format_tb(exc.__traceback__))


@contextmanager
def diagnostic_run(device_id, phase):
    ident = uuid.uuid4().hex[:12]
    safe_device = re.sub(r"[^A-Za-z0-9_-]", "_", device_id)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = LOG_DIR / f"{stamp}_{safe_device}_{phase}_{ident}.jsonl"
    path.touch(mode=0o600)
    token = _active.set(dict(path=path, id=ident, device=device_id, phase=phase, started=time.monotonic()))
    try:
        record("run_started")
        yield str(path)
    except Exception as exc:
        record_exception("run_exception", exc)
        raise
    finally:
        record("run_ended")
        _active.reset(token)
