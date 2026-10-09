"""Diagnostic and terminal tools available during one step repair."""

import log_collector
from tools.simulate import SimulatedState


def collect_logs(device_id: str, reason: str = "", sim: SimulatedState | None = None) -> dict:
    if sim is not None:
        return log_collector.capture_simulated(reason)
    return log_collector.capture_logcat(device_id, reason)


def step_done(fixed: bool, summary: str, state, needs_human: bool = False) -> dict:
    result = {"fixed": fixed, "summary": summary, "needs_human": needs_human}
    state.fix_result = result
    return result
