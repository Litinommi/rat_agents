"""Higher-level "test flow" tools that combine ADB/uiautomator2 primitives with
run state: collecting logs, retrying the playback test, and closing out
the run with a report. These are the tools that give the agent loop its
explicit stop condition (generate_report) and its bounded recovery path
(retry_test).
"""

import log_collector
import report_generator
from diagnostics import current_log_path
from config import MAX_RETRIES
from tools import youtube_tools
from tools.simulate import SimulatedState


def collect_logs(device_id: str, reason: str = "", sim: SimulatedState | None = None) -> dict:
    if sim is not None:
        return log_collector.capture_simulated(reason)
    return log_collector.capture_logcat(device_id, reason)


def retry_test(
    device_id: str,
    reason: str,
    state,
    sim: SimulatedState | None = None,
) -> dict:
    """Re-run the playback steps (stop -> relaunch -> open URL -> clear ads)
    for the run's video URL. Bounded by MAX_RETRIES, enforced here
    regardless of what the model asks for. Fullscreen / stats for nerds
    are left to the agent to redo, since they may need its UI fallback.
    """
    if state.retries_used >= MAX_RETRIES:
        return {
            "success": False,
            "error": f"Retry limit reached ({MAX_RETRIES} retries used) - do not retry again, "
            "call generate_report instead.",
            "retries_used": state.retries_used,
            "retries_remaining": 0,
        }

    state.retries_used += 1
    state.attempts += 1

    steps = [("stop_video", youtube_tools.stop_video(device_id, sim=sim))]
    steps.append(("launch_youtube", youtube_tools.launch_youtube(device_id, sim=sim)))
    if sim is not None:
        sim.note_video_started()
        steps.append(("open_video_url", {"success": True, "url": state.video_url}))
    else:
        steps.append(("open_video_url", youtube_tools.open_video_url(device_id, state.video_url)))
        steps.append(("wait_for_ads", youtube_tools.wait_for_ads(device_id)))

    return {
        "reason": reason,
        "steps": [{"tool": name, "result": result} for name, result in steps],
        "retries_used": state.retries_used,
        "retries_remaining": MAX_RETRIES - state.retries_used,
        "attempt_number": state.attempts,
    }


def generate_report(
    device_id: str,
    status: str,
    summary: str,
    state,
    root_cause: str | None = None,
    recovery_action: str | None = None,
) -> dict:
    """Terminal tool. Building the report also marks the run finished, so
    the agent loop in llm_client.py stops after this call executes.
    """
    report = report_generator.build_report(
        device_id=device_id,
        goal=state.goal,
        status=status,
        summary=summary,
        attempts=max(state.attempts, 1),
        duration_seconds=state.duration_seconds,
        failure_observed=state.failure_observed,
        root_cause=root_cause,
        recovery_action=recovery_action,
        tool_call_count=state.tool_call_count,
        # Snapshot history before the registry records this terminal tool result.
        tool_history=list(state.tool_history),
    )
    report["diagnostic_log"] = current_log_path()
    report["initial_failure"] = getattr(state, "initial_failure", None)
    saved_path = report_generator.save_report(report)
    report["saved_to"] = saved_path

    state.finished = True
    state.final_report = report
    return report
