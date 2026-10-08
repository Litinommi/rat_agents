"""Builds, prints, and persists the final test report."""

import json
from datetime import datetime, timezone

from config import REPORT_DIR


def build_report(
    *,
    device_id: str,
    goal: str,
    status: str,
    summary: str,
    attempts: int,
    duration_seconds: int,
    failure_observed: bool,
    root_cause: str | None,
    recovery_action: str | None,
    tool_call_count: int,
    tool_history: list[dict],
) -> dict:
    report = {
        "device_id": device_id,
        "goal": goal,
        "status": status,
        "summary": summary,
        "attempts": attempts,
        "duration_seconds": duration_seconds,
        "failure_observed": failure_observed,
        "root_cause": root_cause,
        "recovery_action": recovery_action,
        "recovery_successful": failure_observed and status == "PASS",
        "tool_call_count": tool_call_count,
        "tool_history": tool_history,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    return report


def save_report(report: dict) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_device = "".join(c if c.isalnum() else "_" for c in report["device_id"])
    path = REPORT_DIR / f"{timestamp}_{safe_device}_report.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return str(path)


def print_report(report: dict) -> None:
    print()
    print("=" * 40)
    print(" FINAL RESULT")
    print("=" * 40)
    print(f"Device: {report['device_id']}")
    print(f"Goal: {report['goal']}")
    print(f"Status: {report['status']}")
    print(f"Attempts: {report['attempts']}")
    print(f"Duration: {report['duration_seconds']} seconds")
    if report.get("diagnostic_log"):
        print(f"Diagnostic log: {report['diagnostic_log']}")
    if report.get("saved_to"):
        print(f"Report file: {report['saved_to']}")
    print(f"Failure observed: {'Yes' if report['failure_observed'] else 'No'}")
    if report["failure_observed"]:
        print(f"Root cause: {report['root_cause'] or 'unknown'}")
        print(f"Recovery: {'Successful' if report['recovery_successful'] else 'Not successful'}")
    print("=" * 40)
