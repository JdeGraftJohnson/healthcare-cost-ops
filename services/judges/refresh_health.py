"""Refresh-health judge — placeholder for now; reads a refresh log if
present and reports success rate over the last 30 days.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .base import Finding, JudgeResult


def run(spec: dict[str, Any], refresh_log_path: str | None = None) -> JudgeResult:
    findings: list[Finding] = []
    if not refresh_log_path or not Path(refresh_log_path).exists():
        findings.append(Finding("OK", "REFRESH_NOT_TRACKED",
                                "no refresh log present (local-render mode)"))
        return JudgeResult(judge="refresh_health", tier="deterministic", score=1.0,
                           weight=0.5, findings=findings)
    log = json.loads(Path(refresh_log_path).read_text(encoding="utf-8"))
    total = len(log)
    ok = sum(1 for row in log if row.get("status") == "Success")
    rate = ok / total if total else 1.0
    if rate < 0.95:
        findings.append(Finding("WARN", "LOW_REFRESH_RATE",
                                f"{rate:.0%} success over last {total} refreshes (<95%)"))
    if not findings:
        findings.append(Finding("OK", "REFRESH_HEALTH", f"{rate:.0%} success rate"))
    return JudgeResult(judge="refresh_health", tier="deterministic", score=rate,
                       weight=0.5, findings=findings)
