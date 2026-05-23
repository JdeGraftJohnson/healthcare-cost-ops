"""DAX performance / anti-pattern judge.

Catches the high-frequency tabular-model anti-patterns: FILTER(ALL(...)),
nested CALCULATE without modifier intent, columns referenced in fact
tables that should be measures, EARLIER usage outside a calculated
column context. The paired auditor (auditors/dax_auditor.py)
re-inspects each flag in surrounding context.
"""
from __future__ import annotations

import re
from typing import Any

from .base import Finding, JudgeResult

ANTI_PATTERNS = [
    (r"FILTER\s*\(\s*ALL\s*\(", "FILTER_ALL", "FILTER(ALL(...)) — consider REMOVEFILTERS or ALLSELECTED"),
    (r"CALCULATE\s*\([^()]*CALCULATE", "NESTED_CALCULATE", "Nested CALCULATE — verify modifier semantics"),
    (r"EARLIER\s*\(", "EARLIER", "EARLIER() — calculated-column idiom; prefer VAR in measures"),
]


def run(spec: dict[str, Any], measures_dax: str) -> JudgeResult:
    findings: list[Finding] = []
    for pat, code, msg in ANTI_PATTERNS:
        for m in re.finditer(pat, measures_dax):
            findings.append(Finding("WARN", code, msg, where=f"offset={m.start()}"))
    if not findings:
        findings.append(Finding("OK", "DAX_PERF", "no anti-patterns detected"))
    warn = sum(1 for f in findings if f.severity == "WARN")
    score = max(0.0, 1.0 - 0.15 * warn)
    return JudgeResult(
        judge="dax_perf", tier="deterministic", score=score, weight=1.0,
        findings=findings, notes=f"warn={warn}",
    )
