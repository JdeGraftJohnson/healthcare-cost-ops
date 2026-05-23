"""Template-fit judge — measures the alignment between the spec's
declared templates and what the generator actually emitted.

For each template id in spec.templates, check that every required
measure name from services.dax.templates.CATALOGUE appears in the
generated DAX.
"""
from __future__ import annotations

import re
from typing import Any

from ..dax.templates import CATALOGUE
from .base import Finding, JudgeResult


def run(spec: dict[str, Any], measures_dax: str) -> JudgeResult:
    findings: list[Finding] = []
    requested = set(spec.get("templates", []))
    if not requested:
        findings.append(Finding("WARN", "NO_TEMPLATES", "spec.templates is empty"))
        return JudgeResult(judge="template_fit", tier="deterministic", score=0.5,
                           weight=1.0, findings=findings)

    matched = 0
    expected_total = 0
    for entry in CATALOGUE:
        if entry["id"] not in requested:
            continue
        for m in entry["measures"]:
            expected_total += 1
            pat = re.compile(r"\[\s*" + re.escape(m) + r"\s*\]\s*:?=")
            if pat.search(measures_dax):
                matched += 1
            else:
                findings.append(Finding("WARN", "MEASURE_MISSING",
                                        f"template {entry['id']} expects measure [{m}] not found"))
    score = matched / expected_total if expected_total else 0.0
    if score == 1.0:
        findings.append(Finding("OK", "TEMPLATE_FIT", f"{matched}/{expected_total} measures emitted"))
    return JudgeResult(judge="template_fit", tier="deterministic", score=score,
                       weight=1.0, findings=findings,
                       notes=f"matched={matched} expected={expected_total}")
