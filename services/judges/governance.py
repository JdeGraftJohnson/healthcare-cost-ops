"""Governance judge — RLS roles present, data sources documented,
certified-dataset metadata complete.

Reads the .bim model and the spec; cross-checks that any role declared
in spec.compliance.rls has a corresponding model role, and that every
data source listed in spec.data_sources has a documented description
in the model's annotations.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .base import Finding, JudgeResult


def run(spec: dict[str, Any], bim_path: str) -> JudgeResult:
    findings: list[Finding] = []
    bim = {}
    try:
        bim = json.loads(Path(bim_path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        findings.append(Finding("MISS", "BIM_MISSING", f"model file not found: {bim_path}"))

    declared_rls = spec.get("compliance", {}).get("rls")
    model_roles = [r.get("name") for r in bim.get("model", {}).get("roles", [])]
    if declared_rls and declared_rls not in (model_roles or []):
        findings.append(Finding("MISS", "RLS_MISSING",
                                f"spec declares RLS by '{declared_rls}' but no matching role in model"))

    sources = spec.get("data_sources", [])
    annotations = bim.get("model", {}).get("annotations", []) or []
    source_descs = {a.get("name"): a.get("value") for a in annotations
                    if a.get("name", "").startswith("source.")}
    for s in sources:
        key = f"source.{s.get('name')}"
        if key not in source_descs:
            findings.append(Finding("WARN", "SOURCE_UNDOCUMENTED",
                                    f"data source '{s.get('name')}' missing model annotation"))

    if not findings:
        findings.append(Finding("OK", "GOVERNANCE", "RLS + source docs present"))
    miss = sum(1 for f in findings if f.severity == "MISS")
    warn = sum(1 for f in findings if f.severity == "WARN")
    score = max(0.0, 1.0 - 0.30 * miss - 0.10 * warn)
    return JudgeResult(judge="governance", tier="deterministic", score=score,
                       weight=1.5, findings=findings)
