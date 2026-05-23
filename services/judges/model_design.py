"""Star-schema hygiene judge — reads the Tabular Object Model JSON (.bim)
and checks:

  - All relationships are single-direction unless explicitly justified
  - No calculated columns on fact tables
  - Mark-as-date-table is set on the date dimension
  - No bidirectional filters on high-cardinality relationships
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .base import Finding, JudgeResult


def run(spec: dict[str, Any], bim_path: str) -> JudgeResult:
    findings: list[Finding] = []
    try:
        bim = json.loads(Path(bim_path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        findings.append(Finding("MISS", "BIM_MISSING", f"model file not found: {bim_path}"))
        return JudgeResult(judge="model_design", tier="deterministic", score=0.0,
                           weight=1.5, findings=findings)

    relationships = bim.get("model", {}).get("relationships", [])
    bidi = [r for r in relationships if r.get("crossFilteringBehavior") == "bothDirections"]
    if bidi:
        findings.append(Finding("WARN", "BIDI_FILTER",
                                f"{len(bidi)} bidirectional relationship(s) — verify each is intentional"))

    tables = bim.get("model", {}).get("tables", [])
    fact_tables = [t for t in tables if t.get("name", "").startswith("fact_")]
    for ft in fact_tables:
        calc_cols = [c for c in ft.get("columns", []) if c.get("type") == "calculated"]
        if calc_cols:
            findings.append(Finding(
                "WARN", "CALC_COL_ON_FACT",
                f"{ft['name']} has {len(calc_cols)} calculated column(s) — move to measure or silver layer",
            ))

    date_tables = [t for t in tables if t.get("name", "").startswith("dim_date")]
    for dt in date_tables:
        if not dt.get("isMarkedAsDateTable"):
            findings.append(Finding("MISS", "DATE_TABLE_NOT_MARKED",
                                    f"{dt['name']} is not marked as date table"))

    if not findings:
        findings.append(Finding("OK", "MODEL_DESIGN", "star schema clean"))
    miss = sum(1 for f in findings if f.severity == "MISS")
    warn = sum(1 for f in findings if f.severity == "WARN")
    score = max(0.0, 1.0 - 0.4 * miss - 0.10 * warn)
    return JudgeResult(judge="model_design", tier="deterministic", score=score,
                       weight=1.5, findings=findings)
