"""Paired auditor for dax_syntax + dax_perf.

Re-reads each flagged DAX measure in surrounding context and either
confirms or dismisses the finding. The deterministic judge can't tell
whether FILTER(ALL(...)) is an anti-pattern or a legitimate denominator
context clear; this auditor can.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .. import _llm_stub
from ..base import JudgeResult

RUBRIC = """
For each deterministic finding, decide:
  CONFIRM  — the issue is real, severity stays.
  DISMISS  — false positive given context, severity → OK.
  ESCALATE — issue is real and severity should go up (WARN → MISS).

Emit one <finding> per audit decision with the new severity.
"""


def run(spec: dict[str, Any], dax_syntax_result_path: str, dax_perf_result_path: str,
        measures_path: str) -> JudgeResult:
    parts: list[str] = ["## Deterministic findings\n"]
    for p in (dax_syntax_result_path, dax_perf_result_path):
        if Path(p).exists():
            parts.append(Path(p).read_text(encoding="utf-8"))
    parts.append("\n## Measures\n")
    if Path(measures_path).exists():
        parts.append(Path(measures_path).read_text(encoding="utf-8"))
    return _llm_stub.run_llm_judge("dax_auditor", RUBRIC, "\n".join(parts), tier_weight=1.0)
