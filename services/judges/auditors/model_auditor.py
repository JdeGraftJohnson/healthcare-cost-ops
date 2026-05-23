"""Paired auditor for model_design + governance."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .. import _llm_stub
from ..base import JudgeResult

RUBRIC = """
Re-inspect each model-design and governance finding against the .bim
file and the spec. CONFIRM / DISMISS / ESCALATE as appropriate. Pay
special attention to bidirectional relationships — many are legitimate
on small dim tables but flagged blanket-style by the deterministic
judge.
"""


def run(spec: dict[str, Any], model_findings_path: str, governance_findings_path: str,
        bim_path: str) -> JudgeResult:
    parts: list[str] = []
    for p in (model_findings_path, governance_findings_path, bim_path):
        if Path(p).exists():
            parts.append(f"### {Path(p).name}\n" + Path(p).read_text(encoding="utf-8"))
    return _llm_stub.run_llm_judge("model_auditor", RUBRIC, "\n\n".join(parts), tier_weight=1.0)
