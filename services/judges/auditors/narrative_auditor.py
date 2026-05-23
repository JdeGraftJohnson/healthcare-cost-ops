"""Paired auditor for business_narrative.

Verifies that every number cited in the narrative is actually derivable
from the measure set — catches hallucinated metrics.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .. import _llm_stub
from ..base import JudgeResult

RUBRIC = """
For every numeric claim in the narrative, find the supporting measure
and value. If the claim cannot be supported, flag MISS with code
'HALLUCINATED_METRIC'. Cross-reference is the authoritative check — the
narrative may sound right but be subtly wrong on which measure it cites.
"""


def run(spec: dict[str, Any], narrative_path: str, measures_path: str) -> JudgeResult:
    parts: list[str] = []
    if Path(narrative_path).exists():
        parts.append("### Narrative\n" + Path(narrative_path).read_text(encoding="utf-8"))
    if Path(measures_path).exists():
        parts.append("### Measures\n" + Path(measures_path).read_text(encoding="utf-8"))
    return _llm_stub.run_llm_judge("narrative_auditor", RUBRIC, "\n\n".join(parts), tier_weight=1.0)
