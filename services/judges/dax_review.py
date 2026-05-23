"""LLM judge — idiomatic DAX review (qualitative, beyond syntax)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from . import _llm_stub
from .base import JudgeResult

RUBRIC = """
- VAR / RETURN used for any expression evaluated more than once.
- DIVIDE() not raw /.
- Time-intelligence uses SAMEPERIODLASTYEAR / DATESYTD / DATESINPERIOD with a marked date table.
- KEEPFILTERS used when intent is to intersect, not replace, filter context.
- Measure names are PascalCase or 'Friendly Name' with brackets — never 'fact_table[col]'.
"""


def run(spec: dict[str, Any], measures_path: str) -> JudgeResult:
    text = Path(measures_path).read_text(encoding="utf-8") if Path(measures_path).exists() else ""
    return _llm_stub.run_llm_judge("dax_review", RUBRIC, text, tier_weight=1.5)
