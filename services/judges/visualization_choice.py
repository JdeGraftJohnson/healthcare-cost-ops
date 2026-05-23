"""LLM judge — chart-type appropriateness vs data shape."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import _llm_stub
from .base import JudgeResult

RUBRIC = """
- Time series → line. Category comparison → bar. Part-of-whole ≤ 5 categories → stacked or 100% bar
  (NOT pie). Geographic → filled map. Distribution → histogram.
- Score down for: pie/donut with > 5 categories, dual-axis line charts on incommensurable scales,
  3D charts, exploded pies.
"""


def run(spec: dict[str, Any], layout_path: str) -> JudgeResult:
    text = Path(layout_path).read_text(encoding="utf-8") if Path(layout_path).exists() else "{}"
    return _llm_stub.run_llm_judge("visualization_choice", RUBRIC, text, tier_weight=1.0)
