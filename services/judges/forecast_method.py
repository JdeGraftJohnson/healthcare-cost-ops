"""LLM judge — forecast methodology + uncertainty disclosure."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from . import _llm_stub
from .base import JudgeResult

RUBRIC = """
- Forecast methodology is explicitly named (SARIMA, Prophet, ensemble, naive).
- Holdout-window MAPE is reported.
- Confidence band (80% and/or 95%) is shown on every forecast visual.
- Native Power BI exponential-smoothing forecast is NOT used (insufficient documentation).
"""


def run(spec: dict[str, Any], narrative_path: str) -> JudgeResult:
    text = Path(narrative_path).read_text(encoding="utf-8") if Path(narrative_path).exists() else ""
    return _llm_stub.run_llm_judge("forecast_method", RUBRIC, text, tier_weight=1.5)
