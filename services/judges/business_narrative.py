"""LLM judge — executive narrative quality on a single page."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from . import _llm_stub
from .base import JudgeResult

RUBRIC = """
- The narrative must lead with the headline number, then trend, then forward outlook.
- Every number stated must be in the underlying measure set (no hallucinated metrics).
- Tone is executive: 3-5 sentences, no jargon.
- Action-oriented closing sentence.
"""


def run(spec: dict[str, Any], narrative_path: str) -> JudgeResult:
    text = Path(narrative_path).read_text(encoding="utf-8") if Path(narrative_path).exists() else ""
    return _llm_stub.run_llm_judge("business_narrative", RUBRIC, text, tier_weight=1.5)
