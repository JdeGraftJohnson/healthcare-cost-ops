"""LLM judge — healthcare-domain literacy.

Checks that the dashboard uses the right domain vocabulary for its
declared `spec.domain`. For healthcare: NDC (National Drug Code), HCPCS,
ICD-10, HEDIS, CMS, T-MSIS. Parameterized so non-healthcare runs swap
in the right vocabulary list.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from . import _llm_stub
from .base import JudgeResult

DOMAIN_VOCAB = {
    "healthcare": ["NDC", "HCPCS", "ICD-10", "HEDIS", "PMPM", "T-MSIS", "CMS", "FDA Orange Book"],
    "finance":    ["CUSIP", "ISIN", "PnL", "EOD", "T+1", "FINRA", "Reg NMS"],
    "retail":     ["SKU", "UPC", "MSRP", "shrinkage", "comp-store"],
}


def run(spec: dict[str, Any], narrative_path: str) -> JudgeResult:
    domain = spec.get("domain", "healthcare")
    vocab = DOMAIN_VOCAB.get(domain, [])
    rubric = (
        f"For the '{domain}' domain, the following terms should appear at least once where "
        f"contextually appropriate: {', '.join(vocab)}. Penalize buzzword overuse and "
        f"wrong-domain vocabulary."
    )
    text = Path(narrative_path).read_text(encoding="utf-8") if Path(narrative_path).exists() else ""
    return _llm_stub.run_llm_judge("domain_relevance", rubric, text, tier_weight=1.0)
