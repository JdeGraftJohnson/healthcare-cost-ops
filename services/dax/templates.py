"""DAX template library — loads .dax files from templates/ and parameterizes
them against a `dashboard_spec.yml`.

Each template is a thin file with named measures. This module loads them,
applies the spec's column-name mapping, and emits the resulting DAX string
plus metadata used by the judge harness (`services/judges/template_fit.py`).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

TEMPLATE_DIR = Path(__file__).resolve().parent.parent.parent / "templates"

# Public catalogue — id, label, business question, measures emitted
CATALOGUE: list[dict] = [
    {"id": "01", "label": "Executive Overview", "question": "Headline KPIs and trend",
     "measures": ["Total Reimbursement", "Total Prescriptions", "Cost per Rx",
                  "Total Reimbursement YTD", "Total Reimbursement PY", "YoY Growth %"]},
    {"id": "02", "label": "Top N + Other", "question": "Which N categories drive cost",
     "measures": ["N Selected", "Spend Top N", "Spend Other", "Top N Share %"]},
    {"id": "03", "label": "YoY Growth", "question": "Same period last year delta",
     "measures": ["Reimb PY", "Reimb YoY $", "Reimb YoY %", "Reimb YoY Direction"]},
    {"id": "04", "label": "State Choropleth", "question": "Where is cost concentrated",
     "measures": ["Reimb (Suppressed-Aware)", "Reimb per Capita", "Suppression Rate %"]},
    {"id": "05", "label": "PMPM", "question": "Normalized cost per member month",
     "measures": ["Member Months", "PMPM", "PMPY", "PMPM YoY $"]},
    {"id": "06", "label": "Forecast Band", "question": "12-month forward projection",
     "measures": ["Forecast Point", "Forecast Lo80", "Forecast Hi80",
                  "Reimb Actual or Forecast", "Forecast MAPE %"]},
    {"id": "07", "label": "Pareto 80/20", "question": "Cumulative share",
     "measures": ["Drug Rank", "Cumulative Spend", "Cumulative Share %",
                  "Above 80% Threshold"]},
    {"id": "08", "label": "Brand vs Generic", "question": "Substitution opportunity",
     "measures": ["Generic Reimb", "Brand Reimb", "Generic Share %",
                  "Cross-Class Median Generic Share", "Substitution Opportunity Flag"]},
    {"id": "09", "label": "Outlier Flag", "question": "ATR-style break detection",
     "measures": ["Reimb 12M StdDev", "Reimb Δ from Prev", "Outlier Flag"]},
    {"id": "10", "label": "Quality Scorecard", "question": "Self-monitoring page",
     "measures": ["Last Refresh", "Refresh Success Rate 30d",
                  "Judge Composite", "Judge Verdict", "RLS Roles Present"]},
]


@dataclass
class RenderedTemplate:
    id: str
    label: str
    measures: list[str]
    dax: str


def _load(template_id: str) -> str:
    matches = list(TEMPLATE_DIR.glob(f"{template_id}_*.dax"))
    if not matches:
        raise FileNotFoundError(f"no template file matches id={template_id} in {TEMPLATE_DIR}")
    return matches[0].read_text(encoding="utf-8")


def render(template_id: str, column_map: Mapping[str, str] | None = None) -> RenderedTemplate:
    """Load a template, apply column-name substitutions from `column_map`."""
    raw = _load(template_id)
    if column_map:
        for src, dst in column_map.items():
            raw = re.sub(rf"\b{re.escape(src)}\b", dst, raw)
    meta = next(c for c in CATALOGUE if c["id"] == template_id)
    return RenderedTemplate(
        id=template_id,
        label=meta["label"],
        measures=list(meta["measures"]),
        dax=raw,
    )


def render_many(template_ids: Iterable[str], column_map: Mapping[str, str] | None = None) -> list[RenderedTemplate]:
    return [render(tid, column_map) for tid in template_ids]
