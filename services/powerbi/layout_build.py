"""Build a page-layout JSON describing the dashboard's visual structure.

The layout JSON is consumed by:
  - the accessibility judge (color contrast + visual titles)
  - the visualization_choice LLM judge (chart-type vs data shape)
  - the optional .pbit packager (templates/pbit_assembler.py — TODO)

Each page maps one or more template ids to a visual block on a canvas grid.
The format intentionally stays JSON-native rather than Power BI's binary
`Report/Layout` so the harness can inspect it in plain text.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from ..dax.templates import CATALOGUE
from ..judges.base import load_spec

LOG = logging.getLogger("layout_build")

# Default visual choice per template — drives the visualization_choice judge.
VISUAL_BY_TEMPLATE = {
    "01": [("card", "Total Reimbursement"), ("card", "Total Prescriptions"),
           ("card", "Cost per Rx"), ("card", "YoY Growth %"),
           ("line", "Total Reimbursement over Time")],
    "02": [("bar", "Top N Drug Classes by Spend")],
    "03": [("clustered_column", "Reimb vs Reimb PY by Quarter")],
    "04": [("filled_map", "Reimb per Capita by State")],
    "05": [("card", "PMPM"), ("card", "PMPY"), ("line", "PMPM Trend")],
    "06": [("area_with_band", "Actuals + 12-Month Forecast")],
    "07": [("combo_bar_line", "Pareto: Spend + Cumulative %")],
    "08": [("stacked_bar", "Generic vs Brand Share"),
           ("table",        "Substitution Opportunity by Drug Class")],
    "09": [("table", "Outlier Drug Classes (this period)")],
    "10": [("card", "Last Refresh"), ("card", "Refresh Success Rate 30d"),
           ("card", "Judge Composite"), ("kpi", "Judge Verdict")],
}

# Cyfi-neutral palette — high contrast against dark canvas.
COLOR_PAIRS = [
    {"fg": "#FFFFFF", "bg": "#0B1020", "where": "page-background-text"},   # 18.5:1
    {"fg": "#7DD3FC", "bg": "#0B1020", "where": "primary-accent"},          # 11.3:1
    {"fg": "#A7F3D0", "bg": "#0B1020", "where": "success-accent"},          # 13.6:1
    {"fg": "#FCA5A5", "bg": "#0B1020", "where": "warning-accent"},          # 8.4:1
]


def build_layout(spec: dict[str, Any]) -> dict[str, Any]:
    templates = spec.get("templates", [])
    pages: list[dict[str, Any]] = []

    # Page 1: Executive Overview — always template 01
    if "01" in templates:
        pages.append({
            "name": "Executive Overview",
            "ordinal": 1,
            "visuals": [
                {"id": f"p1-v{i+1}", "type": kind, "title": title, "grid": {"row": (i // 4) + 1, "col": (i % 4) + 1}}
                for i, (kind, title) in enumerate(VISUAL_BY_TEMPLATE["01"])
            ],
        })

    # Page 2: Cost concentration — 02, 04, 07
    cost_visuals = []
    for tid in ("02", "04", "07"):
        if tid in templates:
            cost_visuals.extend(VISUAL_BY_TEMPLATE.get(tid, []))
    if cost_visuals:
        pages.append({
            "name": "Cost Concentration",
            "ordinal": 2,
            "visuals": [
                {"id": f"p2-v{i+1}", "type": kind, "title": title, "grid": {"row": (i // 2) + 1, "col": (i % 2) + 1}}
                for i, (kind, title) in enumerate(cost_visuals)
            ],
        })

    # Page 3: Trend + Forecast — 03, 06
    trend_visuals = []
    for tid in ("03", "06"):
        if tid in templates:
            trend_visuals.extend(VISUAL_BY_TEMPLATE.get(tid, []))
    if trend_visuals:
        pages.append({
            "name": "Trend and Forecast",
            "ordinal": 3,
            "visuals": [
                {"id": f"p3-v{i+1}", "type": kind, "title": title, "grid": {"row": i + 1, "col": 1}}
                for i, (kind, title) in enumerate(trend_visuals)
            ],
        })

    # Page 4: Opportunity — 08, 09
    opp_visuals = []
    for tid in ("08", "09"):
        if tid in templates:
            opp_visuals.extend(VISUAL_BY_TEMPLATE.get(tid, []))
    if opp_visuals:
        pages.append({
            "name": "Substitution and Outliers",
            "ordinal": 4,
            "visuals": [
                {"id": f"p4-v{i+1}", "type": kind, "title": title, "grid": {"row": i + 1, "col": 1}}
                for i, (kind, title) in enumerate(opp_visuals)
            ],
        })

    # Page 5: Quality Scorecard — 10
    if "10" in templates:
        pages.append({
            "name": "Quality Scorecard",
            "ordinal": 5,
            "visuals": [
                {"id": f"p5-v{i+1}", "type": kind, "title": title, "grid": {"row": (i // 4) + 1, "col": (i % 4) + 1}}
                for i, (kind, title) in enumerate(VISUAL_BY_TEMPLATE["10"])
            ],
        })

    return {
        "name": spec.get("name"),
        "audience": spec.get("audience"),
        "pages": pages,
        "color_pairs": COLOR_PAIRS,
        "visuals": [v for p in pages for v in p["visuals"]],  # flat for the judge
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--spec", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-5s %(name)s %(message)s")
    spec = load_spec(a.spec)
    layout = build_layout(spec)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(layout, indent=2), encoding="utf-8")
    LOG.info("wrote %s (pages=%d, visuals=%d)", out, len(layout["pages"]), len(layout["visuals"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
