"""Build a Tabular Object Model JSON (.bim) file from a dashboard spec
plus a rendered DAX measure set.

The output conforms to the TOM 1500-compatibility JSON schema (the format
used by AS Azure, SSAS 2019+, and Power BI dataset metadata). It is the
artifact the model_design / governance / pii_leak judges read; downstream,
it can be deployed to Power BI Service via the REST API or imported into
Tabular Editor for visual inspection.

Reference: https://learn.microsoft.com/en-us/analysis-services/tom/tabular-object-model
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

from ..dax.templates import CATALOGUE, render_many
from ..judges.base import load_spec

LOG = logging.getLogger("model_build")

DAX_MEASURE_RE = re.compile(
    r"\[\s*(?P<name>[^\]]+?)\s*\]\s*:?=\s*(?P<body>.+?)(?=\n\[|\Z)",
    re.DOTALL,
)


def _parse_measures(dax_text: str) -> list[dict[str, str]]:
    """Extract measure name + body pairs from a DAX file or concatenated set."""
    measures: list[dict[str, str]] = []
    for m in DAX_MEASURE_RE.finditer(dax_text):
        name = m.group("name").strip()
        body = m.group("body").strip().rstrip("\n").strip()
        if body.endswith(",") or body.endswith("//"):
            body = body.rstrip(",/ \n")
        measures.append({"name": name, "expression": body})
    return measures


def build_bim(spec: dict[str, Any], measures_dax: str) -> dict[str, Any]:
    """Build a TOM-compatible model dict from spec + measures DAX."""
    name = spec.get("name", "dashboard")
    domain = spec.get("domain", "healthcare")

    # Fact + dim tables — schema mirrors silver Parquet output
    tables: list[dict[str, Any]] = []

    tables.append({
        "name": "fact_sdud",
        "columns": [
            {"name": "state_code",        "dataType": "string",   "sourceColumn": "state_code"},
            {"name": "ndc11",             "dataType": "string",   "sourceColumn": "ndc11"},
            {"name": "year",              "dataType": "int64",    "sourceColumn": "year"},
            {"name": "quarter",           "dataType": "int64",    "sourceColumn": "quarter"},
            {"name": "utilization_type",  "dataType": "string",   "sourceColumn": "utilization_type"},
            {"name": "suppressed",        "dataType": "boolean",  "sourceColumn": "suppressed"},
            {"name": "units",             "dataType": "double",   "sourceColumn": "units"},
            {"name": "rx_count",          "dataType": "double",   "sourceColumn": "rx_count"},
            {"name": "total_reimb",       "dataType": "double",   "sourceColumn": "total_reimb",
             "formatString": "\"$\"#,0;(\"$\"#,0)"},
            {"name": "medicaid_reimb",    "dataType": "double",   "sourceColumn": "medicaid_reimb"},
            {"name": "non_medicaid_reimb","dataType": "double",   "sourceColumn": "non_medicaid_reimb"},
        ],
        "partitions": [{
            "name": "fact_sdud-partition",
            "source": {"type": "m", "expression": [
                "let",
                "    Source = Parquet.Document(File.Contents(\"silver/fact_sdud.parquet\"))",
                "in Source"
            ]},
        }],
        "measures": _parse_measures(measures_dax),
    })

    tables.append({
        "name": "dim_state",
        "columns": [
            {"name": "state_code", "dataType": "string", "sourceColumn": "state_code", "isKey": True},
            {"name": "population", "dataType": "int64",  "sourceColumn": "population"},
        ],
    })
    tables.append({
        "name": "dim_drug",
        "columns": [
            {"name": "ndc11",          "dataType": "string", "sourceColumn": "ndc11", "isKey": True},
            {"name": "drug_name",      "dataType": "string", "sourceColumn": "drug_name"},
            {"name": "drug_class",     "dataType": "string", "sourceColumn": "drug_class"},
            {"name": "brand_generic",  "dataType": "string", "sourceColumn": "brand_generic"},
        ],
    })
    tables.append({
        "name": "dim_date",
        "isMarkedAsDateTable": True,
        "columns": [
            {"name": "date",     "dataType": "dateTime", "sourceColumn": "date", "isKey": True},
            {"name": "year",     "dataType": "int64",    "sourceColumn": "year"},
            {"name": "quarter",  "dataType": "int64",    "sourceColumn": "quarter"},
            {"name": "year_qtr", "dataType": "int64",    "sourceColumn": "year_qtr"},
        ],
    })
    tables.append({
        "name": "fact_forecast",
        "columns": [
            {"name": "state_code",    "dataType": "string", "sourceColumn": "state_code"},
            {"name": "ndc11",         "dataType": "string", "sourceColumn": "ndc11"},
            {"name": "year",          "dataType": "int64",  "sourceColumn": "year"},
            {"name": "quarter",       "dataType": "int64",  "sourceColumn": "quarter"},
            {"name": "point",         "dataType": "double", "sourceColumn": "point"},
            {"name": "lo80",          "dataType": "double", "sourceColumn": "lo80"},
            {"name": "hi80",          "dataType": "double", "sourceColumn": "hi80"},
            {"name": "method",        "dataType": "string", "sourceColumn": "method"},
            {"name": "mape_holdout",  "dataType": "double", "sourceColumn": "mape_holdout"},
        ],
    })

    relationships = [
        {"name": "fact_sdud_to_dim_state",
         "fromTable": "fact_sdud", "fromColumn": "state_code",
         "toTable": "dim_state",   "toColumn":   "state_code"},
        {"name": "fact_sdud_to_dim_drug",
         "fromTable": "fact_sdud", "fromColumn": "ndc11",
         "toTable": "dim_drug",    "toColumn":   "ndc11"},
        {"name": "fact_forecast_to_dim_state",
         "fromTable": "fact_forecast", "fromColumn": "state_code",
         "toTable": "dim_state",       "toColumn":   "state_code"},
        {"name": "fact_forecast_to_dim_drug",
         "fromTable": "fact_forecast", "fromColumn": "ndc11",
         "toTable": "dim_drug",        "toColumn":   "ndc11"},
    ]

    roles: list[dict[str, Any]] = []
    rls = spec.get("compliance", {}).get("rls")
    if rls:
        roles.append({
            "name": rls,
            "modelPermission": "read",
            "tablePermissions": [{
                "name": "fact_sdud",
                "filterExpression": f"fact_sdud[{rls}] = USERNAME()",
            }],
        })

    # Source documentation as model annotations — governance judge looks here
    annotations = []
    for s in spec.get("data_sources", []):
        annotations.append({
            "name": f"source.{s.get('name')}",
            "value": json.dumps({k: v for k, v in s.items() if k != "name"}),
        })
    annotations.append({"name": "domain", "value": domain})
    annotations.append({"name": "harness", "value": "healthcare-cost-ops"})

    return {
        "name": name,
        "compatibilityLevel": 1500,
        "model": {
            "culture": "en-US",
            "dataAccessOptions": {"legacyRedirects": True},
            "defaultPowerBIDataSourceVersion": "powerBI_V3",
            "sourceQueryCulture": "en-US",
            "tables": tables,
            "relationships": relationships,
            "roles": roles,
            "annotations": annotations,
        },
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--spec", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--measures", help="optional pre-rendered DAX file; default = render all spec.templates")
    p.add_argument("--verbose", "-v", action="store_true")
    a = p.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if a.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-5s %(name)s %(message)s",
    )
    spec = load_spec(a.spec)

    if a.measures:
        measures_dax = Path(a.measures).read_text(encoding="utf-8")
    else:
        rendered = render_many(spec.get("templates", []))
        measures_dax = "\n\n".join(r.dax for r in rendered)
        out_dax = Path(a.out).with_suffix(".dax")
        out_dax.parent.mkdir(parents=True, exist_ok=True)
        out_dax.write_text(measures_dax, encoding="utf-8")
        LOG.info("rendered %d templates → %s", len(rendered), out_dax)

    bim = build_bim(spec, measures_dax)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(bim, indent=2), encoding="utf-8")
    LOG.info("wrote %s (tables=%d, measures=%d, roles=%d)",
             out, len(bim["model"]["tables"]),
             sum(len(t.get("measures", [])) for t in bim["model"]["tables"]),
             len(bim["model"]["roles"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
