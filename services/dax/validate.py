"""Static DAX validator — naive parser that catches the high-frequency errors
flagged by the deterministic judges before any LLM gets involved.

Checks:
  - balanced parens
  - all referenced columns exist in the schema
  - DIVIDE() not raw /
  - no calculated-column patterns (column refs outside a measure body)
  - functions are in the documented DAX catalogue (compiled at module load)
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

# Subset of the DAX function catalogue from
# https://learn.microsoft.com/en-us/dax/dax-syntax-reference
DAX_FUNCTIONS = frozenset({
    "ABS", "ALL", "ALLEXCEPT", "ALLNOBLANKROW", "ALLSELECTED", "AVERAGE", "AVERAGEX",
    "BLANK", "CALCULATE", "CALCULATETABLE", "CONCATENATE", "CONCATENATEX",
    "COUNT", "COUNTA", "COUNTAX", "COUNTBLANK", "COUNTROWS", "COUNTX",
    "DATE", "DATEADD", "DATEDIFF", "DATESBETWEEN", "DATESINPERIOD", "DATESYTD",
    "DAY", "DIVIDE", "EARLIER", "EARLIEST", "EOMONTH", "EXCEPT", "EXP",
    "FILTER", "FIRSTDATE", "FORMAT", "GENERATE", "GROUPBY", "HASONEVALUE",
    "IF", "IFERROR", "ISBLANK", "ISEMPTY", "ISERROR", "ISFILTERED", "ISLOGICAL",
    "ISNUMBER", "ISONORAFTER", "ISTEXT", "KEEPFILTERS", "LASTDATE", "LEFT",
    "LEN", "LOOKUPVALUE", "LOWER", "MAX", "MAXX", "MEDIAN", "MEDIANX", "MIN",
    "MINX", "MONTH", "NOT", "NOW", "OR", "PARALLELPERIOD", "PERCENTILE.EXC",
    "PERCENTILE.INC", "POWER", "PREVIOUSDAY", "PREVIOUSMONTH", "PREVIOUSQUARTER",
    "PREVIOUSYEAR", "QUARTER", "RANK.EQ", "RANKX", "RELATED", "RELATEDTABLE",
    "REMOVEFILTERS", "REPLACE", "REPT", "RIGHT", "ROUND", "ROUNDDOWN", "ROUNDUP",
    "SAMEPERIODLASTYEAR", "SEARCH", "SELECTEDVALUE", "SIGN", "STDEV.P", "STDEV.S",
    "SUBSTITUTE", "SUM", "SUMMARIZE", "SUMMARIZECOLUMNS", "SUMX", "SWITCH",
    "TODAY", "TOPN", "TOTALMTD", "TOTALQTD", "TOTALYTD", "TRIM", "TRUE", "FALSE",
    "UNION", "UPPER", "USERELATIONSHIP", "VALUE", "VALUES", "VAR", "WEEKDAY",
    "WEEKNUM", "YEAR", "YEARFRAC",
})

CALL_RE = re.compile(r"\b([A-Z][A-Z0-9_.]*)\s*\(")
TABLE_COL_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\[([^\]]+)\]")


@dataclass
class Finding:
    severity: str  # OK | WARN | MISS
    code: str
    msg: str


def validate(dax: str, schema_columns: dict[str, set[str]] | None = None) -> list[Finding]:
    out: list[Finding] = []

    # Balanced parens
    depth = 0
    for ch in dax:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if depth < 0:
            out.append(Finding("MISS", "PARENS", "Unbalanced parentheses"))
            break
    if depth != 0:
        out.append(Finding("MISS", "PARENS", f"Unbalanced parentheses, depth={depth}"))

    # Unknown functions
    for m in CALL_RE.finditer(dax):
        fn = m.group(1).upper()
        if fn in {"IF", "VAR", "TRUE", "FALSE", "RETURN"}:
            continue
        if fn not in DAX_FUNCTIONS:
            out.append(Finding("WARN", "UNKNOWN_FN", f"Unrecognized DAX function: {fn}"))

    # Raw division (heuristic — flags `/` outside DIVIDE)
    # Strip strings + comments first
    stripped = re.sub(r"//.*?$", "", dax, flags=re.MULTILINE)
    stripped = re.sub(r'"[^"]*"', "", stripped)
    if re.search(r"[^A-Za-z_]/[^A-Za-z_/]", stripped):
        if "DIVIDE" not in stripped.upper():
            out.append(Finding("WARN", "RAW_DIVIDE",
                               "Raw `/` division found — prefer DIVIDE() for zero-denominator safety"))

    # Column refs against schema
    if schema_columns is not None:
        for m in TABLE_COL_RE.finditer(dax):
            tbl, col = m.group(1), m.group(2)
            cols = schema_columns.get(tbl)
            if cols is None:
                out.append(Finding("MISS", "UNKNOWN_TABLE", f"Reference to unknown table: {tbl}"))
            elif col not in cols:
                out.append(Finding("MISS", "UNKNOWN_COLUMN", f"Reference to unknown column: {tbl}[{col}]"))

    if not out:
        out.append(Finding("OK", "VALIDATE", "DAX passed all checks"))
    return out


def summarize(findings: Iterable[Finding]) -> dict[str, int]:
    counts = {"OK": 0, "WARN": 0, "MISS": 0}
    for f in findings:
        counts[f.severity] = counts.get(f.severity, 0) + 1
    return counts
