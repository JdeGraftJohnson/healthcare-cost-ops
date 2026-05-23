"""PII / PHI leakage judge.

Scans the Tabular model, measure names, and any rendered narrative for
patterns that look like patient identifiers: MRN, SSN, DOB, email, phone.
SDUD itself is aggregated and contains no PHI; this judge enforces that
no downstream artifact accidentally introduces it.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .base import Finding, JudgeResult

PII_PATTERNS = [
    (r"\b\d{3}-\d{2}-\d{4}\b",               "SSN_PATTERN",   "Possible SSN"),
    (r"\b\d{3}-\d{3}-\d{4}\b",               "PHONE_PATTERN", "Possible phone number"),
    (r"\b\d{2}/\d{2}/\d{4}\b",               "DOB_PATTERN",   "Possible DOB"),
    (r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", "EMAIL", "Possible email address"),
    (r"\bMRN[:\s]?\d{4,}\b",                 "MRN",           "Possible Medical Record Number"),
]


def run(spec: dict[str, Any], artifact_paths: list[str]) -> JudgeResult:
    findings: list[Finding] = []
    for p in artifact_paths:
        path = Path(p)
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pat, code, msg in PII_PATTERNS:
            for m in re.finditer(pat, text, re.IGNORECASE):
                findings.append(Finding("MISS", code, f"{msg} in {path.name}",
                                        where=f"{path.name}:offset={m.start()}"))
    if not findings:
        findings.append(Finding("OK", "PII_LEAK", "no PII patterns detected"))
    miss = sum(1 for f in findings if f.severity == "MISS")
    score = 0.0 if miss else 1.0
    return JudgeResult(judge="pii_leak", tier="deterministic", score=score,
                       weight=3.0, findings=findings, notes=f"miss={miss}")
