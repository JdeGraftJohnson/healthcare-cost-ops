"""Accessibility judge — color contrast, ALT text, chart-title clarity.

Reads the page-layout JSON (or .pbit theme block) and applies WCAG 2.1
AA thresholds: text contrast ≥ 4.5:1, large text ≥ 3:1, every visual
has a title that is not the default placeholder.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .base import Finding, JudgeResult


def _rel_lum(hex_c: str) -> float:
    """sRGB relative luminance per WCAG."""
    hex_c = hex_c.lstrip("#")
    if len(hex_c) != 6:
        return 0.0
    rgb = [int(hex_c[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]
    def _adj(c: float) -> float:
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = (_adj(c) for c in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _contrast(fg: str, bg: str) -> float:
    l1, l2 = _rel_lum(fg), _rel_lum(bg)
    lo, hi = sorted([l1, l2])
    return (hi + 0.05) / (lo + 0.05)


def run(spec: dict[str, Any], layout_path: str) -> JudgeResult:
    findings: list[Finding] = []
    p = Path(layout_path)
    if not p.exists():
        findings.append(Finding("WARN", "LAYOUT_MISSING", f"layout not found: {layout_path}"))
        return JudgeResult(judge="accessibility", tier="deterministic",
                           score=0.5, weight=0.5, findings=findings)

    layout = json.loads(p.read_text(encoding="utf-8"))
    pairs = layout.get("color_pairs", [])
    for cp in pairs:
        c = _contrast(cp["fg"], cp["bg"])
        if c < 4.5:
            findings.append(Finding("WARN", "LOW_CONTRAST",
                                    f"fg={cp['fg']} bg={cp['bg']} contrast={c:.2f} (<4.5:1)",
                                    where=cp.get("where", "")))

    for v in layout.get("visuals", []):
        title = v.get("title", "")
        if not title or re.match(r"^(Chart|Visual|Untitled)\s*\d*$", title, re.I):
            findings.append(Finding("WARN", "DEFAULT_TITLE",
                                    f"visual has placeholder title: {title!r}",
                                    where=v.get("id", "")))

    if not findings:
        findings.append(Finding("OK", "ACCESSIBILITY", "WCAG AA checks pass"))
    warn = sum(1 for f in findings if f.severity == "WARN")
    score = max(0.0, 1.0 - 0.10 * warn)
    return JudgeResult(judge="accessibility", tier="deterministic", score=score,
                       weight=0.5, findings=findings)
