"""Shared types + loaders for the dashboard-ops judge harness.

Mirrors the proposal-ops `base.py` contract: every judge returns a
JudgeResult; the orchestrator merges them by severity and weight.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Finding:
    severity: str   # "OK" | "WARN" | "MISS"
    code: str       # short stable identifier (e.g., "DAX_PARSE")
    msg: str
    where: str = ""  # file:line or measure name


@dataclass
class JudgeResult:
    judge: str
    tier: str       # "deterministic" | "llm" | "auditor"
    score: float    # 0.0-1.0
    weight: float = 1.0
    findings: list[Finding] = field(default_factory=list)
    notes: str = ""

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        return d

    def save(self, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(self.to_json(), indent=2), encoding="utf-8")


def load_spec(path: str | Path) -> dict[str, Any]:
    import yaml
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def banded_verdict(composite: float, has_miss: bool) -> str:
    if has_miss or composite < 0.70:
        return "Re-work"
    if composite < 0.85:
        return "Tighten"
    return "Ship"
