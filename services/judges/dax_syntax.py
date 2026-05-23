"""DAX syntax judge — parses every generated measure with the static
validator in :mod:`services.dax.validate`. Promotes UNKNOWN_FN /
UNKNOWN_COLUMN findings to [MISS] severity.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ..dax.validate import validate
from .base import Finding, JudgeResult


def run(spec: dict[str, Any], measures_dax: str, schema_columns: dict[str, set[str]] | None = None) -> JudgeResult:
    raw_findings = validate(measures_dax, schema_columns)
    findings = [Finding(f.severity, f.code, f.msg) for f in raw_findings]
    miss = sum(1 for f in findings if f.severity == "MISS")
    warn = sum(1 for f in findings if f.severity == "WARN")
    n = max(len(findings), 1)
    score = max(0.0, 1.0 - 0.30 * miss / n - 0.10 * warn / n)
    return JudgeResult(
        judge="dax_syntax", tier="deterministic", score=score, weight=2.0,
        findings=findings,
        notes=f"miss={miss} warn={warn} ok={n - miss - warn}",
    )


def cli(spec_path: str, measures_path: str, out_path: str) -> None:
    from .base import load_spec
    spec = load_spec(spec_path)
    measures = Path(measures_path).read_text(encoding="utf-8")
    r = run(spec, measures)
    r.save(Path(out_path))


if __name__ == "__main__":
    import sys
    cli(sys.argv[1], sys.argv[2], sys.argv[3])
