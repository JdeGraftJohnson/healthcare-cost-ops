"""Top-level orchestration: --plan emits an audit_plan.json that lists each
judge to run; --merge consumes per-judge JSON outputs and produces audit.md
+ composite_scorecard.md.

The judge-orchestrator Claude subagent (see docs/AGENT_PIPELINE.md) reads
audit_plan.json, fans out the deterministic + LLM + auditor judges, and
writes their results back to disk. This Python entrypoint never blocks on
input() and never spawns Claude itself — same contract as
feedback_auto_pipeline_orchestrator.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .base import JudgeResult, banded_verdict, load_spec

LOG = logging.getLogger("audit_runner")

DETERMINISTIC = [
    "dax_syntax", "dax_perf", "model_design", "pii_leak",
    "accessibility", "governance", "refresh_health", "template_fit",
]
LLM = [
    "business_narrative", "visualization_choice",
    "forecast_method", "dax_review", "domain_relevance",
]
AUDITORS = ["dax_auditor", "model_auditor", "narrative_auditor"]

WEIGHTS = {
    # deterministic
    "dax_syntax": 2.0, "dax_perf": 1.0, "model_design": 1.5, "pii_leak": 3.0,
    "accessibility": 0.5, "governance": 1.5, "refresh_health": 0.5,
    "template_fit": 1.0,
    # llm
    "business_narrative": 1.5, "visualization_choice": 1.0,
    "forecast_method": 1.5, "dax_review": 1.5, "domain_relevance": 1.0,
    # auditors
    "dax_auditor": 1.0, "model_auditor": 1.0, "narrative_auditor": 1.0,
}


def plan(spec_path: str, out_dir: Path) -> Path:
    spec = load_spec(spec_path)
    plan_obj = {
        "spec": spec_path,
        "run_id": spec.get("name", "unnamed"),
        "tiers": {
            "deterministic": [{"judge": j, "module": f"services.judges.{j}"} for j in DETERMINISTIC],
            "llm":           [{"judge": j, "module": f"services.judges.{j}"} for j in LLM],
            "auditor":       [{"judge": j, "module": f"services.judges.auditors.{j}"} for j in AUDITORS],
        },
        "weights": WEIGHTS,
    }
    out = out_dir / "judges" / "audit_plan.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(plan_obj, indent=2), encoding="utf-8")
    LOG.info("wrote plan → %s", out)
    return out


def merge(out_dir: Path) -> tuple[Path, Path]:
    judge_dir = out_dir / "judges"
    results: list[JudgeResult] = []
    for f in sorted(judge_dir.glob("*.json")):
        if f.name == "audit_plan.json":
            continue
        try:
            obj = json.loads(f.read_text())
            results.append(JudgeResult(
                judge=obj["judge"], tier=obj["tier"],
                score=float(obj["score"]), weight=float(obj.get("weight", 1.0)),
                findings=[type("F", (), x) for x in obj.get("findings", [])],
                notes=obj.get("notes", ""),
            ))
        except Exception as e:
            LOG.warning("skipping %s: %s", f, e)

    # Composite
    total_w = sum(r.weight for r in results) or 1.0
    composite = sum(r.score * r.weight for r in results) / total_w
    has_miss = any(getattr(fi, "severity", "OK") == "MISS" for r in results for fi in r.findings)
    verdict = banded_verdict(composite, has_miss)

    # audit.md — severity-banded findings
    md = ["# AUDIT.md", "", f"_run id_: `{out_dir.name}`", ""]
    for sev in ("MISS", "WARN", "OK"):
        md.append(f"## [{sev}]")
        for r in results:
            for fi in r.findings:
                if getattr(fi, "severity", "OK") == sev:
                    md.append(f"- **{r.judge}** · `{getattr(fi, 'code', '')}` — {getattr(fi, 'msg', '')}")
        md.append("")
    audit_md = judge_dir / "audit.md"
    audit_md.write_text("\n".join(md), encoding="utf-8")

    # composite_scorecard.md
    sc = ["# COMPOSITE_SCORECARD.md", "",
          f"**Verdict: {verdict}** · composite = `{composite:.3f}`",
          "", "| Judge | Tier | Score | Weight | Weighted |",
          "| - | - | -: | -: | -: |"]
    for r in results:
        sc.append(f"| {r.judge} | {r.tier} | {r.score:.2f} | {r.weight:.1f} | {r.score * r.weight:.2f} |")
    score_md = judge_dir / "composite_scorecard.md"
    score_md.write_text("\n".join(sc), encoding="utf-8")

    LOG.info("merged %d results → %s + %s (verdict=%s)", len(results), audit_md, score_md, verdict)
    return audit_md, score_md


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--plan", action="store_true")
    p.add_argument("--merge", action="store_true")
    p.add_argument("--spec", help="dashboard_spec.yml (required for --plan)")
    p.add_argument("--out", default="out", help="run output directory")
    p.add_argument("--verbose", "-v", action="store_true")
    a = p.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if a.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-5s %(name)s %(message)s",
    )

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    if a.plan:
        if not a.spec:
            print("--plan requires --spec", file=sys.stderr); return 2
        plan(a.spec, out)
    if a.merge:
        merge(out)
    if not (a.plan or a.merge):
        print("pass --plan and/or --merge", file=sys.stderr); return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
