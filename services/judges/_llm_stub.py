"""Shared stub for the five LLM judges.

When ANTHROPIC_API_KEY is set, the judge sends a tightly-scoped prompt to
Claude and parses the structured response. When unset (offline build),
the judge returns a neutral 0.75 placeholder so the harness still runs
end-to-end. Each LLM judge module wraps this with its own rubric.
"""
from __future__ import annotations

import logging
import os
import textwrap
from typing import Any

from .base import Finding, JudgeResult

LOG = logging.getLogger("llm_judge")

PROMPT_FRAME = textwrap.dedent("""
    You are evaluating a Power BI dashboard artifact against a fixed
    rubric. Score from 0.00 to 1.00. Output strictly:

    <score>0.NN</score>
    <findings>
      <finding severity="WARN|MISS|OK" code="SHORT">message</finding>
      ...
    </findings>
    <notes>one-line summary</notes>
""").strip()


def run_llm_judge(judge_name: str, rubric: str, context: str,
                  tier_weight: float = 1.5) -> JudgeResult:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        LOG.warning("%s: ANTHROPIC_API_KEY unset; emitting neutral placeholder", judge_name)
        return JudgeResult(
            judge=judge_name, tier="llm", score=0.75, weight=tier_weight,
            findings=[Finding("OK", "PLACEHOLDER",
                              "LLM judge skipped (no API key); set ANTHROPIC_API_KEY to enable")],
            notes="placeholder run",
        )

    try:
        from anthropic import Anthropic
    except ImportError:
        return JudgeResult(
            judge=judge_name, tier="llm", score=0.75, weight=tier_weight,
            findings=[Finding("WARN", "NO_SDK", "anthropic SDK not installed")],
            notes="anthropic SDK missing",
        )

    client = Anthropic()
    resp = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=1024,
        system=PROMPT_FRAME,
        messages=[{"role": "user", "content": f"## Rubric\n{rubric}\n\n## Context\n{context}"}],
    )
    text = resp.content[0].text
    return _parse_xml(judge_name, text, tier_weight)


def _parse_xml(judge_name: str, text: str, weight: float) -> JudgeResult:
    import re
    m_score = re.search(r"<score>\s*([0-9.]+)\s*</score>", text)
    score = float(m_score.group(1)) if m_score else 0.5
    findings: list[Finding] = []
    for m in re.finditer(
        r'<finding\s+severity="(OK|WARN|MISS)"\s+code="([^"]+)">(.+?)</finding>',
        text, re.S,
    ):
        findings.append(Finding(m.group(1), m.group(2), m.group(3).strip()))
    m_notes = re.search(r"<notes>(.+?)</notes>", text, re.S)
    notes = m_notes.group(1).strip() if m_notes else ""
    return JudgeResult(judge=judge_name, tier="llm", score=score, weight=weight,
                       findings=findings or [Finding("OK", "LLM_OK", "no findings emitted")],
                       notes=notes)
