"""Narrative writer — emits the executive captions per page.

In offline mode, produces a deterministic narrative skeleton from the
measure names + spec. With ANTHROPIC_API_KEY set, asks Claude to fill
the skeleton with a 3-5 sentence executive paragraph per page.

The narrative_auditor checks the result against the actual measure set
to catch hallucinated metrics (see services/judges/auditors/narrative_auditor.py).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import textwrap
from pathlib import Path
from typing import Any

from ..judges.base import load_spec

LOG = logging.getLogger("narrative_build")


def offline_skeleton(spec: dict[str, Any], layout: dict[str, Any]) -> dict[str, Any]:
    narrative: dict[str, str] = {}
    for page in layout.get("pages", []):
        n = page["name"]
        bullets = [v["title"] for v in page["visuals"][:4]]
        narrative[n] = (
            f"[PLACEHOLDER NARRATIVE FOR PAGE: {n}] "
            f"Visuals on this page: {', '.join(bullets)}. "
            f"Set ANTHROPIC_API_KEY to generate the executive caption."
        )
    return {"pages": narrative, "mode": "offline_skeleton"}


def llm_narrative(spec: dict[str, Any], layout: dict[str, Any], measures_dax: str) -> dict[str, Any]:
    from anthropic import Anthropic
    client = Anthropic()
    out: dict[str, str] = {}
    system = textwrap.dedent("""
        You are writing executive captions for a Power BI dashboard.
        Constraints:
          - 3-5 sentences per page.
          - Lead with the headline number/measure, then trend, then forward outlook.
          - Reference ONLY measures that exist in the supplied DAX file.
          - No jargon, no bullet lists, no headings.
    """).strip()
    for page in layout.get("pages", []):
        user = (
            f"Page: {page['name']}\n"
            f"Audience: {spec.get('audience')}\n"
            f"Visuals: {[v['title'] for v in page['visuals']]}\n\n"
            f"DAX measures available:\n```\n{measures_dax[:4000]}\n```"
        )
        resp = client.messages.create(
            model="claude-opus-4-7", max_tokens=512, system=system,
            messages=[{"role": "user", "content": user}],
        )
        out[page["name"]] = resp.content[0].text.strip()
    return {"pages": out, "mode": "claude"}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--spec", required=True)
    p.add_argument("--layout", required=True)
    p.add_argument("--measures", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-5s %(name)s %(message)s")

    spec = load_spec(a.spec)
    layout = json.loads(Path(a.layout).read_text(encoding="utf-8"))
    measures = Path(a.measures).read_text(encoding="utf-8") if Path(a.measures).exists() else ""

    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            doc = llm_narrative(spec, layout, measures)
        except Exception as e:
            LOG.warning("llm narrative failed (%s); falling back to skeleton", e)
            doc = offline_skeleton(spec, layout)
    else:
        doc = offline_skeleton(spec, layout)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    LOG.info("wrote %s (mode=%s, pages=%d)", out, doc["mode"], len(doc["pages"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
