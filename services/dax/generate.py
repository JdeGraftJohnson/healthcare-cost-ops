"""Claude → DAX generator.

Given a `dashboard_spec.yml` business question that doesn't match a template
catalogue entry, ask Claude to produce a DAX measure following the same
shape as the templates. The output is then handed to `validate.py`.

This module is a thin client over the existing Claude SDK setup used by
proposal-ops; it doesn't ship its own model client. The harness either uses
`anthropic.Anthropic()` if `ANTHROPIC_API_KEY` is set or falls back to the
template library if not (so deterministic judges still work offline).
"""
from __future__ import annotations

import logging
import os
import textwrap

LOG = logging.getLogger("dax_generate")

SYSTEM_PROMPT = textwrap.dedent("""
    You are a Power BI DAX expert. Output ONE measure definition per request.

    Hard constraints:
    - Use only documented DAX functions from
      https://learn.microsoft.com/en-us/dax/dax-syntax-reference
    - No calculated columns; measures only.
    - Always wrap division in DIVIDE() to handle zero denominators.
    - Reference the marked Date dimension (dim_date) for time intelligence.
    - Never use FILTER(ALL(...)) unless explicitly clearing slicer context
      for a denominator — and add a one-line comment explaining why.
    - Output format: a single triple-backtick DAX block, no prose around it.
""").strip()


def generate_measure(question: str, schema_yaml: str) -> str:
    """Return a DAX measure as raw text. Falls back to a stub if no API key."""
    try:
        from anthropic import Anthropic
    except ImportError:
        LOG.warning("anthropic SDK not installed; returning placeholder DAX")
        return _placeholder(question)

    if not os.environ.get("ANTHROPIC_API_KEY"):
        LOG.warning("ANTHROPIC_API_KEY unset; returning placeholder DAX")
        return _placeholder(question)

    client = Anthropic()
    resp = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": f"Schema:\n```yaml\n{schema_yaml}\n```\n\nBusiness question:\n{question}",
        }],
    )
    return resp.content[0].text


def _placeholder(question: str) -> str:
    return f"// PLACEHOLDER for: {question}\n// (set ANTHROPIC_API_KEY to generate)\n[Placeholder Measure] := BLANK ()\n"
