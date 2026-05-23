# Next-session plan — Clinical RAG Evaluation Harness

Self-contained brief for a fresh Claude Code session. Builds the lead-headline
portfolio artifact for the **ML/AI Engineering** roles that career-ops has
surfaced. Forecasting (`services/forecast/`) is now the supporting artifact;
this is the lead.

## Why this and not more forecasting

JD vocabulary histogram across all 502 career-ops reports
(`/Users/john/career-ops-main/reports/`):

| Phrase | Mentions across 502 reports |
| --- | --- |
| RAG / retrieval-augmented | 1,223 |
| MCP server | 693 |
| inspect_petri | 188 |
| LangSmith | 138 |
| LangChain | 115 |
| drift detection | 97 |
| model monitoring | 59 |
| fine-tuning | 42 |
| LLM evaluation | 11 |
| Pinecone / Weaviate / LlamaIndex / Langfuse / MLflow / WandB | 5–10 each |
| **SARIMA / Prophet / LightGBM / N-BEATS / TFT** | **0** |

435 of 502 reports mention some flavor of RAG / retrieval / vector DB /
LLM-evaluation. That is the lead-archetype this project targets. Dominant
ask is "AI Platform / LLMOps — Clinical AI" (Cohere Health, Maven, Onsera,
Causaly, Flatiron, Novo Nordisk, GSK, Gilead).

## Project: `clinical-rag-eval`

A **PyTorch-shipped, MCP-server-wrapped, LangSmith-traced, inspect_petri-
graded** clinical RAG evaluation harness. Runs a fixed evaluation suite
against the user's existing clinical-rag corpus (DOI-cited biomedical
evidence) and emits a severity-banded evaluation report.

Mirrors the operator's existing `proposal-ops-judges` architecture
(deterministic + LLM-as-judge + paired auditor scalable-oversight pattern)
but applied to clinical RAG rather than government proposals — a
1:1 transfer of the operator's strongest existing pattern into the
JD-target domain.

## Target repo

- **GitHub:** `JdeGraftJohnson/clinical-rag-eval`
- **Visibility:** public
- **License:** MIT (code), corpus stays as a small sample manifest of 50
  public PubMed IDs; no PHI, no licensed corpus shipped
- **Status badge:** `Live` (matches the existing clinical-rag card)

## Layout

```
clinical-rag-eval/
  README.md
  pyproject.toml
  configs/
    eval_suite_v1.yml          # questions × expected DOIs × must-cite × refusal cases
    retrievers.yml             # embedding model + chunker config per retriever
    judges.yml                 # which deterministic + LLM judges to run
  corpus/
    sample_manifest.json       # 50 PubMed IDs (public)
    ingest.py                  # PubMed E-utils + Crossref + Semantic Scholar pull
  src/clinical_rag_eval/
    retrieval/
      embed.py                 # sentence-transformers (PyTorch) + BGE-large
      hybrid.py                # dense + BM25 + Cohere rerank
      graph_walk.py            # MeSH knowledge-graph fallback when ANN thin
    generation/
      answer.py                # cite-or-refuse system prompt
      refusal_gate.py          # evidence-thin detector
    judges/
      deterministic/
        citation_coverage.py   # every claim has a DOI
        doi_resolves.py        # Crossref round-trip
        refusal_consistency.py # refused-when-thin invariant
        pii_scrubber.py        # zero PHI leakage in answers
        format_compliance.py   # required answer-section structure
      llm/
        faithfulness.py        # Anthropic judge: answer ⊆ retrieved evidence
        relevance.py           # answer relevance to question
        clinical_safety.py     # contraindication / dosage / red-flag check
        evidence_quality.py    # GRADE-style strength rating
      auditors/                # paired LLM auditor for each deterministic judge
        citation_audit.py
        refusal_audit.py
        safety_audit.py
    drift/
      track_residuals.py       # per-question score deltas vs baseline
      psi_ks.py                # reuse pattern from healthcare-cost-ops/services/forecast/monitor.py
    mcp/
      server.py                # FastMCP server exposing: list_questions, run_eval, get_report
      tools.py
    tracking/
      langsmith_hook.py        # trace every run; emit Inspect-format JSON
      mlflow_hook.py           # mirror metrics into MLflow
    cli.py                     # `clinical-rag-eval run --suite eval_suite_v1.yml`
  tests/
    test_judges_smoke.py
    test_retrieval_smoke.py
    test_mcp_server.py
  examples/
    sample_run/
      input.json
      audit_report.md
      audit_record.json
      drift_report.json
  docs/
    ARCHITECTURE.md
    JUDGE_CATALOG.md           # each judge + scalable-oversight justification
    INSPECT_AI_INTEGRATION.md
    LANGSMITH_INTEGRATION.md
    MCP_USAGE.md
    LIMITATIONS.md
```

## Mandatory JD-vocabulary surfaces (verbatim in README + code)

Each row maps one or more JD-report phrases to a concrete artifact.

| JD phrase | Concrete artifact |
| --- | --- |
| **PyTorch** | `retrieval/embed.py` uses `sentence-transformers` (PyTorch backend) for BGE-large embedding. README names PyTorch explicitly. |
| **RAG / retrieval-augmented** | `src/clinical_rag_eval/retrieval/` (hybrid dense+BM25+rerank) + `generation/answer.py` (cite-or-refuse). README headline. |
| **MCP server** | `mcp/server.py` — FastMCP server exposing `list_questions / run_eval / get_report` tools. Documented in `docs/MCP_USAGE.md`. |
| **LangSmith** | `tracking/langsmith_hook.py` — every retrieval + generation + judge call traced with structured tags. |
| **LangChain** | Retrieval graph composed as LangChain `Runnable`s (light usage, easy to lift if it later weighs the project down). |
| **LlamaIndex** | Optional secondary retriever in `retrieval/hybrid.py` — switch via `retrievers.yml`. |
| **Langfuse** | Optional alternative trace sink behind a feature flag. |
| **inspect_petri** | `tracking/langsmith_hook.py` emits Inspect-format JSON so `inspect_petri eval --task clinical_rag_eval` runs the same suite. |
| **drift detection / model monitoring** | `drift/psi_ks.py` (lift from `services/forecast/monitor.py`) tracking judge-score residuals per question over time. |
| **fine-tuning** | `docs/LIMITATIONS.md` notes the retrieval-encoder LoRA path (out of scope for v1) — names the technique without scope-creep. |
| **prompt engineering** | `generation/answer.py` system prompt is version-pinned; prompt-versioning is documented. |
| **guardrails / hallucination** | `judges/llm/faithfulness.py` + `judges/llm/clinical_safety.py` + `judges/deterministic/citation_coverage.py`. |
| **scalable oversight** | `judges/auditors/` — paired LLM auditors confirm/dismiss every deterministic finding (1:1 with `proposal-ops-judges` pattern). |
| **MLflow / Weights & Biases** | `tracking/mlflow_hook.py` mirrors metrics if `MLFLOW_TRACKING_URI` is set. |
| **vector database / Pinecone / Weaviate / Qdrant** | `retrieval/embed.py` swappable backend (default Chroma for OSS, Qdrant adapter behind config flag). |
| **Anthropic SDK / Claude API** | LLM judges call Claude via the `anthropic` SDK with prompt caching. |

If a phrase isn't in the table, it doesn't need to be in the project.

## Eval suite v1 — content

20–30 clinical questions in `configs/eval_suite_v1.yml`, each with:

```yaml
- id: q_sglt2_hfpef
  question: "What is the evidence for SGLT2 inhibitor use in HFpEF?"
  expected_citations:
    - 10.1056/NEJMoa2107038       # EMPEROR-Preserved (must cite)
    - 10.1056/NEJMoa2206286       # DELIVER (should cite)
  must_mention: ["dapagliflozin", "empagliflozin", "ejection fraction"]
  refusal_expected: false
  evidence_quality_expected: high
  safety_red_flags: ["dose adjustment in CKD"]
```

Plus 4–6 deliberate **refusal cases** (questions for which the corpus has no
evidence, where the model must refuse). Refusal-consistency is its own judge.

The 20–30 questions come from the operator's pre-generated Q&A pairs once the
OneDrive blocker from `docs/project-treatment-plan.md` resolves. If not yet
available, the next session authors them from the public PubMed sample
manifest (~3 hrs).

## Connection to the existing portfolio

| Existing project | Role in this build |
| --- | --- |
| `proposal-ops-judges` (github) | **Pattern source.** Each judge here mirrors the proposal-ops scalable-oversight structure (Python deterministic + LLM judge + paired LLM auditor). |
| `johndegraft.app` clinical-rag card | **Front-end consumer.** The Try It Out demo (see `project-treatment-plan.md` § clinical-rag) reads `examples/sample_run/audit_report.md` as the embedded eval result. |
| `healthcare-cost-ops/services/forecast/monitor.py` | **Drift module to lift.** Same PSI + KS pattern, different target (judge-score residuals instead of forecast residuals). |
| `healthcare-cost-ops/services/forecast/tracking.py` | **Tracking shim to lift.** Same JSONL + MLflow pattern. |
| `asi.kb_chunks` Cosmos store (memory) | **Optional** corpus sink for the user's own larger biomedical corpus once OneDrive sync resolves. |

## Build order for the fresh session

1. `corpus/ingest.py` + `sample_manifest.json` → produces a small Chroma index (~30 min)
2. `retrieval/embed.py` + `retrieval/hybrid.py` (~2 hrs)
3. `generation/answer.py` + `generation/refusal_gate.py` (~1.5 hrs)
4. `judges/deterministic/` (5 judges, ~3 hrs)
5. `judges/llm/` (4 judges with Anthropic SDK + prompt caching, ~3 hrs)
6. `judges/auditors/` (3 paired auditors, ~2 hrs)
7. `drift/` (lift from `forecast/monitor.py`, ~1 hr)
8. `mcp/server.py` (FastMCP, ~2 hrs)
9. `tracking/langsmith_hook.py` + `tracking/mlflow_hook.py` (~1.5 hrs)
10. `cli.py` + `tests/` (~2 hrs)
11. `README.md` + `docs/` (~2 hrs — every JD-vocabulary surface gets named)
12. Sample run committed to `examples/sample_run/` (~1 hr)

**Total: ~21 hrs** (≈ 3 working days).

## Acceptance criteria

- `clinical-rag-eval run --suite configs/eval_suite_v1.yml` exits 0 on a clean
  install and writes `audit_report.md` + `audit_record.json` + `drift_report.json`.
- All 5 deterministic judges + 4 LLM judges + 3 paired auditors instantiate.
- MCP server starts and exposes its 3 tools (validate with `mcp dev`).
- LangSmith trace appears for at least one full eval run when `LANGCHAIN_API_KEY`
  is set; absence is graceful (no crash).
- README's "What this looks like in a JD" table cites every phrase from the
  vocabulary histogram above.
- Public repo passes a scrub check (no `LangChainAPIKey.txt`, no Anthropic key,
  no real client names, no Cosmos / Azure resource names).

## Strategic placement

This project becomes the **lead CV bullet** for the AI/ML JDs career-ops
surfaced. The forecasting module remains as a **supporting bullet** —
"Production ML forecasting pipeline with PyTorch Transformer + drift
monitoring + MLflow tracking" — which is exactly the secondary-skill posture
the JDs ask for. The pair tells one story: end-to-end ML platform engineer
who builds both the LLM-evaluation half **and** the classical-ML/forecasting
half of a clinical-AI stack.

## Open questions (resolve before next-session execution)

1. **OneDrive Q&A pairs** — if the operator's pre-generated clinical Q&A
   pairs (from `johndegraft-app/docs/project-treatment-plan.md` § clinical-rag)
   are available by the time the next session runs, use them as the
   `expected_*` fields in `eval_suite_v1.yml`. Otherwise, the next session
   authors 20–30 from the public PubMed sample.
2. **Existing `/rag` backend repo** — if the live `https://johndegraft.app/rag`
   endpoint already has source code somewhere (still unconfirmed),
   `clinical-rag-eval` evaluates it directly. Otherwise it ships a reference
   answer-generator inside `generation/` and evaluates that.
3. **Anthropic key budget** — the LLM judges cost about $0.40 per full
   30-question eval pass at Sonnet 4.6 rates. Confirm acceptable monthly
   budget before scheduling a recurring eval cron.
4. **Cosmos sink** — should drift findings persist to `healthcare_kb.docs` or
   stay in repo-local `examples/sample_run/`? Default: repo-local for the
   public portfolio version; Cosmos sink is a private fork-only feature.
