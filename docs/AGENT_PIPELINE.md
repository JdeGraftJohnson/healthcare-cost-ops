# Agent Pipeline — Claude system for dashboard delivery

A direct port of the `proposal-ops` pattern (`/Users/john/Documents/Claude/Projects/AI Government Proposal/services/judges/`) re-aimed at Power BI artifacts. The roles, handoffs, and severity bands are the same; only the rubrics change.

---

## Pipeline phases

```
spec (dashboard_spec.yml)
  │
  ▼
[1] intake          parse spec → enumerate datasets, KPIs, templates, audience
  │
  ▼
[2] ingest          SDUD pull → Azure bronze → DuckDB silver → Parquet
  │
  ▼
[3] forecast        SARIMA + Prophet ensemble per drug class, 12-mo horizon
  │
  ▼
[4] generate        Claude → DAX measure set per template
                    Claude → model.bim (Tabular Object Model)
                    Claude → page-layout JSON + narrative captions
  │
  ▼
[5] audit_runner --plan
     │
     ├──► deterministic judges (8, parallel)
     │      dax_syntax · dax_perf · model_design · pii_leak
     │      accessibility · governance · refresh_health · template_fit
     │
     ├──► LLM judges (5, parallel)
     │      business_narrative · visualization_choice · forecast_method
     │      dax_review · domain_relevance
     │
     └──► paired auditors (3, parallel)
            dax_auditor · model_auditor · narrative_auditor
  │
  ▼
[6] audit_runner --merge       → audit.md (severity bands)
                                → composite_scorecard.md (weighted verdict)
  │
  ▼
[7] auto_fix (optional)        non-invasive remediation of [WARN] findings
  │
  ▼
[8] render                     model.bim → .pbit template → optional PBI Service push
```

---

## Severity bands (mirrors proposal-ops)

| Tag | Meaning | Gate behavior |
| - | - | - |
| `[OK]`   | Check passed | informational |
| `[WARN]` | Quality issue, ship-blocking only above threshold | `auto_fix` may remediate |
| `[MISS]` | Critical — DAX syntax error, PHI leak, RLS missing | hard gate |

Composite verdict:

- **Ship** — weighted score ≥ 0.85, no `[MISS]`
- **Tighten** — 0.70–0.85, no `[MISS]`
- **Re-work** — < 0.70 or any `[MISS]`

---

## Agent roster

Mirroring `proposal-ops`'s Claude Code subagent definitions in `.claude/agents/`. Each subagent has a frontmatter block, a tightly-scoped tool set, and a clear "when to invoke" rule.

| Subagent | Tools | Invoked by | Job |
| - | - | - | - |
| `dashboard-intake` | Read, Write, Grep | `/dashboard:scope` | Parse the spec, enumerate KPIs, infer template fit |
| `dax-generator` | Read, Write | `/dashboard:build` | Generate DAX measure set from spec + templates |
| `model-builder` | Read, Write, Bash | `/dashboard:build` | Build Tabular Object Model `.bim` JSON |
| `narrative-writer` | Read, Write | `/dashboard:build` | Generate executive captions per page |
| `judge-orchestrator` | Read, Bash, Grep, Glob | `/dashboard:audit` | Fan out judges, collect results, write `audit.md` |
| `dax-auditor` | Read, Grep, Glob | judge-orchestrator | Re-inspect `dax_syntax` + `dax_perf` findings |
| `model-auditor` | Read, Grep, Glob | judge-orchestrator | Re-inspect `model_design` + `governance` |
| `narrative-auditor` | Read, Grep, Glob | judge-orchestrator | Re-inspect narrative for hallucinated metrics |
| `dashboard-rewrite-evaluator` | Read, Bash | post-audit | Score a single page or measure set against rubric |

The `judge-orchestrator` here is the **same pattern** as the proposal-ops orchestrator: it reads an `audit_plan.json` written by the Python `audit_runner.py --plan`, fans out the paired auditors in parallel, collects their XML results, and writes the merged `audit.md`. The Python `audit_runner.py --merge` then produces `composite_scorecard.md`.

---

## Handoff contracts

Each phase writes a single artifact that the next phase reads. No phase looks at intermediate state inside another phase's working directory.

```
[1] intake          → out/<run>/intake.json
[2] ingest          → silver/*.parquet
[3] forecast        → silver/forecast.parquet
[4] generate        → out/<run>/measures.dax, out/<run>/model.bim, out/<run>/narrative.json
[5] audit (plan)    → out/<run>/judges/audit_plan.json
[5] audit (run)     → out/<run>/judges/<judge>.json (one per evaluator)
[6] merge           → out/<run>/audit.md, out/<run>/composite_scorecard.md
[7] auto_fix        → out/<run>/model.fixed.bim, out/<run>/auto_fix.log
[8] render          → out/<run>/dashboard.pbit
```

This is the same contract pattern as proposal-ops's `audit_plan.json` / `AUDIT.md` / `COMPOSITE_SCORECARD.md`. The phase outputs are JSON or Markdown, never pickle or pyobj — so any phase can be rerun standalone, and the orchestrator never has to share Python state across subprocess boundaries (see `feedback_auto_pipeline_orchestrator` — never use `input()` blocking pauses).

---

## Why deterministic judges run before LLM judges

Same logic as proposal-ops:

1. **Cheap fast feedback.** A DAX parse error is detectable in milliseconds — no point spending a Claude turn on it.
2. **Calibration.** The deterministic layer gives the LLM judges a known-good context. The narrative judge isn't asked "is this good?" in a vacuum; it's asked "given that 7 of 8 deterministic checks passed, is the narrative consistent with the numbers?"
3. **Reproducibility.** Deterministic results are byte-stable and become the regression baseline. LLM results have model-version drift — they get re-run on every harness invocation but the deterministic floor doesn't move under your feet.

---

## Why paired auditors exist

Same logic as proposal-ops's auditor tier: deterministic judges catch the **shape** of the problem; auditors catch the **interpretation**.

Example: `dax_perf.py` flags `FILTER(ALL(...))` as a perf anti-pattern. But there's a legitimate case — clearing slicer context to compute a denominator for a % share measure. The deterministic judge can't tell those apart. The `dax-auditor` reads the surrounding measure context and confirms-or-dismisses each flag.

This is the **scalable-oversight** piece: the deterministic judge is fast and broad; the auditor is narrow and judgment-driven. Together they cover the surface area without either being the single point of failure.

---

## Reusing for any future dashboard

The dashboard-ops system is parameterized by `dashboard_spec.yml`:

```yaml
name: medicaid_sdud_2026
domain: healthcare
audience: state Medicaid program managers
data_sources:
  - name: sdud
    type: medicaid_open_data_api
    years: [2020, 2021, 2022, 2023, 2024, 2025]
  - name: state_population
    type: us_census_acs5
    vintage: 2024
business_questions:
  - "What is total Medicaid prescription spend, broken out by state and drug class, with 12-month forecast?"
  - "Which 20% of drug classes drive 80% of cost, and how has that mix shifted YoY?"
  - "Where is the brand-to-generic substitution opportunity largest?"
templates: [01, 02, 03, 04, 06, 07, 08, 09, 10]
compliance:
  phi: false      # SDUD is aggregated state-level, no patient identifiers
  rls: state      # if pushed to PBI Service, row-level by state
forecast:
  horizon_months: 12
  methods: [sarima, prophet]
  ensemble: equal_weight
```

To build a non-healthcare dashboard (e.g., a financial-services PnL view), drop in a different `domain`, point at a different data source, list the templates that apply, and re-run. The judges that are domain-neutral (DAX, model, accessibility, governance, refresh, narrative, viz choice, forecast method) run unchanged. The `domain_relevance.py` judge swaps its prompt rubric based on `domain:` and checks for the appropriate vocabulary (NDC/HCPCS for healthcare; CUSIP/ISIN for finance; etc.).

This is the same reuse story as proposal-ops: `domain_relevance` is the only judge that knows what business it's in. Everything else is structural.

---

## Reference material consulted

- `~/Documents/GitHub/.claude/` and `~/Git/claude_docs/` — agent-teams, sub-agents, plugins, hooks-guide, skills, headless. The handoff pattern here uses the same JSON-artifact-between-phases idiom from the agent-teams doc.
- `~/Git/proposal-ops-judges/` — base.py, audit_runner.py, the judges/auditors split.
- `~/Documents/Claude/Projects/AI Government Proposal/` — full proposal-ops parent system: orchestrator code, severity-band conventions, prompt scaffolds.
- <https://learn.microsoft.com/en-us/dax/dax-syntax-reference> — DAX function arity + return types for the static analyzer in `services/judges/dax_syntax.py`.
- <https://learn.microsoft.com/en-us/power-bi/create-reports/sample-datasets> — visual rhythm baselines (see `docs/DASHBOARD_TEMPLATES.md`).
