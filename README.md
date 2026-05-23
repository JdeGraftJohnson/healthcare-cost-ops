# healthcare-cost-ops

**Portfolio project + reusable Claude system for Power BI dashboard delivery.**

Two deliverables in one repo:

1. **Healthcare prescription cost & forecast dashboard** built on the CMS Medicaid State Drug Utilization Data (SDUD) public API, ingested through the existing ASI Azure stack (Blob → DuckDB silver → Parquet → Power BI semantic model), with DAX measures generated and validated by a Claude LLM pipeline.
2. **`dashboard-ops` LLM evaluation harness** — a port of the [`proposal-ops-judges`](https://github.com/JdeGraftJohnson/proposal-ops-judges) pattern (21 evaluators across deterministic / LLM / paired-auditor tiers) re-aimed at Power BI artifacts: DAX correctness, model design, visualization choice, accessibility, governance, and forecasting methodology. Reusable on any future dashboard.

The healthcare dashboard is the **canonical first run** that exercises the harness end-to-end. Subsequent dashboards plug in by writing a new `dashboard_spec.yml` and re-running the pipeline.

---

## Why this design

Power BI delivery in industry today is judged on more than "does the report open." The job-description evidence (`docs/JD_EVIDENCE.md`, sampled from 16k healthcare-sector listings) shows employers expect:

- **DAX rigor** — calculated columns vs measures, row context vs filter context, `CALCULATE` semantics, time-intelligence patterns.
- **Tabular model hygiene** — star schema, single-direction relationships, role-playing dimensions, no calculated columns in fact tables.
- **Forecasting + KPI framing** — not just descriptive charts; an executive-grade story (current spend → trend → forward 12-month projection → driver attribution).
- **Compliance posture** — PHI / HIPAA awareness, row-level security, data-source documentation.
- **Reproducibility** — pipeline-as-code, parameterized data sources, idempotent refresh.

The Claude harness scores each of these dimensions automatically against a fixed rubric and emits a banded composite verdict (`Ship` / `Tighten` / `Re-work`), the same auditable pattern as `proposal-ops-judges`.

---

## Repo layout

```
services/
  ingest/
    sdud_pull.py             Medicaid SDUD year-partitioned CSV → Azure Blob bronze
    sdud_silver.py           DuckDB cleaning: NDC normalize, drug-class join, $/Rx
    forecast.py              SARIMA + Prophet ensemble per drug class, 12-mo horizon
  dax/
    templates.py             Library of 25+ DAX measure templates (cost, trend, forecast)
    generate.py              Claude → DAX measure from natural-language spec
    validate.py              Static-analyze DAX against Tabular Editor's BPA rules
  powerbi/
    model_build.py           Build .bim (Tabular Object Model JSON) from spec
    deploy.py                Push to Power BI Service via REST
  judges/
    base.py                  JudgeResult dataclass + shared loaders
    audit_runner.py          --plan / --merge / --run-all
    # deterministic (8)
    dax_syntax.py            DAX parses, references only existing columns
    dax_perf.py              No calculated columns on fact, no FILTER(ALL(...))
    model_design.py          Star schema check, single-direction filters
    pii_leak.py              Patient identifiers in semantic model labels
    accessibility.py         Color contrast, chart-title clarity, ALT text
    governance.py            RLS roles present, data-source documentation
    refresh_health.py        Last-refresh, gateway, scheduled cadence
    template_fit.py          Spec → template alignment (which of top-10 templates)
    # LLM judges (5)
    business_narrative.py    Exec summary quality, story arc, action-orientation
    visualization_choice.py  Chart-type appropriateness vs data shape
    forecast_method.py       Model choice + uncertainty disclosure
    dax_review.py            Idiomatic DAX vs anti-patterns
    domain_relevance.py      Healthcare context: HEDIS, ICD, NDC, HCPCS usage
    # paired auditors (3)
    auditors/
      dax_auditor.py         Re-checks dax_syntax + dax_perf findings on the .bim
      model_auditor.py       Re-checks model_design + governance
      narrative_auditor.py   Re-checks business_narrative for hallucinated metrics

templates/
  01_executive_overview.dax            Headline KPIs + spend trend
  02_top_n_drug_classes.dax            Variable-N ranking with "Other" bucket
  03_yoy_growth.dax                    Time-intelligence with SAMEPERIODLASTYEAR
  04_state_choropleth.dax              State-level $/Rx and per-capita normalization
  05_cost_per_member_month.dax         PMPM standard healthcare metric
  06_forecast_band.dax                 Forecast point + 80% / 95% confidence ribbons
  07_pareto_drug_spend.dax             80/20 cumulative-share view
  08_generic_brand_mix.dax             Brand vs generic substitution opportunity
  09_outlier_flag.dax                  ATR-style volatility flag on cost trajectory
  10_quality_scorecard.dax             Refresh / RLS / completeness composite

examples/
  medicaid_sdud_2026/                  Canonical first dashboard run (all artifacts)
    dashboard_spec.yml
    out/
      audit.md
      composite_scorecard.md
      forecast.parquet
      model.bim

docs/
  JD_EVIDENCE.md                       PowerBI/healthcare JD requirements survey
  DASHBOARD_TEMPLATES.md               Top-10 use cases mapped to MS sample datasets
  AGENT_PIPELINE.md                    How the Claude pipeline scopes → builds → audits
  DAX_PATTERNS.md                      Idiomatic DAX patterns from MS DAX syntax ref
  POWERBI_DEPLOY.md                    Local-render-only deploy (no PHI to PBI Service yet)

scripts/
  build-dashboard.sh                   One-command: ingest → generate → audit → render
  refresh-sdud.sh                      Incremental SDUD pull for new monthly drops
```

---

## Quickstart (healthcare canonical run)

```bash
python -m services.ingest.sdud_pull --years 2020,2021,2022,2023,2024,2025 \
  --out azure://&lt;STORAGE_ACCOUNT&gt;/healthcare/bronze/

python -m services.ingest.sdud_silver \
  --bronze azure://&lt;STORAGE_ACCOUNT&gt;/healthcare/bronze/ \
  --silver azure://&lt;STORAGE_ACCOUNT&gt;/healthcare/silver/

python -m services.ingest.forecast \
  --silver azure://&lt;STORAGE_ACCOUNT&gt;/healthcare/silver/ \
  --horizon 12

python -m services.judges.audit_runner --plan \
  --spec examples/medicaid_sdud_2026/dashboard_spec.yml

python -m services.powerbi.model_build \
  --spec examples/medicaid_sdud_2026/dashboard_spec.yml \
  --out examples/medicaid_sdud_2026/out/model.bim
```

The audit emits `examples/medicaid_sdud_2026/out/audit.md` (severity-banded findings) and `composite_scorecard.md` (weighted verdict). A live walkthrough lands at `johndegraft.app/projects/healthcare-dashboard` once Phase 5 ships.

---

## Status

| Phase | Scope | State |
| - | - | - |
| 0 | Repo scaffold, JD evidence, scoping docs | **DONE** (`docs/JD_EVIDENCE.md`, this README) |
| 1 | SDUD ingest → Azure bronze → DuckDB silver | drafted, `services/ingest/sdud_pull.py` |
| 2 | DAX template library + Claude generator | drafted, `services/dax/templates.py` |
| 3 | Judge harness (8 deterministic + 5 LLM + 3 auditors) | scaffolded, ported from proposal-ops |
| 4 | Forecast layer (SARIMA + Prophet ensemble) | designed in `docs/AGENT_PIPELINE.md` |
| 5 | Portfolio walkthrough page on johndegraft.app | pending |

---

## Reusing for non-healthcare dashboards

Write a new `dashboard_spec.yml` describing data sources, business questions, KPIs, and audience. The pipeline ingests, generates DAX from the template library + Claude, builds a Tabular model, audits it, and emits a scorecard. The judges and templates are domain-neutral; only `domain_relevance.py` is healthcare-specific and is parameterized by spec.

License: MIT.
