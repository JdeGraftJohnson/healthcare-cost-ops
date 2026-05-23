# Job-Description Evidence — Power BI for Healthcare Analytics

Sampled from `~/career-ops-main/web-scraper/universal_scraper_health.csv` (16,001 healthcare-sector listings as of 2026-05-23). Filter: title or description mentions Power BI, DAX, tabular model, paginated report, measure, KPI, dashboard, or visualization. **37 listings matched.**

This document is the source-of-truth for which dashboard patterns to build and which evaluation dimensions the judge harness should score.

---

## Top skill mentions (across the 37 matched listings)

| Skill / Term | Hits | Implication for our build |
| - | -: | - |
| SQL | 23 | Silver layer must be queryable in T-SQL idiom; DuckDB + ANSI SQL is fine |
| SSIS | 17 | Legacy ETL. We use Azure Functions + DuckDB, but DOCS should explain mapping |
| Snowflake | 10 | Many shops dual-source. Spec model so source is parameterized |
| Cost | 9 | Cost analysis is the dominant business question — anchor our canonical dashboard here |
| Pharmacy | 7 | Pharmacy/drug analytics is a recurring sub-domain — SDUD is a perfect fit |
| PHI | 7 | HIPAA/PHI handling is consistently called out — judges must score PII leakage + RLS |
| Forecast | 5 | Forecasting is expected, not optional — SARIMA + Prophet ensemble is justified |
| Power BI | 5 (+1 "powerbi") | Direct tool match |
| KPI | 4 | Executive scorecards are a recurring template — top-of-page KPI strip is required |
| Drug | 4 | Reinforces pharmacy/SDUD anchor |
| SQL Server | 3 | Tabular model deploys via SSAS or PBI Service — DOCS must cover both |
| Databricks | 3 | Lakehouse pattern — our DuckDB silver is conceptually equivalent |
| Tableau | 2 | Cross-tool fluency expected. Our templates are tool-agnostic in spec form |
| HIPAA | 1 | Combined with PHI = 8 hits. Compliance is not a nice-to-have |
| Claims | 1 | Claims data is the other major sub-domain; SDUD is a public proxy |
| Fabric | 1 | Microsoft Fabric is emerging — we expose Parquet so OneLake ingestion is one step |

---

## Inferred role expectations (synthesized from sample descriptions)

1. **Own the semantic layer.** Build the Tabular model, define measures, manage relationships. Not just dragging fields onto a canvas.
2. **Write idiomatic DAX.** Measure-first thinking, time-intelligence patterns, CALCULATE + filter modification, avoiding `FILTER(ALL(...))` and other anti-patterns.
3. **Translate business questions into KPIs.** Multiple JDs frame this as "partner with finance / clinical / ops stakeholders to define metrics."
4. **Forecasting + variance analysis.** Budget vs actual, trend with confidence bands, driver attribution.
5. **Governance.** Row-level security, data lineage, refresh monitoring, certified-dataset designation.
6. **Storytelling.** Executive summary page, drill-through paths, narrative captions.
7. **HIPAA awareness.** Mask MRN / SSN / DOB; use de-identified or aggregated data where possible; document data sources.

---

## What the judge harness must score (mapped from the above)

| Expectation | Judge type | Module |
| - | - | - |
| DAX correctness | deterministic + LLM + auditor | `dax_syntax.py`, `dax_review.py`, `auditors/dax_auditor.py` |
| Anti-pattern avoidance | deterministic | `dax_perf.py` |
| Star-schema hygiene | deterministic | `model_design.py` |
| Executive narrative | LLM | `business_narrative.py` + `auditors/narrative_auditor.py` |
| Chart-type fit | LLM | `visualization_choice.py` |
| Forecast methodology | LLM | `forecast_method.py` |
| Healthcare domain literacy | LLM | `domain_relevance.py` (NDC, HCPCS, ICD, HEDIS) |
| PHI / HIPAA hygiene | deterministic | `pii_leak.py` |
| RLS + governance | deterministic | `governance.py` |
| Refresh health | deterministic | `refresh_health.py` |
| Accessibility | deterministic | `accessibility.py` |
| Spec-template fit | deterministic | `template_fit.py` |

Total: **8 deterministic + 5 LLM + 3 paired auditors = 16 evaluators.** Smaller surface area than proposal-ops (21) because Power BI artifacts are more structurally constrained than free-form proposal prose.

---

## Sample listings (representative)

Trimmed to ~250 chars each; full rows in the source CSV.

- **The Economist Group — Insights Product Manager, Analytics Engineering.** "...partner with stakeholders to translate insight needs into Power BI dashboards; own the semantic model and DAX measure library..."
- **BNY — SVP, Data Governance.** "Build the Business Information Model and data architecture for a global financial services firm; KPI catalog, lineage, certified datasets..." (governance-heavy; informs `governance.py`)
- **Intelerad — Database SRE Specialist.** "...medical imaging; SQL Server, monitor refresh, troubleshoot tabular models, ensure PHI handling..." (informs `refresh_health.py`, `pii_leak.py`)
- **Paloma Health — Technical Support & Operations Agent.** "...maintain operational dashboards in Power BI, escalate refresh failures, document data sources..." (governance + refresh)

---

## Microsoft Power BI sample-dataset reference

Per <https://learn.microsoft.com/en-us/power-bi/create-reports/sample-datasets>, Microsoft ships eight reference samples that set the visual / DAX baseline industry expects:

1. **Customer Profitability** — profit waterfall, RLS demo
2. **Human Resources** — headcount KPIs, time-intelligence
3. **IT Spend Analysis** — variance vs budget (directly relevant to our cost-analysis frame)
4. **Opportunity Analysis** — funnel + cohort
5. **Procurement Analysis** — vendor concentration, Pareto (template #07)
6. **Retail Analysis** — geo map + period-over-period
7. **Sales & Returns** — drillthrough story
8. **Supplier Quality Analysis** — scorecard + outlier flagging

Our template library (`templates/01_executive_overview.dax` ... `templates/10_quality_scorecard.dax`) borrows the visual rhythm and KPI framing from samples **3 (IT Spend)**, **5 (Procurement Pareto)**, **6 (Retail geo)**, and **8 (Supplier scorecard)** — the four samples whose mechanics map directly to prescription cost analysis.
