# Forecast pipeline — v0.4+ roadmap

Self-contained backlog for the next focused session. Each item is scoped
enough to execute without reading the whole audit thread. Three tiers,
ordered by combined `(JD-vocabulary signal × operational value × ease)`:

- **Tier 1** — production-deploy blockers. Should be in v0.4.
- **Tier 2** — scale + research-grade. v0.5.
- **Tier 3** — nice-to-have. opportunistic.

The completed v0.1→v0.3 work is in `FORECAST_PIPELINE.md` and the audit
trail is in `FORECAST_PIPELINE_AUDIT.md` + `FORECAST_PIPELINE_AUDIT_DEEP.md`.

---

## Tier 1 — production-deploy blockers

### T1.1 — Distributed SARIMA fan-out on Azure Container Apps — **CODE+INFRA DONE (2026-05-23), DEPLOY GATED**

Code, container image definition, Bicep infra, runbook all landed.
End-to-end local mode (`--mode local`) validated against the real Azure
silver: 3 partitions × 114 series → 2,736 forecast rows in 3.9 s.

Components:
- `services/forecast/distributed.py` — partition + assemble + poll helpers
- `services/forecast/aca_worker.py` — per-partition CLI worker
- `services/forecast/distributed_driver.py` — fan-out + concat CLI
  (`--mode local|aca`)
- `services/forecast/aca/Dockerfile` + `requirements.txt` — slim
  python:3.12-slim image with statsmodels + duckdb (no LightGBM /
  Prophet / torch — only what SARIMA + naive baselines need)
- `infra/forecast_sarima_caj.bicep` — CAJ `caj-forecast-sarima-prod1` +
  user-assigned MI `id-caj-forecast-prod1` + container-scoped RBAC
- `docs/FORECAST_DISTRIBUTED_RUNBOOK.md` — build, deploy, invoke,
  troubleshoot

What's NOT done: `az acr build`, `az deployment group create`, and an
actual fan-out invocation in prod1. Those are operator-gated cloud
spend (~$0.80 per full SDUD run, ~$0.003 per ACR build per
`feedback_azure_cloud_job_runs`). Run the runbook when ready.

**Original specification (preserved for reference):**

ProcessPoolExecutor parallelism shipped in v0.4
(`SarimaModel(n_workers=N)`), but local cores top out at 4-8. The SDUD
panel is ~15k series × ~5s per fit = ~21 hr sequential. Need cloud
fan-out.

**Approach:**
1. New module `services/forecast/distributed.py` with a
   `partition_panel(panel, n_partitions)` that splits by group hash so
   each partition has roughly equal series count.
2. New container job `services/forecast/aca/Dockerfile` building a
   minimal Python image with statsmodels + duckdb + the forecast package
   on `mcr.microsoft.com/azure-functions/python:4-python3.12`.
3. Each ACA job invocation runs `python -m services.forecast.aca_worker
   --partition-id N --total M --silver az://...`; writes its slice of
   `forecast.parquet` to `silver/snapshot=.../forecast_partition_N.parquet`.
4. Driver job (last to complete) `concat`s the partitions and writes the
   final unified `forecast.parquet`.

**Effort:** L (2-3 days). Touches infra (ACA bicep), packaging
(Dockerfile, requirements pin), and code (the partition driver).

**Cost estimate:** ~$0.40/run on Consumption ACA at 4 vCPU × 8 GiB × 30
min for the full SDUD panel. Confirm with operator before scheduled cron
([[feedback_azure_cloud_job_runs]]).

**Dependencies:** real silver in Azure (currently the supplements silver
is the only one populated; SDUD silver in `silver/snapshot=2026-05-xx/`
must exist).

**Watch out for:**
- ProcessPool + ACA Linux base image: fork start method is unreliable;
  set `multiprocessing.set_start_method("spawn", force=True)` at worker
  entrypoint.
- statsmodels imports are heavy; warm a `pip install --no-cache-dir`
  + pre-cached wheel directory in the image build to keep cold-start under
  30s.

---

### T1.2 — MLflow Model Registry promotion semantics — **DONE (2026-05-23)**

Shipped alongside the finalize CAJ pattern:

- `services/forecast/registry.py` — wraps `mlflow.tracking.MlflowClient`;
  `register_and_promote()` registers a bundle, looks up current Production,
  applies the `_is_better(challenger, champion, tol, higher_is_better)`
  rule, transitions to Production (archiving the previous) when the
  challenger improves by > `champion_tol`, otherwise Staging.
- `services/forecast/finalize.py` — downstream CAJ entrypoint that calls
  `register_and_promote` after writing the final `forecast.parquet`.
- `infra/forecast_finalize_caj.bicep` — `caj-forecast-finalize-prod1`.
- `logs/promotions.jsonl` — audit trail written even when MLflow is
  absent (graceful skip; decision is still recorded).
- Tests: 7 new tests covering `_is_better` direction logic, the no-MLflow
  ledger-only path, partials → ensemble round-trip, empty-partial
  marker handling. 40/40 total pipeline tests passing.

**Known gap (v0.5):** the worker's `--score-window` arg is not yet
wired; until it is, `final_test_metrics_by_method` will be empty when
finalize consumes fan-out partials (the partials' forecast dates land
past panel-end, not in the holdout window).

**Original specification (preserved for reference):**

v0.4 ships `services/forecast/tracking.py` that mirrors
metrics to MLflow if `MLFLOW_TRACKING_URI` is set. No model artifacts go
to the registry yet, no Champion / Staging / Production transitions.

**Approach:**
1. Extend `tracking.py:track()` to call `mlflow.register_model(...)` with
   the persisted bundle from `persist.save_bundle()` as the artifact.
2. New CLI `python -m services.forecast.promote --name supplements_price_monthly
   --to production --run-id <id>` that calls
   `MlflowClient.transition_model_version_stage(...)`.
3. Champion-challenger (`compare.py`) auto-promotes the challenger to
   `Staging` when it beats the champion by >tol and demotes the prior
   `Production` to `Archived`.
4. Add an alembic-style migration log in `logs/promotions.jsonl` so the
   promotion history is queryable without an MLflow server up.

**Effort:** M (1 day).

**JD signal:** "MLOps", "model registry", "champion-challenger",
"promotion semantics" — top 20 of the surveyed JDs.

**Watch out for:** MLflow's local file backend is fine for development
but the model registry features require a real backend (SQL backend +
artifact store). Default to S3-compatible (Cloudflare R2 works) so it
runs free-tier.

---

### T1.3 — Power BI integration runbook + binding contract

**Status:** Pipeline writes `forecast.parquet` in a long-form schema; the
Power BI session is supposed to consume it but the contract isn't formally
documented anywhere outside the README.

**Approach:** Write `docs/POWERBI_FORECAST_BINDING.md` with:
1. Schema contract: column names, types, NULL-handling, datetime tz.
2. M-query template that reads `forecast.parquet` from Azure Blob over
   `Web.Contents` with OAuth (signed-in user).
3. DAX measures for: point forecast, 80% band lower, 80% band upper, 95%
   band lower, 95% band upper, method name (for the methodology tooltip
   drill-through).
4. Page-layout JSON template (`templates/06_forecast.json`) that uses
   those measures.
5. Versioning policy: when `forecast.parquet` schema changes, write the
   change with a `schema_version` number; Power BI template checks the
   version and surfaces a banner if mismatched.

**Effort:** M (1 day). No code; pure spec + template.

**Owner overlap:** the Power BI session (parallel Claude Code) consumes
this. Coordinate the merge: this session writes the doc, that session
implements the M-query.

---

### T1.4 — Real Azure silver swap path — **DONE (2026-05-23)**

Shipped. The forecast pipeline now reads the real supplements silver
from `stasiprod1eus2/healthcare/silver/snapshot=2026-05-23-v2/` in the
target-prod1 tenant. Mechanism:

- `services/forecast/data.py:_resolve_env()` interpolates `${VAR:-default}`
  patterns from environment before DuckDB sees the SQL.
- `services/forecast/data.py:_ensure_azure_extension()` installs the
  DuckDB azure extension and registers a `credential_chain` secret per
  distinct `az://<account>` URL in the SQL. credential_chain picks up
  `az login` / managed identity / env-var auth transparently — no secrets
  in config, code, or the JSONL ledger.
- `configs/supplements_price.yml` SQL now defaults to the Azure URL,
  with a `SUPPLEMENTS_SILVER_URL` env override for offline dev.
- `docs/AZURE_SILVER_ACCESS.md` documents the auth chain, the
  managed-identity-vs-SAS tradeoff, and the RBAC grant pattern.

Real-data first-run results (977 rows, 114 series after sparsity filter):

```
                  BACKTEST            FINAL-TEST (untouched holdout)
method        MAPE   MASE  cov80    MAPE   MASE  cov80
drift        1.49%   0.84  76.5%   4.15%  0.75  69.9%
sarima       1.53%   1.00  84.2%   4.18%  0.75  73.9%
mean         1.65%   1.10  72.5%   4.18%  0.75  60.0%
seasonal_n   1.75%   1.10  73.2%   4.18%  0.75  62.4%
```

Two real-vs-synthetic schema gotchas surfaced and fixed on the first
real-data run:

1. `off_categories_tags` is `VARCHAR[]` (OFF tag array), not a
   comma-joined `VARCHAR` like the synthetic. SQL now uses
   `off_categories_tags[len(off_categories_tags)]` to pick the leaf.
2. 2 of 977 rows have `NULL price_date`. SQL now filters
   `price_date IS NOT NULL`.

`examples/supplements_2026/build_sample.py` should be updated to mirror
the real schema (queued as a follow-up — minor).

---

## Tier 2 — scale + research-grade

### T2.1 — Hierarchical reconciliation (Hyndman-Athanasopoulos §11)

**Status:** Forecasts at the leaf level (state × drug_class for SDUD;
category × brand for supplements) don't reconcile with parent-level
aggregates. A drug-class forecast summed across states won't equal the
direct forecast of the national class total.

**Approach:**
1. New module `services/forecast/reconcile.py` implementing MinT
   (Minimum-Trace) reconciliation per Wickramasuriya et al. (2019).
2. Config block:
   ```yaml
   hierarchy:
     levels: [national, region, state, state_drug_class]
     mapping_sql: |
       SELECT state_code, region, 'national' AS national FROM state_dim
   reconcile:
     method: mint_shrink         # mint_shrink | ols | wls | bottom_up
   ```
3. Pipeline runs the leaf-level forecast as today, then reconciles
   up-and-down so all levels agree.
4. Metrics emitted per level: reconciled MAPE vs direct MAPE per level
   shows the reconciliation gain.

**Effort:** L (2 days). Real ML work; needs validation against
`hierarchicalforecast` package (Nixtla) for correctness.

**JD signal:** "hierarchical forecasting" appears in M5-class JDs. Medium
priority.

---

### T2.2 — Quantile-regression backends (replace sigma + conformal stack)

**Status:** Current bands are `point ± k × sigma` (model-native) → 
empirical-quantile (v0.3) → conformal-widened (v0.3 multi-α). That's
three calibration layers on top of a point forecast. Quantile-regression
backends predict each α directly.

**Approach:**
1. `LightGBMQuantileModel(alphas=[0.05, 0.20, 0.50, 0.80, 0.95])` — fit
   `objective="quantile", alpha=α` for each level; ship as a separate
   backend.
2. `QuantileTransformerModel` — modify the Transformer head from
   1 → len(alphas) outputs; use pinball loss per α.
3. Drop the conformal layer when a quantile backend is selected; keep it
   for the non-quantile backends.

**Effort:** L (3 days). Each backend wants its own validation.

**JD signal:** "quantile regression" and "probabilistic forecasting" —
medium frequency in the surveyed JDs.

---

### T2.3 — GPU Transformer training pipeline

**Status:** Transformer runs CPU-only by default; v0.4 batched the
in-sample residual but training still uses small batches sized for CPU.

**Approach:**
1. Detect `torch.cuda.is_available()` in `TransformerModel.__init__`; if
   yes, default `batch_size=1024`, `epochs=80`, `d_model=128`.
2. Add `services/forecast/aca_gpu/Dockerfile` based on
   `nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04`.
3. ACA GPU SKU (V100 / A100) for short training jobs; document cost.

**Effort:** M (1 day if Colab T4; L if production ACA GPU).

**Cost estimate:** Colab T4 free tier handles up to ~50k series easily;
ACA GPU runs ~$2-4/hr depending on SKU. Confirm before scheduling.

---

### T2.4 — Structural-break detection (COVID-19 anchor)

**Status:** Healthcare panels (SDUD especially) have obvious COVID-19
breakpoints. Current pipeline treats them as noise; SARIMA struggles
because the stationarity assumption fails through 2020-2021.

**Approach:**
1. New module `services/forecast/structural.py` running BinSeg + PELT
   changepoint detection per series (via `ruptures` package).
2. For each detected breakpoint, fit two regimes (pre + post) and emit
   regime-conditional forecasts.
3. Visualize breakpoints in the Power BI methodology tooltip.

**Effort:** M (1.5 days). The detection is straightforward; the
two-regime forecast plumbing is the real work.

**Watch out for:** Overfitting changepoints to noise. Require minimum
segment length (e.g. 12 months) and minimum mean-shift magnitude.

---

## Tier 3 — opportunistic

### T3.1 — SHAP-style explainability for LightGBM forecasts

LightGBM ships `predict(X, pred_contrib=True)` for SHAP-equivalent
contributions. Wire into Power BI as a "why is this forecast what it is"
drill-through. ~3 hours.

### T3.2 — Attention-weight introspection for Transformer

`nn.MultiheadAttention(..., need_weights=True)` exposes per-head
attention; render a heatmap of which past periods drove the forecast.
~4 hours.

### T3.3 — Real-time prediction API

FastAPI app `services/forecast/serve/app.py` that loads a persisted
bundle (from T1.2) and answers `POST /predict` with a fresh forecast.
Caches the loaded bundle per-process. ~6 hours.

### T3.4 — Holt-Winters and ETS backends

`statsmodels.tsa.holtwinters.ExponentialSmoothing` ships these natively;
adding them is ~80 LOC each. ~3 hours total.

### T3.5 — Robust outlier handling in input

Winsorize at the 1st/99th percentile per series before fit. Materially
helps SARIMA stability on series with one-off spikes (data quality
issues, supply shocks). ~2 hours.

### T3.6 — Direct multi-step LightGBM (parallel to recursive)

Current LightGBM is recursive (predict h=1, feed back, predict h=2, ...).
Direct multi-step trains h separate models, one per horizon step. More
robust on long horizons but h× the training cost. ~6 hours.

### T3.7 — Probabilistic backtests with sampling

Currently each backtest fold gives a single forecast. Sampling several
predictions per series per fold (via Bayesian SARIMA, MC dropout
Transformer, quantile bootstrap LightGBM) gives a distribution-of-MAPE.
~1 day.

### T3.8 — `inspect_petri` integration

Emit `inspect_ai` task format from `eval.py` so the forecast harness
becomes a callable Inspect eval. Mirrors the path that
`clinical-rag-eval` is taking. ~4 hours.

---

## Open architectural questions for v0.5

1. **Forecast-vs-actual diff loop:** Once production runs are landing
   weekly, the silver gets new actuals. Should the pipeline auto-trigger
   a backtest scoring of the previous forecast against the now-landed
   actuals, and feed that into the drift monitor as "live coverage"?
   Currently drift is only against backtest residuals.

2. **Forecast as a service vs forecast as a job:** Current architecture
   is batch (cron + parquet). T3.3 would add a serving layer. Decide
   whether the operator wants both or only batch — affects whether
   bundle-persistence becomes load-bearing or stays optional.

3. **Single panel vs multi-panel ensembling:** Right now SDUD and
   supplements are two pipelines that don't talk. There's a plausible
   "drug-class ↔ supplement-category" co-prediction story (people
   substitute supplements for prescriptions; aggregate movement is
   correlated). Worth scoping a joint panel.

4. **Reconciliation order:** If T2.1 lands, where does it sit relative
   to conformal calibration? MinT reconciliation can violate conformal
   coverage guarantees because it shifts the point. Need a clean answer.

---

## Coordination with parallel sessions

- **`services/ingest/` session** (DSLD + OFF Open-Prices): forecast does
  not touch this directory. Anytime that session lands a new silver
  snapshot, update the path defaults in `services/forecast/configs/*.yml`.
- **`services/powerbi/` session** (Power BI assembler + Fabric publish):
  T1.3 (binding runbook) is owned by us; the M-query and DAX implementation
  is owned by them.
- **`clinical-rag-eval` repo** (separate repo): drift module and tracking
  module were lifted from here. If we change those interfaces, propagate
  the change to that repo with a follow-up PR.

---

## Estimated total v0.4 effort

| Tier | Items | Effort | Cumulative |
|---|---|---|---|
| T1 | 4 items (distributed SARIMA, MLflow registry, Power BI doc, Azure silver swap) | ~6 days | 6 days |
| T2 | 4 items (reconciliation, quantile backends, GPU, structural) | ~8 days | 14 days |
| T3 | 8 opportunistic items | ~3 days total | 17 days |

Recommend executing T1 in one focused session (4-6 days actual), then
re-scoping T2 with the operator before committing.
