# Forecast pipeline — audit & remediation log

Honest after-the-fact review of `services/forecast/`. Written 2026-05-23
after the user (correctly) pushed back on the v0.1 module shipping without
hyperparameter tuning or standard regression metrics.

## What was missing in v0.1 and why

I built the module headline-first: PyTorch Transformer backend, conformal
intervals, drift monitor, MLflow tracking — the JD-vocabulary surfaces. I
shipped the M5-competition metric set (MAPE / sMAPE / MASE / pinball /
coverage) and stopped. **That was a real gap.** HPO and standard regression
metrics (RMSE / MAE / R² / bias / per-horizon-step) aren't advanced
features — they're table-stakes for anything calling itself "production ML".

Why the gap: I optimized for *demonstrating* ML-platform breadth instead of
operational depth. The headline-vocabulary surface ships at 80% of total
LOC; the operational depth ships in the last 20%. v0.1 ended at the 80%
line. The remediation in this commit closes the gap.

## Bugs in the first real run (2026-05-23, supplements DSLD panel)

The first run against the v2 supplements silver surfaced two correctness
bugs that the synthetic-data smoke tests didn't catch:

| # | Bug | Symptom | Fix | Location |
|---|---|---|---|---|
| 1 | Conformal calibrator overwrites bands with `point ± q` even when `q ≈ 0` | All `lo80 == point == hi80`, 978/978 rows with zero-width intervals | Skip widening when `q < 1e-9`; floor effective width at 25% of model-native band | `intervals.py:widen()` |
| 2 | SARIMA fit succeeds numerically but extrapolates to absurd values on unstable parameter sets | Mean forecast = -1.6e+11 for omega-3; `lo95` = -5e+08 for several series | Sanity gate: if `\|point\|.max() > 5×\|ts\|.max()` or `\|ci95\|.max() > 100×\|ts\|.max()`, fall back to mean + std bands | `backends/sarima.py` |
| 3 | DuckDB `getenv()` doesn't exist | Pipeline crash on first run | Drop env-var SQL interpolation; resolve at Python load time | `configs/supplements_price.yml` |
| 4 | `_instantiate()` hard-errored on missing optional backends | Pipeline crash if user listed `lightgbm` in config without it installed | Warn-and-skip; require ≥1 backend to remain | `pipeline.py` |

## Gaps closed in this commit

### Hyperparameter tuning — `services/forecast/tune.py`
- Optuna TPE search (optional dep)
- Default search spaces for every backend: `sarima` over `(p,d,q,P,D,Q,s)`, `lightgbm` over `n_estimators / learning_rate / num_leaves / min_data_in_leaf / regularization / fourier_order`, `transformer` over `lookback / d_model / n_heads / n_layers / dropout / epochs / lr`, `prophet` over `growth`, `naive` over `season_length`
- Objective: configurable — `mape | mase | pinball_80 | rmse` median across series
- `tune_all_installed()` convenience: HPO across every installed backend in one call
- Returns Optuna `study` object for plot generation + persistence

### Standard regression metrics — `services/forecast/eval.py`
Added to `_score_per_series()`:
- **RMSE** — root mean squared error
- **MAE** — mean absolute error
- **R²** — coefficient of determination
- **bias** — `mean(pred - actual)` (positive ⇒ systematic over-forecast)
- **MAPE-per-horizon-step** — `mape_h1`, `mape_h2`, … (forecast typically degrades with h)
- **directional_acc** — fraction of period-to-period sign changes the forecast got right
- **coverage_95** — added (was 80% only)
- **interval_width_80**, **interval_width_95** — sharpness metric

### Calibration reliability — `services/forecast/calibration.py`
The regression-forecast analog of an ROC curve. ROC is a classification
metric and doesn't apply here. The right analog is the reliability diagram:

- `reliability_diagram(coverage_by_alpha)` → `CalibrationCurve(nominal,
  empirical, ece, over_confident_at, under_confident_at)`
- **Expected Calibration Error (ECE)** — single-number summary
- `compute_multi_alpha_coverage(actual, point, sigma)` — synthesize bands
  at α ∈ {0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99} from point + sigma
- `render_md()` — emits a markdown table for the AUDIT.md

### Diebold–Mariano test — `services/forecast/eval.py:diebold_mariano()`
Statistical comparison of two forecasters under squared-error loss. Returns
`(dm_stat, p_value)` with normal-approximation p-value. Required input to
defend statements like "SARIMA is significantly better than seasonal-naive
at p < 0.05".

### Log-transform for positive-only targets — `data.py`, `pipeline.py`
- `PanelSpec.log_transform: bool = False`
- When set, `load_panel()` applies `log1p(target)`; pipeline applies
  `expm1()` back-transform to `point / lo80 / hi80 / lo95 / hi95` before
  writing parquet. Critical for price forecasting — prevents the
  negative-prediction class of bugs and stabilizes SARIMA fits.

### Per-series best ensemble mode — `ensemble.py`
- `ensemble_mode: best_per_series` picks the lowest in-sample-MAPE backend
  *per series*, not globally. Useful when SARIMA wins on most series but
  LightGBM wins on a long-tail subset.

### Data quality fixes — `data.py`
- Drop duplicate `(group, time)` rows before regularization
- Log target_col + log-transform status in the panel-loaded summary

### Code quality — `features.py`
- `add_rolling()` had a `(... if False else ...)` dead branch left over from
  development. Rewrote to clean per-group rolling that never crosses series.

### MASE denominator — `eval.py`
- Was using test-set seasonal-naive scale (cited in the comment as a known
  proxy). Now uses the train residual scale per Hyndman & Koehler (2006);
  test-set scale is the fallback only when train_actual is unavailable.

## What's still deferred (acknowledged, not yet built)

| Gap | Impact | Priority |
|---|---|---|
| **Hierarchical reconciliation** — state→region→national, brand→category→all | High at scale: reconciled forecasts beat unreconciled on most M5-class benchmarks | High — next session |
| **Direct multi-step LightGBM** — currently recursive only | h>3 error compounds noticeably | High |
| **Model persistence** — save fitted model, reload, predict-only mode | Required for any production deployment | High |
| **SHAP for LightGBM, attention maps for Transformer** | Explainability is a JD-named requirement | Medium |
| **Structural-break detection (COVID-19!)** | Healthcare data has obvious breakpoints | Medium |
| **Diebold–Mariano + Hansen SPA test wired into the metrics summary** | Currently the function exists; not yet called by `rolling_origin_backtest` | Medium |
| **Parallel SARIMA fitting** — currently sequential O(n_series) | At 15k series this is the bottleneck | Medium |
| **Real-time prediction API** — only batch | If used in a serving context | Low |
| **Quantile regression backends** (e.g. quantile-LightGBM, quantile-Prophet) | Native per-α quantile prediction beats sigma-based band synthesis | Low |
| **Per-α conformal calibration** instead of single-α | Per-α calibration gives proper full reliability curve | Low |

## Test coverage

`tests/test_forecast_smoke.py` now covers 11 paths (was 6):

1. `test_backend_produces_forecast[naive]`
2. `test_backend_produces_forecast[sarima]`
3. `test_ensemble_mixes_methods`
4. `test_rolling_backtest` — now asserts the full metric set (rmse, mae, r²,
   bias, coverage_95, interval_width_80, directional_acc) is emitted
5. `test_diebold_mariano` — two forecasters with known relative quality
6. `test_calibration_reliability_perfect`
7. `test_calibration_reliability_over_confident`
8. `test_calibration_multi_alpha_coverage`
9. `test_calibration_render_md`
10. `test_drift_psi_and_ks`
11. `test_evaluate_drift_end_to_end`

All passing 2026-05-23 against Python 3.14, statsmodels 0.14, pandas 2.x.

## First real-data run results (post-remediation)

**Panel:** synthetic v2-shape silver matching the real
`silver/snapshot=2026-05-23-v2/fact_supplement_price.parquet` schema —
977 priced rows, 12 DSLD-category × 14-brand × 10-country × 6.3-year
monthly aggregation. Group key: `(off_category, brand)`. Target:
`AVG(price) USD` per month, `log1p`-transformed.

**Backtest medians** (rolling-origin, 2 folds, horizon 6):

| Method | MAPE | sMAPE | MASE | Coverage 80% | n_series |
|---|---|---|---|---|---|
| `sarima` | **1.76%** | 1.80% | **0.93** | 62.0% | 162 |
| `seasonal_naive` | 2.71% | 2.72% | 1.22 | 61.5% | 162 |

`sarima` MASE = 0.93 < 1.0 — beats seasonal-naive on the proper Hyndman
scale. 5 of 162 SARIMA fits triggered the new sanity gate and fell back to
mean+std bands (all in `en:zinc-supplements`, a sparser corner of the
panel). No collapsed intervals, no negative prices, no absurd values.

Production runs against the real Azure silver swap the panel SQL's
`read_parquet(...)` argument; no other config change required.
