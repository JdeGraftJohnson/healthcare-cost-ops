"""Config-driven runner: load → backtest → fit → ensemble → conformal → write."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import duckdb
import pandas as pd
import yaml

from services.forecast.backends import available as available_backends
from services.forecast.base import ForecastModel, ForecastResult
from services.forecast.data import PanelSpec, load_panel, split_train_test
from services.forecast.ensemble import ensemble
from services.forecast.eval import rolling_origin_backtest, _score_per_series
from services.forecast.intervals import (
    calibrate as calibrate_conformal,
    calibrate_multi as calibrate_multi_conformal,
    calibrate_empirical_bands,
)
from services.forecast.tracking import track as track_run

LOG = logging.getLogger("forecast.pipeline")


@dataclass(frozen=True)
class PipelineConfig:
    name: str
    panel: PanelSpec
    horizon: int
    backends: dict[str, dict]              # name -> kwargs
    ensemble_mode: str
    backtest_folds: int
    conformal: bool
    conformal_alpha: float
    season_length: int
    out_parquet: str
    out_metrics_json: str
    seed: int = 42
    track: bool = True
    run_dm_matrix: bool = True              # Diebold-Mariano matrix in metrics JSON
    final_holdout_periods: int = 0          # 0 = backwards-compatible (no holdout).
                                            # >0 = carve final-test window before backtest;
                                            # final-test metrics are reported separately
                                            # and conformal calibration runs on this window.
    conformal_alphas: tuple = (0.20,)       # multi-alpha calibration: tuple of alphas
                                            # in (0, 1). Each gets its own conformal q
                                            # and feeds the reliability diagram.


def load_config(path: str | Path) -> PipelineConfig:
    raw = yaml.safe_load(Path(path).read_text())
    panel = PanelSpec(
        sql=raw["panel"]["sql"],
        group_cols=tuple(raw["panel"]["group_cols"]),
        time_col=raw["panel"]["time_col"],
        target_col=raw["panel"]["target_col"],
        freq=raw["panel"]["freq"],
        min_obs_per_series=int(raw["panel"].get("min_obs_per_series", 24)),
        fill_gaps=raw["panel"].get("fill_gaps", "zero"),
        log_transform=bool(raw["panel"].get("log_transform", False)),
    )
    return PipelineConfig(
        name=raw["name"],
        panel=panel,
        horizon=int(raw["horizon"]),
        backends=raw.get("backends", {"naive": {}}),
        ensemble_mode=raw.get("ensemble_mode", "inverse_mape"),
        backtest_folds=int(raw.get("backtest_folds", 3)),
        conformal=bool(raw.get("conformal", True)),
        conformal_alpha=float(raw.get("conformal_alpha", 0.20)),
        season_length=int(raw.get("season_length", 12)),
        out_parquet=raw["out_parquet"],
        out_metrics_json=raw["out_metrics_json"],
        seed=int(raw.get("seed", 42)),
        track=bool(raw.get("track", True)),
        run_dm_matrix=bool(raw.get("run_dm_matrix", True)),
        final_holdout_periods=int(raw.get("final_holdout_periods", 0)),
        conformal_alphas=tuple(
            float(a) for a in raw.get("conformal_alphas", [raw.get("conformal_alpha", 0.20)])
        ),
    )


def _instantiate(name: str, kwargs: dict) -> ForecastModel | None:
    import inspect
    reg = available_backends()
    if name not in reg:
        LOG.warning(
            "backend %r requested but not installed (have %s). Skipping. "
            "Install [forecast] extras to enable.", name, list(reg),
        )
        return None
    cls = reg[name]
    # Filter kwargs to those the backend's __init__ actually accepts.
    accepted = set(inspect.signature(cls.__init__).parameters.keys()) - {"self"}
    safe = {k: v for k, v in kwargs.items() if k in accepted}
    return cls(**safe)


def run_pipeline(cfg: PipelineConfig, con: duckdb.DuckDBPyConnection) -> dict:
    import time as _time
    import random as _random
    import numpy as _np
    _random.seed(cfg.seed); _np.random.seed(cfg.seed)
    try:
        import torch as _torch
        _torch.manual_seed(cfg.seed)
    except ImportError:
        pass
    _t0 = _time.time()
    LOG.info("forecast pipeline start name=%s horizon=%d seed=%d holdout=%d",
             cfg.name, cfg.horizon, cfg.seed, cfg.final_holdout_periods)
    panel = load_panel(con, cfg.panel)
    group_cols = list(cfg.panel.group_cols)

    models: dict[str, ForecastModel] = {}
    for name, kw in cfg.backends.items():
        kw = dict(kw or {})
        if "seed" not in kw:
            kw["seed"] = cfg.seed
        inst = _instantiate(name, kw)
        if inst is not None:
            models[name] = inst
    if not models:
        raise RuntimeError("no forecast backends available — install at least one")
    LOG.info("instantiated backends: %s", list(models))

    # ────────────────────────────────────────────────────────────────────────
    # Final-test holdout (Issue 1.1 + 1.3): carve a window from the panel tail
    # that NEITHER backtest NOR conformal calibration can see. The holdout is
    # used once at the end for unbiased out-of-sample metrics + as the
    # conformal calibration window.
    # ────────────────────────────────────────────────────────────────────────
    dev_panel, holdout_panel = _split_final_holdout(
        panel, time_col=cfg.panel.time_col,
        holdout_periods=cfg.final_holdout_periods,
    )
    LOG.info("dev_panel rows=%d  holdout_panel rows=%d (holdout_periods=%d)",
             len(dev_panel), len(holdout_panel), cfg.final_holdout_periods)

    # Backtest on dev only. MAPEs from here feed ensemble weights and the
    # paired-comparison matrix. Holdout is untouched.
    folds, summary = rolling_origin_backtest(
        panel=dev_panel, models=models, group_cols=group_cols,
        time_col=cfg.panel.time_col, target_col=cfg.panel.target_col,
        horizon=cfg.horizon, n_folds=cfg.backtest_folds,
        freq=cfg.panel.freq, season_length=cfg.season_length,
    )
    metrics_by_method = (
        summary.groupby("method")
               .agg(mape=("mape", "median"),
                    smape=("smape", "median"),
                    mase=("mase", "median"),
                    rmse=("rmse", "median"),
                    mae=("mae", "median"),
                    r2=("r2", "median"),
                    bias=("bias", "median"),
                    directional_acc=("directional_acc", "mean"),
                    coverage_80=("coverage_80", "mean"),
                    coverage_95=("coverage_95", "mean"),
                    n_series=("method", "count"))
               .reset_index().to_dict(orient="records")
    )
    LOG.info("backtest medians: %s", metrics_by_method)

    # ────────────────────────────────────────────────────────────────────────
    # Fit on dev_panel and produce holdout-horizon forecasts. These feed:
    #   (1) empirical-quantile band calibration per backend
    #   (2) multi-α conformal calibration on ensemble
    #   (3) final-test metrics (the unbiased out-of-sample scorecard)
    # ────────────────────────────────────────────────────────────────────────
    holdout_horizon = cfg.final_holdout_periods or cfg.horizon
    raw_for_holdout: list[ForecastResult] = []
    for name, model in models.items():
        try:
            raw_for_holdout.extend(model.fit_predict(
                dev_panel, group_cols=group_cols,
                time_col=cfg.panel.time_col, target_col=cfg.panel.target_col,
                horizon=holdout_horizon, freq=cfg.panel.freq,
            ))
        except Exception as e:
            LOG.warning("backend %s failed on dev_panel: %s", name, e)

    # Empirical-quantile band calibration (Issue 1.4): replaces sigma-based
    # bands with quantiles of the actual residual distribution per backend.
    # Residuals are computed against the holdout if available, else against
    # the latest backtest fold's predictions.
    band_cal = None
    if not holdout_panel.empty and raw_for_holdout:
        resid_df = _residuals_from_holdout(
            raw_for_holdout, holdout_panel,
            group_cols=group_cols, time_col=cfg.panel.time_col,
            target_col=cfg.panel.target_col,
        )
        if len(resid_df):
            band_cal = calibrate_empirical_bands(resid_df, alphas=(0.20, 0.05))
            LOG.info("empirical-quantile bands calibrated for methods: %s",
                     list(band_cal.quantiles_by_method.keys()))

    # Final-test metrics: score the dev-panel-fit forecasts against the
    # held-out actual values. Single shot — the holdout is then "spent".
    final_test_summary = pd.DataFrame()
    if not holdout_panel.empty and raw_for_holdout:
        final_test_summary = _score_per_series(
            results=raw_for_holdout, test=holdout_panel,
            group_cols=group_cols, time_col=cfg.panel.time_col,
            target_col=cfg.panel.target_col,
            season_length=cfg.season_length, train=dev_panel,
        )
    final_test_by_method = (
        final_test_summary.groupby("method")
                          .agg(mape=("mape", "median"),
                               smape=("smape", "median"),
                               mase=("mase", "median"),
                               rmse=("rmse", "median"),
                               mae=("mae", "median"),
                               r2=("r2", "median"),
                               bias=("bias", "median"),
                               directional_acc=("directional_acc", "mean"),
                               coverage_80=("coverage_80", "mean"),
                               coverage_95=("coverage_95", "mean"),
                               n_series=("method", "count"))
                          .reset_index().to_dict(orient="records")
        if len(final_test_summary) else []
    )
    if final_test_by_method:
        LOG.info("FINAL-TEST medians (out-of-sample, holdout untouched): %s",
                 final_test_by_method)

    # ────────────────────────────────────────────────────────────────────────
    # Fit on the FULL panel (dev + holdout) and forecast cfg.horizon periods
    # past the panel end. This is the operator-facing forecast.
    # ────────────────────────────────────────────────────────────────────────
    raw_results: list[ForecastResult] = []
    fallback_series: set = set()
    for name, model in models.items():
        try:
            raw_results.extend(model.fit_predict(
                panel, group_cols=group_cols,
                time_col=cfg.panel.time_col, target_col=cfg.panel.target_col,
                horizon=cfg.horizon, freq=cfg.panel.freq,
            ))
        except Exception as e:
            LOG.warning("backend %s failed on full panel: %s", name, e)

    # Apply empirical-quantile bands (replaces sigma-based bands per backend).
    if band_cal is not None:
        raw_results = band_cal.apply(raw_results)

    # Ensemble across backends per series.
    ens = ensemble(raw_results, mode=cfg.ensemble_mode, method_name="ensemble")

    # Graceful all-backends-fail (Issue 4.3): every series in the dev panel
    # must produce at least one forecast row. Series that don't get a
    # naive_last_value fallback with metadata.fallback_reason.
    ens = _ensure_coverage(
        ens, panel=panel, group_cols=group_cols,
        time_col=cfg.panel.time_col, target_col=cfg.panel.target_col,
        horizon=cfg.horizon, freq=cfg.panel.freq, fallback_series_out=fallback_series,
    )

    # Multi-α conformal calibration on the holdout window (Issue 1.3 + 5.4).
    multi_q: dict = {}
    if cfg.conformal and not holdout_panel.empty and raw_for_holdout:
        calib_ens = ensemble(raw_for_holdout, mode=cfg.ensemble_mode, method_name="ensemble")
        calib_pred = {r.series_key: r.point for r in calib_ens}
        cal_multi = calibrate_multi_conformal(
            calib_actual=holdout_panel, calib_pred=calib_pred,
            time_col=cfg.panel.time_col, target_col=cfg.panel.target_col,
            group_cols=group_cols, alphas=tuple(cfg.conformal_alphas),
        )
        ens = cal_multi.widen(ens)
        multi_q = {f"alpha_{a:.2f}": q for a, q in cal_multi.quantiles_by_alpha.items()}
        LOG.info("multi-alpha conformal calibrated: %s", multi_q)
    elif cfg.conformal and folds:
        # Back-compat: no holdout configured, fall back to original fold-based path.
        last_fold = folds[-1]
        calib_actual = dev_panel[
            (dev_panel[cfg.panel.time_col] > last_fold.cutoff)
            & (dev_panel[cfg.panel.time_col] <= last_fold.cutoff
               + pd.tseries.frequencies.to_offset(cfg.panel.freq) * cfg.horizon)
        ].copy()
        train_calib, _ = split_train_test(
            dev_panel, time_col=cfg.panel.time_col, holdout_periods=cfg.horizon,
        )
        calib_raw: list[ForecastResult] = []
        for name, model in models.items():
            try:
                calib_raw.extend(model.fit_predict(
                    train_calib, group_cols=group_cols,
                    time_col=cfg.panel.time_col, target_col=cfg.panel.target_col,
                    horizon=cfg.horizon, freq=cfg.panel.freq,
                ))
            except Exception as e:
                LOG.warning("calib refit %s failed: %s", name, e)
        calib_ens = ensemble(calib_raw, mode=cfg.ensemble_mode, method_name="ensemble")
        calib_pred = {r.series_key: r.point for r in calib_ens}
        cal = calibrate_conformal(
            calib_actual=calib_actual, calib_pred=calib_pred,
            time_col=cfg.panel.time_col, target_col=cfg.panel.target_col,
            group_cols=group_cols, alpha=cfg.conformal_alphas[0],
        )
        ens = cal.widen(ens)
        LOG.info("single-alpha conformal (back-compat) calibrated alpha=%.2f q=%.4f",
                 cfg.conformal_alphas[0], cal.quantile_abs_resid)

    # Persist forecasts.
    long = pd.concat([r.to_long_df(group_cols) for r in ens], ignore_index=True)
    # Back-transform from log space so the persisted forecast is in original units.
    if cfg.panel.log_transform:
        from services.forecast.data import back_transform
        long = back_transform(long, cfg.panel, ["point", "lo80", "hi80", "lo95", "hi95"])
        LOG.info("back-transformed log forecast to original units")
    out = Path(cfg.out_parquet)
    out.parent.mkdir(parents=True, exist_ok=True)
    long.to_parquet(out, compression="zstd", index=False)
    LOG.info("wrote forecast parquet rows=%d -> %s", len(long), out)

    # Paired MAPE comparison matrix on the most-recent fold's per-series MAPE.
    # NOT the canonical Diebold-Mariano test — DM requires per-period
    # (actual, p_a, p_b) tuples which the current backtest summary does not
    # retain. This is a paired Wilcoxon-shape test on per-series MAPE
    # differences: positive median_diff means `a` has lower MAPE than `b`.
    # Wire the true DM through after refactoring backtest to retain actuals.
    paired_matrix: list[dict] = []
    if cfg.run_dm_matrix and folds:
        last = folds[-1]
        methods = sorted(last.per_series["method"].unique())
        mape_by_method: dict[str, "pd.Series"] = {}
        for m in methods:
            sub = last.per_series[last.per_series["method"] == m].set_index(group_cols)
            mape_by_method[m] = sub["mape"].dropna()
        for i, a in enumerate(methods):
            for b in methods[i + 1 :]:
                ma, mb = mape_by_method[a], mape_by_method[b]
                # Align on common series.
                common = ma.index.intersection(mb.index)
                if len(common) < 4:
                    continue
                diff = (ma.loc[common] - mb.loc[common]).values
                median_diff = float(pd.Series(diff).median())
                # Wilcoxon signed-rank approximation: count of negative diffs
                # under H0 = 0.5 → normal approx p-value.
                n = len(diff)
                neg = int((diff < 0).sum())
                pos = int((diff > 0).sum())
                from math import sqrt as _sqrt, erf as _erf
                if neg + pos > 0:
                    se = _sqrt((neg + pos) * 0.25)
                    z = (neg - 0.5 * (neg + pos)) / max(se, 1e-9)
                    p = 2.0 * (1.0 - 0.5 * (1.0 + _erf(abs(z) / _sqrt(2.0))))
                else:
                    z, p = 0.0, 1.0
                paired_matrix.append({
                    "a": a, "b": b, "n_paired": int(len(common)),
                    "median_mape_diff": median_diff, "z": float(z), "p_value": float(p),
                    "test": "paired_sign",
                    "note": "median_diff < 0 means a's MAPE is lower than b's",
                })

    metrics_out = {
        "name": cfg.name,
        "horizon": cfg.horizon,
        "n_series": int(panel.groupby(group_cols, sort=False).ngroups),
        "backends": list(models.keys()),
        "ensemble_mode": cfg.ensemble_mode,
        "conformal": cfg.conformal,
        "conformal_alphas": list(cfg.conformal_alphas),
        "conformal_quantiles": multi_q if 'multi_q' in dir() else {},
        "final_holdout_periods": cfg.final_holdout_periods,
        "backtest_folds": cfg.backtest_folds,
        "backtest_metrics_by_method": metrics_by_method,
        "final_test_metrics_by_method": final_test_by_method,
        "interval_calibration": (
            "empirical_quantile" if band_cal is not None else "model_native"
        ),
        "paired_method_comparison": paired_matrix,
        "n_fallback_series": len(fallback_series),
        "seed": cfg.seed,
    }
    # Backwards-compat alias (downstream readers expected this key).
    metrics_out["metrics_by_method"] = metrics_by_method
    mpath = Path(cfg.out_metrics_json)
    mpath.parent.mkdir(parents=True, exist_ok=True)
    mpath.write_text(json.dumps(metrics_out, indent=2, default=str))
    LOG.info("wrote metrics -> %s", mpath)

    if cfg.track:
        flat_metrics: dict[str, float] = {}
        for m in metrics_by_method:
            for k in ("mape", "smape", "mase", "rmse", "mae", "r2", "bias",
                      "coverage_80", "coverage_95", "directional_acc"):
                if m.get(k) is not None:
                    flat_metrics[f"backtest__{m['method']}__{k}"] = float(m[k])
        for m in final_test_by_method:
            for k in ("mape", "smape", "mase", "rmse", "mae", "r2", "bias",
                      "coverage_80", "coverage_95", "directional_acc"):
                if m.get(k) is not None:
                    flat_metrics[f"final_test__{m['method']}__{k}"] = float(m[k])
        flat_metrics["n_fallback_series"] = float(len(fallback_series))
        # Per-pipeline ledger file: avoids cross-contamination between runs of
        # different pipelines in the same logs/ dir.
        ledger_path = f"logs/forecast_runs__{cfg.name}.jsonl"
        track_run(
            name=cfg.name,
            params={
                "horizon": cfg.horizon,
                "backends": list(models.keys()),
                "ensemble_mode": cfg.ensemble_mode,
                "conformal": cfg.conformal,
                "conformal_alphas": list(cfg.conformal_alphas),
                "final_holdout_periods": cfg.final_holdout_periods,
                "interval_calibration": metrics_out["interval_calibration"],
                "backtest_folds": cfg.backtest_folds,
                "season_length": cfg.season_length,
                "freq": cfg.panel.freq,
                "group_cols": list(cfg.panel.group_cols),
                "n_series": metrics_out["n_series"],
                "seed": cfg.seed,
            },
            metrics=flat_metrics,
            artifacts={"forecast_parquet": str(out), "metrics_json": str(mpath)},
            ledger_path=ledger_path,
            started_at=_t0,
        )
    return metrics_out


# ────────────────────────────────────────────────────────────────────────────
# Helpers introduced by the audit-queue remediation
# ────────────────────────────────────────────────────────────────────────────

def _split_final_holdout(
    panel: pd.DataFrame, *, time_col: str, holdout_periods: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carve the most-recent N unique periods from the panel as the holdout.

    Returns (dev_panel, holdout_panel). When `holdout_periods == 0`, the
    holdout is empty and `dev_panel == panel` (backwards-compatible).
    """
    if holdout_periods <= 0:
        return panel, panel.iloc[0:0].copy()
    times = sorted(panel[time_col].unique())
    if len(times) <= holdout_periods:
        raise RuntimeError(
            f"panel has {len(times)} unique periods but final_holdout_periods={holdout_periods}"
            " — pick a smaller holdout or grow the panel"
        )
    cutoff = times[-holdout_periods]
    dev = panel[panel[time_col] < cutoff].copy()
    holdout = panel[panel[time_col] >= cutoff].copy()
    return dev, holdout


def _residuals_from_holdout(
    results: list[ForecastResult], holdout_panel: pd.DataFrame, *,
    group_cols: list[str], time_col: str, target_col: str,
) -> pd.DataFrame:
    """Produce a flat (method, residual) DataFrame for empirical-band calibration.

    residual = actual - prediction. One row per (method, series, period).
    """
    rows: list[dict] = []
    actuals = {
        tuple(k if isinstance(k, tuple) else (k,)): g.set_index(time_col)[target_col].astype(float)
        for k, g in holdout_panel.groupby(group_cols, sort=False)
    }
    for r in results:
        a = actuals.get(r.series_key)
        if a is None or len(a) == 0: continue
        common = a.index.intersection(r.point.index)
        if len(common) == 0: continue
        for t in common:
            rows.append({"method": r.method, "residual": float(a.loc[t] - r.point.loc[t])})
    return pd.DataFrame(rows)


def _ensure_coverage(
    ens: list[ForecastResult], *, panel: pd.DataFrame,
    group_cols: list[str], time_col: str, target_col: str,
    horizon: int, freq: str, fallback_series_out: set,
) -> list[ForecastResult]:
    """Every series in the panel must end up with a ForecastResult. If the
    ensemble produced none for a series (because every backend errored or
    the ensemble dropped it), emit a naive-last-value fallback marked with
    `metadata.fallback_reason='all_backends_failed'`.
    """
    have = {r.series_key for r in ens}
    expected = set()
    for k, _ in panel.groupby(group_cols, sort=False):
        expected.add(tuple(k if isinstance(k, tuple) else (k,)))
    missing = expected - have
    if not missing:
        return ens
    offset = pd.tseries.frequencies.to_offset(freq)
    panel_by_key = {tuple(k if isinstance(k, tuple) else (k,)): g for k, g in panel.groupby(group_cols, sort=False)}
    for key in missing:
        grp = panel_by_key[key]
        ts = grp.sort_values(time_col).set_index(time_col)[target_col].astype(float)
        if ts.empty:
            continue
        last_t = ts.index[-1]
        last_v = float(ts.iloc[-1])
        future_idx = pd.date_range(last_t + offset, periods=horizon, freq=freq)
        point = pd.Series([last_v] * horizon, index=future_idx)
        sigma = float(ts.std(ddof=1)) if len(ts) > 1 else abs(last_v) * 0.15 or 1.0
        ens.append(ForecastResult(
            series_key=key, method="ensemble", horizon=horizon,
            point=point,
            lo80=point - 1.2816 * sigma, hi80=point + 1.2816 * sigma,
            lo95=point - 1.96 * sigma,   hi95=point + 1.96 * sigma,
            in_sample_mape=None,
            metadata={"fallback_reason": "all_backends_failed", "fallback_method": "naive_last_value"},
        ))
        fallback_series_out.add(key)
    if missing:
        LOG.warning("emitted naive_last_value fallback for %d series (all backends failed)",
                    len(missing))
    return ens
