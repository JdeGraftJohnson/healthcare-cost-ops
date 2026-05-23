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
from services.forecast.eval import rolling_origin_backtest
from services.forecast.intervals import calibrate as calibrate_conformal
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
    )


def _instantiate(name: str, kwargs: dict) -> ForecastModel:
    reg = available_backends()
    if name not in reg:
        raise RuntimeError(
            f"backend {name!r} not available. Installed backends: {list(reg)}. "
            f"Install [forecast] extras: pip install statsmodels prophet lightgbm"
        )
    return reg[name](**kwargs)


def run_pipeline(cfg: PipelineConfig, con: duckdb.DuckDBPyConnection) -> dict:
    import time as _time
    _t0 = _time.time()
    LOG.info("forecast pipeline start name=%s horizon=%d", cfg.name, cfg.horizon)
    panel = load_panel(con, cfg.panel)
    group_cols = list(cfg.panel.group_cols)

    models: dict[str, ForecastModel] = {
        name: _instantiate(name, kw or {}) for name, kw in cfg.backends.items()
    }
    LOG.info("instantiated backends: %s", list(models))

    # Backtest first so MAPEs feed inverse-mape ensemble weights.
    folds, summary = rolling_origin_backtest(
        panel=panel, models=models, group_cols=group_cols,
        time_col=cfg.panel.time_col, target_col=cfg.panel.target_col,
        horizon=cfg.horizon, n_folds=cfg.backtest_folds,
        freq=cfg.panel.freq, season_length=cfg.season_length,
    )
    metrics_by_method = (
        summary.groupby("method")
               .agg(mape=("mape", "median"),
                    smape=("smape", "median"),
                    mase=("mase", "median"),
                    coverage_80=("coverage_80", "mean"),
                    n_series=("method", "count"))
               .reset_index().to_dict(orient="records")
    )
    LOG.info("backtest medians: %s", metrics_by_method)

    # Fit on the full history and forecast the future horizon.
    raw_results: list[ForecastResult] = []
    for name, model in models.items():
        raw_results.extend(model.fit_predict(
            panel, group_cols=group_cols,
            time_col=cfg.panel.time_col, target_col=cfg.panel.target_col,
            horizon=cfg.horizon, freq=cfg.panel.freq,
        ))

    # Ensemble across backends per series.
    ens = ensemble(raw_results, mode=cfg.ensemble_mode, method_name="ensemble")

    # Conformal calibration on the most-recent holdout fold's predictions.
    if cfg.conformal and folds:
        last_fold = folds[-1]
        calib_actual = panel[
            (panel[cfg.panel.time_col] > last_fold.cutoff)
            & (panel[cfg.panel.time_col] <= last_fold.cutoff
               + pd.tseries.frequencies.to_offset(cfg.panel.freq) * cfg.horizon)
        ].copy()
        # Use ensemble point forecasts at calibration cutoff. Refit on truncated
        # train to keep calibration honest.
        train_calib, _ = split_train_test(panel, time_col=cfg.panel.time_col,
                                          holdout_periods=cfg.horizon)
        calib_raw: list[ForecastResult] = []
        for name, model in models.items():
            calib_raw.extend(model.fit_predict(
                train_calib, group_cols=group_cols,
                time_col=cfg.panel.time_col, target_col=cfg.panel.target_col,
                horizon=cfg.horizon, freq=cfg.panel.freq,
            ))
        calib_ens = ensemble(calib_raw, mode=cfg.ensemble_mode, method_name="ensemble")
        calib_pred = {r.series_key: r.point for r in calib_ens}
        cal = calibrate_conformal(
            calib_actual=calib_actual, calib_pred=calib_pred,
            time_col=cfg.panel.time_col, target_col=cfg.panel.target_col,
            group_cols=group_cols, alpha=cfg.conformal_alpha,
        )
        ens = cal.widen(ens)
        LOG.info("conformal calibrated  alpha=%.2f  q_abs_resid=%.4f",
                 cfg.conformal_alpha, cal.quantile_abs_resid)

    # Persist forecasts.
    long = pd.concat([r.to_long_df(group_cols) for r in ens], ignore_index=True)
    out = Path(cfg.out_parquet)
    out.parent.mkdir(parents=True, exist_ok=True)
    long.to_parquet(out, compression="zstd", index=False)
    LOG.info("wrote forecast parquet rows=%d -> %s", len(long), out)

    metrics_out = {
        "name": cfg.name,
        "horizon": cfg.horizon,
        "n_series": int(panel.groupby(group_cols, sort=False).ngroups),
        "backends": list(models.keys()),
        "ensemble_mode": cfg.ensemble_mode,
        "conformal": cfg.conformal,
        "backtest_folds": cfg.backtest_folds,
        "metrics_by_method": metrics_by_method,
    }
    mpath = Path(cfg.out_metrics_json)
    mpath.parent.mkdir(parents=True, exist_ok=True)
    mpath.write_text(json.dumps(metrics_out, indent=2, default=str))
    LOG.info("wrote metrics -> %s", mpath)

    flat_metrics: dict[str, float] = {}
    for m in metrics_by_method:
        for k in ("mape", "smape", "mase", "coverage_80"):
            if m.get(k) is not None:
                flat_metrics[f"{m['method']}__{k}"] = float(m[k])
    track_run(
        name=cfg.name,
        params={
            "horizon": cfg.horizon,
            "backends": list(models.keys()),
            "ensemble_mode": cfg.ensemble_mode,
            "conformal": cfg.conformal,
            "conformal_alpha": cfg.conformal_alpha,
            "backtest_folds": cfg.backtest_folds,
            "season_length": cfg.season_length,
            "freq": cfg.panel.freq,
            "group_cols": list(cfg.panel.group_cols),
            "n_series": metrics_out["n_series"],
        },
        metrics=flat_metrics,
        artifacts={"forecast_parquet": str(out), "metrics_json": str(mpath)},
        started_at=_t0,
    )
    return metrics_out
