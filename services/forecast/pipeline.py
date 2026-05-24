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
    seed: int = 42
    track: bool = True
    run_dm_matrix: bool = True              # Diebold-Mariano matrix in metrics JSON


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
    LOG.info("forecast pipeline start name=%s horizon=%d seed=%d", cfg.name, cfg.horizon, cfg.seed)
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
        "backtest_folds": cfg.backtest_folds,
        "metrics_by_method": metrics_by_method,
        "paired_method_comparison": paired_matrix,
        "seed": cfg.seed,
    }
    mpath = Path(cfg.out_metrics_json)
    mpath.parent.mkdir(parents=True, exist_ok=True)
    mpath.write_text(json.dumps(metrics_out, indent=2, default=str))
    LOG.info("wrote metrics -> %s", mpath)

    if cfg.track:
        flat_metrics: dict[str, float] = {}
        for m in metrics_by_method:
            for k in ("mape", "smape", "mase", "coverage_80"):
                if m.get(k) is not None:
                    flat_metrics[f"{m['method']}__{k}"] = float(m[k])
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
                "conformal_alpha": cfg.conformal_alpha,
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
