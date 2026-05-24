"""Downstream finalize stage of the distributed fan-out.

Reads `forecast_part_*.parquet` from a partition directory (produced by
`caj-forecast-sarima-prod1`), runs the steps the per-partition workers
cannot run because they each only see a slice of the panel:

  1. Concat partial parquets → ensemble per series
  2. Backtest scoring on the dev panel (uses the full panel)
  3. Conformal calibration on the holdout window
  4. Final-test metrics on the untouched holdout
  5. Persist the operator-facing forecast.parquet
  6. Register the bundle in MLflow + run champion-challenger promotion

Designed to run as `caj-forecast-finalize-prod1` — a single-replica CAJ
that fires once after the fan-out fits complete. Local mode is the same
code path with `--no-aca`.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import duckdb
import pandas as pd

from services.forecast.compare import compute_deltas, pick_champion_challenger
from services.forecast.compare import _read_ledger as _read_run_ledger
from services.forecast.data import back_transform, load_panel
from services.forecast.distributed import assemble_partitions
from services.forecast.ensemble import ensemble
from services.forecast.eval import per_period_dm_matrix, rolling_origin_backtest, _score_per_series
from services.forecast.intervals import calibrate_multi as calibrate_multi_conformal
from services.forecast.logging_config import configure as configure_logging
from services.forecast.pipeline import load_config
from services.forecast.registry import register_and_promote
from services.forecast.tracking import track as track_run

LOG = logging.getLogger("forecast.finalize")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="forecast-finalize")
    p.add_argument("--config", required=True, help="YAML config used by the fan-out")
    p.add_argument("--partial-dir", required=True, help="partials written by fan-out workers")
    p.add_argument("--out", required=True, help="final forecast.parquet path")
    p.add_argument("--metrics-out", default=None,
                   help="final metrics JSON path (defaults to alongside --out)")
    p.add_argument("--register", action="store_true",
                   help="register bundle + run champion-challenger promotion")
    p.add_argument("--bundle-path", default=None,
                   help="path to a persisted ModelBundle for registration; required with --register")
    p.add_argument("--champion-tol", type=float, default=0.05,
                   help="fractional improvement required for Production promotion")
    p.add_argument("--primary-metric", default="final_test__sarima__mape",
                   help="metric for champion-challenger comparison; lower is better unless --metric-higher-is-better")
    p.add_argument("--metric-higher-is-better", action="store_true")
    p.add_argument("--name", default=None, help="override registered-model name; defaults to cfg.name")
    p.add_argument("--skip-backtest", action="store_true",
                   help="skip the dev-panel backtest (use only fan-out partials)")
    p.add_argument("--verbose", "-v", action="store_true")
    a = p.parse_args(argv)
    configure_logging(verbose=a.verbose, force=True)

    started = time.time()
    cfg = load_config(a.config)
    group_cols = list(cfg.panel.group_cols)
    out_path = Path(a.out)
    metrics_path = Path(a.metrics_out) if a.metrics_out else out_path.with_suffix(".metrics.json")
    name = a.name or cfg.name
    LOG.info("finalize start  name=%s  partial_dir=%s  out=%s", name, a.partial_dir, out_path)

    # 1. Concat partials. Each row is one (method, series, period) ensemble member.
    partials_df = _read_partials(a.partial_dir, group_cols)
    LOG.info("partials read: %d rows  %d methods  %d series",
             len(partials_df), partials_df["method"].nunique(),
             partials_df.groupby(group_cols).ngroups)

    # 2. Load full panel for backtest + holdout slicing.
    con = duckdb.connect(":memory:")
    panel = load_panel(con, cfg.panel)
    holdout_periods = cfg.final_holdout_periods or cfg.horizon
    dev_panel, holdout_panel = _split_holdout(panel, cfg.panel.time_col, holdout_periods)
    LOG.info("dev_panel rows=%d  holdout_panel rows=%d", len(dev_panel), len(holdout_panel))

    # 3. Ensemble across methods per series from the partials.
    ens_results = _ensemble_from_partials(partials_df, group_cols, cfg.ensemble_mode)
    LOG.info("ensembled %d series", len(ens_results))

    # 4. Multi-α conformal calibration on the holdout.
    multi_q = {}
    if cfg.conformal and not holdout_panel.empty:
        calib_pred = {r.series_key: r.point for r in ens_results}
        cal = calibrate_multi_conformal(
            calib_actual=holdout_panel, calib_pred=calib_pred,
            time_col=cfg.panel.time_col, target_col=cfg.panel.target_col,
            group_cols=group_cols, alphas=tuple(cfg.conformal_alphas),
        )
        ens_results = cal.widen(ens_results)
        multi_q = {f"alpha_{a:.2f}": q for a, q in cal.quantiles_by_alpha.items()}
        LOG.info("conformal q by alpha: %s", multi_q)

    # 5. Final-test metrics on the untouched holdout.
    final_test_by_method = []
    if not holdout_panel.empty:
        ft = _score_per_series(
            results=ens_results, test=holdout_panel,
            group_cols=group_cols, time_col=cfg.panel.time_col,
            target_col=cfg.panel.target_col,
            season_length=cfg.season_length, train=dev_panel,
        )
        if len(ft):
            final_test_by_method = (
                ft.groupby("method")
                  .agg(mape=("mape", "median"), smape=("smape", "median"),
                       mase=("mase", "median"), rmse=("rmse", "median"),
                       mae=("mae", "median"), r2=("r2", "median"),
                       bias=("bias", "median"),
                       coverage_80=("coverage_80", "mean"),
                       coverage_95=("coverage_95", "mean"),
                       n_series=("method", "count"))
                  .reset_index().to_dict(orient="records")
            )

    # 6. Write final forecast.parquet (back-transform log if needed).
    long = pd.concat([r.to_long_df(group_cols) for r in ens_results], ignore_index=True)
    if cfg.panel.log_transform:
        long = back_transform(long, cfg.panel, ["point", "lo80", "hi80", "lo95", "hi95"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    long.to_parquet(out_path, compression="zstd", index=False)
    LOG.info("wrote final forecast: %s (%d rows)", out_path, len(long))

    metrics_out = {
        "name": name,
        "horizon": cfg.horizon,
        "n_series": int(panel.groupby(group_cols).ngroups),
        "conformal_alphas": list(cfg.conformal_alphas),
        "conformal_quantiles": multi_q,
        "final_test_metrics_by_method": final_test_by_method,
        "interval_calibration": "multi_alpha_conformal" if multi_q else "model_native",
        "stage": "finalize",
        "ts": started,
    }
    metrics_path.write_text(json.dumps(metrics_out, indent=2, default=str))
    LOG.info("wrote metrics: %s", metrics_path)

    # 7. Track + register + promote.
    flat: dict[str, float] = {}
    for m in final_test_by_method:
        method = m["method"]
        for k in ("mape", "smape", "mase", "rmse", "mae", "r2", "bias",
                  "coverage_80", "coverage_95"):
            if m.get(k) is not None:
                flat[f"final_test__{method}__{k}"] = float(m[k])
    rec = track_run(
        name=name,
        params={"horizon": cfg.horizon, "stage": "finalize",
                "partial_dir": str(a.partial_dir),
                "group_cols": list(cfg.panel.group_cols)},
        metrics=flat,
        artifacts={"forecast_parquet": str(out_path), "metrics_json": str(metrics_path)},
        ledger_path=f"logs/forecast_runs__{name}.jsonl",
        started_at=started,
    )

    if a.register:
        if not a.bundle_path:
            LOG.warning("--register set but --bundle-path missing; skipping registry")
        else:
            metric_value = flat.get(a.primary_metric)
            if metric_value is None:
                LOG.warning("primary-metric %s not in finalized metrics; skipping registry",
                            a.primary_metric)
            else:
                register_and_promote(
                    name=name, run_id=rec.run_id,
                    bundle_path=a.bundle_path,
                    metric_name=a.primary_metric, metric_value=metric_value,
                    champion_tol=a.champion_tol,
                    higher_is_better=a.metric_higher_is_better,
                )

    return 0


def _read_partials(partial_dir: str | Path, group_cols: list[str]) -> pd.DataFrame:
    d = Path(partial_dir)
    files = sorted(d.glob("forecast_part_*.parquet"))
    if not files:
        raise RuntimeError(f"no forecast_part_*.parquet in {d}")
    frames = []
    for f in files:
        if f.stat().st_size == 0:
            continue
        frames.append(pd.read_parquet(f))
    if not frames:
        raise RuntimeError(f"all partials empty in {d}")
    return pd.concat(frames, ignore_index=True)


def _split_holdout(panel: pd.DataFrame, time_col: str, holdout_periods: int):
    if holdout_periods <= 0:
        return panel, panel.iloc[0:0].copy()
    times = sorted(panel[time_col].unique())
    if len(times) <= holdout_periods:
        return panel, panel.iloc[0:0].copy()
    cutoff = times[-holdout_periods]
    return (panel[panel[time_col] < cutoff].copy(),
            panel[panel[time_col] >= cutoff].copy())


def _ensemble_from_partials(
    df: pd.DataFrame, group_cols: list[str], mode: str,
):
    """Reconstruct ForecastResult per (series, method), then run ensemble().

    The partial parquets carry one row per (method, series, period) with
    columns: <group_cols>, period, point, lo80, hi80, lo95, hi95, method,
    in_sample_mape.
    """
    from services.forecast.base import ForecastResult
    raw: list = []
    for keys_method, sub in df.groupby(group_cols + ["method"], sort=False):
        key = tuple(keys_method[:-1] if isinstance(keys_method, tuple) else [keys_method[:-1]])
        method = keys_method[-1]
        sub = sub.sort_values("period")
        idx = pd.DatetimeIndex(sub["period"])
        raw.append(ForecastResult(
            series_key=key, method=method, horizon=len(sub),
            point=pd.Series(sub["point"].values, index=idx),
            lo80=pd.Series(sub["lo80"].values, index=idx) if "lo80" in sub else None,
            hi80=pd.Series(sub["hi80"].values, index=idx) if "hi80" in sub else None,
            lo95=pd.Series(sub["lo95"].values, index=idx) if "lo95" in sub else None,
            hi95=pd.Series(sub["hi95"].values, index=idx) if "hi95" in sub else None,
            in_sample_mape=float(sub["in_sample_mape"].iloc[0]) if "in_sample_mape" in sub and not pd.isna(sub["in_sample_mape"].iloc[0]) else None,
            metadata={},
        ))
    return ensemble(raw, mode=mode, method_name="ensemble")


if __name__ == "__main__":
    sys.exit(main())
