"""Backtest harness — rolling-origin CV + accuracy/calibration metrics."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from services.forecast.base import ForecastModel, ForecastResult, SeriesKey

LOG = logging.getLogger("forecast.eval")


@dataclass
class FoldResult:
    fold: int
    cutoff: pd.Timestamp
    per_series: pd.DataFrame  # columns: group_cols + [method, mape, smape, mase, pinball_80, coverage_80]


def diebold_mariano(
    actual: np.ndarray, p_a: np.ndarray, p_b: np.ndarray, *, h: int = 1
) -> tuple[float, float]:
    """Two-sided Diebold-Mariano test for equal forecast accuracy under
    squared-error loss. Returns (dm_stat, p_value).
    """
    n = len(actual)
    if n < 4:
        return float("nan"), float("nan")
    d = (actual - p_a) ** 2 - (actual - p_b) ** 2
    d_bar = float(np.mean(d))
    # Long-run variance using h-1 lag terms.
    gamma0 = float(np.var(d, ddof=1))
    if gamma0 <= 0:
        return float("nan"), float("nan")
    gamma_sum = 0.0
    for k in range(1, h):
        if k >= n: break
        gk = float(np.mean((d[k:] - d_bar) * (d[:-k] - d_bar)))
        gamma_sum += 2 * gk
    var_d = (gamma0 + gamma_sum) / n
    if var_d <= 0:
        return float("nan"), float("nan")
    dm = d_bar / np.sqrt(var_d)
    # Two-sided normal approximation.
    from math import erf, sqrt as _sqrt
    p_value = 2.0 * (1.0 - 0.5 * (1.0 + erf(abs(dm) / _sqrt(2.0))))
    return float(dm), float(p_value)


def rolling_origin_backtest(
    panel: pd.DataFrame,
    models: dict[str, ForecastModel],
    *,
    group_cols: Sequence[str],
    time_col: str,
    target_col: str,
    horizon: int,
    n_folds: int,
    freq: str,
    season_length: int = 12,
) -> tuple[list[FoldResult], pd.DataFrame]:
    """Rolling-origin: shift cutoff back by `horizon` for each fold."""
    sorted_times = sorted(panel[time_col].unique())
    if len(sorted_times) < horizon * (n_folds + 1):
        raise RuntimeError(
            f"need ≥ {horizon * (n_folds + 1)} unique periods for {n_folds} folds × horizon {horizon}; "
            f"have {len(sorted_times)}"
        )
    folds: list[FoldResult] = []
    for f in range(n_folds):
        cutoff = pd.Timestamp(sorted_times[-(horizon * (f + 1)) - 1])
        train = panel[panel[time_col] <= cutoff].copy()
        test = panel[(panel[time_col] > cutoff)].copy()
        # truncate test to exactly `horizon` future periods.
        future_periods = sorted(test[time_col].unique())[:horizon]
        test = test[test[time_col].isin(future_periods)].copy()
        LOG.info("fold %d  cutoff=%s  train_rows=%d  test_rows=%d",
                 f, cutoff.date(), len(train), len(test))
        all_results: list[ForecastResult] = []
        for name, model in models.items():
            try:
                all_results.extend(model.fit_predict(
                    train, group_cols=group_cols, time_col=time_col,
                    target_col=target_col, horizon=horizon, freq=freq,
                ))
            except Exception as e:
                LOG.warning("model %s failed in fold %d: %s", name, f, e)
        per_series = _score_per_series(
            results=all_results, test=test, group_cols=group_cols,
            time_col=time_col, target_col=target_col, season_length=season_length,
            train=train,
        )
        per_series["fold"] = f
        per_series["cutoff"] = cutoff
        folds.append(FoldResult(fold=f, cutoff=cutoff, per_series=per_series))

    summary = pd.concat([fr.per_series for fr in folds], ignore_index=True)
    return folds, summary


def _score_per_series(
    *,
    results: list[ForecastResult],
    test: pd.DataFrame,
    group_cols: Sequence[str],
    time_col: str,
    target_col: str,
    season_length: int,
    train: pd.DataFrame | None = None,
) -> pd.DataFrame:
    rows = []
    test_by_key: dict[SeriesKey, pd.Series] = {
        tuple(k if isinstance(k, tuple) else (k,)): g.set_index(time_col)[target_col].astype(float)
        for k, g in test.groupby(list(group_cols), sort=False)
    }
    train_by_key: dict[SeriesKey, np.ndarray] = {}
    if train is not None:
        for k, g in train.groupby(list(group_cols), sort=False):
            key = tuple(k if isinstance(k, tuple) else (k,))
            train_by_key[key] = g.sort_values(time_col)[target_col].astype(float).values
    for r in results:
        actual = test_by_key.get(r.series_key)
        if actual is None or len(actual) == 0:
            continue
        common = actual.index.intersection(r.point.index)
        if len(common) == 0:
            continue
        a = actual.loc[common].values
        p = r.point.loc[common].values
        row = {col: val for col, val in zip(group_cols, r.series_key)}
        row["method"] = r.method
        row["mape"] = _mape(a, p)
        row["smape"] = _smape(a, p)
        row["mase"] = _mase(
            a, p, season_length=season_length,
            train_actual=train_by_key.get(r.series_key),
        )
        # Directional accuracy: did we get the sign of the period-to-period
        # change right? Joins train tail to test head so the first step has a
        # baseline to compare against.
        last_train = (
            float(train_by_key[r.series_key][-1])
            if r.series_key in train_by_key and len(train_by_key[r.series_key]) > 0
            else None
        )
        if last_train is not None and len(a) > 0:
            actual_diffs = np.diff(np.concatenate([[last_train], a]))
            pred_diffs = np.diff(np.concatenate([[last_train], p]))
            row["directional_acc"] = float(np.mean(np.sign(actual_diffs) == np.sign(pred_diffs)))
        # Standard regression metrics: RMSE / MAE / R² / bias.
        row["rmse"] = float(np.sqrt(np.mean((a - p) ** 2)))
        row["mae"] = float(np.mean(np.abs(a - p)))
        row["bias"] = float(np.mean(p - a))           # >0 → over-forecast on average
        row["r2"] = _r2(a, p)
        # Per-horizon-step error breakdown (forecast typically degrades with h).
        for h_idx in range(len(common)):
            if a[h_idx] != 0:
                row[f"mape_h{h_idx+1}"] = float(abs((a[h_idx] - p[h_idx]) / a[h_idx]))
        if r.lo80 is not None and r.hi80 is not None:
            lo = r.lo80.loc[common].values
            hi = r.hi80.loc[common].values
            row["pinball_80"] = _pinball_avg(a, lo, hi, alpha=0.20)
            row["coverage_80"] = float(np.mean((a >= lo) & (a <= hi)))
            row["interval_width_80"] = float(np.mean(hi - lo))
        if r.lo95 is not None and r.hi95 is not None:
            lo95 = r.lo95.loc[common].values
            hi95 = r.hi95.loc[common].values
            row["coverage_95"] = float(np.mean((a >= lo95) & (a <= hi95)))
            row["interval_width_95"] = float(np.mean(hi95 - lo95))
        rows.append(row)
    return pd.DataFrame(rows)


def _r2(a: np.ndarray, p: np.ndarray) -> float | None:
    ss_res = float(np.sum((a - p) ** 2))
    ss_tot = float(np.sum((a - a.mean()) ** 2))
    if ss_tot == 0:
        return None
    return float(1.0 - ss_res / ss_tot)


def _mape(a: np.ndarray, p: np.ndarray) -> float | None:
    mask = a != 0
    if mask.sum() == 0: return None
    return float(np.mean(np.abs((a[mask] - p[mask]) / a[mask])))


def _smape(a: np.ndarray, p: np.ndarray) -> float:
    denom = (np.abs(a) + np.abs(p)) / 2.0
    mask = denom != 0
    if mask.sum() == 0: return 0.0
    return float(np.mean(np.abs(a[mask] - p[mask]) / denom[mask]))


def _mase(
    a: np.ndarray, p: np.ndarray, *, season_length: int,
    train_actual: np.ndarray | None,
) -> float | None:
    """Mean Absolute Scaled Error. Per Hyndman & Koehler (2006), the
    denominator is the in-sample seasonal-naive MAE on the TRAIN set, not on
    the test set. We accept either; train_actual is preferred when available.
    """
    if train_actual is not None and len(train_actual) > season_length:
        scale = float(np.mean(np.abs(train_actual[season_length:] - train_actual[:-season_length])))
    elif len(a) > season_length:
        scale = float(np.mean(np.abs(a[season_length:] - a[:-season_length])))
    elif len(a) > 1:
        scale = float(np.mean(np.abs(np.diff(a))))
    else:
        return None
    if scale == 0: return None
    return float(np.mean(np.abs(a - p)) / scale)


def _pinball_avg(a: np.ndarray, lo: np.ndarray, hi: np.ndarray, *, alpha: float) -> float:
    """Average pinball loss across the lo and hi quantiles of a (1-alpha) band."""
    q_lo = alpha / 2.0
    q_hi = 1.0 - alpha / 2.0
    loss_lo = np.where(a >= lo, q_lo * (a - lo), (1 - q_lo) * (lo - a))
    loss_hi = np.where(a >= hi, q_hi * (a - hi), (1 - q_hi) * (hi - a))
    return float(np.mean(0.5 * (loss_lo + loss_hi)))
