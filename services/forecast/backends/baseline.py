"""Dumb baselines — RandomWalkDrift and Mean. Per M4 / M5 conventions, every
non-trivial forecast claim should clear these two before being reported.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from services.forecast.base import ForecastModel, ForecastResult


class RandomWalkDriftModel(ForecastModel):
    """Last observed value + linear drift estimated from training history."""
    name = "drift"
    requires_regular_grid = True

    def __init__(self, season_length: int = 12, seed: int = 0):
        self.season_length = season_length
        self.seed = seed

    def fit_predict(
        self, panel: pd.DataFrame, *,
        group_cols: Sequence[str], time_col: str,
        target_col: str, horizon: int, freq: str,
    ) -> list[ForecastResult]:
        results: list[ForecastResult] = []
        offset = pd.tseries.frequencies.to_offset(freq)
        for keys, grp in panel.groupby(list(group_cols), sort=False):
            if not isinstance(keys, tuple): keys = (keys,)
            ts = grp.sort_values(time_col).set_index(time_col)[target_col].astype(float)
            if len(ts) < 2:
                drift = 0.0; last = float(ts.iloc[-1]) if len(ts) else 0.0
            else:
                drift = float((ts.iloc[-1] - ts.iloc[0]) / max(len(ts) - 1, 1))
                last = float(ts.iloc[-1])
            future_idx = pd.date_range(ts.index[-1] + offset, periods=horizon, freq=freq)
            point = pd.Series([last + drift * (i + 1) for i in range(horizon)], index=future_idx)
            resid = ts.diff().dropna() - drift
            sigma = float(resid.std(ddof=1)) if len(resid) > 1 else abs(last) * 0.15 or 1.0
            growing_sigma = sigma * np.sqrt(np.arange(1, horizon + 1))
            mape_in = _in_sample_mape_drift(ts.values, drift)
            results.append(ForecastResult(
                series_key=tuple(keys), method=self.name, horizon=horizon, point=point,
                lo80=point - 1.2816 * pd.Series(growing_sigma, index=future_idx),
                hi80=point + 1.2816 * pd.Series(growing_sigma, index=future_idx),
                lo95=point - 1.96 * pd.Series(growing_sigma, index=future_idx),
                hi95=point + 1.96 * pd.Series(growing_sigma, index=future_idx),
                in_sample_mape=mape_in,
                metadata={"drift": drift, "sigma": sigma},
            ))
        return results


class MeanModel(ForecastModel):
    """Forecast = training-window mean. The single dumbest baseline that's
    not actively misleading; required as the floor for any 'beats baseline' claim.
    """
    name = "mean"
    requires_regular_grid = True

    def __init__(self, seed: int = 0):
        self.seed = seed

    def fit_predict(
        self, panel: pd.DataFrame, *,
        group_cols: Sequence[str], time_col: str,
        target_col: str, horizon: int, freq: str,
    ) -> list[ForecastResult]:
        results: list[ForecastResult] = []
        offset = pd.tseries.frequencies.to_offset(freq)
        for keys, grp in panel.groupby(list(group_cols), sort=False):
            if not isinstance(keys, tuple): keys = (keys,)
            ts = grp.sort_values(time_col).set_index(time_col)[target_col].astype(float)
            mu = float(ts.mean()) if len(ts) else 0.0
            sd = float(ts.std(ddof=1)) if len(ts) > 1 else abs(mu) * 0.15 or 1.0
            future_idx = pd.date_range(ts.index[-1] + offset, periods=horizon, freq=freq)
            point = pd.Series([mu] * horizon, index=future_idx)
            mape_in = _in_sample_mape_mean(ts.values, mu)
            results.append(ForecastResult(
                series_key=tuple(keys), method=self.name, horizon=horizon, point=point,
                lo80=point - 1.2816 * sd, hi80=point + 1.2816 * sd,
                lo95=point - 1.96 * sd,   hi95=point + 1.96 * sd,
                in_sample_mape=mape_in,
                metadata={"mu": mu, "sigma": sd},
            ))
        return results


def _in_sample_mape_drift(y: np.ndarray, drift: float) -> float | None:
    if len(y) < 2: return None
    pred = y[:-1] + drift
    actual = y[1:]
    mask = actual != 0
    if mask.sum() == 0: return None
    return float(np.mean(np.abs((actual[mask] - pred[mask]) / actual[mask])))


def _in_sample_mape_mean(y: np.ndarray, mu: float) -> float | None:
    if len(y) == 0: return None
    mask = y != 0
    if mask.sum() == 0: return None
    return float(np.mean(np.abs((y[mask] - mu) / y[mask])))
