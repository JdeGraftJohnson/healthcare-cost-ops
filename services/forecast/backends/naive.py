"""Seasonal-naive baseline. Always available."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from services.forecast.base import ForecastModel, ForecastResult


class SeasonalNaive(ForecastModel):
    name = "seasonal_naive"
    requires_regular_grid = True

    def __init__(self, season_length: int = 12):
        self.season_length = season_length

    def fit_predict(
        self,
        panel: pd.DataFrame,
        *,
        group_cols: Sequence[str],
        time_col: str,
        target_col: str,
        horizon: int,
        freq: str,
    ) -> list[ForecastResult]:
        results: list[ForecastResult] = []
        for keys, grp in panel.groupby(list(group_cols), sort=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            ts = grp.sort_values(time_col).set_index(time_col)[target_col].astype(float)
            if len(ts) < self.season_length:
                point_val = ts.mean() if len(ts) else 0.0
                future_idx = pd.date_range(
                    ts.index[-1] + pd.tseries.frequencies.to_offset(freq),
                    periods=horizon, freq=freq,
                )
                point = pd.Series([point_val] * horizon, index=future_idx)
                sigma = ts.std(ddof=1) if len(ts) > 1 else abs(point_val) * 0.15
            else:
                last_season = ts.values[-self.season_length:]
                reps = int(np.ceil(horizon / self.season_length))
                point_vals = np.tile(last_season, reps)[:horizon]
                future_idx = pd.date_range(
                    ts.index[-1] + pd.tseries.frequencies.to_offset(freq),
                    periods=horizon, freq=freq,
                )
                point = pd.Series(point_vals, index=future_idx)
                resid = ts.values[self.season_length:] - ts.values[:-self.season_length]
                sigma = float(np.std(resid, ddof=1)) if len(resid) > 1 else abs(point_vals.mean()) * 0.15
            sigma = max(sigma, 1e-9)
            lo80 = point - 1.2816 * sigma
            hi80 = point + 1.2816 * sigma
            lo95 = point - 1.96 * sigma
            hi95 = point + 1.96 * sigma
            mape_in = _in_sample_mape_seasonal_naive(ts.values, self.season_length)
            results.append(
                ForecastResult(
                    series_key=tuple(keys),
                    method=self.name,
                    horizon=horizon,
                    point=point,
                    lo80=lo80, hi80=hi80, lo95=lo95, hi95=hi95,
                    in_sample_mape=mape_in,
                    metadata={"sigma": sigma, "season_length": self.season_length},
                )
            )
        return results


def _in_sample_mape_seasonal_naive(y: np.ndarray, s: int) -> float | None:
    if len(y) <= s:
        return None
    fc = y[:-s]
    actual = y[s:]
    mask = actual != 0
    if mask.sum() == 0:
        return None
    return float(np.mean(np.abs((actual[mask] - fc[mask]) / actual[mask])))
