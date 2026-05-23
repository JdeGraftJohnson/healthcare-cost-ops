"""Prophet per-series backend. Requires `prophet`."""
from __future__ import annotations

import logging
import warnings
from typing import Sequence

import numpy as np
import pandas as pd
from prophet import Prophet

from services.forecast.base import ForecastModel, ForecastResult

LOG = logging.getLogger("forecast.prophet")


class ProphetModel(ForecastModel):
    name = "prophet"
    requires_regular_grid = True

    def __init__(
        self,
        yearly_seasonality: bool = True,
        weekly_seasonality: bool = False,
        daily_seasonality: bool = False,
        interval_width: float = 0.80,
        growth: str = "linear",
    ):
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.interval_width = interval_width
        self.growth = growth

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
            ts = grp.sort_values(time_col).rename(columns={time_col: "ds", target_col: "y"})[["ds", "y"]]
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    m80 = Prophet(
                        yearly_seasonality=self.yearly_seasonality,
                        weekly_seasonality=self.weekly_seasonality,
                        daily_seasonality=self.daily_seasonality,
                        interval_width=0.80,
                        growth=self.growth,
                    )
                    m80.fit(ts)
                    future = m80.make_future_dataframe(periods=horizon, freq=freq, include_history=False)
                    f80 = m80.predict(future)
                    m95 = Prophet(
                        yearly_seasonality=self.yearly_seasonality,
                        weekly_seasonality=self.weekly_seasonality,
                        daily_seasonality=self.daily_seasonality,
                        interval_width=0.95,
                        growth=self.growth,
                    )
                    m95.fit(ts)
                    f95 = m95.predict(future)
            except Exception as e:
                LOG.warning("prophet fit failed for %s: %s — falling back to mean", keys, e)
                future_idx = pd.date_range(
                    ts["ds"].iloc[-1] + pd.tseries.frequencies.to_offset(freq),
                    periods=horizon, freq=freq,
                )
                point = pd.Series([ts["y"].mean()] * horizon, index=future_idx)
                sigma = float(ts["y"].std(ddof=1) or abs(ts["y"].mean()) * 0.15 or 1.0)
                results.append(
                    ForecastResult(
                        series_key=tuple(keys),
                        method=self.name,
                        horizon=horizon,
                        point=point,
                        lo80=point - 1.2816 * sigma, hi80=point + 1.2816 * sigma,
                        lo95=point - 1.96 * sigma,   hi95=point + 1.96 * sigma,
                        in_sample_mape=None,
                        metadata={"fallback": "mean"},
                    )
                )
                continue

            idx = pd.DatetimeIndex(f80["ds"])
            point = pd.Series(f80["yhat"].values, index=idx)
            lo80 = pd.Series(f80["yhat_lower"].values, index=idx)
            hi80 = pd.Series(f80["yhat_upper"].values, index=idx)
            lo95 = pd.Series(f95["yhat_lower"].values, index=idx)
            hi95 = pd.Series(f95["yhat_upper"].values, index=idx)

            mape_in = None
            try:
                hist = m80.predict(ts.rename(columns={}))
                actual = ts["y"].values
                pred = hist["yhat"].values
                mask = actual != 0
                if mask.sum() > 0:
                    mape_in = float(np.mean(np.abs((actual[mask] - pred[mask]) / actual[mask])))
            except Exception:
                pass

            results.append(
                ForecastResult(
                    series_key=tuple(keys),
                    method=self.name,
                    horizon=horizon,
                    point=point,
                    lo80=lo80, hi80=hi80, lo95=lo95, hi95=hi95,
                    in_sample_mape=mape_in,
                    metadata={"growth": self.growth},
                )
            )
        return results
