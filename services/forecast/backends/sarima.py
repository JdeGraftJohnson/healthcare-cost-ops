"""SARIMA per-series backend. Requires statsmodels."""
from __future__ import annotations

import logging
import warnings
from typing import Sequence

import numpy as np
import pandas as pd

from statsmodels.tsa.statespace.sarimax import SARIMAX

from services.forecast.base import ForecastModel, ForecastResult

LOG = logging.getLogger("forecast.sarima")


class SarimaModel(ForecastModel):
    name = "sarima"
    requires_regular_grid = True

    def __init__(
        self,
        order: tuple[int, int, int] = (1, 1, 1),
        seasonal_order: tuple[int, int, int, int] = (1, 1, 0, 12),
        enforce_stationarity: bool = False,
        enforce_invertibility: bool = False,
    ):
        self.order = order
        self.seasonal_order = seasonal_order
        self.enforce_stationarity = enforce_stationarity
        self.enforce_invertibility = enforce_invertibility

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
            ts.index = pd.DatetimeIndex(ts.index, freq=freq)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fit = SARIMAX(
                        ts,
                        order=self.order,
                        seasonal_order=self.seasonal_order,
                        enforce_stationarity=self.enforce_stationarity,
                        enforce_invertibility=self.enforce_invertibility,
                    ).fit(disp=False)
                    fc = fit.get_forecast(steps=horizon)
                    point = fc.predicted_mean
                    ci80 = fc.conf_int(alpha=0.20)
                    ci95 = fc.conf_int(alpha=0.05)
            except Exception as e:
                LOG.warning("sarima fit failed for %s: %s — falling back to mean", keys, e)
                future_idx = pd.date_range(
                    ts.index[-1] + pd.tseries.frequencies.to_offset(freq),
                    periods=horizon, freq=freq,
                )
                point = pd.Series([ts.mean()] * horizon, index=future_idx)
                sigma = float(ts.std(ddof=1) or abs(ts.mean()) * 0.15 or 1.0)
                ci80 = pd.DataFrame({"lower": point - 1.2816 * sigma, "upper": point + 1.2816 * sigma})
                ci95 = pd.DataFrame({"lower": point - 1.96 * sigma,   "upper": point + 1.96 * sigma})

            mape_in = None
            try:
                resid = fit.resid.dropna()
                actual = ts.iloc[-len(resid):]
                mask = actual != 0
                if mask.sum() > 0:
                    mape_in = float(np.mean(np.abs(resid[mask] / actual[mask])))
            except Exception:
                pass

            # Sanity gate: SARIMA can extrapolate to absurd values when the
            # fitted parameter set is unstable (unit-root or near-unit-root).
            # If any forecast point exceeds the in-sample range by more than
            # 5x, or any interval bound exceeds 100x in absolute value, fall
            # back to mean + std bands (same shape as the failure path).
            ts_max = float(np.nanmax(np.abs(ts.values))) if len(ts) else 1.0
            point_max = float(np.nanmax(np.abs(point.values)))
            ci95_max = float(np.nanmax(np.abs(ci95.values))) if not ci95.empty else 0.0
            unstable = (point_max > 5.0 * (ts_max + 1.0)) or (ci95_max > 100.0 * (ts_max + 1.0))
            if unstable:
                LOG.warning("sarima produced extreme values for %s (ts_max=%.2f point_max=%.2g ci95_max=%.2g); falling back to mean",
                            keys, ts_max, point_max, ci95_max)
                future_idx = point.index
                point = pd.Series([ts.mean()] * len(future_idx), index=future_idx)
                sigma = float(ts.std(ddof=1) or abs(ts.mean()) * 0.15 or 1.0)
                ci80 = pd.DataFrame({"lower": point - 1.2816 * sigma, "upper": point + 1.2816 * sigma})
                ci95 = pd.DataFrame({"lower": point - 1.96 * sigma,   "upper": point + 1.96 * sigma})

            results.append(
                ForecastResult(
                    series_key=tuple(keys),
                    method=self.name,
                    horizon=horizon,
                    point=point,
                    lo80=ci80.iloc[:, 0], hi80=ci80.iloc[:, 1],
                    lo95=ci95.iloc[:, 0], hi95=ci95.iloc[:, 1],
                    in_sample_mape=mape_in,
                    metadata={
                        "order": str(self.order),
                        "seasonal_order": str(self.seasonal_order),
                    },
                )
            )
        return results
