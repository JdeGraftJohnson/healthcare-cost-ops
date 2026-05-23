"""Split-conformal prediction intervals.

Calibrates a global multiplicative width per backend using residuals on a
held-out calibration window. The conformal width replaces the model's native
interval band when `conformal_intervals: true` is set in the pipeline config.

This is the recommended interval for production: it gives marginal coverage
guarantees independent of the backend's distributional assumptions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from services.forecast.base import ForecastResult, SeriesKey


@dataclass
class ConformalCalibrator:
    alpha: float
    quantile_abs_resid: float

    def widen(self, results: Sequence[ForecastResult]) -> list[ForecastResult]:
        widened = []
        z = 1.96 if abs(self.alpha - 0.05) < 1e-6 else 1.2816
        for r in results:
            lo = r.point - self.quantile_abs_resid
            hi = r.point + self.quantile_abs_resid
            if abs(self.alpha - 0.05) < 1e-6:
                widened.append(_replace(r, lo95=lo, hi95=hi))
            else:
                widened.append(_replace(r, lo80=lo, hi80=hi))
        return widened


def calibrate(
    calib_actual: pd.DataFrame,
    calib_pred: dict[SeriesKey, pd.Series],
    *,
    time_col: str,
    target_col: str,
    group_cols: Sequence[str],
    alpha: float = 0.20,
) -> ConformalCalibrator:
    """Compute the (1-alpha) quantile of absolute residuals across all series."""
    resids: list[float] = []
    for keys, grp in calib_actual.groupby(list(group_cols), sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        actual = grp.set_index(time_col)[target_col].astype(float)
        pred = calib_pred.get(tuple(keys))
        if pred is None:
            continue
        common = actual.index.intersection(pred.index)
        if len(common) == 0:
            continue
        r = (actual.loc[common] - pred.loc[common]).abs().values
        resids.extend(r.tolist())
    if not resids:
        return ConformalCalibrator(alpha=alpha, quantile_abs_resid=0.0)
    q = float(np.quantile(resids, 1.0 - alpha))
    return ConformalCalibrator(alpha=alpha, quantile_abs_resid=q)


def _replace(r: ForecastResult, **kw) -> ForecastResult:
    return ForecastResult(
        series_key=r.series_key, method=r.method, horizon=r.horizon,
        point=r.point,
        lo80=kw.get("lo80", r.lo80), hi80=kw.get("hi80", r.hi80),
        lo95=kw.get("lo95", r.lo95), hi95=kw.get("hi95", r.hi95),
        in_sample_mape=r.in_sample_mape, metadata={**dict(r.metadata), "conformal": True},
    )
