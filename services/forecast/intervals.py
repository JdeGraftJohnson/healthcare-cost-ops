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
        # Skip conformal widening when calibration produced a degenerate q.
        # An exactly-zero or near-zero q means the held-out fold's residuals
        # were vanishingly small, which either reflects a perfect fit (rare
        # in real data — usually means the calibration setup matched the
        # train data too closely) or that the calibration produced no usable
        # residuals. In either case, keeping the model-native bands is more
        # honest than collapsing intervals to zero width.
        if self.quantile_abs_resid <= 1e-9:
            return list(results)
        widened = []
        for r in results:
            # Floor: never let the conformal band be tighter than 25% of the
            # model's own band on each side. Marginal-coverage guarantees are
            # one-sided (widening only); narrowing is unsafe.
            point = r.point
            q = self.quantile_abs_resid
            if abs(self.alpha - 0.05) < 1e-6:
                native_half = (
                    ((r.hi95 - r.lo95).abs() / 2.0) if (r.hi95 is not None and r.lo95 is not None)
                    else 0
                )
                eff = q.__class__(q) if hasattr(q, "__class__") else q
                lo = point - max_series(q, 0.25 * native_half)
                hi = point + max_series(q, 0.25 * native_half)
                widened.append(_replace(r, lo95=lo, hi95=hi))
            else:
                native_half = (
                    ((r.hi80 - r.lo80).abs() / 2.0) if (r.hi80 is not None and r.lo80 is not None)
                    else 0
                )
                lo = point - max_series(q, 0.25 * native_half)
                hi = point + max_series(q, 0.25 * native_half)
                widened.append(_replace(r, lo80=lo, hi80=hi))
        return widened


def max_series(a, b):
    """Element-wise max for scalar/series mixed inputs."""
    import pandas as pd
    if isinstance(b, pd.Series) or isinstance(a, pd.Series):
        import numpy as np
        return np.maximum(a, b)
    return max(a, b)


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
