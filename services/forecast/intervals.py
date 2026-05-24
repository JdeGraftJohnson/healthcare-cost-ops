"""Prediction-interval calibration.

Two complementary mechanisms:

1. Split-conformal at multiple α levels. The classical conformal procedure
   gives distribution-free marginal coverage at one nominal α. Calibrating at
   several α levels in one pass produces a usable reliability diagram and
   lets downstream code render multiple bands (50/80/95/99).

2. Empirical-quantile bands per backend. Sigma-based bands assume normality;
   most price/demand residuals are heavy-tailed and asymmetric. Empirical
   bands draw lo/hi from the actual residual distribution per backend.

Both are honest about their guarantees: conformal gives marginal coverage
(across series, not per-series); empirical-quantile gives in-sample fit
quality (with the train-vs-test caveats of any in-sample diagnostic).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from services.forecast.base import ForecastResult, SeriesKey

LOG = logging.getLogger("forecast.intervals")


# ----------------------------------------------------------------------------
# Multi-alpha conformal calibration
# ----------------------------------------------------------------------------

@dataclass
class MultiAlphaConformal:
    """One q per nominal α. Widens each ForecastResult's bands at every α.

    The `alphas` field is mapped to interval levels:
      alpha=0.50 → 50% interval (lo50/hi50 not persisted; only the requested
                                  pair gets used for widening)
      alpha=0.20 → 80% interval (writes lo80/hi80)
      alpha=0.05 → 95% interval (writes lo95/hi95)

    Coverage is `1 - alpha`. The pipeline currently persists lo80/hi80 and
    lo95/hi95 to match Power BI's expected schema; arbitrary α calibrations
    feed the reliability diagram via `quantiles_by_alpha`.
    """
    quantiles_by_alpha: dict[float, float] = field(default_factory=dict)

    def widen(self, results: Sequence[ForecastResult]) -> list[ForecastResult]:
        # Pick the q for the 80% and 95% bands (closest available alpha).
        q80 = self._q_for(0.20)
        q95 = self._q_for(0.05)
        widened = []
        for r in results:
            updates: dict = {}
            if q80 is not None and q80 > 1e-9:
                native = _native_halfwidth(r.lo80, r.hi80)
                eff = _max_scalar_or_series(q80, 0.25 * native)
                updates["lo80"] = r.point - eff
                updates["hi80"] = r.point + eff
            if q95 is not None and q95 > 1e-9:
                native = _native_halfwidth(r.lo95, r.hi95)
                eff = _max_scalar_or_series(q95, 0.25 * native)
                updates["lo95"] = r.point - eff
                updates["hi95"] = r.point + eff
            widened.append(_replace(r, **updates))
        return widened

    def _q_for(self, target_alpha: float) -> float | None:
        if not self.quantiles_by_alpha:
            return None
        nearest = min(self.quantiles_by_alpha.keys(), key=lambda a: abs(a - target_alpha))
        return self.quantiles_by_alpha[nearest]


def calibrate_multi(
    calib_actual: pd.DataFrame,
    calib_pred: dict[SeriesKey, pd.Series],
    *,
    time_col: str,
    target_col: str,
    group_cols: Sequence[str],
    alphas: Sequence[float] = (0.50, 0.20, 0.10, 0.05, 0.01),
) -> MultiAlphaConformal:
    """Compute (1-alpha) quantiles of |residual| across all series."""
    resids: list[float] = []
    for keys, grp in calib_actual.groupby(list(group_cols), sort=False):
        if not isinstance(keys, tuple): keys = (keys,)
        actual = grp.set_index(time_col)[target_col].astype(float)
        pred = calib_pred.get(tuple(keys))
        if pred is None: continue
        common = actual.index.intersection(pred.index)
        if len(common) == 0: continue
        r = (actual.loc[common] - pred.loc[common]).abs().values
        resids.extend(r.tolist())
    if not resids:
        return MultiAlphaConformal()
    arr = np.asarray(resids)
    return MultiAlphaConformal(
        quantiles_by_alpha={float(a): float(np.quantile(arr, 1.0 - a)) for a in alphas},
    )


# ----------------------------------------------------------------------------
# Empirical-quantile bands per backend
# ----------------------------------------------------------------------------

@dataclass
class EmpiricalBandCalibrator:
    """Per-backend empirical residual quantiles. Replaces sigma-based bands.

    Residuals are collected from rolling-origin backtest folds (or from a
    final-test holdout). Quantiles are pooled across all series of a given
    backend — gives marginal coverage but not per-series. For per-series
    quantiles see `EmpiricalBandPerSeries` (future).
    """
    quantiles_by_method: dict[str, dict[float, tuple[float, float]]] = field(default_factory=dict)

    def apply(self, results: Sequence[ForecastResult]) -> list[ForecastResult]:
        out: list[ForecastResult] = []
        for r in results:
            q_for_method = self.quantiles_by_method.get(r.method)
            if not q_for_method:
                out.append(r)
                continue
            updates = {}
            if 0.20 in q_for_method:
                lo_q, hi_q = q_for_method[0.20]
                updates["lo80"] = r.point + lo_q
                updates["hi80"] = r.point + hi_q
            if 0.05 in q_for_method:
                lo_q, hi_q = q_for_method[0.05]
                updates["lo95"] = r.point + lo_q
                updates["hi95"] = r.point + hi_q
            out.append(_replace(r, **updates))
        return out


def calibrate_empirical_bands(
    actuals_vs_preds: pd.DataFrame,
    *,
    alphas: Sequence[float] = (0.20, 0.05),
) -> EmpiricalBandCalibrator:
    """`actuals_vs_preds` must have columns: method, residual (= actual - pred).

    Each backend's residuals are pooled; for each α, the (α/2) and (1 - α/2)
    quantiles give the lower and upper band offsets.
    """
    cal: dict[str, dict[float, tuple[float, float]]] = {}
    for method, grp in actuals_vs_preds.groupby("method", sort=False):
        resid = grp["residual"].dropna().values
        if len(resid) < 10:
            LOG.warning("empirical-band calibrator: method=%s has only %d residuals; skipping",
                        method, len(resid))
            continue
        cal[method] = {}
        for alpha in alphas:
            lo_q = float(np.quantile(resid, alpha / 2.0))
            hi_q = float(np.quantile(resid, 1.0 - alpha / 2.0))
            cal[method][alpha] = (lo_q, hi_q)
    return EmpiricalBandCalibrator(quantiles_by_method=cal)


# ----------------------------------------------------------------------------
# Backwards-compatible single-alpha API (kept for callers that pre-date multi-α)
# ----------------------------------------------------------------------------

@dataclass
class ConformalCalibrator:
    """Single-α conformal — preserved for back-compat with older pipeline runs.
    New code should use MultiAlphaConformal.
    """
    alpha: float
    quantile_abs_resid: float

    def widen(self, results: Sequence[ForecastResult]) -> list[ForecastResult]:
        if self.quantile_abs_resid <= 1e-9:
            return list(results)
        widened = []
        for r in results:
            point = r.point
            q = self.quantile_abs_resid
            if abs(self.alpha - 0.05) < 1e-6:
                native = _native_halfwidth(r.lo95, r.hi95)
                eff = _max_scalar_or_series(q, 0.25 * native)
                widened.append(_replace(r, lo95=point - eff, hi95=point + eff))
            else:
                native = _native_halfwidth(r.lo80, r.hi80)
                eff = _max_scalar_or_series(q, 0.25 * native)
                widened.append(_replace(r, lo80=point - eff, hi80=point + eff))
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
    resids: list[float] = []
    for keys, grp in calib_actual.groupby(list(group_cols), sort=False):
        if not isinstance(keys, tuple): keys = (keys,)
        actual = grp.set_index(time_col)[target_col].astype(float)
        pred = calib_pred.get(tuple(keys))
        if pred is None: continue
        common = actual.index.intersection(pred.index)
        if len(common) == 0: continue
        r = (actual.loc[common] - pred.loc[common]).abs().values
        resids.extend(r.tolist())
    if not resids:
        return ConformalCalibrator(alpha=alpha, quantile_abs_resid=0.0)
    q = float(np.quantile(resids, 1.0 - alpha))
    return ConformalCalibrator(alpha=alpha, quantile_abs_resid=q)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def _native_halfwidth(lo, hi):
    if lo is None or hi is None:
        return 0
    return (hi - lo).abs() / 2.0


def _max_scalar_or_series(a, b):
    if isinstance(a, pd.Series) or isinstance(b, pd.Series):
        return np.maximum(a, b)
    return max(a, b)


def _replace(r: ForecastResult, **kw) -> ForecastResult:
    meta = {**dict(r.metadata), "interval_method": kw.pop("interval_method", "calibrated")}
    return ForecastResult(
        series_key=r.series_key, method=r.method, horizon=r.horizon,
        point=r.point,
        lo80=kw.get("lo80", r.lo80), hi80=kw.get("hi80", r.hi80),
        lo95=kw.get("lo95", r.lo95), hi95=kw.get("hi95", r.hi95),
        in_sample_mape=r.in_sample_mape, metadata=meta,
    )


def max_series(a, b):
    """Backwards-compat shim — older code imports this name."""
    return _max_scalar_or_series(a, b)
