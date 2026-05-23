"""Combine per-backend forecasts into a single ensemble forecast per series."""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Literal

import numpy as np
import pandas as pd

from services.forecast.base import ForecastResult, SeriesKey


EnsembleMode = Literal["equal_weight", "inverse_mape", "best_only"]


def ensemble(
    results: Iterable[ForecastResult],
    *,
    mode: EnsembleMode = "inverse_mape",
    method_name: str = "ensemble",
) -> list[ForecastResult]:
    by_series: dict[SeriesKey, list[ForecastResult]] = defaultdict(list)
    for r in results:
        by_series[r.series_key].append(r)

    out: list[ForecastResult] = []
    for key, rs in by_series.items():
        if len(rs) == 1:
            out.append(_relabel(rs[0], method_name))
            continue
        weights = _weights(rs, mode)
        # Align by index (all should share future_idx after pipeline regularization).
        idx = rs[0].point.index
        point = sum(w * r.point.reindex(idx).values for w, r in zip(weights, rs))
        lo80  = sum(w * (r.lo80.reindex(idx).values if r.lo80 is not None else r.point.reindex(idx).values) for w, r in zip(weights, rs))
        hi80  = sum(w * (r.hi80.reindex(idx).values if r.hi80 is not None else r.point.reindex(idx).values) for w, r in zip(weights, rs))
        lo95  = sum(w * (r.lo95.reindex(idx).values if r.lo95 is not None else r.point.reindex(idx).values) for w, r in zip(weights, rs))
        hi95  = sum(w * (r.hi95.reindex(idx).values if r.hi95 is not None else r.point.reindex(idx).values) for w, r in zip(weights, rs))
        member_mapes = {r.method: r.in_sample_mape for r in rs}
        out.append(
            ForecastResult(
                series_key=key,
                method=method_name,
                horizon=len(idx),
                point=pd.Series(point, index=idx),
                lo80=pd.Series(lo80, index=idx),
                hi80=pd.Series(hi80, index=idx),
                lo95=pd.Series(lo95, index=idx),
                hi95=pd.Series(hi95, index=idx),
                in_sample_mape=float(np.mean([w * (m or 0) for w, m in zip(weights, member_mapes.values()) if m is not None])) or None,
                metadata={"mode": mode, "members": ",".join(member_mapes.keys()),
                          "weights": ",".join(f"{w:.3f}" for w in weights)},
            )
        )
    return out


def _relabel(r: ForecastResult, method_name: str) -> ForecastResult:
    return ForecastResult(
        series_key=r.series_key, method=method_name, horizon=r.horizon,
        point=r.point, lo80=r.lo80, hi80=r.hi80, lo95=r.lo95, hi95=r.hi95,
        in_sample_mape=r.in_sample_mape,
        metadata={**dict(r.metadata), "members": r.method, "weights": "1.000"},
    )


def _weights(rs: list[ForecastResult], mode: EnsembleMode) -> list[float]:
    n = len(rs)
    if mode == "equal_weight":
        return [1.0 / n] * n
    if mode == "best_only":
        scored = [(i, r.in_sample_mape if r.in_sample_mape is not None else float("inf")) for i, r in enumerate(rs)]
        best = min(scored, key=lambda x: x[1])[0]
        return [1.0 if i == best else 0.0 for i in range(n)]
    # inverse_mape (default). Missing MAPE -> small weight.
    inv = []
    for r in rs:
        m = r.in_sample_mape
        inv.append(1.0 / max(m, 1e-3) if m is not None else 1.0 / 1.0)
    s = sum(inv)
    return [x / s for x in inv]
