"""Core interfaces for the forecast pipeline."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import pandas as pd


SeriesKey = tuple[str, ...]


@dataclass(frozen=True)
class ForecastResult:
    """One backend's forecast for one series."""

    series_key: SeriesKey
    method: str
    horizon: int
    point: pd.Series           # index = forecast timestamps, values = point forecast
    lo80: pd.Series | None = None
    hi80: pd.Series | None = None
    lo95: pd.Series | None = None
    hi95: pd.Series | None = None
    in_sample_mape: float | None = None
    metadata: Mapping[str, float | str | int] = field(default_factory=dict)

    def to_long_df(self, group_cols: Sequence[str]) -> pd.DataFrame:
        df = pd.DataFrame(
            {
                "period": self.point.index,
                "point": self.point.values,
                "lo80":  self.lo80.values  if self.lo80  is not None else None,
                "hi80":  self.hi80.values  if self.hi80  is not None else None,
                "lo95":  self.lo95.values  if self.lo95  is not None else None,
                "hi95":  self.hi95.values  if self.hi95  is not None else None,
            }
        )
        for col, val in zip(group_cols, self.series_key):
            df[col] = val
        df["method"] = self.method
        df["in_sample_mape"] = self.in_sample_mape
        return df


class ForecastModel(ABC):
    """Per-series or global model with a unified fit/predict surface.

    Per-series models (SARIMA, Prophet, naive) fit one model per (group, series).
    Global models (LightGBM) fit one model across all series and predict per key
    by emitting one row per (series_key, period). Both shapes return
    ForecastResult objects keyed by series.
    """

    name: str = "base"
    requires_regular_grid: bool = True

    @abstractmethod
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
        """Fit and forecast `horizon` periods past the last observation per group."""
