"""Feature engineering for global ML forecasters."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def add_calendar_features(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    t = df[time_col]
    df["month"] = t.dt.month.astype("int8")
    df["quarter"] = t.dt.quarter.astype("int8")
    df["year"] = t.dt.year.astype("int16")
    df["dayofyear"] = t.dt.dayofyear.astype("int16")
    df["weekofyear"] = t.dt.isocalendar().week.astype("int8")
    return df


def add_fourier_seasonality(
    df: pd.DataFrame,
    time_col: str,
    *,
    period: float,
    order: int,
    prefix: str,
) -> pd.DataFrame:
    """Sin/cos basis at the given seasonal period (in periods, not days)."""
    t = (df[time_col].astype("int64") // 10**9).astype("float64")
    # Normalize against the smallest timestamp so phase is stable across calls.
    t0 = t.min()
    omega = 2 * np.pi / period
    for k in range(1, order + 1):
        df[f"{prefix}_sin_{k}"] = np.sin(k * omega * (t - t0))
        df[f"{prefix}_cos_{k}"] = np.cos(k * omega * (t - t0))
    return df


def add_lags(
    df: pd.DataFrame,
    *,
    group_cols: Sequence[str],
    target_col: str,
    lags: Sequence[int],
) -> pd.DataFrame:
    g = df.groupby(list(group_cols), sort=False)[target_col]
    for k in lags:
        df[f"lag_{k}"] = g.shift(k)
    return df


def add_rolling(
    df: pd.DataFrame,
    *,
    group_cols: Sequence[str],
    target_col: str,
    windows: Sequence[int],
) -> pd.DataFrame:
    g = df.groupby(list(group_cols), sort=False)[target_col]
    # shift(1) before rolling so the window never sees the current period.
    shifted = g.shift(1)
    for w in windows:
        df[f"rmean_{w}"] = shifted.rolling(w, min_periods=max(2, w // 2)).mean().reset_index(level=0, drop=True) if False else (
            shifted.groupby(df[list(group_cols)[0]] if len(group_cols) == 1 else [df[c] for c in group_cols]).transform(lambda s: s.rolling(w, min_periods=max(2, w // 2)).mean())
        )
        df[f"rstd_{w}"] = shifted.groupby(df[list(group_cols)[0]] if len(group_cols) == 1 else [df[c] for c in group_cols]).transform(lambda s: s.rolling(w, min_periods=max(2, w // 2)).std())
    return df


def encode_group_ids(
    df: pd.DataFrame, group_cols: Sequence[str]
) -> tuple[pd.DataFrame, dict[str, dict]]:
    """Stable integer encoding for each group column. Returns the encoders."""
    encoders: dict[str, dict] = {}
    for col in group_cols:
        cats = pd.Categorical(df[col])
        encoders[col] = {v: i for i, v in enumerate(cats.categories)}
        df[f"{col}_id"] = cats.codes.astype("int32")
    return df, encoders
