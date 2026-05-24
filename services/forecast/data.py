"""Panel loading + regularization for the forecast pipeline."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import duckdb
import pandas as pd

LOG = logging.getLogger("forecast.data")


@dataclass(frozen=True)
class PanelSpec:
    sql: str
    group_cols: Sequence[str]
    time_col: str
    target_col: str
    freq: str                # pandas offset alias: 'MS', 'W-MON', 'QS'
    min_obs_per_series: int  # drop sparse series below this floor
    fill_gaps: str = "zero"  # 'zero' | 'ffill' | 'drop'
    log_transform: bool = False   # fit on log1p(target); back-transform forecasts


def load_panel(con: duckdb.DuckDBPyConnection, spec: PanelSpec) -> pd.DataFrame:
    LOG.info("loading panel via SQL (%d cols group=%s freq=%s)",
             len(spec.group_cols), spec.group_cols, spec.freq)
    df = con.execute(spec.sql).fetch_df()
    if df.empty:
        raise RuntimeError("panel SQL returned zero rows")
    df[spec.time_col] = pd.to_datetime(df[spec.time_col])
    # Data quality: drop any duplicated (group, time) before regularization.
    dup_mask = df.duplicated(subset=list(spec.group_cols) + [spec.time_col], keep="last")
    if dup_mask.any():
        LOG.warning("dropping %d duplicate (group,time) rows", int(dup_mask.sum()))
        df = df[~dup_mask].copy()
    if spec.log_transform:
        import numpy as np
        df[spec.target_col] = np.log1p(df[spec.target_col].clip(lower=0))
        LOG.info("applied log1p transform to %s", spec.target_col)
    df = _regularize(df, spec)
    df = _drop_short_series(df, spec)
    LOG.info("panel rows=%d series=%d span=%s..%s target_col=%s%s",
             len(df), df.groupby(list(spec.group_cols)).ngroups,
             df[spec.time_col].min().date(), df[spec.time_col].max().date(),
             spec.target_col, " (log1p)" if spec.log_transform else "")
    return df


def back_transform(df: pd.DataFrame, spec: PanelSpec, cols: Sequence[str]) -> pd.DataFrame:
    if not spec.log_transform:
        return df
    import numpy as np
    for c in cols:
        if c in df.columns:
            df[c] = np.expm1(df[c])
    return df


def _regularize(df: pd.DataFrame, spec: PanelSpec) -> pd.DataFrame:
    parts = []
    for keys, grp in df.groupby(list(spec.group_cols), sort=False):
        grp = grp.set_index(spec.time_col).sort_index()
        full_idx = pd.date_range(grp.index.min(), grp.index.max(), freq=spec.freq)
        grp = grp.reindex(full_idx)
        if spec.fill_gaps == "zero":
            grp[spec.target_col] = grp[spec.target_col].fillna(0.0)
        elif spec.fill_gaps == "ffill":
            grp[spec.target_col] = grp[spec.target_col].ffill().fillna(0.0)
        elif spec.fill_gaps == "drop":
            grp = grp.dropna(subset=[spec.target_col])
        if not isinstance(keys, tuple):
            keys = (keys,)
        for col, val in zip(spec.group_cols, keys):
            grp[col] = val
        grp.index.name = spec.time_col
        parts.append(grp.reset_index())
    return pd.concat(parts, ignore_index=True)


def _drop_short_series(df: pd.DataFrame, spec: PanelSpec) -> pd.DataFrame:
    g = df.groupby(list(spec.group_cols), sort=False).size()
    keep = g[g >= spec.min_obs_per_series].index
    if len(keep) == len(g):
        return df
    LOG.warning("dropping %d sparse series (<%d obs)",
                len(g) - len(keep), spec.min_obs_per_series)
    df = df.set_index(list(spec.group_cols))
    return df.loc[keep].reset_index()


def split_train_test(
    df: pd.DataFrame,
    *,
    time_col: str,
    holdout_periods: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff = df[time_col].sort_values().unique()[-holdout_periods]
    train = df[df[time_col] < cutoff].copy()
    test  = df[df[time_col] >= cutoff].copy()
    return train, test
