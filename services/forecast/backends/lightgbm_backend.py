"""Global LightGBM forecaster — one model across all series, recursive horizon roll.

Trains one regressor on engineered features (lags, rollings, fourier seasonality,
calendar, group ids). At inference time, expands the last training row per
series forward `horizon` periods, predicting one step at a time and feeding
predictions back into the lag features (recursive forecasting).
"""
from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd
import lightgbm as lgb

from services.forecast.base import ForecastModel, ForecastResult
from services.forecast.features import (
    add_calendar_features,
    add_fourier_seasonality,
    add_lags,
    encode_group_ids,
)

LOG = logging.getLogger("forecast.lightgbm")


class LightGBMModel(ForecastModel):
    name = "lightgbm"
    requires_regular_grid = True

    def __init__(
        self,
        lags: Sequence[int] = (1, 2, 3, 6, 12),
        fourier_order: int = 3,
        season_length: int = 12,
        n_estimators: int = 600,
        learning_rate: float = 0.05,
        num_leaves: int = 64,
        min_data_in_leaf: int = 20,
        reg_alpha: float = 0.0,
        reg_lambda: float = 0.0,
        verbose: int = -1,
    ):
        self.lags = tuple(lags)
        self.fourier_order = fourier_order
        self.season_length = season_length
        self.params = dict(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            min_data_in_leaf=min_data_in_leaf,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            verbose=verbose,
        )

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
        df = panel.copy().sort_values(list(group_cols) + [time_col]).reset_index(drop=True)
        df, encoders = encode_group_ids(df, group_cols)
        df = add_calendar_features(df, time_col)
        df = add_fourier_seasonality(
            df, time_col, period=self.season_length, order=self.fourier_order, prefix="seas"
        )
        df = add_lags(df, group_cols=group_cols, target_col=target_col, lags=self.lags)
        feature_cols = [c for c in df.columns if c not in {time_col, target_col, *group_cols}]
        train_mask = df[[f"lag_{k}" for k in self.lags]].notna().all(axis=1)
        X = df.loc[train_mask, feature_cols]
        y = df.loc[train_mask, target_col]
        model = lgb.LGBMRegressor(**self.params)
        model.fit(X, y, categorical_feature=[c for c in feature_cols if c.endswith("_id")])

        # Recursive multi-step forecast per series.
        results: list[ForecastResult] = []
        offset = pd.tseries.frequencies.to_offset(freq)

        for keys, grp in df.groupby(list(group_cols), sort=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            history = grp[[time_col, target_col] + [c for c in df.columns if c.endswith("_id")]].copy()
            history[target_col] = grp[target_col].astype(float)
            preds = []
            future_idx = []
            last_t = history[time_col].max()
            target_history = history.set_index(time_col)[target_col]
            id_row = {c: grp[c].iloc[-1] for c in df.columns if c.endswith("_id")}
            for _ in range(horizon):
                last_t = last_t + offset
                future_idx.append(last_t)
                row = {time_col: last_t, **id_row}
                # synthesize feature row
                tmp = pd.DataFrame([row])
                tmp = add_calendar_features(tmp, time_col)
                tmp = add_fourier_seasonality(
                    tmp, time_col, period=self.season_length, order=self.fourier_order, prefix="seas"
                )
                # lags from target_history with predicted values appended
                series_with_preds = pd.concat([target_history, pd.Series(preds, index=pd.DatetimeIndex(future_idx[:-1]))])
                for k in self.lags:
                    target_idx = last_t - k * offset
                    val = series_with_preds.get(target_idx)
                    if val is None or pd.isna(val):
                        # fall back to last available value
                        val = series_with_preds.iloc[-1] if len(series_with_preds) else 0.0
                    tmp[f"lag_{k}"] = val
                yhat = float(model.predict(tmp[feature_cols])[0])
                preds.append(yhat)

            point = pd.Series(preds, index=pd.DatetimeIndex(future_idx))

            # Residual sigma from training fit for interval bands.
            resid = y.values - model.predict(X)
            sigma = float(np.std(resid, ddof=1)) if len(resid) > 1 else 1.0
            results.append(
                ForecastResult(
                    series_key=tuple(keys),
                    method=self.name,
                    horizon=horizon,
                    point=point,
                    lo80=point - 1.2816 * sigma, hi80=point + 1.2816 * sigma,
                    lo95=point - 1.96 * sigma,   hi95=point + 1.96 * sigma,
                    in_sample_mape=_train_mape(y.values, model.predict(X)),
                    metadata={"sigma": sigma, "global_model": True, **self.params},
                )
            )
        return results


def _train_mape(y: np.ndarray, yhat: np.ndarray) -> float | None:
    mask = y != 0
    if mask.sum() == 0:
        return None
    return float(np.mean(np.abs((y[mask] - yhat[mask]) / y[mask])))
