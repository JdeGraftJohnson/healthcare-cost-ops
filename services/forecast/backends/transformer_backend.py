"""Global Transformer-encoder forecaster (PyTorch).

A small multi-head self-attention encoder trained across all series with
group-id embeddings. Recursive multi-step rollout at inference. Pure PyTorch;
no pytorch-forecasting / lightning dependency.

Architecture is intentionally compact (~5-layer encoder, d_model 64,
num_heads 4) so it trains on CPU in a few minutes for ~10k-series panels.
Drops in alongside SARIMA / Prophet / LightGBM in the ensemble; the registry
in services/forecast/backends/__init__.py skips it if torch isn't installed.
"""
from __future__ import annotations

import logging
import math
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from services.forecast.base import ForecastModel, ForecastResult
from services.forecast.features import (
    add_calendar_features,
    add_fourier_seasonality,
    encode_group_ids,
)

LOG = logging.getLogger("forecast.transformer")


class _WindowDataset(Dataset):
    def __init__(self, X: np.ndarray, gid: np.ndarray, y: np.ndarray, lookback: int):
        self.X = X
        self.gid = gid
        self.y = y
        self.lookback = lookback
        self.indices = self._build_indices()

    def _build_indices(self) -> list[int]:
        idx = []
        n = len(self.y)
        for i in range(self.lookback, n):
            # Window must stay within the same group.
            if self.gid[i] == self.gid[i - self.lookback]:
                idx.append(i)
        return idx

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        t = self.indices[i]
        window = slice(t - self.lookback, t)
        return (
            torch.from_numpy(self.X[window]).float(),
            torch.tensor(self.gid[t]).long(),
            torch.tensor(self.y[t]).float(),
        )


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class _TransformerForecaster(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_groups: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        dim_ff: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.group_emb = nn.Embedding(max(n_groups, 1), d_model)
        self.pos = _PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Sequential(nn.Linear(d_model, dim_ff), nn.GELU(), nn.Linear(dim_ff, 1))

    def forward(self, x: torch.Tensor, gid: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        h = self.pos(h)
        g = self.group_emb(gid).unsqueeze(1)
        h = h + g
        h = self.encoder(h)
        return self.head(h[:, -1, :]).squeeze(-1)


class TransformerModel(ForecastModel):
    name = "transformer"
    requires_regular_grid = True

    def __init__(
        self,
        lookback: int = 24,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.1,
        epochs: int = 30,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        fourier_order: int = 3,
        season_length: int = 12,
        device: str | None = None,
        seed: int = 0,
    ):
        self.lookback = lookback
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = learning_rate
        self.wd = weight_decay
        self.fourier_order = fourier_order
        self.season_length = season_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = seed

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
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        df = panel.copy().sort_values(list(group_cols) + [time_col]).reset_index(drop=True)
        df, _enc = encode_group_ids(df, group_cols)
        df = add_calendar_features(df, time_col)
        df = add_fourier_seasonality(
            df, time_col, period=self.season_length, order=self.fourier_order, prefix="seas"
        )
        gid_col = [c for c in df.columns if c.endswith("_id")][0]
        feature_cols = [
            c for c in df.columns
            if c not in {time_col, target_col, *group_cols} and not c.endswith("_id")
        ]
        feature_cols = [target_col] + feature_cols  # target as a lagged feature

        # Standardize features per global, target per group (more stable than per-feature).
        feat_mat = df[feature_cols].astype(float).values
        mean = feat_mat.mean(axis=0)
        std = feat_mat.std(axis=0) + 1e-9
        feat_mat = (feat_mat - mean) / std

        per_group_stats: dict[int, tuple[float, float]] = {}
        for gid, grp in df.groupby(gid_col, sort=False):
            mu = float(grp[target_col].mean())
            sd = float(grp[target_col].std(ddof=1) or 1.0)
            per_group_stats[int(gid)] = (mu, sd)

        gid_arr = df[gid_col].values.astype(np.int64)
        y_norm = np.empty(len(df), dtype=np.float32)
        for i, g in enumerate(gid_arr):
            mu, sd = per_group_stats[int(g)]
            y_norm[i] = (df[target_col].iat[i] - mu) / sd
        # also normalize the target column inside feat_mat to keep it consistent
        feat_mat[:, 0] = y_norm

        ds = _WindowDataset(feat_mat, gid_arr, y_norm, lookback=self.lookback)
        if len(ds) == 0:
            LOG.warning("transformer: empty dataset (lookback %d too long) — falling back to seasonal-naive shape",
                        self.lookback)
            return _fallback(df, group_cols, time_col, target_col, horizon, freq)

        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=True, drop_last=False)
        n_groups = int(gid_arr.max()) + 1
        model = _TransformerForecaster(
            n_features=len(feature_cols), n_groups=n_groups,
            d_model=self.d_model, n_heads=self.n_heads, n_layers=self.n_layers,
            dropout=self.dropout,
        ).to(self.device)
        opt = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.wd)
        loss_fn = nn.SmoothL1Loss()
        model.train()
        for ep in range(self.epochs):
            total = 0.0
            n = 0
            for xb, gb, yb in dl:
                xb = xb.to(self.device); gb = gb.to(self.device); yb = yb.to(self.device)
                opt.zero_grad()
                pred = model(xb, gb)
                loss = loss_fn(pred, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                total += loss.item() * xb.size(0); n += xb.size(0)
            if ep == 0 or (ep + 1) % 5 == 0:
                LOG.info("transformer ep=%d loss=%.5f", ep + 1, total / max(n, 1))

        # Recursive multi-step forecast per series.
        results: list[ForecastResult] = []
        offset = pd.tseries.frequencies.to_offset(freq)
        model.eval()
        with torch.no_grad():
            for keys, grp in df.groupby(list(group_cols), sort=False):
                if not isinstance(keys, tuple):
                    keys = (keys,)
                grp = grp.sort_values(time_col)
                gid = int(grp[gid_col].iloc[-1])
                mu, sd = per_group_stats[gid]
                history_feat = (grp[feature_cols].astype(float).values - mean) / std
                last_t = grp[time_col].max()
                preds_norm: list[float] = []
                future_idx: list[pd.Timestamp] = []
                lookback_buf = history_feat[-self.lookback:].copy() if len(history_feat) >= self.lookback else np.pad(
                    history_feat, ((self.lookback - len(history_feat), 0), (0, 0)), mode="edge"
                )
                static_feat_template = history_feat[-1].copy()
                for _ in range(horizon):
                    x = torch.from_numpy(lookback_buf).float().unsqueeze(0).to(self.device)
                    g = torch.tensor([gid]).long().to(self.device)
                    yhat_norm = float(model(x, g).cpu().numpy()[0])
                    preds_norm.append(yhat_norm)
                    last_t = last_t + offset
                    future_idx.append(last_t)
                    next_row = static_feat_template.copy()
                    next_row[0] = yhat_norm  # target lag
                    lookback_buf = np.vstack([lookback_buf[1:], next_row])
                preds = np.array(preds_norm) * sd + mu
                point = pd.Series(preds, index=pd.DatetimeIndex(future_idx))
                # Issue 3.2 — batched in-sample residual: stack every valid
                # window into one forward pass instead of iterating per index.
                in_sample_resid_norm = []
                n_windows = max(0, len(history_feat) - self.lookback)
                if n_windows > 0:
                    windows = np.stack([
                        history_feat[i - self.lookback:i]
                        for i in range(self.lookback, len(history_feat))
                    ])
                    x_batch = torch.from_numpy(windows).float().to(self.device)
                    g_batch = torch.full((n_windows,), gid, dtype=torch.long, device=self.device)
                    yhat_batch = model(x_batch, g_batch).cpu().numpy()
                    targets = history_feat[self.lookback:, 0]
                    in_sample_resid_norm = (targets - yhat_batch).tolist()
                if in_sample_resid_norm:
                    sigma = float(np.std(in_sample_resid_norm, ddof=1) * sd) or abs(preds.mean()) * 0.15 or 1.0
                else:
                    sigma = abs(preds.mean()) * 0.15 or 1.0
                mape_in = _mape_normalized(in_sample_resid_norm, history_feat[self.lookback:, 0], sd, mu)
                results.append(
                    ForecastResult(
                        series_key=tuple(keys),
                        method=self.name,
                        horizon=horizon,
                        point=point,
                        lo80=point - 1.2816 * sigma, hi80=point + 1.2816 * sigma,
                        lo95=point - 1.96 * sigma,   hi95=point + 1.96 * sigma,
                        in_sample_mape=mape_in,
                        metadata={
                            "d_model": self.d_model, "n_heads": self.n_heads,
                            "n_layers": self.n_layers, "lookback": self.lookback,
                            "epochs": self.epochs, "device": self.device,
                            "global_model": True,
                        },
                    )
                )
        return results


def _mape_normalized(
    resid_norm: list[float],
    target_norm: np.ndarray,
    sd: float,
    mu: float,
) -> float | None:
    if not resid_norm: return None
    actual = target_norm * sd + mu
    mask = actual != 0
    if mask.sum() == 0: return None
    pred = (target_norm - np.array(resid_norm)) * sd + mu
    return float(np.mean(np.abs((actual[mask] - pred[mask]) / actual[mask])))


def _fallback(df, group_cols, time_col, target_col, horizon, freq):
    from services.forecast.backends.naive import SeasonalNaive
    return SeasonalNaive().fit_predict(
        df, group_cols=group_cols, time_col=time_col,
        target_col=target_col, horizon=horizon, freq=freq,
    )
