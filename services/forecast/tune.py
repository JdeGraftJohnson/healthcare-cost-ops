"""Hyperparameter tuning — Optuna-backed Bayesian search per backend.

Optuna is gracefully optional. When installed, `tune_backend()` runs a TPE
search over the backend's hyperparameter space, scored against rolling-origin
backtest median MAPE on the passed panel. Returns the best params + Optuna
study object for inspection / persistence.

Search spaces are conservative defaults; override via `param_space` kwarg.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Mapping, Sequence

import pandas as pd

from services.forecast.backends import available as available_backends
from services.forecast.eval import rolling_origin_backtest

LOG = logging.getLogger("forecast.tune")


_DEFAULT_SPACES: dict[str, Callable[[Any], dict[str, Any]]] = {
    "sarima": lambda trial: dict(
        order=(
            trial.suggest_int("p", 0, 2),
            trial.suggest_int("d", 0, 2),
            trial.suggest_int("q", 0, 2),
        ),
        seasonal_order=(
            trial.suggest_int("P", 0, 1),
            trial.suggest_int("D", 0, 1),
            trial.suggest_int("Q", 0, 1),
            trial.suggest_categorical("s", [4, 12]),
        ),
    ),
    "prophet": lambda trial: dict(
        yearly_seasonality=True,
        growth=trial.suggest_categorical("growth", ["linear", "flat"]),
    ),
    "lightgbm": lambda trial: dict(
        n_estimators=trial.suggest_int("n_estimators", 200, 1000, step=200),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        num_leaves=trial.suggest_int("num_leaves", 16, 128, log=True),
        min_data_in_leaf=trial.suggest_int("min_data_in_leaf", 10, 50),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
        lags=(1, 2, 3, 6, 12),
        fourier_order=trial.suggest_int("fourier_order", 1, 4),
        season_length=12,
    ),
    "transformer": lambda trial: dict(
        lookback=trial.suggest_int("lookback", 12, 36, step=6),
        d_model=trial.suggest_categorical("d_model", [32, 64, 128]),
        n_heads=trial.suggest_categorical("n_heads", [2, 4, 8]),
        n_layers=trial.suggest_int("n_layers", 2, 4),
        dropout=trial.suggest_float("dropout", 0.0, 0.3),
        epochs=trial.suggest_int("epochs", 10, 40, step=10),
        learning_rate=trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True),
    ),
    "naive": lambda trial: dict(season_length=trial.suggest_categorical("season_length", [4, 12])),
}


def tune_backend(
    backend: str,
    panel: pd.DataFrame,
    *,
    group_cols: Sequence[str],
    time_col: str,
    target_col: str,
    horizon: int,
    freq: str,
    season_length: int = 12,
    n_trials: int = 30,
    n_folds: int = 2,
    timeout_s: int | None = None,
    seed: int = 42,
    objective: str = "mape",      # 'mape' | 'mase' | 'pinball_80' | 'rmse'
    param_space: Callable | None = None,
) -> dict[str, Any]:
    """Run TPE search and return {best_params, best_value, study}.

    The objective scalar is the median across all series in the latest fold.
    """
    try:
        import optuna
    except ImportError as e:
        raise RuntimeError(
            "Optuna is required for HPO. Install with: pip install optuna"
        ) from e

    reg = available_backends()
    if backend not in reg:
        raise RuntimeError(f"backend {backend!r} not installed; cannot tune")
    cls = reg[backend]
    space_fn = param_space or _DEFAULT_SPACES.get(backend)
    if space_fn is None:
        raise RuntimeError(f"no default param-space defined for backend {backend!r}; pass param_space=")

    def _obj(trial: "optuna.Trial") -> float:
        kwargs = space_fn(trial)
        model = cls(**kwargs)
        try:
            _folds, summary = rolling_origin_backtest(
                panel, {backend: model},
                group_cols=group_cols, time_col=time_col,
                target_col=target_col, horizon=horizon,
                n_folds=n_folds, freq=freq, season_length=season_length,
            )
        except Exception as e:
            LOG.warning("trial %d failed: %s", trial.number, e)
            return float("inf")
        col = objective
        if col not in summary.columns:
            raise RuntimeError(f"objective {col!r} not in backtest summary columns {list(summary.columns)}")
        val = summary[col].median()
        if pd.isna(val):
            return float("inf")
        return float(val)

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(_obj, n_trials=n_trials, timeout=timeout_s, show_progress_bar=False)
    LOG.info("HPO done. best %s=%.4f  best_params=%s",
             objective, study.best_value, study.best_params)
    return {
        "backend": backend,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "n_trials": len(study.trials),
        "objective": objective,
        "study": study,
    }


def tune_all_installed(
    panel: pd.DataFrame,
    *,
    group_cols: Sequence[str],
    time_col: str,
    target_col: str,
    horizon: int,
    freq: str,
    n_trials: int = 20,
    objective: str = "mape",
) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for name in available_backends():
        try:
            out[name] = tune_backend(
                backend=name, panel=panel,
                group_cols=group_cols, time_col=time_col,
                target_col=target_col, horizon=horizon,
                freq=freq, n_trials=n_trials, objective=objective,
            )
        except Exception as e:
            LOG.warning("tune %s failed: %s", name, e)
    return out
