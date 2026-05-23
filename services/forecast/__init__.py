"""Forecast pipeline — classical + global-ML ensemble with conformal intervals.

Reads a silver panel from DuckDB, fits one or more backends per series,
ensembles point forecasts, calibrates prediction intervals via split-conformal,
backtests with rolling-origin CV, and writes a unified forecast.parquet.

Two configurations ship in services/forecast/configs/:
  - sdud_spend.yml         prescription-drug spend (Medicaid SDUD)
  - supplements_price.yml  supplement unit-price events (DSLD + OFF)

Backends are optional. Seasonal-naive always runs; SARIMA / Prophet / LightGBM
register only if their package is importable.
"""
from services.forecast.base import ForecastModel, ForecastResult, SeriesKey
from services.forecast.pipeline import run_pipeline

__all__ = ["ForecastModel", "ForecastResult", "SeriesKey", "run_pipeline"]
