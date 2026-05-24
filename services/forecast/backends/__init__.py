"""Forecast backends — registered lazily based on package availability."""
from __future__ import annotations

import logging
from typing import Type

from services.forecast.base import ForecastModel
from services.forecast.backends.naive import SeasonalNaive

LOG = logging.getLogger("forecast.backends")


BACKEND_NAMES = ("naive", "mean", "drift", "sarima", "prophet", "lightgbm", "transformer")


def available() -> dict[str, Type[ForecastModel]]:
    from services.forecast.backends.baseline import MeanModel, RandomWalkDriftModel
    reg: dict[str, Type[ForecastModel]] = {
        "naive": SeasonalNaive,
        "mean": MeanModel,
        "drift": RandomWalkDriftModel,
    }
    try:
        from services.forecast.backends.sarima import SarimaModel
        reg["sarima"] = SarimaModel
    except ImportError as e:
        LOG.info("sarima backend unavailable: %s", e)
    try:
        from services.forecast.backends.prophet_backend import ProphetModel
        reg["prophet"] = ProphetModel
    except ImportError as e:
        LOG.info("prophet backend unavailable: %s", e)
    try:
        from services.forecast.backends.lightgbm_backend import LightGBMModel
        reg["lightgbm"] = LightGBMModel
    except ImportError as e:
        LOG.info("lightgbm backend unavailable: %s", e)
    try:
        from services.forecast.backends.transformer_backend import TransformerModel
        reg["transformer"] = TransformerModel
    except ImportError as e:
        LOG.info("transformer backend unavailable: %s", e)
    return reg
