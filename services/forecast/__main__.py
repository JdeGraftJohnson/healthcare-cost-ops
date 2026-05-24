"""CLI: python -m services.forecast --config services/forecast/configs/sdud_spend.yml"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import duckdb

from services.forecast.logging_config import configure as configure_logging
from services.forecast.pipeline import load_config, run_pipeline


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="forecast")
    p.add_argument("--config", required=True, help="Path to pipeline YAML")
    p.add_argument("--duckdb", default=":memory:", help="DuckDB file (default :memory:)")
    p.add_argument("--verbose", "-v", action="store_true")
    a = p.parse_args(argv)
    configure_logging(verbose=a.verbose, force=True)
    cfg = load_config(a.config)
    con = duckdb.connect(a.duckdb)
    run_pipeline(cfg, con)
    return 0


if __name__ == "__main__":
    sys.exit(main())
