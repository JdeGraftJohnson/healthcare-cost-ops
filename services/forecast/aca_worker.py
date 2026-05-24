"""ACA worker: fits SARIMA on its assigned partition of a panel.

Invoked once per partition. Env vars:
  PARTITION_ID            (required) int — which slice of the panel to take
  TOTAL_PARTITIONS        (required) int — N total workers
  PANEL_CONFIG            (required) path to the YAML config
  PARTIAL_OUT_DIR         (required) where to write forecast_part_NNN.parquet
  RECORD_OUT_DIR          (required) where to write partition_NNN.json record

This script is the CMD of the container image. It mirrors the local
pipeline (load → partition → fit → write) but for one partition only,
and on a much smaller dependency footprint (SARIMA + naive baselines only;
no LightGBM, Prophet, Transformer, MLflow).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path

import duckdb

from services.forecast.backends import available as available_backends
from services.forecast.data import load_panel
from services.forecast.distributed import PartitionRunRecord, partition_panel
from services.forecast.logging_config import configure as configure_logging
from services.forecast.pipeline import load_config

LOG = logging.getLogger("forecast.aca_worker")


def _env_int(name: str) -> int:
    v = os.environ.get(name)
    if v is None or not v.strip():
        raise RuntimeError(f"required env var {name!r} is unset")
    return int(v)


def _env_path(name: str) -> Path:
    v = os.environ.get(name)
    if not v:
        raise RuntimeError(f"required env var {name!r} is unset")
    return Path(v)


def run() -> int:
    partition_id = _env_int("PARTITION_ID")
    n_partitions = _env_int("TOTAL_PARTITIONS")
    config_path = _env_path("PANEL_CONFIG")
    partial_dir = _env_path("PARTIAL_OUT_DIR"); partial_dir.mkdir(parents=True, exist_ok=True)
    record_dir = _env_path("RECORD_OUT_DIR"); record_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    record_path = record_dir / f"partition_{partition_id:03d}.json"
    artifact_path = partial_dir / f"forecast_part_{partition_id:03d}.parquet"

    cfg = load_config(config_path)
    con = duckdb.connect(":memory:")
    try:
        panel = load_panel(con, cfg.panel)
        my_panel = partition_panel(
            panel, group_cols=list(cfg.panel.group_cols),
            partition_id=partition_id, n_partitions=n_partitions,
        )
        n_series = my_panel.groupby(list(cfg.panel.group_cols), sort=False).ngroups
        LOG.info("partition %d/%d  panel rows=%d  series=%d",
                 partition_id, n_partitions, len(my_panel), n_series)

        if len(my_panel) == 0:
            artifact_path.write_bytes(b"")  # marker file
            PartitionRunRecord(
                partition_id=partition_id, n_partitions=n_partitions,
                n_series=0, n_rows=0,
                started_at=started, ended_at=time.time(), status="completed",
                artifact=str(artifact_path),
            ).to_json()
            record_path.write_text(PartitionRunRecord(
                partition_id=partition_id, n_partitions=n_partitions,
                n_series=0, n_rows=0, started_at=started, ended_at=time.time(),
                status="completed", artifact=str(artifact_path),
            ).to_json())
            return 0

        # ACA workers run a slim backend set. SARIMA is the headline; naive
        # + drift + mean are cheap and ship by default to keep the partial
        # output ensemble-able by the driver.
        reg = available_backends()
        backends = {}
        for name in ("naive", "mean", "drift", "sarima"):
            if name in reg:
                # Sequential inside the worker; ACA is the unit of parallelism.
                backends[name] = reg[name]() if name == "mean" else reg[name](
                    season_length=cfg.season_length
                ) if name in ("naive", "drift") else reg[name]()

        results = []
        for name, m in backends.items():
            try:
                results.extend(m.fit_predict(
                    my_panel,
                    group_cols=list(cfg.panel.group_cols),
                    time_col=cfg.panel.time_col,
                    target_col=cfg.panel.target_col,
                    horizon=cfg.horizon, freq=cfg.panel.freq,
                ))
            except Exception as e:
                LOG.warning("backend %s failed on partition %d: %s", name, partition_id, e)

        long = pd.concat([r.to_long_df(list(cfg.panel.group_cols)) for r in results], ignore_index=True)
        if cfg.panel.log_transform:
            from services.forecast.data import back_transform
            long = back_transform(long, cfg.panel, ["point", "lo80", "hi80", "lo95", "hi95"])
        long.to_parquet(artifact_path, compression="zstd", index=False)

        rec = PartitionRunRecord(
            partition_id=partition_id, n_partitions=n_partitions,
            n_series=int(n_series), n_rows=int(len(long)),
            started_at=started, ended_at=time.time(),
            status="completed", artifact=str(artifact_path),
        )
        record_path.write_text(rec.to_json())
        LOG.info("partition %d completed: %d forecast rows → %s",
                 partition_id, len(long), artifact_path)
        return 0
    except Exception as e:
        rec = PartitionRunRecord(
            partition_id=partition_id, n_partitions=n_partitions,
            n_series=0, n_rows=0,
            started_at=started, ended_at=time.time(),
            status="failed", error=f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
        )
        record_path.write_text(rec.to_json())
        LOG.exception("partition %d FAILED", partition_id)
        return 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="aca_worker")
    p.add_argument("--verbose", "-v", action="store_true")
    a = p.parse_args(argv)
    configure_logging(verbose=a.verbose, force=True)
    return run()


# Late import so logging_config installs before pandas chatter.
import pandas as pd  # noqa: E402


if __name__ == "__main__":
    sys.exit(main())
