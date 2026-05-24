"""Driver for distributed SARIMA fan-out.

Two launch modes:

  local      Spawn N subprocesses on this machine; each runs aca_worker.
             For development and CI smoke tests.

  aca        `az containerapp job start` once per partition; poll the
             per-partition record files for completion.

Both modes share the same partition function (services/forecast/distributed.py)
and the same partial-parquet assembly. Difference is only the worker launch.

  python -m services.forecast.distributed_driver \
      --config services/forecast/configs/supplements_price.yml \
      --partitions 4 \
      --mode local \
      --partial-dir /tmp/forecast/partial \
      --record-dir /tmp/forecast/records \
      --out /tmp/forecast/forecast.parquet
"""
from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from services.forecast.distributed import assemble_partitions, poll_completion
from services.forecast.logging_config import configure as configure_logging

LOG = logging.getLogger("forecast.driver")


def _spawn_local(
    *, config: Path, partitions: int, partial_dir: Path, record_dir: Path, verbose: bool,
) -> list[subprocess.Popen]:
    procs = []
    for pid in range(partitions):
        env = os.environ.copy()
        env.update({
            "PARTITION_ID": str(pid),
            "TOTAL_PARTITIONS": str(partitions),
            "PANEL_CONFIG": str(config),
            "PARTIAL_OUT_DIR": str(partial_dir),
            "RECORD_OUT_DIR": str(record_dir),
        })
        cmd = [sys.executable, "-m", "services.forecast.aca_worker"]
        if verbose: cmd.append("-v")
        p = subprocess.Popen(cmd, env=env)
        procs.append(p)
        LOG.info("spawned local worker pid=%d (partition %d/%d)", p.pid, pid, partitions)
    return procs


def _spawn_aca(
    *, config_url: str, partitions: int, partial_dir: str, record_dir: str,
    job_name: str, rg: str, image_tag: str,
) -> list[dict]:
    """Trigger N invocations of an ACA Job. Returns per-partition execution
    metadata for the driver to log; completion is detected via record files.
    """
    launched = []
    for pid in range(partitions):
        env_vars = [
            f"PARTITION_ID={pid}",
            f"TOTAL_PARTITIONS={partitions}",
            f"PANEL_CONFIG={config_url}",
            f"PARTIAL_OUT_DIR={partial_dir}",
            f"RECORD_OUT_DIR={record_dir}",
        ]
        cmd = [
            "az", "containerapp", "job", "start",
            "--name", job_name,
            "--resource-group", rg,
            "--image", image_tag,
            "--env-vars", *env_vars,
            "-o", "json",
        ]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            LOG.error("ACA start failed for partition %d: %s", pid, e.stderr)
            raise
        launched.append({"partition_id": pid, "az_output": r.stdout.strip()[:200]})
        LOG.info("ACA invocation queued for partition %d/%d", pid, partitions)
    return launched


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="forecast-driver")
    p.add_argument("--config", required=True, help="path to YAML config")
    p.add_argument("--partitions", type=int, required=True, help="N partitions")
    p.add_argument("--mode", choices=("local", "aca"), default="local")
    p.add_argument("--partial-dir", required=True)
    p.add_argument("--record-dir", required=True)
    p.add_argument("--out", required=True, help="final unified forecast.parquet")
    # ACA-mode-only
    p.add_argument("--job-name", default="caj-forecast-sarima-prod1")
    p.add_argument("--rg", default="rg-asi-prod1-eus2")
    p.add_argument("--image-tag", default="acrasiprod1eus2.azurecr.io/asi-forecast:latest")
    # Polling
    p.add_argument("--timeout-s", type=int, default=3600)
    p.add_argument("--poll-interval-s", type=int, default=15)
    p.add_argument("--clean", action="store_true", help="wipe partial/record dirs before run")
    p.add_argument("--verbose", "-v", action="store_true")
    a = p.parse_args(argv)
    configure_logging(verbose=a.verbose, force=True)

    partial_dir = Path(a.partial_dir); record_dir = Path(a.record_dir)
    if a.clean:
        for d in (partial_dir, record_dir):
            if d.exists():
                shutil.rmtree(d)
                LOG.info("cleaned %s", d)
    partial_dir.mkdir(parents=True, exist_ok=True)
    record_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    if a.mode == "local":
        procs = _spawn_local(
            config=Path(a.config), partitions=a.partitions,
            partial_dir=partial_dir, record_dir=record_dir, verbose=a.verbose,
        )
        for proc in procs:
            proc.wait()
    elif a.mode == "aca":
        _spawn_aca(
            config_url=a.config, partitions=a.partitions,
            partial_dir=str(partial_dir), record_dir=str(record_dir),
            job_name=a.job_name, rg=a.rg, image_tag=a.image_tag,
        )

    records = poll_completion(
        a.partitions, record_dir=record_dir,
        timeout_s=a.timeout_s, poll_interval_s=a.poll_interval_s,
    )
    failed = [r for r in records if r.status != "completed"]
    if failed:
        LOG.error("%d/%d partitions failed", len(failed), len(records))
        for r in failed:
            LOG.error("  partition %d: %s", r.partition_id, (r.error or "")[:200])
        return 2

    assemble_partitions(partial_dir, out_path=a.out)
    LOG.info("driver done. partitions=%d  total_series=%d  elapsed=%.1fs",
             len(records), sum(r.n_series for r in records), time.time() - started)
    return 0


if __name__ == "__main__":
    sys.exit(main())
