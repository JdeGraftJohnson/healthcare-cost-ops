"""Partition + assemble helpers for distributed SARIMA on Azure Container Apps.

Architecture:

  ┌───────────────────────────────┐
  │ driver (local or single ACA)  │
  │   partition by series hash    │
  │   fan-out N workers           │
  │   poll completion             │
  │   concat partial parquets     │
  └───────────────────────────────┘
        │ az containerapp job start --env PARTITION_ID=k TOTAL=N
        ▼
  ┌───────────────────────────────┐
  │ caj-forecast-sarima-prod1     │
  │   reads silver from Azure     │
  │   filters to its partition    │
  │   fits SARIMA per series      │
  │   writes forecast_part_NNN    │
  └───────────────────────────────┘

The partition function uses xxh3 (built-in `hash` in Python 3) modulo N
so the same series always lands on the same worker. The driver concats
`forecast_part_*.parquet` files into the unified `forecast.parquet`.

Local mode (no ACA): run N workers in subprocesses for development. The
driver's API is the same; only the launch mechanism differs.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd

LOG = logging.getLogger("forecast.distributed")


def series_partition_id(series_key: tuple, n_partitions: int) -> int:
    """Stable hash-based partition assignment. Same series always lands on
    the same worker; cheap to compute; uniform distribution under typical
    series-key cardinality.
    """
    s = "|".join(str(k) for k in series_key)
    h = hashlib.blake2b(s.encode(), digest_size=8).digest()
    return int.from_bytes(h, "big") % n_partitions


def partition_panel(
    panel: pd.DataFrame,
    *,
    group_cols: Sequence[str],
    partition_id: int,
    n_partitions: int,
) -> pd.DataFrame:
    """Keep only rows whose series_key hashes to `partition_id`."""
    if n_partitions <= 0:
        raise ValueError("n_partitions must be > 0")
    keep = []
    for keys, _grp in panel.groupby(list(group_cols), sort=False):
        key = keys if isinstance(keys, tuple) else (keys,)
        if series_partition_id(key, n_partitions) == partition_id:
            keep.append(key)
    if not keep:
        return panel.iloc[0:0].copy()
    keep_df = pd.DataFrame(keep, columns=list(group_cols))
    return panel.merge(keep_df, on=list(group_cols), how="inner")


def assemble_partitions(
    partial_dir: str | Path,
    *,
    pattern: str = "forecast_part_*.parquet",
    out_path: str | Path,
) -> Path:
    """Concat partial parquets into the unified forecast.parquet."""
    d = Path(partial_dir)
    files = sorted(d.glob(pattern))
    if not files:
        raise RuntimeError(f"no partial parquets matching {pattern} in {d}")
    frames = [pd.read_parquet(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, compression="zstd", index=False)
    LOG.info("assembled %d partials → %s (rows=%d)", len(files), out, len(df))
    return out


@dataclass
class PartitionRunRecord:
    partition_id: int
    n_partitions: int
    n_series: int
    n_rows: int
    started_at: float
    ended_at: float
    status: str   # 'completed' | 'failed'
    error: str | None = None
    artifact: str | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str)


def poll_completion(
    expected_partitions: int,
    *,
    record_dir: str | Path,
    timeout_s: int = 3600,
    poll_interval_s: int = 30,
) -> list[PartitionRunRecord]:
    """Wait until every partition has written its record file. Returns the
    parsed records in partition_id order.

    Used by the driver when fanning out across ACA jobs — each worker
    writes `<record_dir>/partition_<id>.json` on exit (success or fail).
    The driver polls until all N appear or `timeout_s` elapses.
    """
    d = Path(record_dir)
    d.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + timeout_s
    seen: dict[int, PartitionRunRecord] = {}
    while time.time() < deadline and len(seen) < expected_partitions:
        for f in d.glob("partition_*.json"):
            try:
                raw = json.loads(f.read_text())
                rec = PartitionRunRecord(**raw)
                seen[rec.partition_id] = rec
            except (json.JSONDecodeError, TypeError):
                continue
        if len(seen) < expected_partitions:
            time.sleep(poll_interval_s)
    if len(seen) < expected_partitions:
        missing = [i for i in range(expected_partitions) if i not in seen]
        raise TimeoutError(
            f"only {len(seen)}/{expected_partitions} partitions completed in {timeout_s}s; "
            f"missing partition_ids={missing}"
        )
    return [seen[i] for i in range(expected_partitions)]
