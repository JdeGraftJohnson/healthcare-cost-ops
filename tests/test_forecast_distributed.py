"""Smoke tests for the distributed-SARIMA fan-out helpers."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from services.forecast.distributed import (
    PartitionRunRecord,
    assemble_partitions,
    partition_panel,
    poll_completion,
    series_partition_id,
)


def _panel(n_series=12, n_periods=24, seed=0):
    rng = np.random.default_rng(seed)
    times = pd.date_range("2024-01-01", periods=n_periods, freq="MS")
    rows = []
    for i in range(n_series):
        for t in times:
            rows.append({"cat": f"C{i // 4}", "brand": f"B{i % 4}",
                        "period": t, "y": float(rng.normal(50 + i, 5))})
    return pd.DataFrame(rows)


def test_partition_id_is_stable():
    a = series_partition_id(("vitamin-d", "now foods"), 4)
    b = series_partition_id(("vitamin-d", "now foods"), 4)
    assert a == b
    assert 0 <= a < 4


def test_partition_id_distributes_uniformly():
    counts = [0] * 8
    for i in range(2000):
        pid = series_partition_id((f"cat{i % 47}", f"brand{i % 31}"), 8)
        counts[pid] += 1
    # Each bucket should hold ~250; allow ±30%.
    for c in counts:
        assert 175 <= c <= 325, f"uneven partition: counts={counts}"


def test_partition_panel_keeps_only_assigned_series():
    panel = _panel(n_series=12, n_periods=6)
    all_keys = set(tuple(k) for k, _ in panel.groupby(["cat", "brand"], sort=False))
    reconstructed_keys = set()
    for pid in range(4):
        my = partition_panel(panel, group_cols=["cat", "brand"],
                              partition_id=pid, n_partitions=4)
        for k, _ in my.groupby(["cat", "brand"], sort=False):
            reconstructed_keys.add(tuple(k))
    assert reconstructed_keys == all_keys
    # Sum of partition sizes equals total series count (no series duplicated, none dropped).
    sizes = [
        partition_panel(panel, group_cols=["cat", "brand"], partition_id=pid, n_partitions=4)
            .groupby(["cat", "brand"]).ngroups
        for pid in range(4)
    ]
    assert sum(sizes) == len(all_keys)
    # With only 12 series across 4 partitions, perfect balance isn't guaranteed.
    # The uniformity property is tested separately on a larger sample size
    # (test_partition_id_distributes_uniformly with 2000 keys).


def test_assemble_partitions(tmp_path):
    d = tmp_path / "partial"
    d.mkdir()
    # Three partial frames.
    for pid in range(3):
        pd.DataFrame({"a": [pid] * 5, "b": list(range(5))}) \
          .to_parquet(d / f"forecast_part_{pid:03d}.parquet")
    out = tmp_path / "merged.parquet"
    assemble_partitions(d, out_path=out)
    merged = pd.read_parquet(out)
    assert len(merged) == 15
    assert set(merged["a"].unique()) == {0, 1, 2}


def test_poll_completion_returns_records_in_order(tmp_path):
    # Pre-write 3 records out of order; poll_completion should return them
    # sorted by partition_id.
    for pid in [2, 0, 1]:
        rec = PartitionRunRecord(
            partition_id=pid, n_partitions=3,
            n_series=10, n_rows=60, started_at=time.time(),
            ended_at=time.time() + 1, status="completed",
            artifact=f"/tmp/forecast_part_{pid}.parquet",
        )
        (tmp_path / f"partition_{pid:03d}.json").write_text(rec.to_json())
    out = poll_completion(3, record_dir=tmp_path, timeout_s=1, poll_interval_s=1)
    assert [r.partition_id for r in out] == [0, 1, 2]
    assert all(r.status == "completed" for r in out)


def test_poll_completion_raises_on_timeout(tmp_path):
    # Only 1 of 3 records present; should timeout.
    rec = PartitionRunRecord(
        partition_id=0, n_partitions=3,
        n_series=0, n_rows=0, started_at=time.time(),
        ended_at=time.time(), status="completed",
    )
    (tmp_path / "partition_000.json").write_text(rec.to_json())
    with pytest.raises(TimeoutError, match="missing partition_ids="):
        poll_completion(3, record_dir=tmp_path, timeout_s=1, poll_interval_s=1)
