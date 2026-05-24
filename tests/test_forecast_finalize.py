"""Smoke tests for the finalize stage + registry promotion logic."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from services.forecast.registry import (
    PromotionRecord,
    _is_better,
    register_and_promote,
)


def test_is_better_lower_is_better_no_champion():
    assert _is_better(0.05, None, tol=0.05, higher_is_better=False) is True


def test_is_better_lower_is_better_within_tol():
    # Champion 0.04; challenger 0.039 → 2.5% improvement < 5% tol.
    assert _is_better(0.039, 0.04, tol=0.05, higher_is_better=False) is False


def test_is_better_lower_is_better_beats_tol():
    # Champion 0.04; challenger 0.037 → 7.5% improvement > 5% tol.
    assert _is_better(0.037, 0.04, tol=0.05, higher_is_better=False) is True


def test_is_better_higher_is_better():
    # Coverage style: higher is better. Champion 0.72; challenger 0.78 ≈ +8.3%.
    assert _is_better(0.78, 0.72, tol=0.05, higher_is_better=True) is True


def test_register_and_promote_no_mlflow_writes_ledger_row(tmp_path, monkeypatch):
    """Without MLFLOW_TRACKING_URI, promotion is skipped but the ledger
    still records the decision so the audit trail is complete."""
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    ledger = tmp_path / "promotions.jsonl"
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    rec = register_and_promote(
        name="test_pipeline",
        run_id="1700000000-abcdef",
        bundle_path=bundle,
        metric_name="final_test__sarima__mape",
        metric_value=0.042,
        champion_tol=0.05,
        promotions_ledger=ledger,
    )
    assert rec.decision == "skipped_no_mlflow"
    assert rec.run_id == "1700000000-abcdef"
    rows = ledger.read_text().strip().splitlines()
    assert len(rows) == 1
    parsed = json.loads(rows[0])
    assert parsed["decision"] == "skipped_no_mlflow"
    assert parsed["challenger_value"] == 0.042


def test_finalize_ensemble_from_partials(tmp_path):
    """End-to-end smoke of the finalize ensemble path on tiny partials."""
    from services.forecast.finalize import _ensemble_from_partials
    # Two methods × 2 series × 3 horizon steps.
    rows = []
    fut = pd.date_range("2025-01-01", periods=3, freq="MS")
    for method, in_mape in [("sarima", 0.04), ("seasonal_naive", 0.07)]:
        for cat, brand in [("c1", "b1"), ("c2", "b2")]:
            for t in fut:
                rows.append({
                    "off_category": cat, "brand": brand, "period": t,
                    "point": 50.0, "lo80": 45.0, "hi80": 55.0,
                    "lo95": 42.0, "hi95": 58.0,
                    "method": method, "in_sample_mape": in_mape,
                })
    df = pd.DataFrame(rows)
    ens = _ensemble_from_partials(df, ["off_category", "brand"], "inverse_mape")
    assert len(ens) == 2
    for r in ens:
        assert r.method == "ensemble"
        # SARIMA has lower in-sample MAPE so it should weigh more — ensemble
        # point should still be ~50 (both methods agree on 50) but the
        # metadata should record both members.
        assert "members" in r.metadata
        members = set(r.metadata["members"].split(","))
        assert members == {"sarima", "seasonal_naive"}


def test_finalize_read_partials_handles_empty_marker(tmp_path):
    from services.forecast.finalize import _read_partials
    d = tmp_path / "partial"
    d.mkdir()
    # Empty marker file (worker writing a 0-byte partition_NNN.parquet for
    # an empty partition).
    (d / "forecast_part_001.parquet").write_bytes(b"")
    # One real partial.
    pd.DataFrame({"off_category": ["c1"], "brand": ["b1"], "period": [pd.Timestamp("2025-01-01")],
                  "point": [50.0], "method": ["sarima"]}) \
      .to_parquet(d / "forecast_part_000.parquet")
    df = _read_partials(d, ["off_category", "brand"])
    assert len(df) == 1
    assert df["method"].iloc[0] == "sarima"


def test_finalize_read_partials_raises_when_all_empty(tmp_path):
    from services.forecast.finalize import _read_partials
    d = tmp_path / "partial"
    d.mkdir()
    (d / "forecast_part_000.parquet").write_bytes(b"")
    with pytest.raises(RuntimeError, match="all partials empty"):
        _read_partials(d, ["off_category", "brand"])
