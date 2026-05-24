"""Smoke tests — synthetic panel, every available backend produces a forecast."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from services.forecast.backends import available
from services.forecast.calibration import (
    compute_multi_alpha_coverage,
    reliability_diagram,
    render_md,
)
from services.forecast.ensemble import ensemble
from services.forecast.eval import diebold_mariano, rolling_origin_backtest
from services.forecast.monitor import compute_psi, ks_two_sample, evaluate_drift


def _synth_panel(n_series: int = 4, n_periods: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    periods = pd.date_range("2020-01-01", periods=n_periods, freq="MS")
    rows = []
    for i in range(n_series):
        trend = np.linspace(50, 80, n_periods) + rng.normal(0, 1, n_periods)
        seasonal = 10 * np.sin(2 * np.pi * np.arange(n_periods) / 12)
        y = trend + seasonal + rng.normal(0, 2, n_periods) + 5 * i
        for p, val in zip(periods, y):
            rows.append({"state": f"S{i}", "cat": "X", "period": p, "y": float(val)})
    return pd.DataFrame(rows)


GROUP_COLS = ["state", "cat"]
TIME_COL = "period"
TARGET_COL = "y"
FREQ = "MS"
HORIZON = 6


@pytest.fixture(scope="module")
def panel():
    return _synth_panel()


@pytest.mark.parametrize("backend_name", list(available().keys()))
def test_backend_produces_forecast(panel, backend_name):
    cls = available()[backend_name]
    if backend_name == "naive":
        model = cls(season_length=12)
    elif backend_name == "transformer":
        model = cls(lookback=18, epochs=4, batch_size=64, d_model=32, n_heads=4, n_layers=2)
    else:
        model = cls()
    results = model.fit_predict(
        panel, group_cols=GROUP_COLS, time_col=TIME_COL,
        target_col=TARGET_COL, horizon=HORIZON, freq=FREQ,
    )
    assert len(results) == 4
    for r in results:
        assert len(r.point) == HORIZON
        assert r.point.notna().all()
        if r.lo80 is not None:
            assert (r.lo80 <= r.point + 1e-6).all()
            assert (r.hi80 >= r.point - 1e-6).all()


def _build(name, cls):
    if name == "naive":
        return cls(season_length=12)
    if name == "transformer":
        return cls(lookback=18, epochs=4, batch_size=64, d_model=32, n_heads=4, n_layers=2)
    return cls()


def test_ensemble_mixes_methods(panel):
    reg = available()
    raw = []
    for name, cls in reg.items():
        m = _build(name, cls)
        raw.extend(m.fit_predict(
            panel, group_cols=GROUP_COLS, time_col=TIME_COL,
            target_col=TARGET_COL, horizon=HORIZON, freq=FREQ,
        ))
    ens = ensemble(raw, mode="inverse_mape", method_name="ens")
    assert len(ens) == 4
    for r in ens:
        assert r.method == "ens"
        assert "members" in r.metadata


def test_rolling_backtest(panel):
    reg = available()
    models = {n: _build(n, cls) for n, cls in reg.items()}
    folds, summary = rolling_origin_backtest(
        panel, models, group_cols=GROUP_COLS, time_col=TIME_COL,
        target_col=TARGET_COL, horizon=HORIZON, n_folds=2,
        freq=FREQ, season_length=12,
    )
    assert len(folds) == 2
    # All standard regression metrics emitted.
    expected = {"method", "mape", "smape", "mase", "rmse", "mae", "r2", "bias",
                "coverage_80", "coverage_95", "interval_width_80",
                "directional_acc"}
    missing = expected - set(summary.columns)
    assert not missing, f"missing eval metrics: {missing}"
    assert summary["mape"].notna().any()


def test_diebold_mariano():
    rng = np.random.default_rng(3)
    a = rng.normal(50, 10, 60)
    # Two forecasters: one is the actual + small noise (good), one is the
    # actual + large noise (bad). DM should flag the difference as significant.
    p_good = a + rng.normal(0, 1, 60)
    p_bad  = a + rng.normal(0, 8, 60)
    dm, p = diebold_mariano(a, p_good, p_bad, h=1)
    assert not np.isnan(dm)
    assert dm < 0           # negative means good < bad in squared-error loss
    assert p < 0.05


def test_calibration_reliability_perfect():
    # Perfect calibration: empirical == nominal.
    cov = {0.50: 0.50, 0.80: 0.80, 0.95: 0.95}
    curve = reliability_diagram(cov)
    assert curve.ece == 0.0
    assert curve.over_confident_at == []
    assert curve.under_confident_at == []


def test_calibration_reliability_over_confident():
    # Over-confident: empirical < nominal at 80% and 95%.
    cov = {0.50: 0.50, 0.80: 0.65, 0.95: 0.80}
    curve = reliability_diagram(cov)
    assert curve.ece > 0.0
    assert 0.80 in curve.over_confident_at
    assert 0.95 in curve.over_confident_at


def test_calibration_multi_alpha_coverage():
    rng = np.random.default_rng(2)
    n = 500
    point = rng.normal(100, 10, n)
    sigma = np.full(n, 5.0)
    actual = point + rng.normal(0, 5, n)
    cov = compute_multi_alpha_coverage(actual, point, sigma, alphas=(0.50, 0.80, 0.95))
    # With true sigma matching the noise, coverage should be close to nominal.
    for alpha, emp in cov.items():
        assert abs(emp - alpha) < 0.06, f"coverage at α={alpha}: {emp}"


def test_calibration_render_md():
    cov = {0.5: 0.48, 0.8: 0.74, 0.95: 0.90}
    md = render_md(reliability_diagram(cov))
    assert "Expected Calibration Error" in md
    assert "0.50" in md and "0.80" in md and "0.95" in md


def test_drift_psi_and_ks():
    rng = np.random.default_rng(7)
    ref = rng.normal(0.0, 1.0, size=400)
    cur_same = rng.normal(0.0, 1.0, size=400)
    cur_shift = rng.normal(0.8, 1.4, size=400)
    assert compute_psi(ref, cur_same) < 0.1
    assert compute_psi(ref, cur_shift) > 0.25
    _, p_same = ks_two_sample(ref, cur_same)
    _, p_shift = ks_two_sample(ref, cur_shift)
    assert p_same > 0.05
    assert p_shift < 0.01


def test_evaluate_drift_end_to_end():
    rng = np.random.default_rng(11)
    n = 200
    ref = pd.DataFrame({
        "state": ["S0"] * n + ["S1"] * n,
        "cat":   ["X"]  * n + ["X"]  * n,
        "method": ["ensemble"] * (2 * n),
        "residual": np.concatenate([rng.normal(0, 1, n), rng.normal(0, 1, n)]),
        "actual":   np.concatenate([rng.normal(50, 5, n), rng.normal(80, 5, n)]),
        "lo80":     np.concatenate([rng.normal(45, 5, n), rng.normal(75, 5, n)]),
        "hi80":     np.concatenate([rng.normal(55, 5, n), rng.normal(85, 5, n)]),
    })
    cur = ref.copy()
    cur.loc[cur["state"] == "S0", "residual"] = rng.normal(3.0, 2.0, n)
    findings = evaluate_drift(ref_residuals=ref, cur_residuals=cur, group_cols=["state", "cat"])
    severities = {(f.series_key, f.severity) for f in findings}
    assert (("S0", "X"), "alert") in severities
