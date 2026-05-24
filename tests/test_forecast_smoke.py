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


# ── Final-test holdout split (Issue 1.1) ────────────────────────────────────

def test_split_final_holdout(panel):
    from services.forecast.pipeline import _split_final_holdout
    dev, holdout = _split_final_holdout(panel, time_col="period", holdout_periods=6)
    # Holdout has exactly 6 unique periods × 4 series = 24 rows.
    assert holdout["period"].nunique() == 6
    assert dev["period"].nunique() == 60 - 6
    # No temporal overlap.
    assert dev["period"].max() < holdout["period"].min()


def test_split_final_holdout_zero_returns_full_panel(panel):
    from services.forecast.pipeline import _split_final_holdout
    dev, holdout = _split_final_holdout(panel, time_col="period", holdout_periods=0)
    assert len(holdout) == 0
    assert len(dev) == len(panel)


def test_split_final_holdout_raises_when_too_large(panel):
    from services.forecast.pipeline import _split_final_holdout
    with pytest.raises(RuntimeError, match="periods"):
        _split_final_holdout(panel, time_col="period", holdout_periods=999)


# ── Multi-alpha conformal (Issue 5.4) ───────────────────────────────────────

def test_multi_alpha_conformal_quantiles_are_monotone():
    from services.forecast.intervals import calibrate_multi
    # Synthesize calibration data: actual = pred + N(0, sigma).
    rng = np.random.default_rng(5)
    n = 80
    times = pd.date_range("2024-01-01", periods=n, freq="MS")
    actual_df = pd.DataFrame({
        "state": ["S0"] * n,
        "cat":   ["X"]  * n,
        "period": times,
        "y": rng.normal(50, 5, n) + 10 * np.sin(np.arange(n) * 0.5),
    })
    pred_series = pd.Series(50 + 10 * np.sin(np.arange(n) * 0.5), index=times)
    calib = calibrate_multi(
        calib_actual=actual_df,
        calib_pred={("S0", "X"): pred_series},
        time_col="period", target_col="y", group_cols=["state", "cat"],
        alphas=(0.50, 0.20, 0.10, 0.05, 0.01),
    )
    qs = calib.quantiles_by_alpha
    # Lower alpha → wider band → larger q.
    assert qs[0.50] < qs[0.20] < qs[0.10] < qs[0.05] < qs[0.01]


def test_multi_alpha_conformal_widens_at_80_and_95(panel):
    from services.forecast.intervals import MultiAlphaConformal
    reg = available()
    model = reg["naive"](season_length=12)
    results = model.fit_predict(
        panel, group_cols=GROUP_COLS, time_col=TIME_COL,
        target_col=TARGET_COL, horizon=HORIZON, freq=FREQ,
    )
    # Stiff multi-alpha calibrator: q=5 at alpha=0.20, q=10 at alpha=0.05.
    cal = MultiAlphaConformal(quantiles_by_alpha={0.20: 5.0, 0.05: 10.0})
    widened = cal.widen(results)
    for orig, new in zip(results, widened):
        # Bands should be wider than they were natively (this is a smoke check
        # — calibrator floor will use max of q vs 25% of native band).
        new_w80 = (new.hi80 - new.lo80).abs().mean()
        orig_w80 = (orig.hi80 - orig.lo80).abs().mean() if orig.lo80 is not None else 0
        assert new_w80 >= max(orig_w80 * 0.25, 5.0) - 0.01


# ── Empirical-quantile bands (Issue 1.4) ────────────────────────────────────

def test_empirical_band_calibration_quantiles():
    from services.forecast.intervals import calibrate_empirical_bands
    rng = np.random.default_rng(11)
    # Method A: tight residuals, method B: wide residuals.
    df = pd.DataFrame({
        "method": ["A"] * 200 + ["B"] * 200,
        "residual": np.concatenate([rng.normal(0, 1, 200), rng.normal(0, 5, 200)]),
    })
    cal = calibrate_empirical_bands(df, alphas=(0.20, 0.05))
    a_lo80, a_hi80 = cal.quantiles_by_method["A"][0.20]
    b_lo80, b_hi80 = cal.quantiles_by_method["B"][0.20]
    # B's 80% band should be ~5x wider than A's.
    assert (b_hi80 - b_lo80) > 3 * (a_hi80 - a_lo80)
    # 95% band should be wider than 80% band for each method.
    a_lo95, a_hi95 = cal.quantiles_by_method["A"][0.05]
    assert (a_hi95 - a_lo95) > (a_hi80 - a_lo80)


def test_empirical_band_applies_to_forecast(panel):
    from services.forecast.intervals import EmpiricalBandCalibrator
    reg = available()
    model = reg["naive"](season_length=12)
    results = model.fit_predict(
        panel, group_cols=GROUP_COLS, time_col=TIME_COL,
        target_col=TARGET_COL, horizon=HORIZON, freq=FREQ,
    )
    # Asymmetric bands: lo offset = -3, hi offset = +7 → all forecasts widen
    # to (point - 3, point + 7).
    cal = EmpiricalBandCalibrator(quantiles_by_method={
        "seasonal_naive": {0.20: (-3.0, 7.0), 0.05: (-6.0, 12.0)},
    })
    out = cal.apply(results)
    for r in out:
        widths = (r.hi80 - r.lo80).abs()
        # 7 - (-3) = 10 ± floating point.
        assert (widths - 10.0).abs().max() < 1e-9


# ── Graceful all-backends-fail (Issue 4.3) ──────────────────────────────────

def test_ensure_coverage_fills_missing_series():
    from services.forecast.pipeline import _ensure_coverage
    # Build a 2-series panel; ensemble produced no result for series ('S1','X').
    rng = np.random.default_rng(0)
    n = 24
    periods = pd.date_range("2023-01-01", periods=n, freq="MS")
    panel = pd.concat([
        pd.DataFrame({"state": ["S0"] * n, "cat": ["X"] * n, "period": periods, "y": rng.normal(50, 5, n)}),
        pd.DataFrame({"state": ["S1"] * n, "cat": ["X"] * n, "period": periods, "y": rng.normal(80, 8, n)}),
    ], ignore_index=True)
    # Ensemble only has S0.
    from services.forecast.base import ForecastResult
    fut = pd.date_range("2025-01-01", periods=3, freq="MS")
    ens = [ForecastResult(
        series_key=("S0", "X"), method="ensemble", horizon=3,
        point=pd.Series([50.0, 50.0, 50.0], index=fut),
        lo80=pd.Series([45.0]*3, index=fut), hi80=pd.Series([55.0]*3, index=fut),
        lo95=pd.Series([42.0]*3, index=fut), hi95=pd.Series([58.0]*3, index=fut),
        in_sample_mape=None,
    )]
    fallback: set = set()
    out = _ensure_coverage(
        ens, panel=panel, group_cols=["state", "cat"],
        time_col="period", target_col="y",
        horizon=3, freq="MS", fallback_series_out=fallback,
    )
    keys = {r.series_key for r in out}
    assert ("S0", "X") in keys and ("S1", "X") in keys
    assert ("S1", "X") in fallback
    s1 = [r for r in out if r.series_key == ("S1", "X")][0]
    assert s1.metadata.get("fallback_reason") == "all_backends_failed"


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
