"""Calibration / reliability diagram — the regression-forecast analog of ROC.

ROC plots TPR vs FPR across discrimination thresholds and is meaningful only
for binary classification. For probabilistic regression forecasts the analogue
is the reliability diagram: for each nominal coverage level α ∈ {0.50, 0.60,
0.70, 0.80, 0.90, 0.95, 0.99}, compute the empirical coverage rate on a
held-out window. A perfectly-calibrated forecaster lies on the y=x diagonal.

Use after a backtest run: pass `(actual, lo_for_alpha, hi_for_alpha)` tuples
per α level and the module returns a tidy DataFrame ready for plotting plus a
single-number Expected Calibration Error (ECE).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CalibrationCurve:
    nominal: np.ndarray
    empirical: np.ndarray
    ece: float            # Expected Calibration Error (mean absolute gap)
    over_confident_at: list[float]    # alphas where empirical < nominal - 0.05
    under_confident_at: list[float]   # alphas where empirical > nominal + 0.05

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "nominal_coverage":   self.nominal,
            "empirical_coverage": self.empirical,
            "gap":                self.empirical - self.nominal,
        })


def reliability_diagram(
    coverage_by_alpha: Mapping[float, float],
    *,
    over_tol: float = 0.05,
) -> CalibrationCurve:
    """coverage_by_alpha maps nominal alpha -> empirical interval coverage.

    Example input:
      {0.50: 0.48, 0.80: 0.74, 0.95: 0.90, 0.99: 0.97}
    """
    nominal = np.array(sorted(coverage_by_alpha.keys()))
    empirical = np.array([coverage_by_alpha[a] for a in nominal])
    ece = float(np.mean(np.abs(empirical - nominal)))
    over = [float(a) for a, e in zip(nominal, empirical) if e < a - over_tol]
    under = [float(a) for a, e in zip(nominal, empirical) if e > a + over_tol]
    return CalibrationCurve(
        nominal=nominal, empirical=empirical, ece=ece,
        over_confident_at=over, under_confident_at=under,
    )


def compute_multi_alpha_coverage(
    actual: np.ndarray,
    point: np.ndarray,
    sigma: np.ndarray,
    *,
    alphas: Sequence[float] = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99),
) -> Mapping[float, float]:
    """Assume normal residuals and synthesize bands at multiple nominal levels.

    Production code should plug in true quantile predictions per α. This helper
    is for backends that emit only a point + sigma estimate.
    """
    from scipy.stats import norm
    out: dict[float, float] = {}
    for alpha in alphas:
        z = float(norm.ppf(0.5 + alpha / 2.0))
        lo = point - z * sigma
        hi = point + z * sigma
        out[alpha] = float(np.mean((actual >= lo) & (actual <= hi)))
    return out


def render_md(curve: CalibrationCurve, *, title: str = "Calibration reliability") -> str:
    lines = [f"## {title}\n",
             f"- Expected Calibration Error (ECE): **{curve.ece:.4f}**",
             f"- Over-confident at α ∈ {curve.over_confident_at or 'none'}",
             f"- Under-confident at α ∈ {curve.under_confident_at or 'none'}",
             "",
             "| nominal α | empirical coverage | gap |",
             "|---|---|---|"]
    for n, e in zip(curve.nominal, curve.empirical):
        lines.append(f"| {n:.2f} | {e:.3f} | {e - n:+.3f} |")
    return "\n".join(lines)
