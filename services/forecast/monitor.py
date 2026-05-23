"""Drift monitoring for the forecast pipeline.

After a forecast is shipped, each new actual that lands gets compared against
the model's prediction. Three drift signals:

  - PSI on the residual distribution between backtest-fold residuals (reference)
    and the rolling-window of live residuals (current).
  - KS test on the same.
  - Coverage drift: 80% prediction-interval empirical coverage in the live
    window vs. the calibration coverage.

Emits a `drift_report.json` per series-and-method, plus a global summary. Designed
to be run on a schedule (Azure Container App job) and posted to the operator's
SMS-approval flow when severity crosses a threshold (cf. feedback_sms_approval).
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

LOG = logging.getLogger("forecast.monitor")


@dataclass
class DriftFinding:
    series_key: tuple
    method: str
    n_ref: int
    n_cur: int
    psi: float
    ks_stat: float
    ks_pvalue: float
    coverage_ref: float | None
    coverage_cur: float | None
    severity: str  # 'ok' | 'watch' | 'alert'


def compute_psi(ref: np.ndarray, cur: np.ndarray, *, bins: int = 10) -> float:
    """Population Stability Index between two samples. >0.25 is a strong shift."""
    if len(ref) == 0 or len(cur) == 0:
        return 0.0
    edges = np.quantile(ref, np.linspace(0, 1, bins + 1))
    edges[0] = -np.inf
    edges[-1] = np.inf
    edges = np.unique(edges)
    if len(edges) < 3:
        return 0.0
    ref_hist, _ = np.histogram(ref, bins=edges)
    cur_hist, _ = np.histogram(cur, bins=edges)
    ref_pct = ref_hist / max(ref_hist.sum(), 1)
    cur_pct = cur_hist / max(cur_hist.sum(), 1)
    eps = 1e-6
    ref_pct = np.where(ref_pct == 0, eps, ref_pct)
    cur_pct = np.where(cur_pct == 0, eps, cur_pct)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def ks_two_sample(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Kolmogorov-Smirnov 2-sample. Returns (stat, asymptotic p-value)."""
    if len(a) == 0 or len(b) == 0:
        return 0.0, 1.0
    a_sorted = np.sort(a); b_sorted = np.sort(b)
    all_v = np.concatenate([a_sorted, b_sorted])
    cdf_a = np.searchsorted(a_sorted, all_v, side="right") / len(a_sorted)
    cdf_b = np.searchsorted(b_sorted, all_v, side="right") / len(b_sorted)
    d = float(np.max(np.abs(cdf_a - cdf_b)))
    n_e = len(a) * len(b) / (len(a) + len(b))
    lam = (math.sqrt(n_e) + 0.12 + 0.11 / math.sqrt(n_e)) * d
    # Marsaglia–Tsang–Wang asymptotic approximation.
    p = 2.0 * sum((-1) ** (j - 1) * math.exp(-2.0 * (lam ** 2) * (j ** 2)) for j in range(1, 101))
    p = float(min(max(p, 0.0), 1.0))
    return d, p


def _coverage(actual: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float | None:
    if len(actual) == 0: return None
    return float(np.mean((actual >= lo) & (actual <= hi)))


def _severity(psi: float, ks_p: float, cov_ref: float | None, cov_cur: float | None) -> str:
    if psi > 0.25 or ks_p < 0.01: return "alert"
    cov_drop = (cov_ref or 0) - (cov_cur or 0)
    if psi > 0.10 or ks_p < 0.05 or cov_drop > 0.10: return "watch"
    return "ok"


def evaluate_drift(
    *,
    ref_residuals: pd.DataFrame,   # cols: <group_cols>, method, residual, [actual, lo80, hi80]
    cur_residuals: pd.DataFrame,
    group_cols: Sequence[str],
) -> list[DriftFinding]:
    findings: list[DriftFinding] = []
    by_key_method_ref = ref_residuals.groupby(list(group_cols) + ["method"], sort=False)
    by_key_method_cur = cur_residuals.groupby(list(group_cols) + ["method"], sort=False)
    cur_keys = set(by_key_method_cur.groups.keys())
    for key, grp_ref in by_key_method_ref:
        if key not in cur_keys: continue
        grp_cur = by_key_method_cur.get_group(key)
        psi = compute_psi(grp_ref["residual"].values, grp_cur["residual"].values)
        ks_s, ks_p = ks_two_sample(grp_ref["residual"].values, grp_cur["residual"].values)
        cov_ref = cov_cur = None
        if {"actual", "lo80", "hi80"}.issubset(grp_ref.columns):
            cov_ref = _coverage(grp_ref["actual"].values, grp_ref["lo80"].values, grp_ref["hi80"].values)
        if {"actual", "lo80", "hi80"}.issubset(grp_cur.columns):
            cov_cur = _coverage(grp_cur["actual"].values, grp_cur["lo80"].values, grp_cur["hi80"].values)
        method = key[-1]
        series_key = key[:-1]
        findings.append(DriftFinding(
            series_key=series_key, method=method,
            n_ref=len(grp_ref), n_cur=len(grp_cur),
            psi=psi, ks_stat=ks_s, ks_pvalue=ks_p,
            coverage_ref=cov_ref, coverage_cur=cov_cur,
            severity=_severity(psi, ks_p, cov_ref, cov_cur),
        ))
    return findings


def write_report(findings: list[DriftFinding], path: str | Path) -> dict:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "n_findings": len(findings),
        "by_severity": {
            "alert": sum(1 for f in findings if f.severity == "alert"),
            "watch": sum(1 for f in findings if f.severity == "watch"),
            "ok":    sum(1 for f in findings if f.severity == "ok"),
        },
        "findings": [
            {**asdict(f), "series_key": list(f.series_key)} for f in findings
        ],
    }
    out.write_text(json.dumps(summary, indent=2, default=str))
    LOG.info("drift report written: %s  alert=%d watch=%d",
             out, summary["by_severity"]["alert"], summary["by_severity"]["watch"])
    return summary
