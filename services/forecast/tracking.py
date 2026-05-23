"""Experiment-tracking shim.

Records each pipeline run (config hash, backend versions, backtest medians,
artifact paths) to a local JSONL ledger. If `mlflow` is installed and a
tracking URI is configured, mirrors the same record into an MLflow run so
the same code works in a CI/MLOps context without any caller changes.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import sys
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Mapping

LOG = logging.getLogger("forecast.tracking")


@dataclass
class RunRecord:
    run_id: str
    name: str
    started_at: float
    ended_at: float
    config_hash: str
    params: dict[str, Any]
    metrics: dict[str, float]
    artifacts: dict[str, str]
    env: dict[str, str] = field(default_factory=dict)
    status: str = "completed"


def _hash_config(d: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:12]


def _env() -> dict[str, str]:
    out: dict[str, str] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    for pkg in ("pandas", "numpy", "duckdb", "statsmodels", "prophet", "lightgbm", "torch"):
        try:
            mod = __import__(pkg)
            out[pkg] = getattr(mod, "__version__", "unknown")
        except Exception:
            pass
    return out


def track(
    *,
    name: str,
    params: Mapping[str, Any],
    metrics: Mapping[str, float],
    artifacts: Mapping[str, str],
    ledger_path: str | Path = "logs/forecast_runs.jsonl",
    mlflow_uri: str | None = None,
    mlflow_experiment: str = "forecast",
    started_at: float | None = None,
) -> RunRecord:
    started = started_at or time.time()
    ended = time.time()
    cfg_hash = _hash_config(dict(params))
    run_id = f"{int(started)}-{cfg_hash}"
    record = RunRecord(
        run_id=run_id,
        name=name,
        started_at=started,
        ended_at=ended,
        config_hash=cfg_hash,
        params=dict(params),
        metrics={k: float(v) for k, v in metrics.items() if v is not None},
        artifacts=dict(artifacts),
        env=_env(),
    )
    lp = Path(ledger_path)
    lp.parent.mkdir(parents=True, exist_ok=True)
    with lp.open("a") as f:
        f.write(json.dumps(asdict(record), default=str) + "\n")
    LOG.info("forecast run logged: %s (%s)", run_id, lp)

    uri = mlflow_uri or os.environ.get("MLFLOW_TRACKING_URI")
    if uri:
        try:
            import mlflow
            mlflow.set_tracking_uri(uri)
            mlflow.set_experiment(mlflow_experiment)
            with mlflow.start_run(run_name=name) as run:
                for k, v in record.params.items():
                    mlflow.log_param(k, v)
                for k, v in record.metrics.items():
                    mlflow.log_metric(k, v)
                for art_name, art_path in record.artifacts.items():
                    if Path(art_path).exists():
                        mlflow.log_artifact(art_path, artifact_path=art_name)
                mlflow.set_tag("config_hash", cfg_hash)
                LOG.info("mlflow run: %s", run.info.run_id)
        except ImportError:
            LOG.info("mlflow not installed; skipping remote log (set MLFLOW_TRACKING_URI to enable)")
        except Exception as e:
            LOG.warning("mlflow log failed: %s", e)
    return record
