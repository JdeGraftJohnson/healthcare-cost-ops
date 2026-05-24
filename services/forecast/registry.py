"""MLflow Model Registry promotion semantics.

Thin wrapper over `mlflow` that registers fitted forecast bundles and
moves them through `Staging` → `Production` based on champion-challenger
comparison (`services/forecast/compare.py`).

Gracefully no-op when `MLFLOW_TRACKING_URI` is unset — local dev runs
don't need a real MLflow backend. When set, registration + transitions
are atomic per-run-id.

Promotion semantics:
  • Every successful pipeline run registers its persisted bundle as a
    new model version under name = pipeline cfg.name.
  • If `metrics[primary_metric]` improves on the current Production by
    more than `champion_tol`, the new version is transitioned to
    Production and the previous Production is archived.
  • Otherwise the new version is transitioned to Staging for review.
  • All transitions emit a row in logs/promotions.jsonl regardless of
    whether MLflow is wired up.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

LOG = logging.getLogger("forecast.registry")


Stage = Literal["None", "Staging", "Production", "Archived"]


@dataclass
class PromotionRecord:
    name: str
    version: str | None
    run_id: str
    metric_name: str
    challenger_value: float
    champion_value: float | None
    pct_improvement: float | None
    decision: Literal["promoted_to_production", "promoted_to_staging", "registered_only", "skipped_no_mlflow"]
    archived_previous: list[str]
    ts: float

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str)


def _is_better(challenger: float, champion: float | None, *, tol: float,
               higher_is_better: bool) -> bool:
    if champion is None:
        return True
    if higher_is_better:
        return challenger > champion * (1.0 + tol)
    return challenger < champion * (1.0 - tol)


def _read_ledger_metric(ledger_path: str | Path, run_id: str, metric_name: str) -> float | None:
    p = Path(ledger_path)
    if not p.exists():
        return None
    with p.open() as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("run_id") == run_id:
                return rec.get("metrics", {}).get(metric_name)
    return None


def register_and_promote(
    *,
    name: str,
    run_id: str,
    bundle_path: str | Path,
    metric_name: str = "final_test__sarima__mape",
    metric_value: float,
    champion_tol: float = 0.05,
    higher_is_better: bool = False,
    promotions_ledger: str | Path = "logs/promotions.jsonl",
    mlflow_tracking_uri: str | None = None,
) -> PromotionRecord:
    """Register `bundle_path` and decide Staging vs Production transition.

    Always writes a row to `promotions_ledger` so the decision history is
    auditable without MLflow being up.
    """
    started = time.time()
    uri = mlflow_tracking_uri or os.environ.get("MLFLOW_TRACKING_URI")
    archived: list[str] = []

    if not uri:
        rec = PromotionRecord(
            name=name, version=None, run_id=run_id,
            metric_name=metric_name, challenger_value=metric_value,
            champion_value=None, pct_improvement=None,
            decision="skipped_no_mlflow", archived_previous=[],
            ts=started,
        )
        _write_ledger(promotions_ledger, rec)
        LOG.info("registry: MLFLOW_TRACKING_URI not set; skipping (ledger row written)")
        return rec

    try:
        import mlflow
        from mlflow.tracking import MlflowClient
        from mlflow.entities.model_registry.model_version_status import ModelVersionStatus
    except ImportError:
        LOG.warning("registry: mlflow not installed; skipping promotion")
        rec = PromotionRecord(
            name=name, version=None, run_id=run_id,
            metric_name=metric_name, challenger_value=metric_value,
            champion_value=None, pct_improvement=None,
            decision="skipped_no_mlflow", archived_previous=[],
            ts=started,
        )
        _write_ledger(promotions_ledger, rec)
        return rec

    mlflow.set_tracking_uri(uri)
    client = MlflowClient()

    # Register the bundle as a new model version.
    bundle_uri = f"file://{Path(bundle_path).resolve()}"
    try:
        mv = client.create_model_version(
            name=name, source=bundle_uri, run_id=run_id,
            description=f"Forecast bundle for {name} at {run_id}",
        )
        version = mv.version
        # Wait for it to leave PENDING_REGISTRATION (typically < 1s).
        for _ in range(30):
            mv = client.get_model_version(name=name, version=version)
            if mv.status == ModelVersionStatus.to_string(ModelVersionStatus.READY):
                break
            time.sleep(0.5)
    except mlflow.exceptions.RestException as e:
        if "RESOURCE_DOES_NOT_EXIST" in str(e):
            client.create_registered_model(name)
            mv = client.create_model_version(name=name, source=bundle_uri, run_id=run_id)
            version = mv.version
        else:
            raise

    # Look up current Production version's metric.
    champion_value = None
    try:
        prods = client.get_latest_versions(name=name, stages=["Production"])
        if prods:
            champion_run = client.get_run(prods[0].run_id)
            champion_value = champion_run.data.metrics.get(metric_name)
    except Exception as e:
        LOG.warning("registry: failed to read champion metric: %s", e)

    if _is_better(metric_value, champion_value, tol=champion_tol,
                  higher_is_better=higher_is_better):
        # Promote to Production; archive any previous Production versions.
        for old in client.get_latest_versions(name=name, stages=["Production"]):
            client.transition_model_version_stage(
                name=name, version=old.version, stage="Archived",
                archive_existing_versions=False,
            )
            archived.append(old.version)
        client.transition_model_version_stage(
            name=name, version=version, stage="Production",
            archive_existing_versions=False,
        )
        decision = "promoted_to_production"
        pct_imp = (
            None if champion_value is None
            else (champion_value - metric_value) / champion_value
                  if not higher_is_better
                  else (metric_value - champion_value) / champion_value
        )
    else:
        client.transition_model_version_stage(
            name=name, version=version, stage="Staging",
            archive_existing_versions=False,
        )
        decision = "promoted_to_staging"
        pct_imp = (
            (champion_value - metric_value) / champion_value
            if not higher_is_better else (metric_value - champion_value) / champion_value
        ) if champion_value else None

    rec = PromotionRecord(
        name=name, version=version, run_id=run_id,
        metric_name=metric_name, challenger_value=metric_value,
        champion_value=champion_value, pct_improvement=pct_imp,
        decision=decision, archived_previous=archived, ts=started,
    )
    _write_ledger(promotions_ledger, rec)
    LOG.info("registry: %s v%s → %s (challenger=%.4f champion=%s)",
             name, version, decision, metric_value,
             f"{champion_value:.4f}" if champion_value is not None else "none")
    return rec


def _write_ledger(path: str | Path, rec: PromotionRecord) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as f:
        f.write(rec.to_json() + "\n")


def get_production_version(name: str) -> dict | None:
    """Return the current Production version dict, or None if MLflow is
    unavailable / no Production version exists. Useful for the
    finalize-or-rollback path.
    """
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not uri:
        return None
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
        mlflow.set_tracking_uri(uri)
        client = MlflowClient()
        prods = client.get_latest_versions(name=name, stages=["Production"])
        if not prods:
            return None
        mv = prods[0]
        return {
            "name": mv.name, "version": mv.version, "run_id": mv.run_id,
            "source": mv.source, "current_stage": mv.current_stage,
        }
    except Exception as e:
        LOG.warning("registry: get_production_version failed: %s", e)
        return None
