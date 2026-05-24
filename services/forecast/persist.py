"""Native-format model persistence per backend.

Saves and reloads fitted models using each backend's native serializer:

  statsmodels  → SARIMAXResults.save()           (.pkl, statsmodels native)
  prophet      → JSON via prophet.serialize       (.json)
  lightgbm     → Booster.save_model()             (.txt, LightGBM native)
  torch        → state_dict() + arch hyperparams (.pt + .json sidecar)
  naive/mean/drift → pickled config + history    (.pkl)

A `ModelBundle` packs every per-series fit for a single backend into one
artifact directory. Pipeline orchestration can persist after fit and later
reload-and-predict without refitting.

Bundle layout:
  <dir>/
    bundle.json                  # backend name, group_cols, freq, etc.
    series/
      <safe_series_key>.{ext}    # one fit per series (statsmodels)
    global.bin                   # one file for global models (lightgbm, torch)
"""
from __future__ import annotations

import hashlib
import json
import logging
import pickle
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

LOG = logging.getLogger("forecast.persist")


@dataclass
class BundleManifest:
    backend: str
    name: str                                 # operator-facing name (e.g. "supplements_price_v3")
    group_cols: list[str]
    time_col: str
    target_col: str
    freq: str
    season_length: int
    log_transform: bool
    fit_at: str                               # ISO timestamp
    n_series: int
    config_hash: str
    backend_version: str = ""
    notes: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)


def _safe_key(series_key: tuple) -> str:
    """Filesystem-safe encoding of a series_key tuple."""
    h = hashlib.sha1(repr(series_key).encode()).hexdigest()[:10]
    raw = "_".join(str(k).replace("/", "-").replace(" ", "_")[:32] for k in series_key)
    return f"{raw}__{h}"


def save_bundle(
    *,
    backend: str,
    name: str,
    group_cols: Sequence[str],
    time_col: str,
    target_col: str,
    freq: str,
    season_length: int,
    log_transform: bool,
    fits_by_series: dict[tuple, Any] | None = None,
    global_state: Any = None,
    out_dir: str | Path,
    notes: str = "",
    backend_version: str = "",
) -> Path:
    """Persist a fitted backend. Either `fits_by_series` (per-series) or
    `global_state` (global model) must be provided; both is allowed.
    """
    from datetime import datetime, timezone
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    n_series = len(fits_by_series or {})
    cfg_hash = hashlib.sha256(
        json.dumps({"backend": backend, "freq": freq, "group_cols": list(group_cols),
                    "season_length": season_length, "log_transform": log_transform},
                   sort_keys=True).encode()
    ).hexdigest()[:12]
    manifest = BundleManifest(
        backend=backend, name=name, group_cols=list(group_cols),
        time_col=time_col, target_col=target_col, freq=freq,
        season_length=season_length, log_transform=log_transform,
        fit_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        n_series=n_series, config_hash=cfg_hash, backend_version=backend_version,
        notes=notes,
    )
    (p / "bundle.json").write_text(manifest.to_json())

    if fits_by_series:
        series_dir = p / "series"
        series_dir.mkdir(exist_ok=True)
        for key, fit in fits_by_series.items():
            ext, blob = _serialize_per_series(backend, fit)
            (series_dir / f"{_safe_key(key)}.{ext}").write_bytes(blob)
            # Sidecar with the actual key (since the filename is hashed).
            (series_dir / f"{_safe_key(key)}.key.json").write_text(
                json.dumps(list(key))
            )
    if global_state is not None:
        ext, blob = _serialize_global(backend, global_state)
        (p / f"global.{ext}").write_bytes(blob)

    LOG.info("saved bundle backend=%s name=%s n_series=%d -> %s",
             backend, name, n_series, p)
    return p


def load_bundle(in_dir: str | Path) -> tuple[BundleManifest, dict]:
    """Returns (manifest, payload). `payload` is shaped like
    {'fits_by_series': {...}, 'global_state': <obj or None>}.
    """
    p = Path(in_dir)
    manifest_data = json.loads((p / "bundle.json").read_text())
    manifest = BundleManifest(**manifest_data)
    payload: dict[str, Any] = {"fits_by_series": {}, "global_state": None}
    series_dir = p / "series"
    if series_dir.exists():
        for fkey in series_dir.glob("*.key.json"):
            stem = fkey.name.replace(".key.json", "")
            data_files = list(series_dir.glob(f"{stem}.*"))
            data_files = [f for f in data_files if not f.name.endswith(".key.json")]
            if not data_files:
                continue
            key = tuple(json.loads(fkey.read_text()))
            ext = data_files[0].suffix.lstrip(".")
            fit = _deserialize_per_series(manifest.backend, ext, data_files[0].read_bytes())
            payload["fits_by_series"][key] = fit
    global_files = list(p.glob("global.*"))
    if global_files:
        ext = global_files[0].suffix.lstrip(".")
        payload["global_state"] = _deserialize_global(
            manifest.backend, ext, global_files[0].read_bytes(),
        )
    LOG.info("loaded bundle backend=%s name=%s n_series=%d from %s",
             manifest.backend, manifest.name, manifest.n_series, p)
    return manifest, payload


# ── per-backend serializers ────────────────────────────────────────────────

def _serialize_per_series(backend: str, fit: Any) -> tuple[str, bytes]:
    if backend == "sarima":
        try:
            from statsmodels.tsa.statespace.sarimax import SARIMAXResults  # noqa: F401
            import io
            buf = io.BytesIO()
            fit.save(buf)
            return ("pkl", buf.getvalue())
        except Exception:
            return ("pkl", pickle.dumps(fit))
    if backend == "prophet":
        try:
            from prophet.serialize import model_to_json
            return ("json", model_to_json(fit).encode())
        except Exception:
            return ("pkl", pickle.dumps(fit))
    return ("pkl", pickle.dumps(fit))


def _deserialize_per_series(backend: str, ext: str, blob: bytes) -> Any:
    if backend == "sarima" and ext == "pkl":
        try:
            from statsmodels.tsa.statespace.sarimax import SARIMAXResults
            import io
            return SARIMAXResults.load(io.BytesIO(blob))
        except Exception:
            return pickle.loads(blob)
    if backend == "prophet" and ext == "json":
        try:
            from prophet.serialize import model_from_json
            return model_from_json(blob.decode())
        except Exception:
            return pickle.loads(blob)
    return pickle.loads(blob)


def _serialize_global(backend: str, state: Any) -> tuple[str, bytes]:
    if backend == "lightgbm":
        try:
            import io
            buf = io.BytesIO()
            booster = state["model"].booster_ if hasattr(state["model"], "booster_") else state["model"]
            booster.save_model(str(Path("/tmp/.lgb_tmp_save.txt")))
            data = Path("/tmp/.lgb_tmp_save.txt").read_bytes()
            extra = pickle.dumps({"feature_cols": state.get("feature_cols"),
                                  "encoders": state.get("encoders")})
            return ("lgbpkg", pickle.dumps({"model": data, "extra": extra}))
        except Exception:
            return ("pkl", pickle.dumps(state))
    if backend == "transformer":
        try:
            import io, torch
            sd = state["state_dict"]
            arch = state["arch"]
            buf = io.BytesIO()
            torch.save({"state_dict": sd, "arch": arch}, buf)
            return ("pt", buf.getvalue())
        except Exception:
            return ("pkl", pickle.dumps(state))
    return ("pkl", pickle.dumps(state))


def _deserialize_global(backend: str, ext: str, blob: bytes) -> Any:
    if backend == "lightgbm" and ext == "lgbpkg":
        try:
            import lightgbm as lgb
            pkg = pickle.loads(blob)
            Path("/tmp/.lgb_tmp_load.txt").write_bytes(pkg["model"])
            booster = lgb.Booster(model_file="/tmp/.lgb_tmp_load.txt")
            extra = pickle.loads(pkg["extra"])
            return {"model": booster, **extra}
        except Exception:
            return pickle.loads(blob)
    if backend == "transformer" and ext == "pt":
        try:
            import io, torch
            return torch.load(io.BytesIO(blob), map_location="cpu")
        except Exception:
            return pickle.loads(blob)
    return pickle.loads(blob)
