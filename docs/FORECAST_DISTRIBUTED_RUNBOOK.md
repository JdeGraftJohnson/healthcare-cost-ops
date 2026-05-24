# Forecast pipeline — distributed SARIMA on Azure Container Apps

How to fan out per-series SARIMA fits across N Azure Container App Job
invocations so the SDUD-scale panel (~15k series × ~5s per fit ≈ 21 hr
sequential) finishes in ~30 min wall-clock.

T1.1 of `docs/FORECAST_ROADMAP.md`. Pattern mirrors existing
`caj-fl-bench-prod1` and `caj-embed-finbert-prod1`.

## Architecture

```
  ┌──────────────────────────────────┐
  │ driver (local CLI or single-shot)│
  │   • partition by series hash      │
  │   • az containerapp job start × N │
  │   • poll partition_NNN.json       │
  │   • concat forecast_part_NNN.*    │
  └──────────────────────────────────┘
                │ env: PARTITION_ID, TOTAL_PARTITIONS,
                │      PANEL_CONFIG, PARTIAL_OUT_DIR, RECORD_OUT_DIR
                ▼
  ┌──────────────────────────────────┐
  │ caj-forecast-sarima-prod1  × N   │
  │   • read silver via DuckDB+MI     │
  │   • partition_panel(my_id, N)     │
  │   • fit per-series SARIMA + base  │
  │   • write forecast_part_NNN.pq    │
  │   • write partition_NNN.json      │
  └──────────────────────────────────┘
```

Partition key: `blake2b(group_cols_joined, 8) % N`. Same series always
lands on the same worker; uniform distribution at any reasonable
cardinality (verified at 2000 keys × 8 partitions in
`tests/test_forecast_distributed.py`).

## Components

| File | Purpose |
|---|---|
| `services/forecast/distributed.py` | `partition_panel()`, `series_partition_id()`, `assemble_partitions()`, `poll_completion()` |
| `services/forecast/aca_worker.py` | CLI worker — reads env vars, fits its slice, writes partial parquet + record JSON |
| `services/forecast/distributed_driver.py` | CLI driver — `--mode local|aca`; fans out N invocations; polls; concats |
| `services/forecast/aca/Dockerfile` | Slim Python 3.12 image with numpy/pandas/duckdb/statsmodels + the forecast package |
| `services/forecast/aca/requirements.txt` | Pinned deps for the worker image (no LightGBM/Prophet/torch) |
| `infra/forecast_sarima_caj.bicep` | CAJ + user-assigned MI + container-scoped RBAC |

## Local smoke test (no Azure spend)

```bash
python -m services.forecast.distributed_driver \
  --config services/forecast/configs/supplements_price.yml \
  --partitions 3 \
  --mode local \
  --partial-dir /tmp/forecast/partial \
  --record-dir /tmp/forecast/records \
  --out /tmp/forecast/forecast.parquet \
  --clean -v
```

Validated 2026-05-23 against the real `stasiprod1eus2/healthcare/silver/snapshot=2026-05-23-v2/`:
3 partitions × 114 series → 2,736 forecast rows in 3.9 s wall-clock.

## Build the container image

```bash
# Tag with date + short SHA so we can roll back.
TAG="$(date -u +%Y%m%d)-$(git rev-parse --short HEAD)"

# Build in ACR (no local Docker daemon needed; honors feedback_github_actions_unavailable).
az acr build \
  --registry acrasiprod1eus2 \
  --image asi-forecast:${TAG} \
  --image asi-forecast:latest \
  --file services/forecast/aca/Dockerfile \
  .
```

Cost estimate: ACR Tasks Standard tier ~$0.001/minute, build takes ~3 min →
**< $0.01 per build**.

## Deploy the CAJ (one-time)

```bash
az deployment group create \
  --resource-group rg-asi-prod1-eus2 \
  --template-file infra/forecast_sarima_caj.bicep \
  --parameters environmentName=cae-asi-prod1-eus2
```

Bicep creates:
- User-assigned managed identity `id-caj-forecast-prod1`
- 3 role assignments (Storage Blob Data Reader + Contributor on the
  `healthcare` container; AcrPull on `acrasiprod1eus2`)
- `caj-forecast-sarima-prod1` CAJ — Manual trigger, 1 vCPU / 2 GiB,
  60-min timeout

Re-runs are idempotent (Bicep `existing` references + GUID role-assignment
names).

## Trigger a distributed run

```bash
# Step 1: invoke N replicas of the CAJ
python -m services.forecast.distributed_driver \
  --config 'az://stasiprod1eus2.blob.core.windows.net/healthcare/configs/supplements_price.yml' \
  --partitions 16 \
  --mode aca \
  --partial-dir 'az://stasiprod1eus2.blob.core.windows.net/healthcare/forecast_runs/$(date +%Y-%m-%d)/partial' \
  --record-dir  'az://stasiprod1eus2.blob.core.windows.net/healthcare/forecast_runs/$(date +%Y-%m-%d)/records' \
  --out         'az://stasiprod1eus2.blob.core.windows.net/healthcare/silver/snapshot=$(date +%Y-%m-%d)/forecast.parquet' \
  --job-name caj-forecast-sarima-prod1 \
  --rg rg-asi-prod1-eus2 \
  --image-tag acrasiprod1eus2.azurecr.io/asi-forecast:latest \
  --timeout-s 3600 --poll-interval-s 30 \
  -v
```

The driver polls record blobs until all N return `status:completed`.
On any failure, it exits non-zero with the offending partition IDs.

## Cost estimate per full SDUD run

| Component | Unit cost | Quantity | Subtotal |
|---|---|---|---|
| CAJ Consumption (1 vCPU + 2 GiB) | $0.000024/vCPU-s + $0.0000025/GiB-s | 16 partitions × 30 min | **~$0.80** |
| ACR Tasks build | $0.001/min | 3 min | $0.003 |
| Storage egress (read silver, write partials) | $0.087/GB after 100 GB/month free | < 1 GB | $0 |
| Storage read transactions | $0.0044 per 10k | < 10k | < $0.01 |
| **Total per scheduled run** | | | **~$0.81** |

Per `feedback_azure_cloud_job_runs`: get sign-off before scheduling a
recurring cron. Single-shot manual triggers under $1 are operator-
authorized via this runbook.

## Polling + completion semantics

Each worker writes `partition_NNN.json` on exit (success or fail) with:

```json
{
  "partition_id": 7,
  "n_partitions": 16,
  "n_series": 932,
  "n_rows": 5592,
  "started_at": 1779600000.123,
  "ended_at": 1779601812.456,
  "status": "completed",
  "artifact": "az://.../forecast_part_007.parquet"
}
```

`poll_completion()` raises `TimeoutError` if all N records don't appear
within `--timeout-s`. The driver then exits non-zero and the caller can:

1. Re-trigger only the missing partitions (driver flag TBD — current
   workflow is to wipe `--partial-dir` and retry from scratch).
2. Inspect the ACA replica logs:
   ```bash
   az containerapp job execution list \
     --name caj-forecast-sarima-prod1 \
     --resource-group rg-asi-prod1-eus2 \
     --query "[].{name:name, status:properties.status, start:properties.startTime}" -o table
   az containerapp job execution show \
     --name caj-forecast-sarima-prod1 \
     --resource-group rg-asi-prod1-eus2 \
     --job-execution-name <execution-name>
   ```

## Failure modes + fixes

| Symptom | Likely cause | Fix |
|---|---|---|
| `_duckdb.IOException: opening container failed` | MI lacks Storage Blob Data Reader on `healthcare` container | Re-deploy the bicep; the role-assignment guid is idempotent |
| `MissingTokenError` in worker | `AZURE_CLIENT_ID` not injected | Bicep sets this; if missing, check `userAssignedIdentities` on the job |
| Single partition runs > timeout | Partition has a long-tail of slow SARIMA fits | Bump `--partitions` (smaller per-partition load) or `--timeout-s` |
| `forecast_part_*.parquet` empty | Partition had 0 series (small panel + many partitions) | Reduce `--partitions` |
| All partitions completed but `n_series=0` | Silver URL wrong, or NULL `price_date` filtered everything | Verify with `az storage blob exists`; check the SQL WHERE clause |

## Coordination with the rest of the pipeline

This driver runs the **fit** stage only. It does NOT run:
- Backtest (rolling-origin) — needs the full panel; runs on the driver host or as a separate single-replica CAJ
- Ensemble / conformal calibration — driver-side after assembly
- Final-test scoring — driver-side after assembly
- MLflow logging — driver-side (the per-partition record JSON is the
  per-replica trace; MLflow run aggregates them)

The full v0.4 production cron would chain:

```
caj-forecast-sarima-prod1 × N  (this CAJ, T1.1)
        ↓ records assembled
caj-forecast-finalize-prod1    (single replica, runs ensemble + conformal
                                + final-test + writes forecast.parquet)
        ↓ writes
mlflow.register_model           (T1.2 — separate runbook)
```

`caj-forecast-finalize-prod1` is a separate CAJ — out of scope for T1.1.

## Tests

```bash
pytest tests/test_forecast_distributed.py -v
# 6 tests cover series_partition_id stability + uniformity, partition_panel
# round-trip, assemble_partitions concat, poll_completion ordering + timeout.
```

End-to-end local mode tested manually against real Azure silver (above).
ACA mode requires the image to be built + bicep deployed — gated by
operator cost sign-off.
