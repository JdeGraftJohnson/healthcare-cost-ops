# Azure silver access — forecast pipeline

How the forecast module reads real silver parquets from Azure Blob.
Settled 2026-05-23 during T1.4 (`docs/FORECAST_ROADMAP.md`).

## Tenant + storage

- **Tenant:** target-prod1 `4f80e7d4-e5ec-4174-b98a-89899a6cf056`
- **Subscription:** `f51a19c6-da25-45ae-8980-2a1e5dbff1e2`
- **Resource group:** `rg-asi-prod1-eus2` (eastus2)
- **Storage account:** `stasiprod1eus2`
  - Standard_LRS, StorageV2, Hot, HNS-enabled (ADLS Gen2)
  - Shared with the financial-system workloads (~202 GB used of 5 PiB)
- **Container:** `healthcare`
- **Canonical silver:** `silver/snapshot=2026-05-23-v2/`
  - `dim_supplement.parquet` (227 KB)
  - `fact_supplement_price.parquet` (57 KB, 977 priced rows)
  - `supplement_ingredient_long.parquet` (597 KB)

## Auth — credential chain (recommended)

The forecast pipeline uses **DuckDB's `azure` extension + `credential_chain`
secret provider**, which transparently picks up whichever Azure identity is
already signed in:

1. `AZURE_*` environment variables (CI / pinned scenarios)
2. Managed Identity (when running on Azure ACA / VM / Functions)
3. `az login` CLI session (interactive development)
4. Visual Studio / IntelliJ tokens
5. Azure PowerShell

In practice for this repo:

- **Local dev:** run `az login` once. Pipeline picks it up automatically.
- **Azure Container App job:** assign a system-assigned managed identity
  with `Storage Blob Data Reader` on the `healthcare` container; no env
  vars or connection strings required.

The pipeline never logs, persists, or echoes the credential. Per
`feedback_no_pii_in_verify`: no `AZURE_STORAGE_CONNECTION_STRING`,
`*_KEY`, or `*_SAS` ever appears in the JSONL ledger
(`services/forecast/tracking.py:_ENV_ALLOW` enforces this).

## Auth — SAS token (fallback)

If a managed identity isn't available and `az login` isn't feasible (e.g.
a third-party CI runner without device-code support), generate a
read-only SAS token:

```bash
EXPIRY=$(date -u -v+30d +%Y-%m-%dT%H:%MZ)   # 30-day TTL
SAS=$(az storage container generate-sas \
  --account-name stasiprod1eus2 \
  --name healthcare \
  --permissions rl \
  --expiry "$EXPIRY" \
  --auth-mode login --as-user \
  -o tsv)
export AZURE_STORAGE_SAS_TOKEN="$SAS"
```

Then in the pipeline override the credential setup to use SAS:

```python
con.execute(
    "CREATE SECRET az_creds (TYPE azure, "
    "PROVIDER config, ACCOUNT_NAME 'stasiprod1eus2', "
    f"SAS_TOKEN '{os.environ['AZURE_STORAGE_SAS_TOKEN']}')"
)
```

(The current `data.py:_ensure_azure_extension` only wires the
credential-chain path. SAS fallback is documented but not yet wired; add a
follow-up issue if a CI runner needs it.)

## Tradeoffs

| Approach | Pros | Cons |
|---|---|---|
| **credential_chain (default)** | No secrets in code/config; honors `az login`, managed identity, env vars in priority order; works locally and on Azure | Requires the runner to have *some* AAD identity that matches the storage RBAC; opaque failure mode when no chain item matches |
| SAS token | Works from anywhere with no Azure SDK login flow; explicit | Has an expiry (30 days here); the token itself is a bearer credential — anyone with it can read; rotation overhead |
| Account key | Works everywhere; never expires | Most dangerous; full account access; should never be in env or code |
| Connection string | Same as account key | Same as account key |

**Decision: credential_chain only.** The SAS path is documented for
emergency use but not wired by default. Account keys and connection
strings are not permitted for this pipeline.

## Config

`services/forecast/configs/supplements_price.yml` reads the silver via:

```yaml
panel:
  sql: |
    SELECT ...
    FROM read_parquet('${SUPPLEMENTS_SILVER_URL:-az://stasiprod1eus2.blob.core.windows.net/healthcare/silver/snapshot=2026-05-23-v2/fact_supplement_price.parquet}')
    WHERE price IS NOT NULL
      AND price > 0
      AND price_date IS NOT NULL
```

The `${VAR:-default}` interpolation is implemented in
`services/forecast/data.py:_resolve_env()`. Override the silver path for
testing against a different snapshot:

```bash
SUPPLEMENTS_SILVER_URL='az://stasiprod1eus2.blob.core.windows.net/healthcare/silver/snapshot=2026-06-15/fact_supplement_price.parquet' \
  python -m services.forecast --config services/forecast/configs/supplements_price.yml
```

Or to point at a local snapshot for offline dev:

```bash
SUPPLEMENTS_SILVER_URL='examples/supplements_2026/out/silver/snapshot=2026-05-23-v2/fact_supplement_price.parquet' \
  python -m services.forecast --config services/forecast/configs/supplements_price.yml
```

## Granting RBAC to a new identity

When a new managed identity or service principal needs to read the silver:

```bash
PRINCIPAL_ID=<sp object id, e.g. asi-github-actions>
az role assignment create \
  --assignee "$PRINCIPAL_ID" \
  --role "Storage Blob Data Reader" \
  --scope /subscriptions/f51a19c6-da25-45ae-8980-2a1e5dbff1e2/resourceGroups/rg-asi-prod1-eus2/providers/Microsoft.Storage/storageAccounts/stasiprod1eus2/blobServices/default/containers/healthcare
```

Container-scoped, not account-scoped — the financial-system workloads in
sibling containers stay independently governed.

## Schema gotchas observed during T1.4

The real silver schema differs from `examples/supplements_2026/build_sample.py`
(synthetic) in two ways that bit the first run:

1. `off_categories_tags` is `VARCHAR[]` in real data (an OFF array of
   `en:...` tags) — the synthetic had a comma-joined `VARCHAR`. The config
   SQL now uses `off_categories_tags[len(off_categories_tags)]` to pick
   the leaf tag.
2. 2 of 977 rows have `NULL price_date`. The config SQL now filters
   `price_date IS NOT NULL`. The synthetic sample has no nulls.

`examples/supplements_2026/build_sample.py` should be updated to mirror
the real schema (VARCHAR[] categories + a couple null dates) so local dev
catches these issues before the Azure read. Tracked as a follow-up.

## What this unlocked

The forecast pipeline now reads the real supplements silver
(977 rows × 32 countries × 369 brands × span 2016-06 → 2026-05) and emits
the same `forecast.parquet` schema Power BI consumes. Compared with the
synthetic baseline:

- Forecast quality drops significantly on real data — backtest MAPE rises
  from ~1.7% (synthetic SARIMA) to ~1.5% (real SARIMA) but the holdout
  result shows all methods collapse to ~4.2% on the untouched window.
  Methods are statistically harder to separate.
- SARIMA's empirical 80% coverage is *higher* on real data (84% backtest,
  74% holdout vs 62-64% on synthetic) — the synthetic was over-confident.
- The sanity-gate fallback triggers ~13× per run on real data (vs ~5×
  synthetic) because real prices have heavier tails than the synthetic
  `N(1, 0.08)` noise model.

These differences are the value of running against real data instead of
shipping a "looks good on synthetic" benchmark.
