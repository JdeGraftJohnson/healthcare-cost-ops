"""Container entrypoint for caj-medicaid-rx-gis-prod1.

Runs aggregate_state_rx for the latest SDUD year, then uploads the JSON to
the configured Azure Blob container via a write-capable SAS URL.

Required env:
    BLOB_SAS_URL — full SAS URL targeting the public container,
                   e.g. https://stasiprod1eus2.blob.core.windows.net/healthcare-public?<token>
                   The job writes us-rx-by-state.json inside this container.

Optional env:
    SDUD_YEAR — defaults to 2024 (CMS lag; 2025 partial)
"""
from __future__ import annotations
import os, sys, pathlib, json, subprocess

YEAR = int(os.environ.get("SDUD_YEAR", "2024"))
SAS_URL = os.environ.get("BLOB_SAS_URL")
if not SAS_URL:
    print("FATAL: BLOB_SAS_URL not set"); sys.exit(2)

WORK_DIR = pathlib.Path("/tmp/job")
WORK_DIR.mkdir(parents=True, exist_ok=True)
OUT = WORK_DIR / "us-rx-by-state.json"

rc = subprocess.call([
    "python3", "/app/services/gis/aggregate_state_rx.py",
    "--year", str(YEAR),
    "--out", str(OUT),
    "--cache-dir", str(WORK_DIR / "cache"),
])
if rc != 0:
    print(f"FATAL: aggregator exited {rc}"); sys.exit(rc)

from azure.storage.blob import ContainerClient

cc = ContainerClient.from_container_url(SAS_URL)
blob_name = "us-rx-by-state.json"
print(f"uploading {OUT} → blob {blob_name}")
with OUT.open("rb") as f:
    cc.upload_blob(name=blob_name, data=f, overwrite=True,
                   content_type="application/json")

# Also upload a year-stamped backup so we keep a history
stamped = f"history/us-rx-by-state-{YEAR}.json"
with OUT.open("rb") as f:
    cc.upload_blob(name=stamped, data=f, overwrite=True,
                   content_type="application/json")
print(f"uploaded {blob_name} + {stamped}")

j = json.loads(OUT.read_text())
print(f"states={j['_meta']['states_count']} total_rx={j['_meta']['total_rx_count']:,} "
      f"total_reimb=${j['_meta']['total_reimb']:,.0f}")
