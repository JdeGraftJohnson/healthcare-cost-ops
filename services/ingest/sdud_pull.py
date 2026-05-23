"""Pull Medicaid State Drug Utilization Data (SDUD) from data.medicaid.gov.

The SDUD program publishes one CSV per calendar year, ~5-10M rows annually.
Each row is (State, NDC, Drug Name, Year, Quarter, Suppression Used,
Units Reimbursed, Number of Prescriptions, Total Amount Reimbursed,
Medicaid Amount Reimbursed, Non Medicaid Amount Reimbursed).

Source API: https://data.medicaid.gov/api/1/metastore/schemas/dataset/items
Direct CSV pattern: https://download.medicaid.gov/data/StateDrugUtilizationData-{YEAR}.csv

Bronze writes are year-partitioned to Azure Blob:
  azure://&lt;STORAGE_ACCOUNT&gt;/healthcare/bronze/sdud/year=YYYY/data.csv.gz
"""
from __future__ import annotations

import argparse
import gzip
import io
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import requests

LOG = logging.getLogger("sdud_pull")

SDUD_URL_LEGACY = "https://download.medicaid.gov/data/StateDrugUtilizationData-{year}.csv"
SDUD_URL_2020PLUS = "https://download.medicaid.gov/data/sdud-{year}-updated.csv"
DATASET_REGISTRY = "https://data.medicaid.gov/api/1/metastore/schemas/dataset/items"
CHUNK = 1 << 20  # 1 MiB streaming


def resolve_download_url(year: int) -> str:
    """Resolve the actual download URL by querying the dataset registry.

    Falls back to the documented pattern if registry lookup fails.
    """
    import re
    try:
        offset = 0
        while offset < 500:
            r = requests.get(f"{DATASET_REGISTRY}?limit=50&offset={offset}", timeout=20)
            r.raise_for_status()
            page = r.json()
            if not page:
                break
            for it in page:
                m = re.match(r"State Drug Utilization Data (\d{4})", it.get("title", ""))
                if m and int(m.group(1)) == year:
                    for d in it.get("distribution", []) or []:
                        u = d.get("downloadURL", "")
                        if u.endswith(".csv"):
                            return u
            if len(page) < 50:
                break
            offset += 50
    except Exception as e:
        LOG.warning("registry lookup failed for %d: %s", year, e)
    return (SDUD_URL_2020PLUS if year >= 2020 else SDUD_URL_LEGACY).format(year=year)


@dataclass
class PullResult:
    year: int
    bytes_in: int
    bytes_out: int
    rows: int
    dest: str


def stream_year(year: int, dest_dir: Path) -> PullResult:
    """Stream one year's SDUD CSV to a local gzip file under dest_dir."""
    url = resolve_download_url(year)
    out = dest_dir / f"year={year}" / "data.csv.gz"
    out.parent.mkdir(parents=True, exist_ok=True)

    bytes_in = 0
    rows = 0
    LOG.info("GET %s", url)
    with requests.get(url, stream=True, timeout=180) as r:
        r.raise_for_status()
        with gzip.open(out, "wb", compresslevel=6) as gz:
            for chunk in r.iter_content(chunk_size=CHUNK):
                if not chunk:
                    continue
                bytes_in += len(chunk)
                rows += chunk.count(b"\n")
                gz.write(chunk)
    bytes_out = out.stat().st_size
    LOG.info(
        "year=%d rows=%d in=%.1fMB out=%.1fMB dest=%s",
        year, rows, bytes_in / 1e6, bytes_out / 1e6, out,
    )
    return PullResult(year=year, bytes_in=bytes_in, bytes_out=bytes_out, rows=rows, dest=str(out))


def upload_to_blob(local: Path, blob_url: str) -> None:
    """Upload a local file to Azure Blob using azcopy (preferred) or azure-storage-blob.

    Defers SDK choice to the caller's environment. If the AZCOPY env exposes a
    SAS-bearing URL we shell out; otherwise we require ``AZURE_STORAGE_CONNECTION_STRING``
    in the environment and use the python SDK.
    """
    azcopy = os.environ.get("AZCOPY_PATH")
    if azcopy and os.path.isfile(azcopy):
        import subprocess
        subprocess.run([azcopy, "copy", str(local), blob_url, "--overwrite=true"], check=True)
        return

    try:
        from azure.storage.blob import BlobClient
    except ImportError as e:
        raise SystemExit(
            "azure-storage-blob not installed and AZCOPY_PATH unset; "
            "pip install azure-storage-blob or set AZCOPY_PATH"
        ) from e

    conn = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
    if not conn:
        raise SystemExit("AZURE_STORAGE_CONNECTION_STRING not set")
    # blob_url is expected as az://<account>/<container>/<path>  →  parse 3-part
    assert blob_url.startswith("az://") or blob_url.startswith("azure://"), blob_url
    rest = blob_url.split("://", 1)[1]
    parts = rest.split("/", 2)
    if len(parts) == 3:
        _account, container, blob_path = parts
    else:
        container, _, blob_path = rest.partition("/")
    bc = BlobClient.from_connection_string(conn, container_name=container, blob_name=blob_path)
    with open(local, "rb") as f:
        bc.upload_blob(f, overwrite=True)
    LOG.info("uploaded %s → %s", local, blob_url)


def run(years: Iterable[int], out: str, upload: bool) -> list[PullResult]:
    out_path = Path(out) if not out.startswith(("az://", "azure://")) else Path("/tmp/sdud-stage")
    out_path.mkdir(parents=True, exist_ok=True)
    results: list[PullResult] = []
    for y in years:
        r = stream_year(y, out_path)
        results.append(r)
        if upload and out.startswith(("az://", "azure://")):
            target = f"{out.rstrip('/')}/sdud/year={y}/data.csv.gz"
            upload_to_blob(Path(r.dest), target)
    return results


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--years", required=True, help="comma-separated, e.g. 2020,2021,2022")
    p.add_argument("--out", required=True, help="local dir or azure://container/prefix")
    p.add_argument("--no-upload", action="store_true", help="skip blob upload even for az:// out")
    p.add_argument("--verbose", "-v", action="store_true")
    a = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if a.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-5s %(name)s %(message)s",
    )
    years = [int(y) for y in a.years.split(",") if y.strip()]
    results = run(years, a.out, upload=not a.no_upload)
    LOG.info(
        "SUMMARY years=%d total_rows=%d total_bytes_in=%.1fMB",
        len(results), sum(r.rows for r in results), sum(r.bytes_in for r in results) / 1e6,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
