"""Build the SDUD silver layer from real CMS CSVs over HTTPS.

Aggregates Medicaid State Drug Utilization Data to quarterly state-level
prescription spend + Rx count. Reads via DuckDB's httpfs extension directly
from `download.medicaid.gov` — no local CSV staging, no synthetic data.

Output schema (one row per state × year × quarter):
  state_code              VARCHAR  (drops 'XX' national aggregate row)
  year                    INT
  quarter                 INT
  period                  DATE     (quarter-start: YYYY-MM-01)
  total_reimb             DOUBLE   (sum of Total Amount Reimbursed)
  medicaid_reimb          DOUBLE
  non_medicaid_reimb      DOUBLE
  rx_count                BIGINT   (sum of Number of Prescriptions)
  units_reimbursed        DOUBLE
  n_ndc                   BIGINT   (distinct NDC codes that quarter)
  suppression_rate        DOUBLE   (fraction of NDC rows that were suppressed)

Per `feedback_real_data_only`: real CMS data only — no build_sample.py
synthetic equivalent for SDUD.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import duckdb

LOG = logging.getLogger("sdud_silver_build")

SDUD_URL_TPL = "https://download.medicaid.gov/data/sdud-{year}-updated.csv"
SDUD_URL_LEGACY = "https://download.medicaid.gov/data/StateDrugUtilizationData-{year}.csv"


def build_silver(
    years: list[int],
    out_path: Path,
    *,
    drop_national_xx: bool = True,
    duckdb_path: str = ":memory:",
) -> dict:
    """Pull each year's CSV directly, aggregate to (state, year, quarter)."""
    con = duckdb.connect(duckdb_path)
    con.execute("INSTALL httpfs"); con.execute("LOAD httpfs")
    LOG.info("aggregating SDUD years=%s direct-from-URL", years)

    union_parts = []
    for y in years:
        url = SDUD_URL_TPL.format(year=y)
        union_parts.append(f"""
            SELECT
                State                                           AS state_code,
                TRY_CAST(Year AS INT)                           AS year,
                TRY_CAST(Quarter AS INT)                        AS quarter,
                TRY_CAST("Total Amount Reimbursed" AS DOUBLE)    AS total_reimb,
                TRY_CAST("Medicaid Amount Reimbursed" AS DOUBLE) AS medicaid_reimb,
                TRY_CAST("Non Medicaid Amount Reimbursed" AS DOUBLE) AS non_medicaid_reimb,
                TRY_CAST("Number of Prescriptions" AS BIGINT)    AS rx_count,
                TRY_CAST("Units Reimbursed" AS DOUBLE)           AS units,
                NDC                                              AS ndc,
                "Suppression Used"                               AS suppression_used
            FROM read_csv(
                '{url}',
                header=true, ignore_errors=true, all_varchar=true
            )
        """)
    full_select = " UNION ALL ".join(union_parts)

    where_xx = "AND state_code != 'XX'" if drop_national_xx else ""

    t = time.time()
    con.execute(f"""
        CREATE OR REPLACE TABLE sdud_aggregated AS
        WITH raw AS (
            {full_select}
        )
        SELECT
            state_code,
            year,
            quarter,
            make_date(year, ((quarter - 1) * 3) + 1, 1)                AS period,
            SUM(CASE WHEN suppression_used = 'false' THEN total_reimb ELSE 0 END)    AS total_reimb,
            SUM(CASE WHEN suppression_used = 'false' THEN medicaid_reimb ELSE 0 END) AS medicaid_reimb,
            SUM(CASE WHEN suppression_used = 'false' THEN non_medicaid_reimb ELSE 0 END) AS non_medicaid_reimb,
            SUM(CASE WHEN suppression_used = 'false' THEN rx_count ELSE 0 END)       AS rx_count,
            SUM(CASE WHEN suppression_used = 'false' THEN units ELSE 0 END)          AS units_reimbursed,
            COUNT(DISTINCT ndc)                                         AS n_ndc,
            SUM(CASE WHEN suppression_used = 'true' THEN 1 ELSE 0 END) * 1.0 /
                NULLIF(COUNT(*), 0)                                     AS suppression_rate
        FROM raw
        WHERE state_code IS NOT NULL
          AND year IS NOT NULL AND quarter IS NOT NULL
          {where_xx}
        GROUP BY state_code, year, quarter
        ORDER BY state_code, year, quarter
    """)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    con.execute(f"COPY sdud_aggregated TO '{out_path}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    elapsed = time.time() - t
    summary = con.execute("""
        SELECT
            COUNT(*) AS rows,
            COUNT(DISTINCT state_code) AS n_states,
            MIN(period) AS first_period,
            MAX(period) AS last_period,
            SUM(total_reimb) AS total_reimb_all,
            SUM(rx_count) AS rx_count_all
        FROM sdud_aggregated
    """).fetch_df().iloc[0].to_dict()
    summary["elapsed_s"] = elapsed
    summary["out_path"] = str(out_path)
    LOG.info("silver built: %s", json.dumps(summary, default=str, indent=2))
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="sdud_silver_build")
    p.add_argument("--years", type=int, nargs="+", required=True)
    p.add_argument("--out", required=True, help="output parquet path")
    p.add_argument("--keep-national-xx", action="store_true",
                   help="keep state_code='XX' (national aggregate) rows")
    p.add_argument("--duckdb-path", default=":memory:")
    p.add_argument("--verbose", "-v", action="store_true")
    a = p.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if a.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-5s %(name)s %(message)s",
    )
    summary = build_silver(
        years=a.years, out_path=Path(a.out),
        drop_national_xx=not a.keep_national_xx,
        duckdb_path=a.duckdb_path,
    )
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
