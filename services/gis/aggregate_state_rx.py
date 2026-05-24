"""Aggregate CMS Medicaid State Drug Utilization Data (2024) to per-state
JSON suitable for a choropleth map. Reads the public CSV directly from
data.medicaid.gov via DuckDB httpfs (with optional local cache), applies the
same drug_class heuristic used in services/powerbi/fabric_publish.py, and
writes a single JSON document keyed by USPS state code.

Output schema (per state):
    {
      "state_code": "CA",
      "state_name": "California",
      "region": "West",
      "division": "Pacific",
      "rx_count": 12345678,
      "total_reimb": 2345678901.23,
      "top_class": "Antiretrovirals / HIV",
      "top_class_share": 0.18,
      "top_drugs": [
        {"product_name": "...", "ndc11": "...", "spend": 12345.0, "rx_count": 6789}
      ]
    }

Usage:
    python3 services/gis/aggregate_state_rx.py \
        --year 2024 \
        --out /Users/john/Git/johndegraft-app/public/demo/healthcare/us-rx-by-state.json
"""
from __future__ import annotations
import argparse, json, pathlib, sys, time

import duckdb

# Reuse the same publisher's state + drug-class taxonomies so the map and the
# Power BI dashboard agree on definitions.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1].parent))
from services.powerbi.fabric_publish import (  # type: ignore
    STATE_REGION_DIVISION, DRUG_CLASS_PREFIXES,
)

STATE_NAMES = {
    "AL": "Alabama","AK":"Alaska","AZ":"Arizona","AR":"Arkansas","CA":"California",
    "CO":"Colorado","CT":"Connecticut","DE":"Delaware","DC":"District of Columbia",
    "FL":"Florida","GA":"Georgia","HI":"Hawaii","ID":"Idaho","IL":"Illinois","IN":"Indiana",
    "IA":"Iowa","KS":"Kansas","KY":"Kentucky","LA":"Louisiana","ME":"Maine","MD":"Maryland",
    "MA":"Massachusetts","MI":"Michigan","MN":"Minnesota","MS":"Mississippi","MO":"Missouri",
    "MT":"Montana","NE":"Nebraska","NV":"Nevada","NH":"New Hampshire","NJ":"New Jersey",
    "NM":"New Mexico","NY":"New York","NC":"North Carolina","ND":"North Dakota","OH":"Ohio",
    "OK":"Oklahoma","OR":"Oregon","PA":"Pennsylvania","RI":"Rhode Island","SC":"South Carolina",
    "SD":"South Dakota","TN":"Tennessee","TX":"Texas","UT":"Utah","VT":"Vermont","VA":"Virginia",
    "WA":"Washington","WV":"West Virginia","WI":"Wisconsin","WY":"Wyoming",
    "PR":"Puerto Rico","VI":"U.S. Virgin Islands","GU":"Guam","AS":"American Samoa","MP":"Northern Mariana Is.",
}

# Supplemental brand-name taxonomy for actual top Medicaid drugs (the
# publisher's prefix list is generic-only; ~87% of brand spend was falling
# into "Other" without these). Order matters: more specific matches first.
GIS_BRAND_PREFIXES: list[tuple[str, list[str]]] = [
    ("HIV / Antiretroviral",
     ["BIKTARVY", "DESCOVY", "TRIUMEQ", "GENVOYA", "DOVATO", "TRUVADA",
      "ODEFSEY", "STRIBILD", "JULUCA", "TIVICAY", "PREZISTA", "ISENTRESS",
      "EPCLUSA", "MAVYRET", "HARVONI", "SUNLENCA", "CABENUVA"]),
    ("Diabetes (GLP-1 / SGLT2)",
     ["OZEMPIC", "MOUNJARO", "RYBELSUS", "TRULICITY", "VICTOZA", "WEGOVY",
      "JARDIANCE", "FARXIGA", "INVOKANA", "STEGLATRO", "BYDUREON", "ZEPBOUND"]),
    ("Diabetes (Insulin / Other)",
     ["HUMALOG", "LANTUS", "NOVOLOG", "TRESIBA", "BASAGLAR", "LEVEMIR",
      "ADMELOG", "TOUJEO", "FIASP", "JANUVIA", "JANUMET", "ACTOS",
      "METFORMIN", "GLIPIZIDE", "GLYBURIDE"]),
    ("Autoimmune / Biologic",
     ["HUMIRA", "ENBREL", "STELARA", "DUPIXENT", "RINVOQ", "SKYRIZI",
      "OTEZLA", "COSENTYX", "TREMFYA", "TALTZ", "XELJANZ", "OLUMIANT",
      "CIMZIA", "REMICADE", "ORENCIA", "SIMPONI"]),
    ("Mental Health",
     ["ABILIFY", "LATUDA", "VRAYLAR", "CAPLYTA", "REXULTI", "INVEGA",
      "RISPERDAL", "SEROQUEL", "ZYPREXA", "GEODON", "CLOZARIL", "HALDOL",
      "FLUOXETINE", "SERTRALINE", "BUPROPION", "VENLAFAXIN", "CITALOPRAM",
      "ESCITALOPR", "QUETIAPINE", "ARIPIPRAZO", "RISPERIDON", "OLANZAPINE",
      "CLOZAPINE", "LITHIUM", "DULOXETINE", "TRAZODONE", "MIRTAZAPIN",
      "PAROXETINE", "DESVENLAFA", "LURASIDONE", "PROZAC", "ZOLOFT",
      "LEXAPRO", "CYMBALTA", "EFFEXOR", "WELLBUTRIN"]),
    ("Cystic Fibrosis",
     ["TRIKAFTA", "KALYDECO", "SYMDEKO", "ORKAMBI"]),
    ("Oncology",
     ["IBRANCE", "IMBRUVICA", "REVLIMID", "POMALYST", "JAKAFI", "VENCLEXTA",
      "TAGRISSO", "VERZENIO", "LYNPARZA", "TRUSELTIQ", "KEYTRUDA", "OPDIVO",
      "DARZALEX", "XTANDI", "ZYTIGA", "ERLEADA", "NUBEQA", "CALQUENCE",
      "BRUKINSA", "TAFINLAR"]),
    ("Anticoagulant",
     ["ELIQUIS", "XARELTO", "PRADAXA", "SAVAYSA", "WARFARIN", "COUMADIN",
      "APIXABAN", "RIVAROXABA"]),
    ("Cardiovascular",
     ["ENTRESTO", "REPATHA", "PRALUENT", "VASCEPA", "LIVALO",
      "LISINOPRIL", "ATORVASTAT", "ROSUVASTAT", "SIMVASTATI", "METOPROLOL",
      "AMLODIPINE", "LOSARTAN", "VALSARTAN", "CARVEDILOL", "FUROSEMIDE",
      "HYDROCHLOR", "DILTIAZEM", "CLOPIDOGRE", "ATENOLOL", "PRAVASTATI",
      "ENALAPRIL", "BENAZEPRIL", "NIFEDIPINE", "RAMIPRIL"]),
    ("Respiratory / Asthma",
     ["TRELEGY", "BREO", "SYMBICORT", "ADVAIR", "DULERA", "BEVESPI",
      "ANORO", "STIOLTO", "INCRUSE", "SPIRIVA", "FLOVENT", "PULMICORT",
      "ALBUTEROL", "VENTOLIN", "PROAIR", "DALIRESP", "FASENRA", "NUCALA",
      "XOLAIR", "MONTELUKAST", "SINGULAIR"]),
    ("ADHD / Stimulant",
     ["VYVANSE", "ADDERALL", "CONCERTA", "FOCALIN", "STRATTERA", "QELBREE",
      "RITALIN", "DAYTRANA", "INTUNIV", "JORNAY"]),
    ("Hepatitis / Liver",
     ["EPCLUSA", "MAVYRET", "HARVONI", "SOVALDI", "VOSEVI", "ZEPATIER",
      "VIEKIRA", "TECHNIVIE"]),
    ("Corticosteroid / Anti-Inflammatory",
     ["DEXAMETHA", "PREDNISONE", "PREDNISOLO", "METHYLPRED", "HYDROCORTI",
      "BETAMETHAS", "TRIAMCINOL", "BUDESONIDE", "FLUTICASON", "MOMETASON"]),
    ("Pain / Opioid",
     ["OXYCODONE", "OXYCONTIN", "HYDROCODON", "MORPHINE", "FENTANYL",
      "TRAMADOL", "PERCOCET", "VICODIN", "SUBOXONE", "METHADONE",
      "GABAPENTIN", "PREGABALIN", "LYRICA", "CYMBALTA"]),
    ("Hormonal / Reproductive",
     ["NUVARING", "MIRENA", "NEXPLANON", "DEPO-PROV", "YAZ", "LO LOESTRIN",
      "TRI-LO", "NUVARING", "ESTRADIOL", "PROGESTERO", "TESTOSTER"]),
]

# Build a single SQL expression mapping product_name prefixes → drug class.
# DRUG_CLASS_PREFIXES shape: [(class_name, [prefix1, prefix2, ...]), ...]
def _drug_class_case_sql(col: str = "product_name") -> str:
    cases = []
    for cls, prefixes in GIS_BRAND_PREFIXES + list(DRUG_CLASS_PREFIXES):
        safe_cls = cls.replace("'", "''")
        for p in prefixes:
            safe_p = p.replace("'", "''")
            cases.append(
                f"WHEN starts_with(upper({col}), '{safe_p}') THEN '{safe_cls}'"
            )
    return "CASE " + " ".join(cases) + " ELSE 'Other / Unknown' END"

SDUD_URL_TMPL = "https://download.medicaid.gov/data/sdud-{year}-updated.csv"
# 2020 has a different URL stem; not needed for default 2024 run.

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache-dir", default="/tmp/sdud-cache")
    ap.add_argument("--top-drugs-per-state", type=int, default=5)
    args = ap.parse_args()

    url = SDUD_URL_TMPL.format(year=args.year)
    cache = pathlib.Path(args.cache_dir) / f"sdud-{args.year}.csv"
    cache.parent.mkdir(parents=True, exist_ok=True)

    if not cache.exists() or cache.stat().st_size < 1_000_000:
        print(f"downloading {url} → {cache}")
        import urllib.request
        t0 = time.time()
        urllib.request.urlretrieve(url, cache)
        print(f"  done in {time.time()-t0:.1f}s, {cache.stat().st_size/1e6:.1f} MB")
    else:
        print(f"using cached {cache} ({cache.stat().st_size/1e6:.1f} MB)")

    con = duckdb.connect()
    con.execute(f"""
        CREATE OR REPLACE VIEW sdud AS
        SELECT
          "State" AS state_code,
          "Product Code" AS ndc11,
          "Product Name" AS product_name,
          CAST("Number of Prescriptions" AS DOUBLE) AS rx_count,
          CAST("Total Amount Reimbursed" AS DOUBLE) AS total_reimb,
          {_drug_class_case_sql('"Product Name"')} AS drug_class
        FROM read_csv('{cache.as_posix()}', header=true, ignore_errors=true)
        WHERE "Product Name" IS NOT NULL
          AND "State" IS NOT NULL
          AND CAST("Total Amount Reimbursed" AS DOUBLE) > 0
    """)
    # Drop placeholder/national-aggregate rows
    con.execute("CREATE OR REPLACE VIEW sdud_clean AS SELECT * FROM sdud WHERE state_code IN ('"
                + "','".join(STATE_NAMES) + "')")

    # 1) per-state totals
    state_totals = con.execute("""
        SELECT state_code,
               SUM(rx_count) AS rx_count,
               SUM(total_reimb) AS total_reimb
        FROM sdud_clean
        GROUP BY state_code
    """).fetchall()

    # 2) top class per state (by spend), excluding "Other / Unknown" — ranked
    #    AFTER filtering so we always surface the #1 categorized class
    top_class_rows = con.execute("""
        WITH by_class AS (
          SELECT state_code, drug_class, SUM(total_reimb) AS spend
          FROM sdud_clean
          WHERE drug_class <> 'Other / Unknown'
          GROUP BY state_code, drug_class
        ),
        ranked AS (
          SELECT *, RANK() OVER (PARTITION BY state_code ORDER BY spend DESC) AS r,
                 SUM(spend) OVER (PARTITION BY state_code) AS state_total
          FROM by_class
        )
        SELECT state_code, drug_class, spend, state_total
        FROM ranked
        WHERE r = 1
    """).fetchall()
    # Also keep a fallback row for states whose #1 is "Other / Unknown"
    top_class_inc_other = con.execute("""
        WITH by_class AS (
          SELECT state_code, drug_class, SUM(total_reimb) AS spend
          FROM sdud_clean
          GROUP BY state_code, drug_class
        ),
        ranked AS (
          SELECT *, RANK() OVER (PARTITION BY state_code ORDER BY spend DESC) AS r,
                 SUM(spend) OVER (PARTITION BY state_code) AS state_total
          FROM by_class
        )
        SELECT state_code, drug_class, spend, state_total
        FROM ranked WHERE r = 1
    """).fetchall()

    top_class_excluding_other = {r[0]: (r[1], r[2], r[3]) for r in top_class_rows}
    top_class_any = {r[0]: (r[1], r[2], r[3]) for r in top_class_inc_other}

    # 3) top-N drugs per state by spend
    top_drugs = con.execute(f"""
        WITH agg AS (
          SELECT state_code, ndc11,
                 ANY_VALUE(product_name) AS product_name,
                 SUM(rx_count) AS rx_count,
                 SUM(total_reimb) AS spend,
                 ANY_VALUE(drug_class) AS drug_class
          FROM sdud_clean
          GROUP BY state_code, ndc11
        ),
        ranked AS (
          SELECT *, RANK() OVER (PARTITION BY state_code ORDER BY spend DESC) AS r
          FROM agg
        )
        SELECT state_code, ndc11, product_name, rx_count, spend, drug_class
        FROM ranked
        WHERE r <= {args.top_drugs_per_state}
        ORDER BY state_code, spend DESC
    """).fetchall()

    drugs_by_state: dict[str, list[dict]] = {}
    for sc, ndc, name, rx, spend, klass in top_drugs:
        drugs_by_state.setdefault(sc, []).append({
            "ndc11": ndc,
            "product_name": name,
            "rx_count": int(rx or 0),
            "spend": round(float(spend or 0.0), 2),
            "drug_class": klass,
        })

    out_rows = []
    for state_code, rx_count, total_reimb in state_totals:
        region, division = STATE_REGION_DIVISION.get(state_code, ("", ""))
        # Prefer the non-Other top class when our heuristic was conclusive
        klass_pref = top_class_excluding_other.get(state_code) or top_class_any.get(state_code)
        if klass_pref is None:
            top_class, top_class_spend, state_total = "Other / Unknown", 0.0, 0.0
        else:
            top_class, top_class_spend, state_total = klass_pref
        share = (top_class_spend / state_total) if state_total else 0.0
        out_rows.append({
            "state_code": state_code,
            "state_name": STATE_NAMES.get(state_code, state_code),
            "region": region,
            "division": division,
            "rx_count": int(rx_count or 0),
            "total_reimb": round(float(total_reimb or 0.0), 2),
            "top_class": top_class,
            "top_class_share": round(share, 4),
            "top_drugs": drugs_by_state.get(state_code, []),
        })

    out_rows.sort(key=lambda r: -r["total_reimb"])
    out = {
        "year": args.year,
        "source": "CMS Medicaid State Drug Utilization Data",
        "source_url": "https://data.medicaid.gov/dataset/state-drug-utilization-data",
        "states": out_rows,
        "_meta": {
            "states_count": len(out_rows),
            "total_rx_count": sum(r["rx_count"] for r in out_rows),
            "total_reimb": round(sum(r["total_reimb"] for r in out_rows), 2),
        },
    }
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}")
    print(f"  states: {len(out_rows)}  total_rx: {out['_meta']['total_rx_count']:,}"
          f"  total_reimb: ${out['_meta']['total_reimb']:,.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
