"""Build a v2-shaped silver panel for local forecast development.

Production reads the real Azure silver at
`silver/snapshot=2026-05-23-v2/fact_supplement_price.parquet`. Locally we
mirror that schema and scale (977 priced rows, 10 countries, DSLD ingredient
vocabulary) using public knowledge of the DSLD category landscape — no
licensed corpus required to develop the forecast pipeline against.

The forecast YAML reads the same parquet via the SUPPLEMENTS_SILVER env var,
so once the operator points it at the real Azure blob, the same code runs
unchanged in production.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

LOG = logging.getLogger("supplements.sample")


DSLD_TOP_CATEGORIES = [
    ("en:vitamin-d-supplements",      "vitamin d"),
    ("en:multivitamins",              "multivitamin"),
    ("en:omega-3-supplements",        "fish oil"),
    ("en:probiotic-supplements",      "probiotic"),
    ("en:magnesium-supplements",      "magnesium"),
    ("en:calcium-supplements",        "calcium"),
    ("en:vitamin-c-supplements",      "vitamin c"),
    ("en:vitamin-b12-supplements",    "vitamin b12"),
    ("en:collagen-supplements",       "collagen"),
    ("en:iron-supplements",           "iron"),
    ("en:zinc-supplements",           "zinc"),
    ("en:turmeric-supplements",       "turmeric"),
]

BRANDS = [
    "now foods", "nature made", "kirkland", "nordic naturals", "thorne",
    "garden of life", "solgar", "puritans pride", "natures bounty",
    "doctors best", "jarrow", "swanson", "life extension", "vital proteins",
]

COUNTRIES = [
    "United States", "United Kingdom", "Canada", "France", "Germany",
    "Spain", "Italy", "Netherlands", "Belgium", "Australia",
]

# Category-level USD base price (typical mid-market 60-day supply, 2020 levels)
CATEGORY_BASE_PRICE = {
    "en:vitamin-d-supplements":      8.50,
    "en:multivitamins":             18.00,
    "en:omega-3-supplements":       22.00,
    "en:probiotic-supplements":     32.00,
    "en:magnesium-supplements":     14.00,
    "en:calcium-supplements":       11.00,
    "en:vitamin-c-supplements":      9.00,
    "en:vitamin-b12-supplements":   12.00,
    "en:collagen-supplements":      28.00,
    "en:iron-supplements":          10.00,
    "en:zinc-supplements":           9.50,
    "en:turmeric-supplements":      17.00,
}


def synthesize(seed: int = 42, n_rows: int = 977, start: str = "2020-01-15", end: str = "2026-04-30") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    span_days = (end_ts - start_ts).days

    rows = []
    for i in range(n_rows):
        cat_tag, _ingredient = DSLD_TOP_CATEGORIES[rng.integers(len(DSLD_TOP_CATEGORIES))]
        brand = BRANDS[rng.integers(len(BRANDS))]
        country = COUNTRIES[rng.integers(len(COUNTRIES))]
        days_offset = int(rng.integers(span_days))
        price_date = (start_ts + pd.Timedelta(days=days_offset)).date()

        base = CATEGORY_BASE_PRICE[cat_tag]
        years_since_start = days_offset / 365.25
        # Mild secular inflation + seasonal wave + brand premium + country FX-ish offset + noise.
        inflation = 1.035 ** years_since_start
        seasonal = 1.0 + 0.06 * np.sin(2 * np.pi * (days_offset / 365.25 + 0.25))
        brand_premium = 1.0 + (hash(brand) % 25) / 100.0
        country_offset = {
            "United States": 1.00, "Canada": 1.05, "United Kingdom": 1.18,
            "France": 1.22, "Germany": 1.20, "Spain": 1.12, "Italy": 1.15,
            "Netherlands": 1.21, "Belgium": 1.20, "Australia": 1.10,
        }[country]
        noise = float(rng.normal(1.0, 0.08))
        price = max(0.5, base * inflation * seasonal * brand_premium * country_offset * noise)

        gtin14 = f"{rng.integers(10**13, 10**14):014d}"
        rows.append({
            "price_id":               f"px_{i:06d}",
            "price_date":             pd.Timestamp(price_date),
            "gtin14":                 gtin14,
            "dsld_id":                int(rng.integers(10000, 999999)),
            "product_name":           f"{brand} {cat_tag.split(':')[-1].replace('-', ' ')}",
            "brand":                  brand,
            "off_categories_tags":    cat_tag,
            "price":                  round(price, 2),
            "price_without_discount": None,
            "is_discounted":          False,
            "discount_type":          None,
            "currency":               "USD",
            "price_per":              "UNIT",
            "location_osm_id":        None,
            "location_osm_type":      None,
            "location_name":          None,
            "country":                country,
            "city":                   None,
            "labels_tags":            None,
            "created_at":             pd.Timestamp(price_date),
            "owner":                  "synthetic_sample",
        })
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="examples/supplements_2026/out/silver/snapshot=2026-05-23-v2")
    p.add_argument("--rows", type=int, default=977, help="match the real v2 snapshot scale")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--verbose", "-v", action="store_true")
    a = p.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if a.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-5s %(name)s %(message)s",
    )
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = synthesize(seed=a.seed, n_rows=a.rows)
    out_path = out_dir / "fact_supplement_price.parquet"
    df.to_parquet(out_path, compression="zstd", index=False)
    LOG.info("wrote %d rows -> %s", len(df), out_path)
    LOG.info("categories=%d brands=%d countries=%d",
             df["off_categories_tags"].nunique(), df["brand"].nunique(),
             df["country"].nunique())
    LOG.info("date span=%s..%s", df["price_date"].min().date(), df["price_date"].max().date())
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
