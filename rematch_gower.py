"""
rematch_gower.py
-----------------
Match EVT films → Gower (ENT_FORECAST_PRD.CURATED.GW_LIFE_TIME) rows.

1. Pulls the Gower extract from Snowflake (sql/gower_export.sql).
2. Builds the EVT films work-set via rematch_comscore.py::load_evt_films —
   same loader Comscore uses, so both sources score against an identical
   EVT film catalogue (concert films + festival distributors excluded).
3. Runs GowerMatcher.build_mapping — writes:
     gower_cache.parquet           keyed on EVT film_id
     gower_review_needed.parquet   borderline+unmatched for triage

On subsequent runs:
  - Films already matched (score ≥ threshold) are skipped.
  - Films previously below threshold are retried — useful when the Gower
    extract has been refreshed or matching rules have changed.
  - Manual overrides (gower_manual_overrides.csv, column `gower_id`) aren't
    wired up with a real file yet — the mechanism is inherited from
    title_matcher.py and works the moment that file exists, mirroring
    Comscore's comscore_manual_overrides.csv.
"""

import pandas as pd

from config import GOWER_SQL_PATH
from base_snowflake import SnowFlakeBase
from config import SF_WAREHOUSE, SF_DATABASE, SF_SCHEMA, SF_RSA_KEY
from gower_matcher import GowerMatcher
from rematch_comscore import load_evt_films

# Test knob — set to an int to match against a random sample, None for full run.
LIMIT_FILMS: int | None = None
RANDOM_SEED = 42

_GW_COLS = [
    "primary_title_no", "title", "rel_date", "snapshot_date",
    "snapshot_type", "life_time_base",
]


def pull_gower() -> pd.DataFrame:
    sb = SnowFlakeBase(SF_WAREHOUSE, SF_SCHEMA, SF_DATABASE)
    sb.create_snowflake_connection(SF_RSA_KEY)
    with open(GOWER_SQL_PATH) as f:
        sql = f.read()
    gw_raw = sb.return_query_output(sql)
    missing = [c for c in _GW_COLS if c not in gw_raw.columns]
    if missing:
        raise ValueError(f"Gower extract missing expected columns: {missing}")
    gw = gw_raw[_GW_COLS].drop_duplicates().reset_index(drop=True)
    gw['rel_date'] = pd.to_datetime(gw['rel_date'], errors='coerce')
    print(f"Gower: {len(gw_raw):,} raw rows → {len(gw):,} unique rows, "
          f"{gw['rel_date'].min().date()} → {gw['rel_date'].max().date()}")
    print("Gower titles by year:")
    print(gw['rel_date'].dt.year.value_counts().sort_index().to_string())
    return gw


def refresh_gower_match(limit: int | None = None, random_seed: int = 42) -> dict:
    """Pull Gower, load EVT films, run the matcher. Returns summary dict for Dagster."""
    gw_df    = pull_gower()
    films_df = load_evt_films()

    if limit is not None:
        films_df = films_df.sample(min(limit, len(films_df)),
                                    random_state=random_seed).reset_index(drop=True)
        print(f"Sampled {len(films_df):,} films (limit={limit}, seed={random_seed})")

    matcher = GowerMatcher(carry_cols=["life_time_base"])
    matcher.build_mapping(films_df, gw_df)

    conf = matcher.mapping_df["match_confidence"].fillna("legacy").value_counts().to_dict()
    return {
        "in_scope":   int(len(matcher.mapping_df)),
        "high":       int(conf.get("high", 0)),
        "borderline": int(conf.get("borderline", 0)),
        "unmatched":  int(conf.get("unmatched", 0)),
        "manual":     int(conf.get("manual", 0)),
        "cache_path": matcher.CACHE_PATH,
    }


if __name__ == "__main__":
    summary = refresh_gower_match(limit=LIMIT_FILMS, random_seed=RANDOM_SEED)
    print(f"\nSummary: {summary}")
