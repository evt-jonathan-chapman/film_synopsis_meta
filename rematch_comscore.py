"""
rematch_comscore.py
-------------------
Match EVT films → Comscore (IBOE_TITLES + IBOE_FLASH_GROSS) rows.

1. Pulls the Comscore extract from Snowflake (sql/comscore_extract.sql).
2. Builds the EVT films work-set from the parquet snapshots (same shape as
   refresh.py::load_films_from_snowflake), joining `film` from film_lookup.
3. Runs ComscoreMatcher.build_mapping — writes:
     comscore_cache.parquet           keyed on EVT film_id
     comscore_review_needed.parquet   borderline+unmatched for triage

On subsequent runs:
  - Films already matched (score ≥ threshold) are skipped.
  - Films previously below threshold are retried — useful when the Comscore
    extract has been refreshed or matching rules have changed.
  - To force a chosen Comscore match for a misfire: edit
    comscore_review_needed.parquet, fill `manual_override_cs_id`, save as
    comscore_manual_overrides.parquet, then re-run.
"""

import glob
import pandas as pd

from config import (
    DATA_DIR, SF_WAREHOUSE, SF_DATABASE, SF_SCHEMA, SF_RSA_KEY,
    RAW_PARQUET_GLOB, RAW_PARQUET_GLOB_TEST, RAW_PARQUET_GLOB_PRED,
    FILM_META_ENRICHED_PATH, COMSCORE_SQL_PATH,
)
from base_snowflake import SnowFlakeBase
from comscore_matcher import ComscoreMatcher
import s3_checkpoint

# Test knob — set to an int to match against a random sample, None for full run.
LIMIT_FILMS: int | None = None
RANDOM_SEED = 42

# Mirrors refresh.py::FILM_META_SKIP_DISTRIBUTORS — festival/event/sports distributors
# that are not commercial cinema releases and will never appear in Comscore.
SKIP_DISTRIBUTORS = {
    "ZZ Japanese Film Festival", "ZZ JEWISH FILM FESTIVAL", "ZZ Russian Film Festival",
    "AU Sydney Science Fiction Film Festival", "AU KOREAN FILM FESTIVAL",
    "AU Taiwan Film Festival", "AU SciFi Film Festival",
    "ZZ Sydney Underground Film Festival Inc", "ZZ Iranian Film Festival Australia",
    "ZZ SOUTH AFRICAN FILM FESTIVAL", "ZZ UKRANIAN FILM FESTIVAL",
    "ZZ JIFF Distribution", "ZZ Flickerfest", "ZZ Gold Coast Film Fantastic Ltd",
    "ZZ SF3 - SMARTFONE FLICK FEST", "ZZ GOETHE INSTITUTE", "ZZ FOR FILMS SAKE",
    "AU Trafalgar Releasing Ltd", "AU Cinema Live", "AU PATHE LIVE",
    "ZZ THE WIGGLES INTERNATIONAL", "AU FATHOM EVENTS",
    "ZZ Fox Sports Venues", "ZZ BeIN SPORTS",
    "ZZ Queensland Cricket Association Ltd", "ZZ ESPN Australia Pty Ltd",
    "AU IMAX THEATRES INTL", "ZZ Nickelodeon Australia Management",
    "ZZ CRUNCHYROLL PTY LTD", "ZZ SBS-ALTERNATE CONTENT",
}


_CS_COLS = [
    "title_global_id", "film_name", "upper_name", "title_aka",
    "us_title_name", "short_name", "synopsis",
    "is_alt_content", "orig_cntry", "cntry_id", "distr_global_id", "release_date",
]


def pull_comscore() -> pd.DataFrame:
    sb = SnowFlakeBase(SF_WAREHOUSE, SF_SCHEMA, SF_DATABASE)
    sb.create_snowflake_connection(SF_RSA_KEY)
    with open(COMSCORE_SQL_PATH) as f:
        sql = f.read()
    cs_raw = sb.return_query_output(sql)
    # The SQL joins IBOE_TITLES to IBOE_FLASH_GROSS_STATE_TITLE, producing one
    # row per AU state for the same title_global_id. Keep only the columns the
    # matcher and diagnostics need, then deduplicate to one row per film.
    missing = [c for c in _CS_COLS if c not in cs_raw.columns]
    if missing:
        raise ValueError(f"Comscore extract missing expected columns: {missing}")
    cs = cs_raw[_CS_COLS].drop_duplicates().reset_index(drop=True)
    cs['release_date'] = pd.to_datetime(cs['release_date'], errors='coerce')
    print(f"Comscore: {len(cs_raw):,} raw rows → {len(cs):,} unique titles, "
          f"{cs['release_date'].min().date()} → {cs['release_date'].max().date()}")
    print("Comscore titles by year:")
    print(cs['release_date'].dt.year.value_counts().sort_index().to_string())
    return cs


def load_evt_films() -> pd.DataFrame:
    """EVT films work-set: full film_lookup catalogue (all EVT films, not just model subset)."""
    films = pd.read_parquet(
        DATA_DIR / "look_ups" / "film_lookup.parquet",
        columns=["film_id", "film", "rel_at", "dstbtr"],
    ).drop_duplicates("film_id").reset_index(drop=True)
    n_null = films["film"].isna().sum()
    if n_null:
        films = films[films["film"].notna()].reset_index(drop=True)
        print(f"Dropped {n_null} rows with null film title")
    print(f"film_lookup: {len(films):,} films")

    # Drop festival/event/sports distributors — not in Comscore, inflates unmatched bucket.
    n_before = len(films)
    films = films[~films["dstbtr"].isin(SKIP_DISTRIBUTORS)].reset_index(drop=True)
    print(f"Dropped {n_before - len(films)} festival/event distributor films")

    # Drop concert films — they're filtered out on the Comscore side too
    # (is_alt_content), so matching against them just inflates the unmatched bucket.
    # film_meta_enriched.parquet is refresh.py's output, which now lives on S3 (see
    # s3_checkpoint.py) rather than this local path — try S3 first, fall back to the
    # local copy (e.g. a main.py-driven local run, or an old cached copy).
    fm = s3_checkpoint.read_parquet("film_meta", "film_meta_enriched.parquet",
                                     columns=["film_id", "adaptation_type"])
    if fm is None and FILM_META_ENRICHED_PATH.exists():
        fm = pd.read_parquet(FILM_META_ENRICHED_PATH, columns=["film_id", "adaptation_type"])
    if fm is not None:
        concert_ids = set(fm.loc[fm["adaptation_type"] == "concert_film", "film_id"])
        n_before = len(films)
        films = films[~films["film_id"].isin(concert_ids)].reset_index(drop=True)
        print(f"EVT films: {len(films):,} (dropped {n_before - len(films)} concert_film entries)")
    else:
        print("film_meta_enriched.parquet not found (S3 or local) — skipping concert film filter")

    films['rel_at'] = pd.to_datetime(films['rel_at'], utc=True, errors='coerce')
    print(f"EVT date range: {films['rel_at'].min().date()} → {films['rel_at'].max().date()}")
    print("EVT films by year:")
    print(films['rel_at'].dt.year.value_counts().sort_index().to_string())
    return films


def refresh_comscore_match(limit: int | None = None, random_seed: int = 42) -> dict:
    """Pull Comscore, load EVT films, run the matcher. Returns summary dict for Dagster."""
    cs_df    = pull_comscore()
    films_df = load_evt_films()

    if limit is not None:
        films_df = films_df.sample(min(limit, len(films_df)),
                                    random_state=random_seed).reset_index(drop=True)
        print(f"Sampled {len(films_df):,} films (limit={limit}, seed={random_seed})")

    matcher = ComscoreMatcher()
    matcher.build_mapping(films_df, cs_df)

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
    summary = refresh_comscore_match(limit=LIMIT_FILMS, random_seed=RANDOM_SEED)
    print(f"\nSummary: {summary}")
