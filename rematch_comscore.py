
"""
rematch_tmdb.py
---------------
Re-score the entire TMDB cache against the catalog with the current matching
algorithm. Phase C — applies the rapidfuzz/WRatio + variant-prefix
normalisation + release-date tie-breaker introduced in session 25e to films
that were matched under the old difflib-based algorithm.

Behaviour:
  - Iterates every row in tmdb_cache.parquet
  - Manual overrides (match_confidence == "manual") are left untouched
  - For each film: re-scores against catalog, picks winner with new algo
  - If new tmdb_id == cached tmdb_id → updates match_confidence + days_diff
    only (no API call, details unchanged)
  - If new tmdb_id != cached tmdb_id → re-fetches details from TMDB
  - Writes updated tmdb_cache.parquet + fresh tmdb_review_needed.parquet
    listing every borderline / unmatched film for manual triage

Pre-requisites:
  - Catalog parquets already downloaded at tmdb/catalog/tmdb_catalog_*.parquet
    (run tmdb_studio_audit.py once if missing — its download_catalog() call
    populates them).
  - TMDB API key in config.yaml (read via StudioLookup.from_config)

Inline flag:
  REFETCH_CHANGED = True   re-fetches details for films whose match changes.
                           Set False for a dry-run that only re-scores and
                           previews the impact in the cache + review file.

Run from IPython:
    %run build_data/rematch_tmdb.py
"""

from evt_back_up.set_paths import paths
paths()

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import CONFIG_PATH, DATA_ROOT, TMDB_DIR

from IPython import get_ipython
def reload_packages():
    ipython = get_ipython()
    if ipython is not None:
        ipython.run_line_magic('load_ext', 'autoreload')
        ipython.run_line_magic('autoreload', '2')
reload_packages()

import glob
import pandas as pd

from helper_fucntions import Helper
from studio_lookup import StudioLookup

# =============================================================================
# Inline flags — edit before running
# =============================================================================
REFETCH_CHANGED = True   # False = dry-run; only updates score/confidence
TRAIN_DATE      = "20260129"
PRED_DATE       = "20260430"

COMSCORE_SQL_PATH = "/Users/jonathanchapman/Documents/git/film_synopsis_meta/sql/comscore_extract.sql"

# =============================================================================
# Config + lookup
# =============================================================================
config = Helper.return_config(str(CONFIG_PATH))
lookup = StudioLookup.from_config(config)
print(f"Cache: {len(lookup._cache):,} films at {lookup.CACHE_PATH}")

# =============================================================================
# Build films_df (train + test + pred universe) — mirrors tmdb_studio_audit.py.
# Used to give re_score_cache real EVT release dates for the date tie-breaker
# and rel_at/dstbtr for the review file.
# =============================================================================
sf = DATA_ROOT / "raw_from_snowflake"
train = pd.read_parquet(sf / TRAIN_DATE / "train" / "train_raw_ds.parquet")
test  = pd.read_parquet(sf / TRAIN_DATE / "test"  / "test_raw_ds.parquet")
pred  = pd.read_parquet(DATA_ROOT / "prediction_from_snowflake" / PRED_DATE / "prediction_raw.parquet")

film_look_up = pd.read_parquet(DATA_ROOT / "look_ups" / "film_lookup.parquet")
flu_dedup    = film_look_up[["film_id", "film"]].drop_duplicates("film_id")

films_df = (
    pd.concat([
        train[["film_id", "rel_at", "dstbtr"]],
        test [["film_id", "rel_at", "dstbtr"]],
        pred [["film_id", "rel_at", "dstbtr"]],
    ])
    .drop_duplicates("film_id")
    .merge(flu_dedup, on="film_id", how="left")
    .reset_index(drop=True)
)
films_df = films_df[films_df["film"].notna()].reset_index(drop=True)
print(f"films_df: {len(films_df):,} films from train({TRAIN_DATE}) + test + pred({PRED_DATE})")

# =============================================================================
# Load full multi-year catalog
# =============================================================================
cat_files = sorted(glob.glob(str(TMDB_DIR / "catalog" / "tmdb_catalog_*.parquet")))
if not cat_files:
    raise FileNotFoundError(
        f"No catalog files at {TMDB_DIR}/catalog/ — run tmdb_studio_audit.py "
        f"once to populate them via lookup.download_catalog()."
    )
catalog = pd.concat([pd.read_parquet(f) for f in cat_files], ignore_index=True)
print(f"Catalog: {len(catalog):,} entries across {len(cat_files)} year files "
      f"({cat_files[0].split('_')[-1][:4]}-{cat_files[-1].split('_')[-1][:4]})")

# =============================================================================
# Run re-score
# =============================================================================
print(f"\nREFETCH_CHANGED = {REFETCH_CHANGED}  "
      f"({'will fetch new details for changed matches' if REFETCH_CHANGED else 'DRY RUN — no API calls'})")
lookup.re_score_cache(catalog, films_df=films_df, refetch_changed=REFETCH_CHANGED)

print(f"\nDone. Review the changes:")
print(f"  cache    → {lookup.CACHE_PATH}")
print(f"  review   → {lookup.REVIEW_PATH}")
print(f"  populate → {lookup.MANUAL_OVERRIDES}  (fill manual_override_tmdb_id then re-run)")
