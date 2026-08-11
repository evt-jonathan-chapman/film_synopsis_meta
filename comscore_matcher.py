"""
comscore_matcher.py
-------------------
Match Comscore box-office rows (IBOE_TITLES + IBOE_FLASH_GROSS) → EVT film_id.

Matching logic mirrors the latest TMDB StudioLookup
(cinema_admits_models/depreciated/tmdb/studio_lookup.py):

  - rapidfuzz ratio + token_sort_ratio on NFKD-normalised titles
  - variant/festival prefix stripping ("GC", "3D", "IMAX", "TFF -", ...)
  - length-ratio guard (rejects short-title impostors like AVATAR→TÁR)
  - ±1-year window over Comscore release_date vs EVT rel_at
  - date-proximity tie-breaker when titles tie
  - confidence tiers: high (>=0.92 and ≤365 days off), borderline, unmatched, manual
  - review file + user-editable manual-override parquet

No API calls — Comscore data is pulled once via SQL and matched in-memory,
so the entire flow is CPU/regex work.

The generic matching engine (title normalisation/scoring, variant
propagation, cache/review/override I/O) lives in title_matcher.py and is
shared with gower_matcher.py::GowerMatcher — this class only supplies
Comscore's column names and output paths.

Outputs (under DATA_DIR/title_matching/comscore/):
  - comscore_cache.parquet           keyed on EVT film_id
  - comscore_review_needed.parquet   borderline+unmatched, top-5 candidates
  - comscore_manual_overrides.csv    user fills manual_override_cs_id
"""

from config import DATA_DIR
from title_matcher import FuzzyTitleMatcher


class ComscoreMatcher(FuzzyTitleMatcher):

    COMSCORE_DIR         = DATA_DIR / "title_matching" / "comscore"
    CACHE_PATH           = str(COMSCORE_DIR / "comscore_cache.parquet")
    REVIEW_PATH          = str(COMSCORE_DIR / "comscore_review_needed.parquet")
    MANUAL_OVERRIDES     = str(COMSCORE_DIR / "comscore_manual_overrides.parquet")  # legacy
    MANUAL_OVERRIDES_CSV = str(COMSCORE_DIR / "comscore_manual_overrides.csv")      # preferred

    SOURCE_LABEL = "Comscore"

    # IBOE_TITLES columns we score against (max wins). Lower-cased by
    # SnowFlakeBase.return_query_output, so use lowercase here.
    # upper_name is a duplicate of film_name capitalised — _normalise_title
    # lowercases anyway, so the extra column doesn't change scores; kept for
    # symmetry with the SQL and in case Comscore ever diverges them.
    TITLE_COLS = ["film_name", "upper_name", "title_aka", "us_title_name", "short_name"]
    DATE_COL   = "release_date"
    ID_COL     = "title_global_id"
    ALT_CONTENT_COL = "is_alt_content"

    # Cache/review/override column names — preserved exactly from before this
    # was refactored onto FuzzyTitleMatcher, since cinema_admits_models reads
    # comscore_cache.parquet directly by these names.
    ID_FIELD             = "cs_id"
    TITLE_FIELD          = "cs_title"
    DATE_FIELD           = "cs_release_date"
    MATCHED_TITLE_FIELD  = "matched_cs_title"
