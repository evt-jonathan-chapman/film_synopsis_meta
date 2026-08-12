"""
gower_matcher.py
----------------
Match Gower box-office rows (DBT.EDW_ENT_PRD.CINR_TBM_GW_LIFE_TIME) → EVT
film_id. Same fuzzy-title engine as comscore_matcher.py::ComscoreMatcher
(shared base: title_matcher.py::FuzzyTitleMatcher) — this class only
supplies Gower's column names and output paths.

Gower is thinner than Comscore: one title column (no aka/us_title/short_name
variants) and no title_global_id-style key — `prmry_title_no` is the
closest thing to a stable per-title identifier, so it's used as ID_COL.
There's no is_alt_content-style flag either, so ALT_CONTENT_COL is left None.

The SQL extract (sql/gower_export.sql) returns up to three snapshot rows per
title (`snapshot_type` in latest/1m_pre_release/3m_pre_release) — one title
can therefore appear several times with the same prmry_title_no. _prep_source
picks the 'latest' snapshot row (falling back to 1m/3m pre-release) before
the shared dedup-by-ID_COL logic runs, so scoring only ever sees one row per
title.

Manual overrides are supported by the shared base (gower_manual_overrides.csv
with a `gower_id` column, mirroring Comscore's cs_id override CSV) but no
such file exists yet — deferred until the review file shows it's needed.

Outputs (under DATA_DIR/title_matching/gower/):
  - gower_cache.parquet           keyed on EVT film_id
  - gower_review_needed.parquet   borderline+unmatched, top-5 candidates
  - gower_manual_overrides.csv    user fills manual_override_gower_id (not yet created)
"""

from config import DATA_DIR
from title_matcher import FuzzyTitleMatcher

_SNAPSHOT_PRIORITY = {"latest": 0, "1m_pre_release": 1, "3m_pre_release": 2}


class GowerMatcher(FuzzyTitleMatcher):

    GOWER_DIR            = DATA_DIR / "title_matching" / "gower"
    CACHE_PATH           = str(GOWER_DIR / "gower_cache.parquet")
    REVIEW_PATH          = str(GOWER_DIR / "gower_review_needed.parquet")
    MANUAL_OVERRIDES     = str(GOWER_DIR / "gower_manual_overrides.parquet")  # legacy, unused for now
    MANUAL_OVERRIDES_CSV = str(GOWER_DIR / "gower_manual_overrides.csv")      # preferred, not yet created

    SOURCE_LABEL = "Gower"

    # Lower-cased by SnowFlakeBase.return_query_output, so use lowercase here.
    TITLE_COLS = ["title"]
    DATE_COL   = "rel_date"
    ID_COL     = "prmry_title_no"
    ALT_CONTENT_COL = None

    # Cache/review/override column names — "gower_" prefix mirrors Comscore's
    # "cs_" convention. ID_FIELD is literally "gower_id" per the requested
    # film_id/cs_id/gower_id bridge schema.
    ID_FIELD             = "gower_id"
    TITLE_FIELD          = "gower_title"
    DATE_FIELD           = "gower_release_date"
    MATCHED_TITLE_FIELD  = "matched_gower_title"

    def _prep_source(self, src_df):
        """Pick one row per prmry_title_no before the shared dedup runs —
        prefer 'latest' snapshot, then 1m/3m pre-release, over anything else."""
        src = src_df.copy()
        if "snapshot_type" in src.columns:
            src["_snap_prio"] = src["snapshot_type"].map(_SNAPSHOT_PRIORITY).fillna(3)
            src = src.sort_values([self.ID_COL, "_snap_prio"]).drop(columns="_snap_prio")
        return super()._prep_source(src)
