"""
backfill_budget_from_tmdb.py
-----------------------------
Cheap (zero-LLM-cost) backfill of null budget_usd rows in
film_meta_enriched.parquet from a locally-cached TMDB fuzzy-title-match
parquet (film_id -> tmdb_id, budget, match_confidence).

Only fills rows where budget_usd IS NULL — never overwrites an existing
LLM-sourced figure. Only trusts match_confidence == 'high' matches (verified
title match, score 1.0/near-1.0) — 'borderline' TMDB matches are excluded
from the silent backfill because a wrong title match would inject a
different film's budget into a real prediction input, which is worse than
leaving it null. budget values of 0/NaN in the TMDB cache are treated as
"TMDB doesn't have it either", not a real zero-dollar budget.

TMDB_CACHE_PATH is a plain constant, not a config.py path, because this
cache isn't part of the regular extraction pipeline (no matcher/driver
script produces it yet — it's a standalone snapshot). See CLAUDE.md's
Comscore/Gower matchers for the pattern to follow if this needs to become a
proper, refreshable matching path later.

Usage:
    python backfill_budget_from_tmdb.py              # dry run, prints impact only
    python backfill_budget_from_tmdb.py --apply       # writes the backfilled parquet
                                                        # (backs up the original first)
"""
import argparse

import pandas as pd

from config import FILM_META_ENRICHED_PATH

TMDB_CACHE_PATH = "/Users/jonathan_chapman/Documents/data/entertainment/tmdb/tmdb_cache.parquet"

TRUSTED_CONFIDENCE = {"high"}


def backfill_budget_from_tmdb(df: pd.DataFrame, tmdb: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (backfilled df, DataFrame of the rows that were filled, for reporting)."""
    usable = tmdb[
        tmdb["match_confidence"].isin(TRUSTED_CONFIDENCE) & tmdb["budget"].notna() & (tmdb["budget"] > 0)
    ][["film_id", "budget", "tmdb_title", "match_score"]]

    out = df.merge(usable, on="film_id", how="left")
    fill_mask = out["budget_usd"].isna() & out["budget"].notna()

    filled_report = out.loc[
        fill_mask, ["film_id", "title", "tmdb_title", "budget", "match_score"]
    ].rename(columns={"budget": "new_budget_usd"})

    out.loc[fill_mask, "budget_usd"] = out.loc[fill_mask, "budget"]
    out.loc[fill_mask, "budget_local"] = out.loc[fill_mask, "budget"]
    out.loc[fill_mask, "budget_currency"] = "USD"
    out.loc[fill_mask, "budget_source"] = "TMDB (reported)"

    out = out.drop(columns=["budget", "tmdb_title", "match_score"])
    return out, filled_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="write the backfilled parquet (backs up the original first)")
    args = parser.parse_args()

    df = pd.read_parquet(FILM_META_ENRICHED_PATH)
    tmdb = pd.read_parquet(TMDB_CACHE_PATH)
    print(f"Loaded {len(df):,} rows from {FILM_META_ENRICHED_PATH}")
    print(f"Loaded {len(tmdb):,} rows from {TMDB_CACHE_PATH}")

    n_before = df["budget_usd"].isna().sum()
    df_backfilled, filled_report = backfill_budget_from_tmdb(df, tmdb)
    n_after = df_backfilled["budget_usd"].isna().sum()

    print(f"\n=== TMDB budget backfill (confidence in {sorted(TRUSTED_CONFIDENCE)}) ===")
    print(f"  null budget_usd: {n_before:,} -> {n_after:,}  (recovered {n_before - n_after:,})")
    print(filled_report.head(20).to_string(index=False))
    if len(filled_report) > 20:
        print(f"  ... and {len(filled_report) - 20} more")

    if not args.apply:
        print("\nDry run — no files written. Re-run with --apply to write the backfilled parquet.")
        return

    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    backup_path = FILM_META_ENRICHED_PATH.with_name(f"film_meta_enriched.pre_tmdb_backfill_{ts}.parquet")
    df.to_parquet(backup_path, engine="pyarrow", index=False)
    print(f"\nBacked up pre-backfill snapshot → {backup_path}")

    df_backfilled.to_parquet(FILM_META_ENRICHED_PATH, engine="pyarrow", index=False)
    print(f"Wrote backfilled parquet → {FILM_META_ENRICHED_PATH} ({len(df_backfilled):,} rows)")


if __name__ == "__main__":
    main()
