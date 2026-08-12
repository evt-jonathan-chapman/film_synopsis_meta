"""
post_refresh_check.py — one consolidated health check across all four
extraction outputs after a refresh.py run. Read-only, no writes, no LLM/
Snowflake calls — just re-checks what a normal run should have produced.

Rolls up the individual spot-checks from this session into one script:
row counts, duplicate film_ids, checkpoint/parquet agreement, genre
vocabulary sanity (film_meta + synopsis), the Documentary/Biography overlap
rule, is_concert coverage, and film_id_variants.parquet's dtype consistency
(the exact bug that broke a film_meta run earlier — see film_variant_merge.py
and cleanup_film_meta.py's persist_variant_map).

Usage:
    python diagnostics/post_refresh_check.py
"""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from config import (
    SYNOPSES_EXTRACTED_PATH, CAST_ENRICHED_PATH, DIRECTOR_ENRICHED_PATH,
    FILM_META_ENRICHED_PATH, FILM_ID_VARIANTS_PATH,
)
from refresh import (
    SYNOPSIS_CHECKPOINT_PATH, CAST_CHECKPOINT_PATH, DIRECTOR_CHECKPOINT_PATH,
    FILM_META_CHECKPOINT_PATH, FILM_META_ERRORS_PATH, CAST_ERRORS_PATH,
    DIRECTOR_ERRORS_PATH,
)
from cleanup_film_meta import normalize_synopsis_genre_list, MIN_GENRE_FILM_COUNT


def _section(title):
    print(f"\n{'=' * 10} {title} {'=' * 10}")


def _load_json(path):
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def check_synopsis():
    _section("Synopsis")
    if not SYNOPSES_EXTRACTED_PATH.exists():
        print("  MISSING:", SYNOPSES_EXTRACTED_PATH)
        return
    df = pd.read_parquet(SYNOPSES_EXTRACTED_PATH)
    print(f"  {len(df):,} rows, {df['film_id'].duplicated().sum()} duplicate film_ids")
    if 'alt_synopsis' in df.columns:
        print(f"  alt_synopsis populated: {df['alt_synopsis'].notna().sum():,} of {len(df):,}")
    else:
        print("  WARNING: alt_synopsis column missing")

    checkpoint = _load_json(SYNOPSIS_CHECKPOINT_PATH)
    if checkpoint is None:
        print(f"  WARNING: checkpoint missing at {SYNOPSIS_CHECKPOINT_PATH}")
    else:
        gap = len(df) - len(checkpoint)
        print(f"  checkpoint: {len(checkpoint):,} entries (parquet has {len(df):,} — gap of {gap:,})")

    genres = df['genres'].apply(lambda g: list(g) if g is not None and hasattr(g, '__iter__') else [])
    normalized = genres.apply(normalize_synopsis_genre_list)
    n_would_change = int((genres.apply(list) != normalized).sum())
    if n_would_change:
        print(f"  NOTE: {n_would_change:,} rows' genres would change if normalize_synopsis_genre_list were applied "
              f"(not wired into the live pipeline yet — still a manual step)")


def check_cast():
    _section("Cast")
    if not CAST_ENRICHED_PATH.exists():
        print("  MISSING:", CAST_ENRICHED_PATH)
        return
    df = pd.read_parquet(CAST_ENRICHED_PATH)
    n_unknown = (df['fame_tier'].astype(str).str.strip().str.lower() == 'unknown').sum()
    print(f"  {len(df):,} actors, {df['actor_name'].duplicated().sum()} duplicate names, "
          f"{n_unknown:,} unknown ({n_unknown/len(df)*100:.1f}%)")

    errors = _load_json(CAST_ERRORS_PATH)
    if errors:
        print(f"  {len(errors):,} entries in cast_errors.json (will auto-retry next run)")

    placeholder = df[df['actor_name'].astype(str).str.upper().str.startswith('VARIOUS')]
    if not placeholder.empty:
        print(f"  WARNING: {len(placeholder)} placeholder-looking actor_name(s) still present: "
              f"{placeholder['actor_name'].tolist()}")


def check_director():
    _section("Director")
    if not DIRECTOR_ENRICHED_PATH.exists():
        print("  MISSING:", DIRECTOR_ENRICHED_PATH)
        return
    df = pd.read_parquet(DIRECTOR_ENRICHED_PATH)
    n_unknown = (df['director_tier'].astype(str).str.strip().str.lower() == 'unknown').sum()
    print(f"  {len(df):,} directors, {df['director_name'].duplicated().sum()} duplicate names, "
          f"{n_unknown:,} unknown ({n_unknown/len(df)*100:.1f}%)")

    errors = _load_json(DIRECTOR_ERRORS_PATH)
    if errors:
        print(f"  {len(errors):,} entries in director_errors.json (will auto-retry next run)")


def check_film_meta():
    _section("Film meta")
    if not FILM_META_ENRICHED_PATH.exists():
        print("  MISSING:", FILM_META_ENRICHED_PATH)
        return
    df = pd.read_parquet(FILM_META_ENRICHED_PATH)
    print(f"  {len(df):,} films, {df['film_id'].duplicated().sum()} duplicate film_ids")

    genre_counts = df['genres'].explode().value_counts()
    below_threshold = genre_counts[genre_counts < MIN_GENRE_FILM_COUNT]
    print(f"  {genre_counts.shape[0]} unique genres "
          f"(min count {genre_counts.min() if len(genre_counts) else 'n/a'}, threshold {MIN_GENRE_FILM_COUNT})")
    if not below_threshold.empty:
        print(f"  WARNING: {len(below_threshold)} genre(s) below the rarity threshold slipped through: "
              f"{below_threshold.to_dict()}")

    doc_bio = df[df['genres'].apply(lambda g: 'Documentary' in g and 'Biography' in g)]
    print(f"  Documentary+Biography overlap: {len(doc_bio)} films (should be 0)")

    if 'is_concert' in df.columns:
        print(f"  is_concert: {df['is_concert'].sum()} films flagged")
    else:
        print("  WARNING: is_concert column missing")

    n_empty_genres = (df['genres'].apply(len) == 0).sum()
    print(f"  {n_empty_genres} films with empty genres list")

    dup_titles = df['title'].astype(str).str.strip().str.upper()
    n_dup_titles = dup_titles.duplicated(keep=False).sum()
    print(f"  {n_dup_titles} rows share a title with another row (residual duplicates — see audit_film_meta.py)")

    checkpoint = _load_json(FILM_META_CHECKPOINT_PATH)
    if checkpoint is not None:
        print(f"  checkpoint: {len(checkpoint):,} entries")
    errors = _load_json(FILM_META_ERRORS_PATH)
    if errors:
        print(f"  {len(errors):,} entries in film_meta_errors.json (will auto-retry next run)")

    n_genres_per_film = df['genres'].apply(len)
    print(f"  genres/film: min={n_genres_per_film.min()} max={n_genres_per_film.max()} "
          f"mode={n_genres_per_film.mode().iloc[0]}")


def check_variant_map():
    _section("Film ID variants")
    if not FILM_ID_VARIANTS_PATH.exists():
        print("  (not created yet — no variants detected so far)")
        return
    df = pd.read_parquet(FILM_ID_VARIANTS_PATH)
    print(f"  {len(df):,} variant film_id(s) mapped, {df['film_id'].duplicated().sum()} duplicate film_ids")
    for col in ('variant_rel_at', 'canonical_rel_at'):
        types = df[col].apply(type).unique()
        flag = " WARNING: mixed types!" if len(types) > 1 else ""
        print(f"  {col} dtypes present: {[t.__name__ for t in types]}{flag}")
    print(df['confirmed_by'].value_counts().to_string())


def main():
    check_synopsis()
    check_cast()
    check_director()
    check_film_meta()
    check_variant_map()
    print()


if __name__ == '__main__':
    main()
