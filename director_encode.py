"""
Director feature encoding pipeline.

Reads director_enriched.parquet (per-director profiles) and the raw Snowflake
parquets, looks up each film's lead director, and outputs film-level features.

Output: director_meta/director_features.parquet
  One row per film_id. Join to training/prediction data on film_id.

Usage:
    python director_encode.py
"""

import glob
import re
from pathlib import Path

import pandas as pd

from config import DIRECTOR_ENRICHED_PATH, DIRECTOR_FEATURES_PATH, RAW_PARQUET_GLOBS_ALL

DIRECTOR_TIER_MAP = {
    'unknown':     0,
    'emerging':    1,
    'established': 2,
    'top_tier':    3,
}

VALID_DIRECTOR_TIER  = set(DIRECTOR_TIER_MAP.keys())
VALID_PRIMARY_MARKET = {'global', 'us_uk', 'korean', 'hindi', 'south_indian', 'european', 'other', 'unknown'}


def encode_director_features(
    enriched_path: Path = DIRECTOR_ENRICHED_PATH,
    raw_globs: list = RAW_PARQUET_GLOBS_ALL,
    out_path: Path = DIRECTOR_FEATURES_PATH,
) -> pd.DataFrame:

    if not enriched_path.exists():
        raise FileNotFoundError(f"director_enriched.parquet not found: {enriched_path}. Run main.py with RUN_DIRECTOR=True first.")

    enriched = pd.read_parquet(enriched_path)

    # Clamp LLM fields
    for col, valid, fallback in [
        ('director_tier',  VALID_DIRECTOR_TIER,  'unknown'),
        ('primary_market', VALID_PRIMARY_MARKET, 'other'),
    ]:
        if col in enriched.columns:
            enriched[col] = enriched[col].str.lower().str.strip()
            mask = ~enriched[col].isin(valid)
            if mask.any():
                print(f"  Clamping {mask.sum()} out-of-schema '{col}' values → '{fallback}'")
            enriched[col] = enriched[col].where(enriched[col].isin(valid), fallback)

    director_lookup: dict = enriched.set_index('director_name').to_dict('index')
    print(f"Loaded {len(director_lookup):,} director profiles from {enriched_path}")

    # Load unique film → director mapping from all raw parquets (train + test + pred)
    if isinstance(raw_globs, str):
        raw_globs = [raw_globs]
    paths = []
    for pattern in raw_globs:
        paths.extend(glob.glob(pattern))
    paths = sorted(set(paths))
    if not paths:
        raise FileNotFoundError(f"No raw parquets found for globs: {raw_globs}")

    film_director_map: dict[int, str] = {}
    for path in paths:
        df = pd.read_parquet(path, columns=['film_id', 'director'])
        df = df.drop_duplicates('film_id')
        for _, row in df.iterrows():
            fid = int(row['film_id'])
            if fid not in film_director_map:
                film_director_map[fid] = row['director'] if pd.notna(row['director']) else ''

    print(f"Films to encode: {len(film_director_map):,}")

    rows = []
    n_matched = 0
    n_unknown = 0

    for film_id, director_str in film_director_map.items():
        # Take first-billed director
        lead = director_str.split(',')[0].strip() if director_str else ''

        if lead and lead in director_lookup:
            profile = director_lookup[lead]
            n_matched += 1
        else:
            profile = {'director_tier': 'unknown', 'primary_market': 'unknown'}
            n_unknown += 1

        rows.append({
            'film_id':          film_id,
            'director_tier':    DIRECTOR_TIER_MAP.get(profile.get('director_tier', 'unknown'), 0),
            'director_market':  profile.get('primary_market', 'unknown'),
        })

    df_features = pd.DataFrame(rows).reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_features.to_parquet(out_path, engine='pyarrow', index=False)

    total = n_matched + n_unknown
    print(f"\nDirector lookup hit rate: {n_matched/total*100:.1f}%  ({n_matched:,} matched / {n_unknown:,} unknown)")
    print(f"Saved {len(df_features):,} film rows → {out_path}")

    return df_features


def diagnostics(df_features: pd.DataFrame) -> None:
    print("\n── Director tier distribution ─────────────────────────────────")
    print(df_features['director_tier'].value_counts().to_string())

    print("\n── Market distribution ────────────────────────────────────────")
    print(df_features['director_market'].value_counts().head(15).to_string())

    print("\n── Top-tier directors ─────────────────────────────────────────")
    print(df_features[df_features['director_tier'] == 3][['film_id', 'director_market']].head(20).to_string(index=False))


if __name__ == '__main__':
    df_features = encode_director_features()
    diagnostics(df_features)
