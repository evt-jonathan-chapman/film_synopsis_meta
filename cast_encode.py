"""
Cast feature encoding pipeline.

Reads cast_enriched.parquet (per-actor fame profiles) and the raw Snowflake
parquets, joins on actor_list (pipe-separated), and aggregates to film-level
features that supplement the existing Snowflake cast_total_admits etc.

Output: cast_meta/outputs/cast_features.parquet
  One row per film_id. Join to training/prediction data on film_id.

Usage:
    python cast_encode.py
"""

import glob
import re

import numpy as np
import pandas as pd

from config import CAST_ENRICHED_PATH, CAST_FEATURES_PATH, RAW_PARQUET_GLOB, RAW_PARQUET_GLOB_TEST

_AND_PREFIX = re.compile(r'^AND\s+', re.IGNORECASE)

def _clean_actor_name(raw: str) -> str:
    name = raw.strip().upper()
    name = _AND_PREFIX.sub('', name).strip()
    if name in ('AND', 'N/A', '') or len(name) <= 1:
        return ''
    return name

# ── Ordinal maps ──────────────────────────────────────────────────────────────

FAME_TIER_MAP = {
    'unknown':      0,
    'emerging':     1,
    'bstar':        2,
    'astar':        3,
    'global_astar': 4,
}

FAME_SOURCE_NON_CINEMA = {'tv_streaming', 'music', 'sports', 'social_media'}


# ── Film-level aggregation ────────────────────────────────────────────────────

def aggregate_cast_features(film_actors: list[dict]) -> dict:
    """
    Given a list of actor profile dicts for one film, return film-level
    aggregate features. Actors not found in enriched data are treated as unknown.
    """
    if not film_actors:
        return _zero_features()

    tiers       = [FAME_TIER_MAP.get(a.get('fame_tier', 'unknown'), 0) for a in film_actors]
    sources     = [a.get('fame_source', 'unknown') for a in film_actors]
    markets     = [a.get('primary_market', 'unknown') for a in film_actors]
    cross_media = [a.get('cross_media') or [] for a in film_actors]

    known_mask = [t > 0 for t in tiers]
    n_known    = sum(known_mask)

    cross_flat = [item for sublist in cross_media for item in sublist]

    return {
        # Fame tier
        'cast_max_fame_tier':        max(tiers),
        'cast_global_astar_count':   sum(t == 4 for t in tiers),
        'cast_astar_plus_count':     sum(t >= 3 for t in tiers),
        'cast_bstar_plus_count':     sum(t >= 2 for t in tiers),
        'cast_fame_score':           sum(tiers),                           # weighted sum across cast

        # Fame source
        'cast_tv_streaming_count':   sum(s == 'tv_streaming' for s in sources),
        'cast_music_count':          sum(s == 'music'        for s in sources),
        'cast_sports_count':         sum(s == 'sports'       for s in sources),
        'cast_social_media_count':   sum(s == 'social_media' for s in sources),
        'cast_non_cinema_count':     sum(s in FAME_SOURCE_NON_CINEMA for s in sources),

        # Market
        'cast_global_market_count':  sum(m == 'global'       for m in markets),
        'cast_us_uk_count':          sum(m == 'us_uk'        for m in markets),
        'cast_hindi_count':          sum(m == 'hindi'        for m in markets),
        'cast_korean_count':         sum(m == 'korean'       for m in markets),
        'cast_south_indian_count':   sum(m == 'south_indian' for m in markets),

        # Cross-media
        'cast_cross_media_count':    sum(bool(cm) for cm in cross_media),
        'cast_has_music_crossover':  int('music'         in cross_flat),
        'cast_has_tv_crossover':     int('tv_streaming'  in cross_flat),

        # Coverage
        'cast_known_count':          n_known,
        'cast_non_cinema_ratio':     (
            sum(s in FAME_SOURCE_NON_CINEMA for s in sources) / n_known
            if n_known > 0 else 0.0
        ),
    }


def _zero_features() -> dict:
    return {k: 0 for k in aggregate_cast_features.__code__.co_consts
            if isinstance(k, str) and k.startswith('cast_')} | {
        'cast_non_cinema_ratio': 0.0,
        'cast_max_fame_tier':    0,
        'cast_fame_score':       0,
        'cast_global_astar_count': 0,
        'cast_astar_plus_count': 0,
        'cast_bstar_plus_count': 0,
        'cast_global_astar_count': 0,
        'cast_tv_streaming_count': 0,
        'cast_music_count': 0,
        'cast_sports_count': 0,
        'cast_social_media_count': 0,
        'cast_non_cinema_count': 0,
        'cast_global_market_count': 0,
        'cast_us_uk_count': 0,
        'cast_hindi_count': 0,
        'cast_korean_count': 0,
        'cast_south_indian_count': 0,
        'cast_cross_media_count': 0,
        'cast_has_music_crossover': 0,
        'cast_has_tv_crossover': 0,
        'cast_known_count': 0,
        'cast_non_cinema_ratio': 0.0,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def encode_cast_features(
    enriched_path: Path = CAST_ENRICHED_PATH,
    raw_glob: str = RAW_PARQUET_GLOB,
    out_path: Path = CAST_FEATURES_PATH,
) -> pd.DataFrame:

    if not enriched_path.exists():
        raise FileNotFoundError(f"cast_enriched.parquet not found: {enriched_path}. Run cast_main.py first.")

    # Load enriched actor profiles — normalise name to uppercase for matching
    enriched = pd.read_parquet(enriched_path)
    enriched['actor_name'] = enriched['actor_name'].str.upper().str.strip()
    enriched = enriched.drop_duplicates('actor_name')
    actor_lookup: dict = enriched.set_index('actor_name').to_dict('index')
    print(f"Loaded {len(actor_lookup):,} actor profiles from {enriched_path}")

    # Load unique film → actor_list mapping from all raw parquets
    paths = sorted(glob.glob(raw_glob))
    if not paths:
        raise FileNotFoundError(f"No raw parquets found: {raw_glob}")

    film_actors_map: dict[int, str] = {}
    film_name_map:   dict[int, str] = {}
    for path in paths:
        df = pd.read_parquet(path, columns=['film_id', 'film', 'actor_list'])
        df = df.drop_duplicates('film_id')
        for _, row in df.iterrows():
            fid = int(row['film_id'])
            if fid not in film_actors_map:
                film_actors_map[fid] = row['actor_list'] if pd.notna(row['actor_list']) else ''
                film_name_map[fid]   = row['film']

    print(f"Films to encode: {len(film_actors_map):,}")

    # Aggregate per film
    rows = []
    n_enriched_total = 0
    n_unknown_total  = 0

    for film_id, actor_list_str in film_actors_map.items():
        actors_raw = [_clean_actor_name(a) for a in str(actor_list_str).split('|')] \
                     if actor_list_str else []
        actors_raw = [a for a in actors_raw if a]

        film_actor_profiles = []
        for actor in actors_raw:
            if actor in actor_lookup:
                film_actor_profiles.append(actor_lookup[actor])
                n_enriched_total += 1
            else:
                film_actor_profiles.append({'fame_source': 'unknown', 'fame_tier': 'unknown',
                                            'primary_market': 'unknown', 'cross_media': []})
                n_unknown_total += 1

        feats = aggregate_cast_features(film_actor_profiles)
        feats['film_id'] = film_id
        feats['film']    = film_name_map[film_id]
        feats['cast_size'] = len(actors_raw)
        rows.append(feats)

    df_features = pd.DataFrame(rows)

    # Reorder: film_id first
    cols = ['film_id', 'film', 'cast_size'] + \
           [c for c in df_features.columns if c not in ('film_id', 'film', 'cast_size')]
    df_features = df_features[cols].reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_features.to_parquet(out_path, engine='pyarrow', index=False)

    total_lookups = n_enriched_total + n_unknown_total
    hit_rate = n_enriched_total / total_lookups * 100 if total_lookups else 0
    print(f"\nActor lookup hit rate: {hit_rate:.1f}%  "
          f"({n_enriched_total:,} matched / {n_unknown_total:,} unknown)")
    print(f"Saved {len(df_features):,} film rows → {out_path}")

    return df_features


def diagnostics(df_features: pd.DataFrame) -> None:
    """Print a summary of the encoded features to validate output."""
    print("\n── Feature distributions ──────────────────────────────────────")
    print(df_features[[
        'cast_max_fame_tier', 'cast_global_astar_count', 'cast_astar_plus_count',
        'cast_non_cinema_count', 'cast_tv_streaming_count', 'cast_music_count',
        'cast_known_count',
    ]].describe().round(2).to_string())

    print("\n── Films with global A-listers ────────────────────────────────")
    top = df_features[df_features['cast_global_astar_count'] > 0].sort_values(
        'cast_global_astar_count', ascending=False
    )[['film', 'cast_global_astar_count', 'cast_astar_plus_count', 'cast_fame_score']].head(20)
    print(top.to_string(index=False))

    print("\n── Films with non-cinema primary cast ─────────────────────────")
    non_cin = df_features[df_features['cast_non_cinema_count'] > 0].sort_values(
        'cast_non_cinema_count', ascending=False
    )[['film', 'cast_non_cinema_count', 'cast_tv_streaming_count',
       'cast_music_count', 'cast_max_fame_tier']].head(20)
    print(non_cin.to_string(index=False))


# ── Run ────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    df_features = encode_cast_features()
    diagnostics(df_features)
