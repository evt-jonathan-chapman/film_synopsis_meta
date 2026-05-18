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
from pathlib import Path

import pandas as pd

from config import CAST_ENRICHED_PATH, CAST_FEATURES_PATH, RAW_PARQUET_GLOBS_ALL, PROC_PARQUET_GLOBS_ALL

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

# Manual corrections for actors whose LLM-extracted fame_tier is wrong.
# Keys are uppercase actor names (matching actor_lookup keys).
# Add entries here for actors the auto-promotion thresholds can't handle
# (e.g. pre-release stars with no AU admits history yet).
ACTOR_FAME_TIER_OVERRIDES: dict[str, dict] = {}

# Auto-promotion thresholds. Training data covers ~2022–2025 (TRAIN_MIN_DATE),
# so per-actor totals are naturally lower than lifetime figures. Thresholds are
# calibrated against the corrected (non-double-counted) distribution where the
# observed max total is ~720K (Chris Pratt, 5 films).
#
# global_astar: uses top3_billed_admits (admits only from films where actor is
# billed 1st/2nd/3rd) — this filters Avatar ensemble cast (position 5+) who
# accumulate admits without being the marquee draw.
#
# astar: uses total admits — fine for supporting cast who build up admits across
# many films without necessarily leading any of them.
ADMITS_ASTAR_FLOOR                = 100_000  # total admits: emerging/unknown → astar
ADMITS_ASTAR_AVG_FLOOR            = 20_000   # avg per film: must also clear this
ADMITS_GLOBAL_ASTAR_TOP3_FLOOR    = 200_000  # top3_billed_admits: gate for global_astar
ADMITS_GLOBAL_ASTAR_TOP3_MIN_FILMS = 2       # min films as top-3 billed to qualify
MIN_FILMS_FOR_PROMOTION           = 2        # min total films for any promotion


# ── Film-level aggregation ────────────────────────────────────────────────────

def aggregate_cast_features(film_actors: list[dict]) -> dict:
    """
    Given a list of actor profile dicts for one film (in billing order), return
    film-level aggregate features. Actors not found in enriched data are treated as unknown.
    """
    if not film_actors:
        return _zero_features()

    tiers   = [FAME_TIER_MAP.get(a.get('fame_tier', 'unknown'), 0) for a in film_actors]
    sources = [a.get('fame_source', 'unknown') for a in film_actors]

    n_total = len(tiers)
    n_known = sum(t > 0 for t in tiers)

    return {
        'cast_lead_tier':          tiers[0],
        'cast_avg_fame_score':     round(sum(tiers) / n_total, 4),
        'cast_global_astar_count': sum(t == 4 for t in tiers),
        'cast_astar_plus_count':   sum(t >= 3 for t in tiers),
        'cast_known_ratio':        round(n_known / n_total, 4),
        'cast_non_cinema_ratio':   (
            round(sum(s in FAME_SOURCE_NON_CINEMA for s in sources) / n_known, 4)
            if n_known > 0 else 0.0
        ),
    }


def _zero_features() -> dict:
    return {
        'cast_lead_tier':          0,
        'cast_avg_fame_score':     0.0,
        'cast_global_astar_count': 0,
        'cast_astar_plus_count':   0,
        'cast_known_ratio':        0.0,
        'cast_non_cinema_ratio':   0.0,
    }


# ── Admits-based auto-promotion ───────────────────────────────────────────────

def compute_actor_admits(proc_globs: list) -> pd.DataFrame:
    """
    Compute per-actor lifetime AU EVT admits from processed parquets.

    Joins actor_list (pipe-separated, from raw parquets) with per-film total
    w1 admits (week_admits_merged_w1, summed across sites from proc parquets).

    Returns a DataFrame with columns:
      actor_name, total_admits, film_count,
      top3_billed_admits, n_top3_billed_films
    top3_billed_* counts only films where the actor is billed in positions 0–2
    (0-indexed), i.e. the lead or co-lead. Used to filter ensemble blockbuster
    actors (e.g. Avatar cast, position 5+) from global_astar promotion.
    """
    # Per-film total w1 admits from proc parquets (merged across format variants)
    proc_paths = []
    for pattern in proc_globs:
        proc_paths.extend(glob.glob(pattern))
    proc_paths = sorted(set(proc_paths))

    if not proc_paths:
        return pd.DataFrame(columns=['actor_name', 'total_admits', 'film_count'])

    film_admits = []
    for path in proc_paths:
        try:
            df = pd.read_parquet(path, columns=['film_id', 'week_admits_merged_w1'])
            film_admits.append(df.groupby('film_id')['week_admits_merged_w1'].sum().reset_index())
        except Exception:
            pass
    if not film_admits:
        return pd.DataFrame(columns=['actor_name', 'total_admits', 'film_count'])

    # Use max across parquets (same film appears in multiple training snapshots —
    # summing would double/triple count).
    film_totals = (
        pd.concat(film_admits, ignore_index=True)
        .groupby('film_id')['week_admits_merged_w1'].max()
        .reset_index()
        .rename(columns={'week_admits_merged_w1': 'film_w1_total'})
    )

    # Actor list from raw parquets (same glob pattern, swap filename)
    raw_paths = [p.replace('train_proc_ds', 'train_raw_ds').replace('test_proc_ds', 'test_raw_ds')
                 for p in proc_paths]
    raw_paths = [p for p in raw_paths if Path(p).exists()]

    actor_rows = []
    for path in raw_paths:
        try:
            df = pd.read_parquet(path, columns=['film_id', 'actor_list'])
            actor_rows.append(df.drop_duplicates('film_id'))
        except Exception:
            pass
    if not actor_rows:
        return pd.DataFrame(columns=['actor_name', 'total_admits', 'film_count'])

    film_actors = pd.concat(actor_rows, ignore_index=True).drop_duplicates('film_id')
    film_actors = film_actors.merge(film_totals, on='film_id', how='inner')

    records = []
    for _, row in film_actors.iterrows():
        if not pd.notna(row['actor_list']) or not row['actor_list']:
            continue
        actors_in_film = [_clean_actor_name(a) for a in str(row['actor_list']).split('|')]
        for pos, actor in enumerate(actors_in_film):
            if actor:
                records.append({
                    'actor_name':  actor,
                    'admits':      row['film_w1_total'],
                    'top3_billed': pos <= 2,   # 0-indexed: positions 0,1,2 = top-3 billed
                })

    if not records:
        return pd.DataFrame(columns=['actor_name', 'total_admits', 'film_count',
                                     'top3_billed_admits', 'n_top3_billed_films'])

    df = pd.DataFrame(records)
    base = df.groupby('actor_name')['admits'].agg(total_admits='sum', film_count='count').reset_index()
    top3 = (
        df[df['top3_billed']]
        .groupby('actor_name')['admits']
        .agg(top3_billed_admits='sum', n_top3_billed_films='count')
        .reset_index()
    )
    return base.merge(top3, on='actor_name', how='left').fillna(
        {'top3_billed_admits': 0, 'n_top3_billed_films': 0}
    )


def apply_admits_promotion(enriched: pd.DataFrame, actor_admits: pd.DataFrame) -> pd.DataFrame:
    """
    Auto-promote actors whose LLM-extracted fame_tier is below what their historical
    admits suggest. Only promotes upward; never demotes.

    global_astar gate uses top3_billed_admits so ensemble blockbuster cast members
    (e.g. Avatar position 5+) don't qualify purely from franchise co-appearances.
    astar gate uses total admits — fine for supporting cast across many films.
    """
    if actor_admits.empty:
        return enriched

    TIER_RANK = FAME_TIER_MAP

    admits_map = actor_admits.set_index('actor_name').to_dict('index')
    promoted = []

    for idx, row in enriched.iterrows():
        actor = row['actor_name']
        info  = admits_map.get(actor)
        if info is None or info['film_count'] < MIN_FILMS_FOR_PROMOTION:
            continue

        total       = info['total_admits']
        avg         = total / max(info['film_count'], 1)
        top3_admits = info.get('top3_billed_admits', 0) or 0
        n_top3      = int(info.get('n_top3_billed_films', 0) or 0)
        cur_tier    = row.get('fame_tier', 'unknown')
        cur_rank    = TIER_RANK.get(cur_tier, 0)

        new_tier = cur_tier
        if (top3_admits >= ADMITS_GLOBAL_ASTAR_TOP3_FLOOR
                and n_top3 >= ADMITS_GLOBAL_ASTAR_TOP3_MIN_FILMS
                and cur_rank < 4):
            new_tier = 'global_astar'
        elif (total >= ADMITS_ASTAR_FLOOR and avg >= ADMITS_ASTAR_AVG_FLOOR
              and cur_rank < 3):
            new_tier = 'astar'

        if new_tier != cur_tier:
            enriched.at[idx, 'fame_tier'] = new_tier
            promoted.append(
                f"  Admits promotion: {actor}  {cur_tier} → {new_tier}"
                f"  (total={total:,.0f}, top3={top3_admits:,.0f}, top3_films={n_top3})"
            )

    for msg in promoted:
        print(msg)
    if promoted:
        print(f"  Auto-promoted {len(promoted)} actors based on historical admits")

    return enriched


# ── Main ──────────────────────────────────────────────────────────────────────

def encode_cast_features(
    enriched_path: Path = CAST_ENRICHED_PATH,
    raw_globs: list = RAW_PARQUET_GLOBS_ALL,
    proc_globs: list = PROC_PARQUET_GLOBS_ALL,
    out_path: Path = CAST_FEATURES_PATH,
) -> pd.DataFrame:

    if not enriched_path.exists():
        raise FileNotFoundError(f"cast_enriched.parquet not found: {enriched_path}. Run cast_main.py first.")

    VALID_FAME_SOURCE   = {'cinema', 'tv_streaming', 'music', 'sports', 'social_media', 'unknown'}
    VALID_FAME_TIER     = {'global_astar', 'astar', 'bstar', 'emerging', 'unknown'}
    VALID_PRIMARY_MARKET = {'global', 'us_uk', 'korean', 'hindi', 'south_indian', 'european', 'other', 'unknown'}
    VALID_CROSS_MEDIA   = {'cinema', 'tv_streaming', 'music', 'sports', 'social_media'}
    VALID_AGE_RANGE     = {'under_35', '35_to_55', 'over_55', 'unknown'}

    # Load enriched actor profiles — normalise name to uppercase for matching
    enriched = pd.read_parquet(enriched_path)
    enriched['actor_name'] = enriched['actor_name'].str.upper().str.strip()
    enriched = enriched.drop_duplicates('actor_name')

    # Clamp LLM fields to allowed values; out-of-schema values → 'unknown'/'other'
    for col, valid, fallback in [
        ('fame_source',    VALID_FAME_SOURCE,    'unknown'),
        ('fame_tier',      VALID_FAME_TIER,      'unknown'),
        ('primary_market', VALID_PRIMARY_MARKET, 'other'),
        ('age_range',      VALID_AGE_RANGE,      'unknown'),
    ]:
        if col in enriched.columns:
            enriched[col] = enriched[col].str.lower().str.strip()
            mask = ~enriched[col].isin(valid)
            if mask.any():
                print(f"  Clamping {mask.sum()} out-of-schema '{col}' values → '{fallback}'")
            enriched[col] = enriched[col].where(enriched[col].isin(valid), fallback)

    if 'cross_media' in enriched.columns:
        def _clamp_cross(val):
            if not isinstance(val, list):
                return []
            return [v for v in val if v in VALID_CROSS_MEDIA]
        enriched['cross_media'] = enriched['cross_media'].apply(_clamp_cross)

    # Apply manual fame-tier overrides (see ACTOR_FAME_TIER_OVERRIDES at top of file)
    for actor_name, fields in ACTOR_FAME_TIER_OVERRIDES.items():
        mask = enriched['actor_name'] == actor_name
        if mask.any():
            for col, val in fields.items():
                enriched.loc[mask, col] = val
            print(f"  Fame-tier override applied: {actor_name} → {fields}")
        else:
            # Actor not yet in cast_enriched — insert a stub row so the override takes effect
            stub = {c: 'unknown' for c in enriched.columns}
            stub['actor_name'] = actor_name
            stub.update(fields)
            enriched = pd.concat([enriched, pd.DataFrame([stub])], ignore_index=True)
            print(f"  Fame-tier override inserted (new stub): {actor_name} → {fields}")

    # Auto-promote under-scored actors based on historical admits
    actor_admits = compute_actor_admits(proc_globs)
    enriched = apply_admits_promotion(enriched, actor_admits)

    actor_lookup: dict = enriched.set_index('actor_name').to_dict('index')
    print(f"Loaded {len(actor_lookup):,} actor profiles from {enriched_path}")

    # Load unique film → actor_list mapping from all raw parquets (train + test + pred)
    if isinstance(raw_globs, str):
        raw_globs = [raw_globs]
    paths = []
    for pattern in raw_globs:
        paths.extend(glob.glob(pattern))
    paths = sorted(set(paths))
    if not paths:
        raise FileNotFoundError(f"No raw parquets found for globs: {raw_globs}")

    film_actors_map: dict[int, str] = {}
    for path in paths:
        df = pd.read_parquet(path, columns=['film_id', 'actor_list'])
        df = df.drop_duplicates('film_id')
        for _, row in df.iterrows():
            fid = int(row['film_id'])
            if fid not in film_actors_map:
                film_actors_map[fid] = row['actor_list'] if pd.notna(row['actor_list']) else ''

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
        feats['film_id']   = film_id
        feats['cast_size'] = len(actors_raw)
        rows.append(feats)

    df_features = pd.DataFrame(rows)

    cols = ['film_id', 'cast_size'] + \
           [c for c in df_features.columns if c not in ('film_id', 'cast_size')]
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
        'cast_lead_tier', 'cast_avg_fame_score', 'cast_global_astar_count',
        'cast_astar_plus_count', 'cast_known_ratio', 'cast_non_cinema_ratio', 'cast_size',
    ]].describe().round(3).to_string())

    print("\n── Top films by avg fame score ─────────────────────────────────")
    top = df_features[df_features['cast_global_astar_count'] > 0].sort_values(
        'cast_avg_fame_score', ascending=False
    )[['film_id', 'cast_lead_tier', 'cast_avg_fame_score',
       'cast_global_astar_count', 'cast_astar_plus_count', 'cast_size']].head(20)
    print(top.to_string(index=False))

    print("\n── Films with high non-cinema ratio ────────────────────────────")
    non_cin = df_features[df_features['cast_non_cinema_ratio'] > 0].sort_values(
        'cast_non_cinema_ratio', ascending=False
    )[['film_id', 'cast_non_cinema_ratio', 'cast_lead_tier', 'cast_avg_fame_score']].head(20)
    print(non_cin.to_string(index=False))


# ── Run ────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    df_features = encode_cast_features()
    diagnostics(df_features)
