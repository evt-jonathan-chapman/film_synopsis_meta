"""
compare_prompt_versions.py

Runs both prompt versions on a fixed set of target films and prints a
side-by-side comparison of key extraction fields plus cast features.

Useful for validating prompts_v2.yaml before switching to production.

Usage:
    python compare_prompt_versions.py
"""

import asyncio
import gc
import os
import re
import sys
from pathlib import Path

import nest_asyncio
nest_asyncio.apply()

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

import litellm
litellm.success_callback = []
litellm.failure_callback = []

sys.path.insert(0, '/Users/jonathanchapman/Documents/git/evt_back_up/base')

from base_snowflake import SnowFlakeBase
from extractor import LlmJsonExtractor
from load_prompts import load_tasks_from_yaml
from models import DEFAULT_MODEL, DEFAULT_FALLBACKS, MODELS
from films import sql as films_sql
from config import (
    SYNOPSES_EXTRACTED_PATH, CAST_FEATURES_PATH, CAST_ENRICHED_PATH,
    DATA_DIR, SF_WAREHOUSE, SF_DATABASE, SF_SCHEMA, SF_RSA_KEY,
)

# ── Config ────────────────────────────────────────────────────────────────────

TARGET_IDS = [57603, 57343, 59530, 60336, 60261, 60560]  # Mandalorian, Michael, Housemaid, Hail Mary, Devil Wears Prada 2, Marty Supreme

PROMPTS_V1        = Path('prompts/prompts.yaml')
PROMPTS_V2        = Path('prompts/prompts_v2.yaml')
OUTPUT_PATH       = DATA_DIR / 'comparisons' / 'prompt_comparison.xlsx'
CAST_PROMPTS_PATH = Path('prompts/cast_prompts.yaml')

_AND_RE = re.compile(r'^AND\s+', re.IGNORECASE)

def _parse_actor_list(raw) -> list[str]:
    """Parse actor_list — handles both JSON array and pipe-separated formats."""
    import json
    actors = []
    raw = str(raw).strip()
    if raw.startswith('['):
        try:
            items = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            items = [raw]
    else:
        items = raw.split('|')
    for a in items:
        name = _AND_RE.sub('', str(a).strip()).strip().upper()
        if name and name not in ('AND', 'N/A') and len(name) > 1:
            actors.append(name)
    return actors

# Fields to compare between v1 and v2
COMPARE_FIELDS = [
    'genres',
    'primary_audience',
    'secondary_audiences',
    'is_sequel',
    'intellectual_property',
    'protagonist_type',
    # v2 new fields
    'ip_strength',
    'adaptation_type',
    'narrative_scope',
]

CAST_DISPLAY_FIELDS = [
    'cast_size',
    'cast_known_count',
    'cast_max_fame_tier',
    'cast_global_astar_count',
    'cast_astar_plus_count',
    'cast_fame_score',
    'cast_tv_streaming_count',
    'cast_music_count',
    'cast_non_cinema_count',
    'cast_hindi_count',
    'cast_korean_count',
]

FAME_TIER_LABELS = {0: 'unknown', 1: 'emerging', 2: 'bstar', 3: 'astar', 4: 'global_astar'}


# ── Snowflake ─────────────────────────────────────────────────────────────────

def load_films_from_snowflake() -> pd.DataFrame:
    sb = SnowFlakeBase(warehouse=SF_WAREHOUSE, database=SF_DATABASE, schema=SF_SCHEMA)
    sb.create_snowflake_connection(SF_RSA_KEY)
    df = pd.read_sql(films_sql.SQL_FILM_DETAILS, sb.engine)
    df['film_id'] = df['film_id'].astype(int)
    df.columns = [c.lower() for c in df.columns]
    print(f"Snowflake: {len(df)} films loaded")
    return df


def select_target_films(df_all: pd.DataFrame) -> pd.DataFrame:
    """Filter to target film_ids."""
    result = df_all[df_all['film_id'].isin(TARGET_IDS)].copy()
    missing = set(TARGET_IDS) - set(result['film_id'].tolist())
    if missing:
        print(f"  Warning: film_ids not found in Snowflake: {missing}")
    print(f"\nTarget films selected ({len(result)}):")
    for _, r in result.iterrows():
        print(f"  {r['film_id']:>6}  {r['film_title']}")
    return result


# ── Extraction ────────────────────────────────────────────────────────────────

def load_existing_extracts(film_ids: list[int]) -> dict:
    """Load existing production extracts from synopses_extracted.parquet."""
    if not SYNOPSES_EXTRACTED_PATH.exists():
        print("  Warning: synopses_extracted.parquet not found — existing extracts unavailable")
        return {}
    df = pd.read_parquet(SYNOPSES_EXTRACTED_PATH)
    df['film_id'] = df['film_id'].astype(int)
    df = df[df['film_id'].isin(film_ids)]
    results = {}
    for _, row in df.iterrows():
        results[int(row['film_id'])] = row.to_dict()
    missing = set(film_ids) - set(results.keys())
    if missing:
        print(f"  Note: {len(missing)} films not yet in parquet (will show '—' for existing): {missing}")
    print(f"  Loaded {len(results)} existing extracts from {SYNOPSES_EXTRACTED_PATH}")
    return results


async def run_v2_extraction(df: pd.DataFrame) -> dict:
    tasks     = load_tasks_from_yaml(PROMPTS_V2)
    model_cfg = MODELS.get(DEFAULT_MODEL, {})
    extractor = LlmJsonExtractor(
        tasks=tasks,
        model=DEFAULT_MODEL,
        fallbacks=DEFAULT_FALLBACKS,
        api_key=os.getenv('OPENAI_KEY'),
        cost_per_1m_input=model_cfg.get('cost_per_1m_input'),
        cost_per_1m_output=model_cfg.get('cost_per_1m_output'),
    )
    print(f"\nExtracting with v2 (prompts_v2.yaml) — {len(df)} films...")
    results = await extractor.arun_multiple_synopses(
        df=df,
        id_col='film_id',
        title_col='film_title',
        synopsis_col='synopsis',
        alt_synopsis_col='alt_synopsis' if 'alt_synopsis' in df.columns else None,
        flatten=True,
        max_concurrency=10,
    )
    if extractor.token_usage:
        u = extractor.token_usage
        print(f"  Cost: ${u.get('cost_usd', 0):.4f}  tokens: {u.get('prompt_tokens', 0):,} in / {u.get('completion_tokens', 0):,} out")
    del extractor
    gc.collect()
    return results


# ── Cast extraction ───────────────────────────────────────────────────────────

def _get_actors_from_films(df_films: pd.DataFrame) -> list[str]:
    """Extract all unique clean actor names from the target films' actor_list."""
    actors = set()
    for val in df_films['actor_list'].dropna():
        actors.update(_parse_actor_list(val))
    return sorted(actors)


async def enrich_target_actors(df_films: pd.DataFrame, df_enriched: pd.DataFrame | None) -> pd.DataFrame:
    """
    Extract cast profiles for any actors in the target films not yet in cast_enriched.
    Saves results back to CAST_ENRICHED_PATH and returns the updated DataFrame.
    """
    all_actors = _get_actors_from_films(df_films)

    existing_names = set()
    if df_enriched is not None and not df_enriched.empty:
        existing_names = set(df_enriched['actor_name'].str.upper().str.strip())

    new_actors = [a for a in all_actors if a not in existing_names]
    print(f"\nCast: {len(all_actors)} actors across target films — {len(existing_names)} already enriched — {len(new_actors)} to extract")

    if not new_actors:
        return df_enriched

    df_new = pd.DataFrame({'actor_name': new_actors, 'synopsis': ''})

    tasks     = load_tasks_from_yaml(CAST_PROMPTS_PATH)
    model_cfg = MODELS.get(DEFAULT_MODEL, {})
    extractor = LlmJsonExtractor(
        tasks=tasks,
        model=DEFAULT_MODEL,
        fallbacks=DEFAULT_FALLBACKS,
        api_key=os.getenv('OPENAI_KEY'),
        cost_per_1m_input=model_cfg.get('cost_per_1m_input'),
        cost_per_1m_output=model_cfg.get('cost_per_1m_output'),
    )

    results = await extractor.arun_multiple_synopses(
        df=df_new,
        id_col='actor_name',
        title_col='actor_name',
        synopsis_col='synopsis',
        alt_synopsis_col=None,
        flatten=True,
        max_concurrency=20,
        min_synopsis_len=0,
    )

    if extractor.token_usage:
        u = extractor.token_usage
        print(f"  Cast cost: ${u.get('cost_usd', 0):.4f}  tokens: {u.get('prompt_tokens', 0):,} in / {u.get('completion_tokens', 0):,} out")
    del extractor
    gc.collect()

    df_results = pd.DataFrame(results.values())
    if df_results.empty:
        print("  Warning: no cast results returned.")
        return df_enriched

    if 'title' in df_results.columns:
        df_results = df_results.rename(columns={'title': 'actor_name'})
    if '_error' in df_results.columns:
        df_results = df_results[df_results['_error'].isna()].drop(
            columns=[c for c in ['_error', '_error_message', '_raw_output', 'synopsis', 'film_id']
                     if c in df_results.columns], errors='ignore'
        )

    out = pd.concat([df_enriched, df_results], ignore_index=True) if df_enriched is not None and not df_enriched.empty else df_results
    out = out.drop_duplicates(subset='actor_name', keep='first').reset_index(drop=True)

    CAST_ENRICHED_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(CAST_ENRICHED_PATH, engine='pyarrow', index=False)
    print(f"  Saved {len(out)} actors → {CAST_ENRICHED_PATH}")
    return out


# ── Display ───────────────────────────────────────────────────────────────────

def _fmt(v) -> str:
    if v is None:
        return '—'
    if isinstance(v, list):
        return ', '.join(str(x) for x in v) if v else '—'
    if isinstance(v, bool):
        return '✓' if v else '✗'
    return str(v) if str(v) else '—'


def print_film_comparison(film_id: int, film_title: str, v1: dict, v2: dict) -> None:
    bar = '━' * 70
    print(f'\n{bar}')
    print(f'  {film_title}  (film_id={film_id})')
    print(bar)
    print(f'  {"Field":<26}  {"existing (parquet)":<30}  {"v2 (prompts_v2.yaml)"}')
    print(f'  {"─"*26}  {"─"*30}  {"─"*30}')

    for field in COMPARE_FIELDS:
        v1_val = _fmt(v1.get(field))
        v2_val = _fmt(v2.get(field))
        changed = ' ◄' if v1_val != v2_val else ''
        # Truncate long values
        v1_disp = v1_val[:29] if len(v1_val) > 29 else v1_val
        v2_disp = v2_val[:29] if len(v2_val) > 29 else v2_val
        print(f'  {field:<26}  {v1_disp:<30}  {v2_disp}{changed}')


def print_cast_features(film_id: int, df_cast: pd.DataFrame | None) -> None:
    print(f'\n  Cast features:')
    if df_cast is None or df_cast.empty:
        print('  (cast_features.parquet not found — run cast_main.py then cast_encode.py first)')
        return

    row = df_cast[df_cast['film_id'] == film_id]
    if row.empty:
        print(f'  (film_id {film_id} not in cast_features.parquet)')
        return

    row = row.iloc[0]
    tier_label = FAME_TIER_LABELS.get(int(row.get('cast_max_fame_tier', 0)), '?')
    for col in CAST_DISPLAY_FIELDS:
        if col in row.index:
            val = row[col]
            suffix = f' ({tier_label})' if col == 'cast_max_fame_tier' else ''
            print(f'  {col:<30}  {val}{suffix}')


def print_cast_actors(film_id: int, df_films: pd.DataFrame, df_enriched: pd.DataFrame | None) -> None:
    """Print individual actor profiles as a markdown table."""
    row = df_films[df_films['film_id'] == film_id]
    if row.empty or pd.isna(row.iloc[0].get('actor_list', '')):
        return

    actors = _parse_actor_list(row.iloc[0]['actor_list'])

    if not actors:
        return

    idx = df_enriched.set_index('actor_name') if df_enriched is not None and 'actor_name' in df_enriched.columns else None

    print(f'\n### Cast ({len(actors)} actors)\n')
    print(f'| {"Actor":<32} | {"Tier":<14} | {"Source":<16} | {"Market":<14} | Cross Media |')
    print(f'|{"-"*34}|{"-"*16}|{"-"*18}|{"-"*16}|{"-"*13}|')

    for actor in actors:
        if idx is not None and actor in idx.index:
            p          = idx.loc[actor]
            tier       = str(p.get('fame_tier',      '—'))
            source     = str(p.get('fame_source',    '—'))
            market     = str(p.get('primary_market', '—'))
            cross      = _fmt(p.get('cross_media',   []))
        else:
            tier = source = market = cross = '—'
        print(f'| {actor:<32} | {tier:<14} | {source:<16} | {market:<14} | {cross} |')


# ── Excel export ──────────────────────────────────────────────────────────────

def save_comparison_excel(
    df_films: pd.DataFrame,
    v1_results: dict,
    v2_results: dict,
    df_cast: pd.DataFrame | None,
    df_enriched: pd.DataFrame | None,
) -> None:
    # ── Summary sheet ────────────────────────────────────────────────────────
    rows = []
    for film_id, v1 in v1_results.items():
        v2 = v2_results.get(film_id, {})
        title_row = df_films[df_films['film_id'] == film_id]
        title = title_row.iloc[0]['film_title'] if not title_row.empty else str(film_id)

        row = {'film_id': film_id, 'film_title': title}
        for field in COMPARE_FIELDS:
            row[f'existing_{field}'] = _fmt(v1.get(field))
            row[f'v2_{field}']       = _fmt(v2.get(field))

        if df_cast is not None:
            cast_row = df_cast[df_cast['film_id'] == film_id]
            if not cast_row.empty:
                for col in CAST_DISPLAY_FIELDS:
                    if col in cast_row.columns:
                        row[col] = cast_row.iloc[0][col]
        rows.append(row)

    def _actor_rows(film_id: int) -> pd.DataFrame:
        film_row = df_films[df_films['film_id'] == film_id]
        if film_row.empty or pd.isna(film_row.iloc[0].get('actor_list', '')):
            return pd.DataFrame()
        actors = _parse_actor_list(film_row.iloc[0]['actor_list'])
        if not actors:
            return pd.DataFrame()
        idx = df_enriched.set_index('actor_name') if df_enriched is not None and 'actor_name' in df_enriched.columns else None
        rows_cast = []
        for actor in actors:
            if idx is not None and actor in idx.index:
                p = idx.loc[actor]
                rows_cast.append({
                    'actor':          actor,
                    'fame_tier':      p.get('fame_tier',      '—'),
                    'fame_source':    p.get('fame_source',    '—'),
                    'primary_market': p.get('primary_market', '—'),
                    'cross_media':    _fmt(p.get('cross_media', [])),
                })
            else:
                rows_cast.append({'actor': actor, 'fame_tier': '—',
                                  'fame_source': '—', 'primary_market': '—', 'cross_media': '—'})
        return pd.DataFrame(rows_cast)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUTPUT_PATH, engine='openpyxl') as writer:
        pd.DataFrame(rows).to_excel(writer, sheet_name='comparison', index=False)

        # ── Per-film cast sheets ─────────────────────────────────────────────
        for _, film_row in df_films.iterrows():
            fid   = int(film_row['film_id'])
            title = film_row['film_title']
            df_actors = _actor_rows(fid)
            if df_actors.empty:
                continue
            # Excel sheet names max 31 chars, no special chars
            sheet = re.sub(r'[\\/*?:\[\]]', '', title)[:28].strip()
            df_actors.to_excel(writer, sheet_name=sheet, index=False)

    print(f'\nSaved → {OUTPUT_PATH}')


# ── Main ──────────────────────────────────────────────────────────────────────

# Load films
df_all   = load_films_from_snowflake()
df_films = select_target_films(df_all)

# if df_films.empty:
#     print("No target films found — check film IDs / titles.")
#     return

# Load existing cast data
df_cast = None
df_enriched = None
if CAST_FEATURES_PATH.exists():
    df_cast = pd.read_parquet(CAST_FEATURES_PATH)
    df_cast['film_id'] = df_cast['film_id'].astype(int)
if CAST_ENRICHED_PATH.exists():
    df_enriched = pd.read_parquet(CAST_ENRICHED_PATH)
    df_enriched['actor_name'] = df_enriched['actor_name'].str.upper().str.strip()

# Extract cast profiles for any actors in target films not yet enriched
df_enriched = asyncio.run(enrich_target_actors(df_films, df_enriched))

# Load existing extracts from parquet (no API cost)
print("\nLoading existing extracts...")
existing_results = load_existing_extracts([int(r['film_id']) for _, r in df_films.iterrows()])

# Run v2 extraction fresh
v2_results = asyncio.run(run_v2_extraction(df_films))

# Print comparison per film
for _, film_row in df_films.iterrows():
    fid   = int(film_row['film_id'])
    title = film_row['film_title']
    existing = existing_results.get(fid, {})
    v2       = v2_results.get(fid, {})

    print_film_comparison(fid, title, existing, v2)
    print_cast_features(fid, df_cast)
    print_cast_actors(fid, df_films, df_enriched)

# Save Excel
save_comparison_excel(df_films, existing_results, v2_results, df_cast, df_enriched)
