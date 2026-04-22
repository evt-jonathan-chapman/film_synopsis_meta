"""
Cast enrichment pipeline.

Reads actor_list (pipe-separated) from raw Snowflake parquets, deduplicates
to unique actors, queries LLM for fame profile per actor, saves to parquet.

Output: cast_meta/outputs/cast_enriched.parquet
  Columns: actor_name, fame_source, fame_tier, primary_market, cross_media,
           film_count, _error, _error_message

Usage: edit the run block at the bottom, then:
    python cast_main.py
"""

import sys
sys.path.insert(0, '/Users/jonathanchapman/Documents/git/evt_back_up/base')

import asyncio
import nest_asyncio
nest_asyncio.apply()
import gc
import glob
import os
from collections import Counter
import pandas as pd
from dotenv import load_dotenv

import litellm
litellm.success_callback = []
litellm.failure_callback = []

load_dotenv()

from extractor import LlmJsonExtractor
from load_prompts import load_tasks_from_yaml
from models import DEFAULT_MODEL, DEFAULT_FALLBACKS, MODELS
from config import CAST_ENRICHED_PATH, RAW_PARQUET_GLOB


# ── Helpers ───────────────────────────────────────────────────────────────────

import re
_AND_PREFIX = re.compile(r'^AND\s+', re.IGNORECASE)

def _clean_actor_name(raw: str) -> str:
    """
    Normalise a single actor token from a pipe-split actor_list.
    - Strip whitespace and uppercase
    - Remove SQL artefact 'AND ' prefix (e.g. '|AND YOON KYE-SANG' → 'YOON KYE-SANG')
      caused by comma-before-AND in source data not fully normalised by the SQL regex
    - Return empty string for tokens that are just 'AND' or empty after cleaning
    """
    name = raw.strip().upper()
    name = _AND_PREFIX.sub('', name).strip()
    # Skip bare 'AND', 'N/A', single characters
    if name in ('AND', 'N/A', '') or len(name) <= 1:
        return ''
    return name


# ── Actor loading ──────────────────────────────────────────────────────────────

def load_unique_actors(glob_pattern: str, min_film_count: int = 3) -> pd.DataFrame:
    """
    Read all raw training parquets, split actor_list by pipe, count how many
    unique films each actor appears in (after uppercasing for deduplication),
    return DataFrame sorted by film_count descending.
    """
    paths = sorted(glob.glob(glob_pattern))
    if not paths:
        raise FileNotFoundError(f"No parquets found matching: {glob_pattern}")

    counts: Counter = Counter()
    for path in paths:
        df = pd.read_parquet(path, columns=['film_id', 'actor_list'])
        df = df.drop_duplicates('film_id')
        for val in df['actor_list'].dropna():
            for actor in str(val).split('|'):
                actor = _clean_actor_name(actor)
                if actor:
                    counts[actor] += 1

    actors_df = (
        pd.DataFrame(counts.items(), columns=['actor_name', 'film_count'])
        .query('film_count >= @min_film_count')
        .sort_values('film_count', ascending=False)
        .reset_index(drop=True)
    )
    print(f"Unique actors (film_count >= {min_film_count}): {len(actors_df)}")
    return actors_df


# ── Main extraction ───────────────────────────────────────────────────────────

async def main(actors_df: pd.DataFrame, sample_size: int = 0):
    CAST_ENRICHED_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Load existing checkpoint
    if CAST_ENRICHED_PATH.exists():
        df_existing = pd.read_parquet(CAST_ENRICHED_PATH)
        existing_names = set(df_existing['actor_name'].str.upper())
        df_new = actors_df[~actors_df['actor_name'].isin(existing_names)].copy()
        print(f"Existing: {len(df_existing):,}  New to process: {len(df_new):,}")
    else:
        df_existing = pd.DataFrame()
        df_new = actors_df.copy()
        print(f"No existing data. Processing all {len(df_new):,} actors.")

    if df_new.empty:
        print("Nothing to process.")
        return

    # Sample
    if sample_size > 0:
        df_new = df_new.head(sample_size).copy()
        print(f"Sampling top {sample_size} actors by film_count.")

    # Build input — actor_name is both ID and title; synopsis is empty (unused by cast prompt)
    df_new = df_new.assign(synopsis='').reset_index(drop=True)

    tasks   = load_tasks_from_yaml(Path('prompts/cast_prompts.yaml'))
    model_cfg = MODELS.get(DEFAULT_MODEL, {})

    extractor = LlmJsonExtractor(
        tasks=tasks,
        model=DEFAULT_MODEL,
        fallbacks=DEFAULT_FALLBACKS,
        api_key=os.getenv('OPENAI_KEY'),
        cost_per_1m_input=model_cfg.get('cost_per_1m_input'),
        cost_per_1m_output=model_cfg.get('cost_per_1m_output'),
    )

    # Use async extraction — much faster for large actor lists
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

    df_results = pd.DataFrame(results.values())
    if df_results.empty:
        print("Warning: no results returned from extractor.")
        return
    # The extractor stores the title under 'title' — rename to actor_name
    if 'title' in df_results.columns:
        df_results = df_results.rename(columns={'title': 'actor_name'})

    # Drop error rows from the main output but log them
    if '_error' in df_results.columns:
        n_errors = df_results['_error'].notna().sum()
        if n_errors:
            print(f"Warning: {n_errors} actors had extraction errors — excluded from output.")
        df_results = df_results[df_results['_error'].isna()].drop(
            columns=[c for c in ['_error', '_error_message', '_raw_output', 'synopsis']
                     if c in df_results.columns],
            errors='ignore'
        )

    # Re-attach film_count
    df_results = df_results.merge(
        df_new[['actor_name', 'film_count']], on='actor_name', how='left'
    )

    # Combine with existing and deduplicate (keep first = existing wins on re-run)
    out_df = pd.concat([df_existing, df_results], ignore_index=True)
    out_df = out_df.drop_duplicates(subset='actor_name', keep='first').reset_index(drop=True)

    out_df.to_parquet(CAST_ENRICHED_PATH, engine='pyarrow', index=False)
    print(f"\nSaved {len(out_df):,} actors → {CAST_ENRICHED_PATH}")

    if extractor.token_usage:
        u = extractor.token_usage
        print(f"Tokens — prompt: {u['prompt_tokens']:,}  "
              f"completion: {u['completion_tokens']:,}  "
              f"cost: ${u.get('cost_usd', 0):.4f}")

    del extractor
    gc.collect()


# ── Run ────────────────────────────────────────────────────────────────────────
# min_film_count=2 skips one-off actors to reduce cost.
# sample_size=50 for a validation run; set to 0 for all actors.

actors_df = load_unique_actors(RAW_PARQUET_GLOB, min_film_count=2)

asyncio.run(main(actors_df, sample_size=50))
