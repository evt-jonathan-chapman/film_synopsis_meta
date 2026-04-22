"""
refresh.py — weekly refresh orchestrator for film_synopsis_meta.

Checks Snowflake for new/updated films and actors, re-extracts only what has
changed, then re-encodes. Designed to run as a Dagster job but can also be
called directly.

Dagster usage (future):
    @asset
    def synopsis_refresh():
        return run_refresh()

Direct usage:
    python refresh.py

    # Or with flags:
    python refresh.py --force-synopsis --force-cast --force-encode
"""

import argparse
import asyncio
import datetime
import gc
import logging
import os
import re
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# Suppress litellm async logging
import litellm
litellm.success_callback = []
litellm.failure_callback = []

from config import (SYNOPSES_EXTRACTED_PATH, CAST_ENRICHED_PATH, CAST_FEATURES_PATH,
                    SF_WAREHOUSE, SF_DATABASE, SF_SCHEMA, SF_RSA_KEY, SQL_PATH)
from extractor import LlmJsonExtractor
from load_prompts import load_tasks_from_yaml
from models import DEFAULT_MODEL, DEFAULT_FALLBACKS, MODELS
from ingest import sync_synopses_sources
from encode_synopsis import encode_synopsis_features
from cast_encode import encode_cast_features

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)s  %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────

PROMPTS_PATH      = Path('prompts/prompts_v2.yaml')
CAST_PROMPTS_PATH = Path('prompts/cast_prompts.yaml')

_AND_PREFIX = re.compile(r'^AND\s+', re.IGNORECASE)


def _clean_actor(raw: str) -> str:
    name = raw.strip().upper()
    name = _AND_PREFIX.sub('', name).strip()
    return '' if name in ('AND', 'N/A', '') or len(name) <= 1 else name


# ── Diff helpers ──────────────────────────────────────────────────────────────

def _get_films_from_snowflake() -> pd.DataFrame | None:
    """
    Pull current film list from Snowflake.
    Returns DataFrame with at least: film_id, film_title, synopsis, actor_list.
    Returns None if Snowflake is unavailable.
    """
    try:
        sys.path.insert(0, '/Users/jonathanchapman/Documents/git/evt_back_up/base')
        from base_snowflake import SnowFlakeBase
        sb = SnowFlakeBase(warehouse=SF_WAREHOUSE, database=SF_DATABASE, schema=SF_SCHEMA)
        sb.create_snowflake_connection(SF_RSA_KEY)

        sql = (SQL_PATH / 'synopsis_look_up.sql').read_text()
        df  = pd.read_sql(sql, sb.engine)
        df['film_id'] = df['film_id'].astype(int)
        log.info(f"Snowflake: {len(df)} films loaded")
        return df
    except Exception as e:
        log.warning(f"Snowflake unavailable ({e}) — using local parquet fallback")
        return None


def _find_new_synopsis_films(df_snowflake: pd.DataFrame) -> pd.DataFrame:
    """
    Compare Snowflake films against local extracted parquet.
    Returns films that are new OR have a changed synopsis.
    """
    if not SYNOPSES_EXTRACTED_PATH.exists():
        log.info("No existing synopsis parquet — all films are new")
        return df_snowflake

    df_existing = pd.read_parquet(SYNOPSES_EXTRACTED_PATH, columns=['film_id', 'synopsis'])
    df_existing['film_id'] = df_existing['film_id'].astype(int)

    # New films
    existing_ids = set(df_existing['film_id'])
    df_new = df_snowflake[~df_snowflake['film_id'].isin(existing_ids)].copy()

    # Changed synopsis
    merged = df_snowflake[df_snowflake['film_id'].isin(existing_ids)].merge(
        df_existing.rename(columns={'synopsis': 'synopsis_old'}),
        on='film_id', how='left',
    )
    synopsis_col = 'synopsis' if 'synopsis' in merged.columns else merged.columns[1]
    changed = merged[merged[synopsis_col].fillna('') != merged['synopsis_old'].fillna('')]

    result = pd.concat([df_new, changed[df_snowflake.columns]], ignore_index=True)
    log.info(f"Synopsis diff: {len(df_new)} new, {len(changed)} updated → {len(result)} to extract")
    return result


def _find_new_actors(df_films: pd.DataFrame) -> list[str]:
    """
    Given the films to process, find actor names not yet in cast_enriched.parquet.
    Returns list of unique normalised actor names.
    """
    all_actors = set()
    for val in df_films.get('actor_list', pd.Series(dtype=str)).dropna():
        for a in str(val).split('|'):
            a = _clean_actor(a)
            if a:
                all_actors.add(a)

    if not CAST_ENRICHED_PATH.exists():
        log.info(f"No existing cast parquet — all {len(all_actors)} actors are new")
        return sorted(all_actors)

    existing = set(
        pd.read_parquet(CAST_ENRICHED_PATH, columns=['actor_name'])
        ['actor_name'].str.upper().str.strip()
    )
    new_actors = sorted(all_actors - existing)
    log.info(f"Cast diff: {len(new_actors)} new actors to enrich")
    return new_actors


# ── Extraction steps ──────────────────────────────────────────────────────────

async def _extract_synopses(df_films: pd.DataFrame) -> None:
    """Run async synopsis extraction for the given films and merge into parquet."""
    if df_films.empty:
        return

    tasks     = load_tasks_from_yaml(PROMPTS_PATH)
    model_cfg = MODELS.get(DEFAULT_MODEL, {})
    extractor = LlmJsonExtractor(
        tasks=tasks,
        model=DEFAULT_MODEL,
        fallbacks=DEFAULT_FALLBACKS,
        api_key=os.getenv('OPENAI_KEY'),
        cost_per_1m_input=model_cfg.get('cost_per_1m_input'),
        cost_per_1m_output=model_cfg.get('cost_per_1m_output'),
    )

    log.info(f"Extracting synopses for {len(df_films)} films...")
    results = await extractor.arun_multiple_synopses(
        df=df_films,
        id_col='film_id',
        title_col='film_title',
        synopsis_col='synopsis',
        alt_synopsis_col='alt_synopsis' if 'alt_synopsis' in df_films.columns else None,
        flatten=True,
        max_concurrency=20,
    )

    df_new = pd.DataFrame(results.values())
    if '_error' in df_new.columns:
        n_err = df_new['_error'].notna().sum()
        if n_err:
            log.warning(f"{n_err} films had extraction errors — excluded")
        df_new = df_new[df_new['_error'].isna()].copy()

    # Merge with existing parquet
    SYNOPSES_EXTRACTED_PATH.parent.mkdir(parents=True, exist_ok=True)
    if SYNOPSES_EXTRACTED_PATH.exists():
        df_existing = pd.read_parquet(SYNOPSES_EXTRACTED_PATH)
        out = (
            pd.concat([df_new, df_existing], ignore_index=True)
            .drop_duplicates(subset='film_id', keep='first')  # new wins on re-run
        )
    else:
        out = df_new

    out.to_parquet(SYNOPSES_EXTRACTED_PATH, engine='pyarrow', index=False)
    log.info(f"Synopsis parquet updated: {len(out)} total films → {SYNOPSES_EXTRACTED_PATH}")

    if extractor.token_usage:
        u = extractor.token_usage
        log.info(f"Tokens — prompt: {u['prompt_tokens']:,}  "
                 f"completion: {u['completion_tokens']:,}  "
                 f"cost: ${u.get('cost_usd', 0):.4f}")
    del extractor
    gc.collect()


async def _enrich_cast(new_actors: list[str]) -> None:
    """Run async cast enrichment for the given actor names."""
    if not new_actors:
        return

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

    df_actors = pd.DataFrame({'actor_name': new_actors, 'synopsis': ''})

    log.info(f"Enriching {len(new_actors)} actors...")
    results = await extractor.arun_multiple_synopses(
        df=df_actors,
        id_col='actor_name',
        title_col='actor_name',
        synopsis_col='synopsis',
        alt_synopsis_col=None,
        flatten=True,
        max_concurrency=20,
    )

    df_new = pd.DataFrame(results.values())
    if 'title' in df_new.columns:
        df_new = df_new.rename(columns={'title': 'actor_name'})
    if '_error' in df_new.columns:
        df_new = df_new[df_new['_error'].isna()].drop(
            columns=[c for c in ['_error', '_error_message', '_raw_output', 'synopsis']
                     if c in df_new.columns], errors='ignore'
        )

    CAST_ENRICHED_PATH.parent.mkdir(parents=True, exist_ok=True)
    if CAST_ENRICHED_PATH.exists():
        df_existing = pd.read_parquet(CAST_ENRICHED_PATH)
        out = (
            pd.concat([df_existing, df_new], ignore_index=True)
            .drop_duplicates(subset='actor_name', keep='first')
        )
    else:
        out = df_new

    out.to_parquet(CAST_ENRICHED_PATH, engine='pyarrow', index=False)
    log.info(f"Cast parquet updated: {len(out)} actors → {CAST_ENRICHED_PATH}")

    del extractor
    gc.collect()


# ── Main orchestrator ─────────────────────────────────────────────────────────

async def _run_refresh_async(
    force_synopsis: bool = False,
    force_cast: bool = False,
    force_encode: bool = False,
) -> dict:

    out_date         = datetime.datetime.today().strftime('%Y%m%d')
    synopsis_updated = False
    cast_updated     = False

    # ── 1. Load source data ──────────────────────────────────────────────────
    df_snowflake = _get_films_from_snowflake()

    # ── 2. Synopsis extraction ───────────────────────────────────────────────
    if df_snowflake is not None:
        df_to_extract = (
            df_snowflake if force_synopsis
            else _find_new_synopsis_films(df_snowflake)
        )
        if not df_to_extract.empty:
            await _extract_synopses(df_to_extract)
            synopsis_updated = True
        else:
            log.info("Synopses up to date — no extraction needed")
    else:
        log.warning("No Snowflake connection — skipping synopsis extraction step")

    # ── 3. Cast enrichment ───────────────────────────────────────────────────
    # Diff against all Snowflake films so actors added to existing films are
    # caught even when force_cast=False. force_cast=True ignores cast_enriched
    # and re-extracts every actor from scratch.
    if df_snowflake is not None:
        if force_cast:
            all_actors: set[str] = set()
            for val in df_snowflake.get('actor_list', pd.Series(dtype=str)).dropna():
                for a in str(val).split('|'):
                    a = _clean_actor(a)
                    if a:
                        all_actors.add(a)
            new_actors = sorted(all_actors)
            log.info(f"force_cast=True — re-enriching all {len(new_actors)} actors")
        else:
            new_actors = _find_new_actors(df_snowflake)

        if new_actors:
            await _enrich_cast(new_actors)
            cast_updated = True
        else:
            log.info("Cast enrichment up to date — no new actors")

    # ── 4. Snowflake sync ────────────────────────────────────────────────────
    if synopsis_updated and df_snowflake is not None:
        try:
            log.info("Syncing updated synopses to Snowflake...")
            sync_synopses_sources(SYNOPSES_EXTRACTED_PATH)
        except Exception as e:
            log.warning(f"Snowflake sync failed (non-fatal): {e}")

    # ── 5. Encode synopsis features ──────────────────────────────────────────
    if synopsis_updated or force_encode:
        log.info("Re-encoding synopsis features...")
        synopsis_out = encode_synopsis_features(out_date=out_date)
        log.info(f"Synopsis features saved → {synopsis_out}")
    else:
        log.info("Synopsis encoding skipped — no changes")

    # ── 6. Encode cast features ──────────────────────────────────────────────
    if cast_updated or force_encode:
        log.info("Re-encoding cast features...")
        cast_out = encode_cast_features()
        log.info(f"Cast features saved → {cast_out}")
    else:
        log.info("Cast encoding skipped — no changes")

    result = {
        'run_date':        out_date,
        'synopsis_updated': synopsis_updated,
        'cast_updated':     cast_updated,
        'encoded':          synopsis_updated or cast_updated or force_encode,
    }
    log.info(f"Refresh complete: {result}")
    return result


def run_refresh(
    force_synopsis: bool = False,
    force_cast: bool = False,
    force_encode: bool = False,
) -> dict:
    """
    Entry point for Dagster or direct calls.

    Args:
        force_synopsis: Re-extract all synopses regardless of diff.
        force_cast:     Re-enrich all actors regardless of diff.
        force_encode:   Re-encode even if nothing changed (useful after
                        editing overrides in encode_synopsis.py).

    Returns:
        Dict with run_date, synopsis_updated, cast_updated, encoded.
    """
    return asyncio.run(_run_refresh_async(force_synopsis, force_cast, force_encode))


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='film_synopsis_meta weekly refresh')
    parser.add_argument('--force-synopsis', action='store_true',
                        help='Re-extract all synopses')
    parser.add_argument('--force-cast',     action='store_true',
                        help='Re-enrich all actors')
    parser.add_argument('--force-encode',   action='store_true',
                        help='Re-encode even if nothing changed')
    args = parser.parse_args()

    run_refresh(
        force_synopsis=args.force_synopsis,
        force_cast=args.force_cast,
        force_encode=args.force_encode,
    )
