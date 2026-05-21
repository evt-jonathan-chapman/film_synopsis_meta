"""
refresh.py — diff-based refresh orchestrator for film_synopsis_meta.

Four independent refresh functions, one per extraction path:
    refresh_synopsis(df_films)
    refresh_cast(df_films)
    refresh_directors(df_films)
    refresh_film_meta(df_films, film_lookup)

Each can be called standalone (df_films=None → loads from Snowflake) or
with a pre-loaded films DataFrame (used by Dagster to share one Snowflake
pull across all four assets).

CLI:
    python refresh.py                                  # run all four
    python refresh.py --only synopsis cast             # subset
    python refresh.py --force-synopsis                 # skip diff for synopsis
"""

import argparse
import asyncio
import datetime
import gc
import json
import logging
import os
import re
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

import litellm
litellm.success_callback = []
litellm.failure_callback = []

from config import (
    DATA_DIR,
    SYNOPSES_EXTRACTED_PATH, CAST_ENRICHED_PATH,
    DIRECTOR_ENRICHED_PATH, FILM_META_ENRICHED_PATH,
    SF_WAREHOUSE, SF_DATABASE, SF_SCHEMA, SF_RSA_KEY,
)
from extractor import LlmJsonExtractor
from film_meta_extractor import FilmMetaExtractor, ActorMetaExtractor, DirectorMetaExtractor
from load_prompts import load_tasks_from_yaml
from models import DEFAULT_MODEL, DEFAULT_FALLBACKS, MODELS
from ingest import sync_synopses_sources
from films import sql as films_sql

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)s  %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
log = logging.getLogger(__name__)

# ── Paths / constants ─────────────────────────────────────────────────────────

PROMPTS_PATH           = Path('prompts/prompts_v2.yaml')
CAST_PROMPTS_PATH      = Path('prompts/cast_prompts.yaml')
DIRECTOR_PROMPTS_PATH  = Path('prompts/director_prompts.yaml')
FILM_META_PROMPTS_PATH = Path('prompts/film_meta_prompts.yaml')

META_MAX_CONCURRENCY = 2   # web_search calls bound by 200k TPM ≈ 13 req/min sustainable
META_BATCH_SIZE       = 25  # films per checkpoint flush — matches main.py
META_BATCH_PAUSE_SECS = 3   # pause between batches to let TPM window reset

FILM_META_CHECKPOINT_PATH = DATA_DIR / 'film_meta' / 'film_meta_progress.json'
FILM_META_ERRORS_PATH     = DATA_DIR / 'film_meta' / 'film_meta_errors.json'
CAST_CHECKPOINT_PATH      = DATA_DIR / 'cast_meta'     / 'cast_progress.json'
CAST_ERRORS_PATH          = DATA_DIR / 'cast_meta'     / 'cast_errors.json'
DIRECTOR_CHECKPOINT_PATH  = DATA_DIR / 'director_meta' / 'director_progress.json'
DIRECTOR_ERRORS_PATH      = DATA_DIR / 'director_meta' / 'director_errors.json'

# Distributors with no extractable cast/budget/studios metadata.
# Mirrors main.py::FILM_META_SKIP_DISTRIBUTORS.
FILM_META_SKIP_DISTRIBUTORS = {
    "ZZ Japanese Film Festival", "ZZ JEWISH FILM FESTIVAL", "ZZ Russian Film Festival",
    "AU Sydney Science Fiction Film Festival", "AU KOREAN FILM FESTIVAL",
    "AU Taiwan Film Festival", "AU SciFi Film Festival",
    "ZZ Sydney Underground Film Festival Inc", "ZZ Iranian Film Festival Australia",
    "ZZ SOUTH AFRICAN FILM FESTIVAL", "ZZ UKRANIAN FILM FESTIVAL",
    "ZZ JIFF Distribution", "ZZ Flickerfest", "ZZ Gold Coast Film Fantastic Ltd",
    "ZZ SF3 - SMARTFONE FLICK FEST", "ZZ GOETHE INSTITUTE", "ZZ FOR FILMS SAKE",
    "AU Trafalgar Releasing Ltd", "AU Cinema Live", "AU PATHE LIVE",
    "ZZ THE WIGGLES INTERNATIONAL", "AU FATHOM EVENTS",
    "ZZ Fox Sports Venues", "ZZ BeIN SPORTS",
    "ZZ Queensland Cricket Association Ltd", "ZZ ESPN Australia Pty Ltd",
    "AU IMAX THEATRES INTL", "ZZ Nickelodeon Australia Management",
    "ZZ CRUNCHYROLL PTY LTD", "ZZ SBS-ALTERNATE CONTENT",
}

_AND_PREFIX = re.compile(r'^AND\s+', re.IGNORECASE)


def _clean_actor(raw: str) -> str:
    name = raw.strip().upper()
    name = _AND_PREFIX.sub('', name).strip()
    return '' if name in ('AND', 'N/A', '') or len(name) <= 1 else name


# ── Source loaders ────────────────────────────────────────────────────────────

def load_films_from_snowflake() -> pd.DataFrame | None:
    """Returns the curated film work-set used by all four extraction paths.

    Mirrors main.py's loader: reads the train/test/prediction parquet snapshots
    (the model-relevant subset of EVT's catalogue — ~4–5k films), drops rows
    without a usable synopsis, then joins authoritative titles + the
    distributor column from Snowflake.

    Function name kept for backwards-compat with dagster_defs.py — note the
    primary source is now the parquets, not Snowflake.

    Returns None only if the parquet snapshots are unreadable.
    """
    import glob

    raw_paths  = sorted(glob.glob(str(DATA_DIR / 'raw_from_snowflake'        / '*' / 'train' / 'train_raw_ds.parquet')))
    raw_paths += sorted(glob.glob(str(DATA_DIR / 'raw_from_snowflake'        / '*' / 'test'  / 'test_raw_ds.parquet')))
    pred_paths = sorted(glob.glob(str(DATA_DIR / 'prediction_from_snowflake' / '*' / 'prediction_raw.parquet')))
    all_paths  = raw_paths + pred_paths

    if not all_paths:
        log.warning(f"No parquet snapshots found under {DATA_DIR}/raw_from_snowflake or /prediction_from_snowflake")
        return None

    parts = []
    for p in all_paths:
        part = pd.read_parquet(p, columns=['film_id', 'synopsis', 'actor_list',
                                           'rel_at', 'director', 'dstbtr'])
        part['rel_at'] = pd.to_datetime(part['rel_at'], utc=True, errors='coerce')
        parts.append(part)
        log.info(f"  {Path(p).relative_to(DATA_DIR)}: {part['film_id'].nunique()} films")

    df = (pd.concat(parts, ignore_index=True)
            .drop_duplicates('film_id')
            .assign(film_id=lambda d: d['film_id'].astype(int)))
    log.info(f"Unique films from parquets: {len(df)}")

    df = df[df['synopsis'].notna() & (df['synopsis'].astype(str).str.len() >= 5)].copy()
    log.info(f"Films with synopsis: {len(df)}")

    try:
        from base_snowflake import SnowFlakeBase
        sb = SnowFlakeBase(warehouse=SF_WAREHOUSE, database=SF_DATABASE, schema=SF_SCHEMA)
        sb.create_snowflake_connection(SF_RSA_KEY)
        titles = pd.read_sql(films_sql.SQL_FILM_DETAILS, sb.engine)[['film_id', 'film_title']]
        titles['film_id'] = titles['film_id'].astype(int)
        df = df.merge(titles, on='film_id', how='left')
        log.info("Film titles joined from Snowflake")
    except Exception as e:
        log.warning(f"Snowflake unavailable ({e}) — using film_id as title fallback")
        df['film_title'] = df['film_id'].astype(str)

    return df


def load_full_film_catalogue() -> pd.DataFrame | None:
    """Full Snowflake film_details — used by the re-release filter so older
    releases (outside our model-relevant snapshot window) can still be matched
    against current films. Mirrors main.py's `_flu_full` query."""
    try:
        from base_snowflake import SnowFlakeBase
        sb = SnowFlakeBase(warehouse=SF_WAREHOUSE, database=SF_DATABASE, schema=SF_SCHEMA)
        sb.create_snowflake_connection(SF_RSA_KEY)
        full = pd.read_sql(films_sql.SQL_FILM_DETAILS, sb.engine)
        full = full.rename(columns={
            'director_list':      'director',
            'distributor_name':   'dstbtr',
            'film_nat_open_date': 'rel_at',
            'film_title':         'film',
        })
        full['film_id'] = full['film_id'].astype(int)
        full['rel_at']  = pd.to_datetime(full['rel_at'], utc=True, errors='coerce')
        log.info(f"Full Snowflake catalogue loaded for re-release lookup: {len(full)} films")
        return full
    except Exception as e:
        log.warning(f"Snowflake unavailable for re-release lookup ({e})")
        return None


def _ensure_films(df_films: pd.DataFrame | None) -> pd.DataFrame | None:
    return df_films if df_films is not None else load_films_from_snowflake()


# ── Diff helpers ──────────────────────────────────────────────────────────────

def _diff_synopsis_films(df_films: pd.DataFrame) -> pd.DataFrame:
    """Films that are new OR whose synopsis text has changed."""
    if not SYNOPSES_EXTRACTED_PATH.exists():
        log.info("No existing synopsis parquet — all films are new")
        return df_films

    existing = pd.read_parquet(SYNOPSES_EXTRACTED_PATH, columns=['film_id', 'synopsis'])
    existing['film_id'] = existing['film_id'].astype(int)
    existing_ids = set(existing['film_id'])

    new = df_films[~df_films['film_id'].isin(existing_ids)]

    merged = df_films[df_films['film_id'].isin(existing_ids)].merge(
        existing.rename(columns={'synopsis': 'synopsis_old'}),
        on='film_id', how='left',
    )
    changed = merged[merged['synopsis'].fillna('') != merged['synopsis_old'].fillna('')]
    result = pd.concat([new, changed[df_films.columns]], ignore_index=True)
    log.info(f"Synopsis diff: {len(new)} new + {len(changed)} updated → {len(result)} to extract")
    return result


def _diff_actors(df_films: pd.DataFrame) -> list[str]:
    """Actors not yet in cast_enriched.parquet OR cast_progress.json checkpoint.

    Reading both means a partially-completed run (checkpoint written, parquet
    not yet flushed) doesn't get re-extracted from scratch.
    """
    all_actors: set[str] = set()
    for val in df_films.get('actor_list', pd.Series(dtype=str)).dropna():
        for a in str(val).split('|'):
            a = _clean_actor(a)
            if a:
                all_actors.add(a)

    done: set[str] = set()
    if CAST_ENRICHED_PATH.exists():
        done |= set(
            pd.read_parquet(CAST_ENRICHED_PATH, columns=['actor_name'])
            ['actor_name'].astype(str).str.upper().str.strip()
        )
    if CAST_CHECKPOINT_PATH.exists():
        with open(CAST_CHECKPOINT_PATH) as f:
            done |= {str(k).upper().strip() for k in json.load(f).keys()}

    new = sorted(all_actors - done)
    log.info(f"Cast diff: {len(new)} new actors  ({len(done)} already done across parquet+checkpoint)")
    return new


def _diff_directors(df_films: pd.DataFrame) -> list[str]:
    """Directors not yet in director_enriched.parquet OR director_progress.json."""
    all_dirs: set[str] = set()
    for val in df_films.get('director', pd.Series(dtype=str)).dropna():
        # Snowflake pipes; raw parquets sometimes comma — handle both.
        parts = re.split(r'[|,]', str(val))
        for d in parts:
            d = d.strip()
            if d:
                all_dirs.add(d)

    done: set[str] = set()
    if DIRECTOR_ENRICHED_PATH.exists():
        done |= set(
            pd.read_parquet(DIRECTOR_ENRICHED_PATH, columns=['director_name'])
            ['director_name'].astype(str).str.strip()
        )
    if DIRECTOR_CHECKPOINT_PATH.exists():
        with open(DIRECTOR_CHECKPOINT_PATH) as f:
            done |= {str(k).strip() for k in json.load(f).keys()}

    new = sorted(all_dirs - done)
    log.info(f"Director diff: {len(new)} new directors  ({len(done)} already done across parquet+checkpoint)")
    return new


def _diff_film_meta(df_films: pd.DataFrame) -> pd.DataFrame:
    """Films not yet in checkpoint JSON or enriched parquet, with skip-distributor filter applied.

    Checks the checkpoint as well as the parquet so a partially-completed run (where
    the checkpoint has been written but the final parquet flush has not yet happened)
    isn't re-extracted.
    """
    df = df_films.copy()
    n0 = len(df)
    if 'dstbtr' in df.columns:
        df = df[~df['dstbtr'].isin(FILM_META_SKIP_DISTRIBUTORS)]
        if n0 - len(df):
            log.info(f"film_meta skip-distributors: -{n0 - len(df)} → {len(df)}")

    done_ids: set[int] = set()
    if FILM_META_CHECKPOINT_PATH.exists():
        with open(FILM_META_CHECKPOINT_PATH) as f:
            done_ids |= {int(k) for k in json.load(f)}
    if FILM_META_ENRICHED_PATH.exists():
        done_ids |= set(
            pd.read_parquet(FILM_META_ENRICHED_PATH, columns=['film_id'])
            ['film_id'].astype(int)
        )

    df = df[~df['film_id'].astype(int).isin(done_ids)]
    log.info(f"film_meta diff: {len(done_ids)} already done — {len(df)} to extract")
    return df


# ── Extraction steps ──────────────────────────────────────────────────────────

async def _extract_synopses(df: pd.DataFrame) -> None:
    if df.empty:
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

    log.info(f"Extracting synopses for {len(df)} films")
    results = await extractor.arun_multiple_synopses(
        df=df,
        id_col='film_id',
        title_col='film_title',
        synopsis_col='synopsis',
        alt_synopsis_col='alt_synopsis' if 'alt_synopsis' in df.columns else None,
        flatten=True,
        max_concurrency=20,
    )

    df_new = pd.DataFrame(results.values())
    if '_error' in df_new.columns:
        n_err = df_new['_error'].notna().sum()
        if n_err:
            log.warning(f"{n_err} films had extraction errors — excluded")
        df_new = df_new[df_new['_error'].isna()].copy()

    SYNOPSES_EXTRACTED_PATH.parent.mkdir(parents=True, exist_ok=True)
    if SYNOPSES_EXTRACTED_PATH.exists():
        existing = pd.read_parquet(SYNOPSES_EXTRACTED_PATH)
        out = (pd.concat([df_new, existing], ignore_index=True)
               .drop_duplicates(subset='film_id', keep='first'))
    else:
        out = df_new
    out.to_parquet(SYNOPSES_EXTRACTED_PATH, engine='pyarrow', index=False)
    log.info(f"Synopsis parquet → {SYNOPSES_EXTRACTED_PATH}  ({len(out)} total)")

    if extractor.token_usage:
        u = extractor.token_usage
        log.info(f"Synopsis tokens — prompt: {u['prompt_tokens']:,}  "
                 f"completion: {u['completion_tokens']:,}  "
                 f"cost: ${u.get('cost_usd', 0):.4f}")
    del extractor
    gc.collect()


async def _enrich_cast(new_actors: list[str]) -> None:
    """Batch-processes actor enrichment with per-batch checkpoint + errors JSON.

    Mirrors _enrich_film_meta: each batch writes cast_progress.json (successes)
    and cast_errors.json (failures, with already-recovered entries purged). A
    crash mid-run loses only the in-flight batch. Final step flushes the
    checkpoint into cast_enriched.parquet.
    """
    if not new_actors:
        return

    CAST_CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    checkpoint: dict = {}
    if CAST_CHECKPOINT_PATH.exists():
        with open(CAST_CHECKPOINT_PATH) as f:
            checkpoint = json.load(f)
        log.info(f"Loaded cast checkpoint: {len(checkpoint)} actors already done")

    tasks       = load_tasks_from_yaml(CAST_PROMPTS_PATH)
    model       = os.environ.get('FILM_META_MODEL', 'gpt-5.4-mini')
    model_cfg   = MODELS.get(model, {})
    extractor   = ActorMetaExtractor(
        task=tasks['actor_profile'],
        model=model,
        api_key=os.getenv('OPENAI_KEY'),
        cost_per_1m_input=model_cfg.get('cost_per_1m_input'),
        cost_per_1m_output=model_cfg.get('cost_per_1m_output'),
    )

    df_actors = pd.DataFrame({'actor_name': new_actors})
    chunks    = [df_actors.iloc[i:i + META_BATCH_SIZE]
                 for i in range(0, len(df_actors), META_BATCH_SIZE)]
    log.info(f"Enriching {len(new_actors)} actors in {len(chunks)} batches of {META_BATCH_SIZE}")
    prev_cost = 0.0

    for batch_num, chunk in enumerate(chunks, 1):
        log.info(f"Cast batch {batch_num}/{len(chunks)}  ({len(chunk)} actors)")
        results = await extractor.arun(
            df=chunk,
            name_col='actor_name',
            max_concurrency=META_MAX_CONCURRENCY,
        )

        batch_errors: dict[str, dict] = {}
        batch_success_keys: set[str] = set()
        for actor_name, data in results.items():
            if not data.get('_error'):
                checkpoint[str(actor_name)] = {**data, 'actor_name': actor_name}
                batch_success_keys.add(str(actor_name))
            else:
                batch_errors[str(actor_name)] = {
                    'actor_name':  actor_name,
                    '_error':      data.get('_error'),
                    '_raw_output': data.get('_raw_output'),
                }

        with open(CAST_CHECKPOINT_PATH, 'w') as f:
            json.dump(checkpoint, f, default=str)
        log.info(f"  Checkpoint saved: {len(checkpoint)} actors → {CAST_CHECKPOINT_PATH}")

        if batch_errors or (batch_success_keys and CAST_ERRORS_PATH.exists()):
            existing_errors: dict = {}
            if CAST_ERRORS_PATH.exists():
                with open(CAST_ERRORS_PATH) as f:
                    existing_errors = json.load(f)
            purged_n = sum(1 for k in batch_success_keys
                           if existing_errors.pop(k, None) is not None)
            existing_errors.update(batch_errors)
            if purged_n or batch_errors:
                with open(CAST_ERRORS_PATH, 'w') as f:
                    json.dump(existing_errors, f, default=str, indent=2)
            msg = f"  Errors this batch: {len(batch_errors)}"
            if purged_n:
                msg += f"  [purged {purged_n} now-recovered]"
            msg += f" → {CAST_ERRORS_PATH}"
            log.info(msg)

        if extractor.token_usage:
            curr  = extractor.token_usage.get('cost_usd', 0.0)
            delta = curr - prev_cost
            prev_cost = curr
            log.info(f"  Batch cost: ${delta:.4f}  |  Run total: ${curr:.4f}")

        if batch_num < len(chunks):
            await asyncio.sleep(META_BATCH_PAUSE_SECS)

    # ── Flush checkpoint → parquet ────────────────────────────────────────────
    df_new = pd.DataFrame(checkpoint.values())
    if df_new.empty:
        log.warning("No cast results in checkpoint")
        return
    df_new = df_new.drop(columns=[c for c in
        ['_error', '_error_message', '_raw_output', 'synopsis', 'title']
        if c in df_new.columns], errors='ignore')
    for col in ['fame_tier', 'fame_source', 'primary_market', 'age_range']:
        if col in df_new.columns:
            df_new[col] = df_new[col].astype(str).str.lower().str.strip()

    CAST_ENRICHED_PATH.parent.mkdir(parents=True, exist_ok=True)
    if CAST_ENRICHED_PATH.exists():
        existing = pd.read_parquet(CAST_ENRICHED_PATH)
        out = (pd.concat([existing, df_new], ignore_index=True)
               .drop_duplicates(subset='actor_name', keep='first'))
    else:
        out = df_new
    out.to_parquet(CAST_ENRICHED_PATH, engine='pyarrow', index=False)
    log.info(f"Cast parquet → {CAST_ENRICHED_PATH}  ({len(out)} actors)")

    if extractor.token_usage:
        u = extractor.token_usage
        log.info(f"Cast tokens — prompt: {u['prompt_tokens']:,}  "
                 f"completion: {u['completion_tokens']:,}  "
                 f"searches: {u.get('search_calls', 0):,}  "
                 f"cost: ${u.get('cost_usd', 0):.4f}")
    del extractor
    gc.collect()


async def _enrich_directors(new_directors: list[str]) -> None:
    """Batch-processes director enrichment with per-batch checkpoint + errors JSON.

    Mirrors _enrich_cast / _enrich_film_meta.
    """
    if not new_directors:
        return

    DIRECTOR_CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    checkpoint: dict = {}
    if DIRECTOR_CHECKPOINT_PATH.exists():
        with open(DIRECTOR_CHECKPOINT_PATH) as f:
            checkpoint = json.load(f)
        log.info(f"Loaded director checkpoint: {len(checkpoint)} directors already done")

    tasks     = load_tasks_from_yaml(DIRECTOR_PROMPTS_PATH)
    model     = os.environ.get('FILM_META_MODEL', 'gpt-5.4-mini')
    model_cfg = MODELS.get(model, {})
    extractor = DirectorMetaExtractor(
        task=tasks['director_profile'],
        model=model,
        api_key=os.getenv('OPENAI_KEY'),
        cost_per_1m_input=model_cfg.get('cost_per_1m_input'),
        cost_per_1m_output=model_cfg.get('cost_per_1m_output'),
    )

    df_dirs = pd.DataFrame({'director_name': new_directors})
    chunks  = [df_dirs.iloc[i:i + META_BATCH_SIZE]
               for i in range(0, len(df_dirs), META_BATCH_SIZE)]
    log.info(f"Enriching {len(new_directors)} directors in {len(chunks)} batches of {META_BATCH_SIZE}")
    prev_cost = 0.0

    for batch_num, chunk in enumerate(chunks, 1):
        log.info(f"Director batch {batch_num}/{len(chunks)}  ({len(chunk)} directors)")
        results = await extractor.arun(
            df=chunk,
            name_col='director_name',
            max_concurrency=META_MAX_CONCURRENCY,
        )

        batch_errors: dict[str, dict] = {}
        batch_success_keys: set[str] = set()
        for name, data in results.items():
            if not data.get('_error'):
                checkpoint[str(name)] = {**data, 'director_name': name}
                batch_success_keys.add(str(name))
            else:
                batch_errors[str(name)] = {
                    'director_name': name,
                    '_error':        data.get('_error'),
                    '_raw_output':   data.get('_raw_output'),
                }

        with open(DIRECTOR_CHECKPOINT_PATH, 'w') as f:
            json.dump(checkpoint, f, default=str)
        log.info(f"  Checkpoint saved: {len(checkpoint)} directors → {DIRECTOR_CHECKPOINT_PATH}")

        if batch_errors or (batch_success_keys and DIRECTOR_ERRORS_PATH.exists()):
            existing_errors: dict = {}
            if DIRECTOR_ERRORS_PATH.exists():
                with open(DIRECTOR_ERRORS_PATH) as f:
                    existing_errors = json.load(f)
            purged_n = sum(1 for k in batch_success_keys
                           if existing_errors.pop(k, None) is not None)
            existing_errors.update(batch_errors)
            if purged_n or batch_errors:
                with open(DIRECTOR_ERRORS_PATH, 'w') as f:
                    json.dump(existing_errors, f, default=str, indent=2)
            msg = f"  Errors this batch: {len(batch_errors)}"
            if purged_n:
                msg += f"  [purged {purged_n} now-recovered]"
            msg += f" → {DIRECTOR_ERRORS_PATH}"
            log.info(msg)

        if extractor.token_usage:
            curr  = extractor.token_usage.get('cost_usd', 0.0)
            delta = curr - prev_cost
            prev_cost = curr
            log.info(f"  Batch cost: ${delta:.4f}  |  Run total: ${curr:.4f}")

        if batch_num < len(chunks):
            await asyncio.sleep(META_BATCH_PAUSE_SECS)

    # ── Flush checkpoint → parquet ────────────────────────────────────────────
    df_new = pd.DataFrame(checkpoint.values())
    if df_new.empty:
        log.warning("No director results in checkpoint")
        return
    df_new = df_new.drop(columns=[c for c in
        ['_error', '_error_message', '_raw_output', 'synopsis', 'title']
        if c in df_new.columns], errors='ignore')
    for col in ['director_tier', 'primary_market']:
        if col in df_new.columns:
            df_new[col] = df_new[col].astype(str).str.lower().str.strip()

    DIRECTOR_ENRICHED_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DIRECTOR_ENRICHED_PATH.exists():
        existing = pd.read_parquet(DIRECTOR_ENRICHED_PATH)
        out = (pd.concat([existing, df_new], ignore_index=True)
               .drop_duplicates(subset='director_name', keep='first'))
    else:
        out = df_new
    out.to_parquet(DIRECTOR_ENRICHED_PATH, engine='pyarrow', index=False)
    log.info(f"Director parquet → {DIRECTOR_ENRICHED_PATH}  ({len(out)} directors)")

    if extractor.token_usage:
        u = extractor.token_usage
        log.info(f"Director tokens — prompt: {u['prompt_tokens']:,}  "
                 f"completion: {u['completion_tokens']:,}  "
                 f"searches: {u.get('search_calls', 0):,}  "
                 f"cost: ${u.get('cost_usd', 0):.4f}")
    del extractor
    gc.collect()


async def _enrich_film_meta(df: pd.DataFrame, film_lookup: pd.DataFrame | None) -> None:
    """Batch-processes film_meta extraction with per-batch checkpoint + error JSON
    writes. Mirrors main.py::enrich_film_meta so Dagster runs are safely
    interruptible — every BATCH_SIZE films, progress is persisted to disk."""
    if df.empty:
        return

    if film_lookup is not None and 'rel_at' in film_lookup.columns:
        try:
            from vendored.cinema_admits_models.re_release_filter import ReReleaseFilter
            rr = ReReleaseFilter()
            flagged = rr.flag(df.rename(columns={'film_title': 'film'}),
                              film_lookup, title_col='film')
            n_rr = int(flagged['rerelease_flag'].sum())
            df = (flagged[flagged['rerelease_flag'] == 0]
                  .rename(columns={'film': 'film_title'})
                  .copy())
            if n_rr:
                log.info(f"film_meta re-releases filtered: -{n_rr} → {len(df)}")
        except Exception as e:
            log.warning(f"Re-release filter skipped ({e})")

    if df.empty:
        return

    FILM_META_CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if FILM_META_CHECKPOINT_PATH.exists():
        with open(FILM_META_CHECKPOINT_PATH) as f:
            checkpoint: dict = json.load(f)
        log.info(f"Loaded film_meta checkpoint: {len(checkpoint)} films already done")
    else:
        checkpoint = {}

    done_ids = {int(k) for k in checkpoint}
    df = df[~df['film_id'].astype(int).isin(done_ids)].copy()
    log.info(f"film_meta: {len(done_ids)} in checkpoint — {len(df)} to extract")
    if df.empty:
        log.info("film_meta enrichment up to date.")
        return

    evt_passthrough = (
        df.set_index('film_id')[['dstbtr', 'rel_at']]
        .rename(columns={'dstbtr': 'evt_dstbtr', 'rel_at': 'evt_rel_at'})
        .to_dict('index')
    )

    tasks     = load_tasks_from_yaml(FILM_META_PROMPTS_PATH)
    model_cfg = MODELS.get(os.environ.get('FILM_META_MODEL', 'gpt-5.4-mini'), {})
    extractor = FilmMetaExtractor(
        task=tasks['film_meta'],
        api_key=os.getenv('OPENAI_KEY'),
        cost_per_1m_input=model_cfg.get('cost_per_1m_input'),
        cost_per_1m_output=model_cfg.get('cost_per_1m_output'),
        # Film budgets come from a diverse set (The Numbers, Box Office Mojo,
        # Variety, Bollywood trades) — ~45% of citations are outside wiki+imdb.
        # Override the env-level allow-list to None so film_meta can escalate
        # beyond wiki/imdb. The prompt nudges it to TRY wiki/imdb first.
        web_search_domains=None,
    )

    chunks = [df.iloc[i:i + META_BATCH_SIZE] for i in range(0, len(df), META_BATCH_SIZE)]
    prev_cost = 0.0
    for batch_num, chunk in enumerate(chunks, 1):
        log.info(f"film_meta batch {batch_num}/{len(chunks)}  ({len(chunk)} films)")
        results = await extractor.arun(
            df=chunk,
            id_col='film_id',
            title_col='film_title',
            rel_at_col='rel_at',
            director_col='director',
            synopsis_col='synopsis',
            max_concurrency=META_MAX_CONCURRENCY,
        )

        title_lookup = chunk.set_index('film_id')['film_title'].to_dict()
        batch_errors: dict[str, dict] = {}
        batch_success_ids: set[str] = set()
        for film_id, data in results.items():
            if not data.get('_error'):
                data['film_id'] = film_id
                pt = evt_passthrough.get(film_id, {})
                data['evt_dstbtr'] = str(pt.get('evt_dstbtr')) if pd.notna(pt.get('evt_dstbtr')) else None
                data['evt_rel_at'] = str(pt.get('evt_rel_at')) if pd.notna(pt.get('evt_rel_at')) else None
                checkpoint[str(film_id)] = data
                batch_success_ids.add(str(film_id))
            else:
                batch_errors[str(film_id)] = {
                    'film_id':     film_id,
                    'film_title':  title_lookup.get(film_id),
                    '_error':      data.get('_error'),
                    '_raw_output': data.get('_raw_output'),
                }

        with open(FILM_META_CHECKPOINT_PATH, 'w') as f:
            json.dump(checkpoint, f, default=str)
        log.info(f"  Checkpoint saved: {len(checkpoint)} films → {FILM_META_CHECKPOINT_PATH}")

        # Errors JSON: merge in new errors AND drop entries for films that just
        # succeeded (covers both this batch's wins and stale entries from prior
        # runs that have since recovered).
        if batch_errors or (batch_success_ids and FILM_META_ERRORS_PATH.exists()):
            existing_errors: dict = {}
            if FILM_META_ERRORS_PATH.exists():
                with open(FILM_META_ERRORS_PATH) as f:
                    existing_errors = json.load(f)
            purged_n = sum(1 for fid in batch_success_ids
                           if existing_errors.pop(fid, None) is not None)
            existing_errors.update(batch_errors)
            if purged_n or batch_errors:
                with open(FILM_META_ERRORS_PATH, 'w') as f:
                    json.dump(existing_errors, f, default=str, indent=2)
            err_counts: dict[str, int] = {}
            for v in batch_errors.values():
                key = str(v.get('_error', 'unknown')).split(':')[0][:40]
                err_counts[key] = err_counts.get(key, 0) + 1
            msg = f"  Errors this batch: {len(batch_errors)}"
            if err_counts:
                msg += f" ({', '.join(f'{k}={n}' for k, n in err_counts.items())})"
            if purged_n:
                msg += f"  [purged {purged_n} now-recovered]"
            msg += f" → {FILM_META_ERRORS_PATH}"
            log.info(msg)

        if extractor.token_usage:
            curr  = extractor.token_usage.get('cost_usd', 0.0)
            delta = curr - prev_cost
            prev_cost = curr
            log.info(f"  Batch cost: ${delta:.4f}  |  Run total: ${curr:.4f}")

        if batch_num < len(chunks):
            await asyncio.sleep(META_BATCH_PAUSE_SECS)

    df_new = pd.DataFrame(checkpoint.values())
    if '_error' in df_new.columns:
        df_new = df_new[df_new['_error'].isna()].drop(
            columns=[c for c in ['_error', '_raw_output'] if c in df_new.columns],
            errors='ignore',
        )

    FILM_META_ENRICHED_PATH.parent.mkdir(parents=True, exist_ok=True)
    if FILM_META_ENRICHED_PATH.exists():
        existing = pd.read_parquet(FILM_META_ENRICHED_PATH)
        out = (pd.concat([existing, df_new], ignore_index=True)
               .drop_duplicates(subset='film_id', keep='last'))
    else:
        out = df_new
    out.to_parquet(FILM_META_ENRICHED_PATH, engine='pyarrow', index=False)
    log.info(f"film_meta parquet → {FILM_META_ENRICHED_PATH}  ({len(out)} films)")

    if extractor.token_usage:
        u = extractor.token_usage
        log.info(f"film_meta tokens — prompt: {u['prompt_tokens']:,}  "
                 f"completion: {u['completion_tokens']:,}  "
                 f"searches: {u.get('search_calls', 0):,}  "
                 f"cost: ${u.get('cost_usd', 0):.4f}")
    del extractor
    gc.collect()


# ── Public refresh functions (one per Dagster asset) ──────────────────────────

def refresh_synopsis(df_films: pd.DataFrame | None = None, force: bool = False) -> dict:
    df_films = _ensure_films(df_films)
    if df_films is None:
        return {'path': 'synopsis', 'updated': False, 'reason': 'snowflake_unavailable'}

    to_extract = df_films if force else _diff_synopsis_films(df_films)
    if to_extract.empty:
        return {'path': 'synopsis', 'updated': False, 'reason': 'up_to_date'}

    asyncio.run(_extract_synopses(to_extract))
    try:
        sync_synopses_sources(SYNOPSES_EXTRACTED_PATH)
    except Exception as e:
        log.warning(f"Snowflake sync failed (non-fatal): {e}")
    return {'path': 'synopsis', 'updated': True, 'films_extracted': len(to_extract)}


def refresh_cast(df_films: pd.DataFrame | None = None, force: bool = False) -> dict:
    df_films = _ensure_films(df_films)
    if df_films is None:
        return {'path': 'cast', 'updated': False, 'reason': 'snowflake_unavailable'}

    if force:
        actors: set[str] = set()
        for val in df_films.get('actor_list', pd.Series(dtype=str)).dropna():
            for a in str(val).split('|'):
                a = _clean_actor(a)
                if a:
                    actors.add(a)
        new_actors = sorted(actors)
        log.info(f"force=True — re-enriching all {len(new_actors)} actors")
    else:
        new_actors = _diff_actors(df_films)

    if not new_actors:
        return {'path': 'cast', 'updated': False, 'reason': 'up_to_date'}

    asyncio.run(_enrich_cast(new_actors))
    return {'path': 'cast', 'updated': True, 'actors_extracted': len(new_actors)}


def refresh_directors(df_films: pd.DataFrame | None = None, force: bool = False) -> dict:
    df_films = _ensure_films(df_films)
    if df_films is None:
        return {'path': 'director', 'updated': False, 'reason': 'snowflake_unavailable'}

    if force:
        dirs: set[str] = set()
        for val in df_films.get('director', pd.Series(dtype=str)).dropna():
            for d in re.split(r'[|,]', str(val)):
                d = d.strip()
                if d:
                    dirs.add(d)
        new_dirs = sorted(dirs)
        log.info(f"force=True — re-enriching all {len(new_dirs)} directors")
    else:
        new_dirs = _diff_directors(df_films)

    if not new_dirs:
        return {'path': 'director', 'updated': False, 'reason': 'up_to_date'}

    asyncio.run(_enrich_directors(new_dirs))
    return {'path': 'director', 'updated': True, 'directors_extracted': len(new_dirs)}


def refresh_film_meta(df_films: pd.DataFrame | None = None, force: bool = False) -> dict:
    df_films = _ensure_films(df_films)
    if df_films is None:
        return {'path': 'film_meta', 'updated': False, 'reason': 'snowflake_unavailable'}

    if force:
        df = df_films.copy()
        if 'dstbtr' in df.columns:
            df = df[~df['dstbtr'].isin(FILM_META_SKIP_DISTRIBUTORS)]
    else:
        df = _diff_film_meta(df_films)

    if df.empty:
        return {'path': 'film_meta', 'updated': False, 'reason': 'up_to_date'}

    # film_lookup for the re-release filter — pulls the FULL Snowflake catalogue,
    # not just the curated work-set, so older releases outside the parquet
    # snapshot window are still available for title matching.
    film_lookup = load_full_film_catalogue()

    asyncio.run(_enrich_film_meta(df, film_lookup))
    return {'path': 'film_meta', 'updated': True, 'films_extracted': len(df)}


# ── Convenience: run all four (CLI / backwards compat) ────────────────────────

def run_refresh(
    force_synopsis: bool = False,
    force_cast: bool = False,
    force_director: bool = False,
    force_film_meta: bool = False,
    only: list[str] | None = None,
) -> dict:
    df_films = load_films_from_snowflake()
    if df_films is None:
        log.error("Snowflake unavailable — aborting")
        return {'error': 'snowflake_unavailable'}

    out = {'run_date': datetime.datetime.today().strftime('%Y%m%d'), 'paths': {}}
    selected = set(only) if only else {'synopsis', 'cast', 'director', 'film_meta'}

    if 'synopsis' in selected:
        out['paths']['synopsis']  = refresh_synopsis(df_films,  force=force_synopsis)
    if 'cast' in selected:
        out['paths']['cast']      = refresh_cast(df_films,      force=force_cast)
    if 'director' in selected:
        out['paths']['director']  = refresh_directors(df_films, force=force_director)
    if 'film_meta' in selected:
        out['paths']['film_meta'] = refresh_film_meta(df_films, force=force_film_meta)

    log.info(f"Refresh complete: {out}")
    return out


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='film_synopsis_meta refresh')
    parser.add_argument('--only', nargs='+',
                        choices=['synopsis', 'cast', 'director', 'film_meta'],
                        help='Restrict to a subset of paths')
    parser.add_argument('--force-synopsis',  action='store_true')
    parser.add_argument('--force-cast',      action='store_true')
    parser.add_argument('--force-director',  action='store_true')
    parser.add_argument('--force-film-meta', action='store_true')
    args = parser.parse_args()

    run_refresh(
        force_synopsis=args.force_synopsis,
        force_cast=args.force_cast,
        force_director=args.force_director,
        force_film_meta=args.force_film_meta,
        only=args.only,
    )
