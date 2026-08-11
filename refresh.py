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

import nest_asyncio
nest_asyncio.apply()

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
import litellm
litellm.success_callback = []
litellm.failure_callback = []
logging.getLogger("LiteLLM").setLevel(logging.ERROR)

from config import (
    DATA_DIR,
    SYNOPSES_EXTRACTED_PATH, CAST_ENRICHED_PATH,
    DIRECTOR_ENRICHED_PATH, FILM_META_ENRICHED_PATH, FILM_ID_VARIANTS_PATH,
    SF_WAREHOUSE, SF_DATABASE, SF_SCHEMA, SF_RSA_KEY,
)
from cleanup_film_meta import clean_film_meta_df
from extractor import LlmJsonExtractor
from film_meta_extractor import FilmMetaExtractor, ActorMetaExtractor, DirectorMetaExtractor
from film_variant_merge import filter_variants
from load_prompts import load_tasks_from_yaml
from models import DEFAULT_MODEL, DEFAULT_FALLBACKS, MODELS
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

# An unknown-fame_tier actor is only worth retrying if some film they're in
# still needs them — i.e. that film's already-resolved cast count is below
# this. If every film they appear in already has this many known co-stars,
# retrying them is low value (billing-order/lead-actor features already have
# enough signal) and just burns web_search calls that mostly won't resolve
# anyway. See the session's cast-backlog investigation for the data behind
# this: at 3, ~72% of the unknown backlog is skippable with zero films
# dropping below 3 known cast members.
MIN_KNOWN_CAST_FOR_RETRY = 3

SYNOPSIS_CHECKPOINT_PATH  = DATA_DIR / 'meta_data' / 'synopsis_v2'   / 'synopsis_progress.json'
FILM_META_CHECKPOINT_PATH = DATA_DIR / 'meta_data' / 'film_meta'     / 'film_meta_progress.json'
FILM_META_ERRORS_PATH     = DATA_DIR / 'meta_data' / 'film_meta'     / 'film_meta_errors.json'
CAST_CHECKPOINT_PATH      = DATA_DIR / 'meta_data' / 'cast_meta'     / 'cast_progress.json'
CAST_ERRORS_PATH          = DATA_DIR / 'meta_data' / 'cast_meta'     / 'cast_errors.json'
DIRECTOR_CHECKPOINT_PATH  = DATA_DIR / 'meta_data' / 'director_meta' / 'director_progress.json'
DIRECTOR_ERRORS_PATH      = DATA_DIR / 'meta_data' / 'director_meta' / 'director_errors.json'

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

# Placeholder tokens Vista sometimes uses in actor_list instead of a real name
# — e.g. CatVideoFest's "VARIOUS CATS FROM THE INTERWEB!", or generic "VARIOUS
# ACTORS" for compilation/tour films. These will never resolve to a fame_tier
# no matter how many times they're retried, so they're dropped at the source.
_PLACEHOLDER_ACTOR_RE = re.compile(r'^VARIOUS\b', re.IGNORECASE)


def _clean_actor(raw: str) -> str:
    name = raw.strip().upper()
    name = _AND_PREFIX.sub('', name).strip()
    if _PLACEHOLDER_ACTOR_RE.match(name):
        return ''
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
        snow = pd.read_sql(films_sql.SQL_FILM_DETAILS, sb.engine)[
            ['film_id', 'film_title', 'ihub_synopsis', 'vista_synopsis']
        ]
        snow['film_id'] = snow['film_id'].astype(int)
        df = df.merge(snow, on='film_id', how='left')

        # IHUB_SYNOPSIS is primary (matches the pre-split COALESCE(m.SYNOPSIS,
        # f.FILM_DESC) behaviour); VISTA_SYNOPSIS becomes alt_synopsis whenever
        # it actually differs, so both sources feed the extractor and both get
        # diffed for changes on the next run (see _diff_synopsis_films).
        ihub  = df['ihub_synopsis'].fillna('').str.strip()
        vista = df['vista_synopsis'].fillna('').str.strip()
        df['synopsis'] = df['ihub_synopsis'].where(ihub != '', df['vista_synopsis'])
        df['alt_synopsis'] = df['vista_synopsis'].where(vista != ihub, None)
        df = df.drop(columns=['ihub_synopsis', 'vista_synopsis'])
        log.info("Film titles + synopsis (IHUB/VISTA) joined from Snowflake")
    except Exception as e:
        log.warning(f"Snowflake unavailable ({e}) — using film_id as title fallback, "
                     f"keeping parquet-sourced synopsis (no alt_synopsis)")
        df['film_title'] = df['film_id'].astype(str)

    return df


def load_full_film_catalogue() -> pd.DataFrame | None:
    """Full film catalogue for the re-release filter — reads film_lookup.parquet
    (same source as rematch_comscore.py::load_evt_films) so no Snowflake connection
    is required."""
    path = DATA_DIR / "look_ups" / "film_lookup.parquet"
    try:
        full = pd.read_parquet(path, columns=["film_id", "film", "rel_at", "dstbtr", "director"])
        full = full[full["film"].notna()].reset_index(drop=True)
        full["film_id"] = full["film_id"].astype(int)
        full["rel_at"]  = pd.to_datetime(full["rel_at"], utc=True, errors="coerce")
        log.info(f"Full film catalogue loaded from parquet: {len(full)} films")
        return full
    except FileNotFoundError:
        log.warning(f"film_lookup.parquet not found at {path} — re-release filter will be skipped")
        return None


def _ensure_films(df_films: pd.DataFrame | None) -> pd.DataFrame | None:
    return df_films if df_films is not None else load_films_from_snowflake()


# ── Diff helpers ──────────────────────────────────────────────────────────────

def _diff_synopsis_films(df_films: pd.DataFrame) -> pd.DataFrame:
    """Films that are new OR whose synopsis (IHUB_SYNOPSIS) OR alt_synopsis
    (VISTA_SYNOPSIS) text has changed — either source changing is enough to
    trigger re-extraction, not just the primary one, since Vista's own
    booking description can get corrected independently of IHUB's."""
    if not SYNOPSES_EXTRACTED_PATH.exists():
        log.info("No existing synopsis parquet — all films are new")
        return df_films

    existing = pd.read_parquet(SYNOPSES_EXTRACTED_PATH)
    existing['film_id'] = existing['film_id'].astype(int)
    has_alt_old = 'alt_synopsis' in existing.columns
    has_alt_new = 'alt_synopsis' in df_films.columns
    existing_cols = ['film_id', 'synopsis'] + (['alt_synopsis'] if has_alt_old else [])
    existing = existing[existing_cols]
    existing_ids = set(existing['film_id'])

    new = df_films[~df_films['film_id'].isin(existing_ids)]

    rename_map = {'synopsis': 'synopsis_old'}
    if has_alt_old:
        rename_map['alt_synopsis'] = 'alt_synopsis_old'
    merged = df_films[df_films['film_id'].isin(existing_ids)].merge(
        existing.rename(columns=rename_map),
        on='film_id', how='left',
    )
    synopsis_changed = merged['synopsis'].fillna('') != merged['synopsis_old'].fillna('')
    if has_alt_new and has_alt_old:
        alt_changed = merged['alt_synopsis'].fillna('') != merged['alt_synopsis_old'].fillna('')
    elif has_alt_new:
        # alt_synopsis wasn't tracked before this run — treat any non-empty
        # alt text as new/changed so it gets picked up once going forward.
        alt_changed = merged['alt_synopsis'].fillna('') != ''
    else:
        alt_changed = pd.Series(False, index=merged.index)

    changed = merged[synopsis_changed | alt_changed]
    result = pd.concat([new, changed[df_films.columns]], ignore_index=True)
    log.info(f"Synopsis diff: {len(new)} new + {len(changed)} updated "
             f"(synopsis or alt_synopsis changed) → {len(result)} to extract")
    return result


def _diff_actors(df_films: pd.DataFrame) -> list[str]:
    """Actors not yet in cast_enriched.parquet OR cast_progress.json checkpoint,
    PLUS already-done actors whose fame_tier came back "unknown" — but only if
    some film they're in still needs them (see MIN_KNOWN_CAST_FOR_RETRY): if
    every film they appear in already has enough resolved co-stars, retrying
    them is low value and skipped, even though they'd otherwise qualify for
    retry forever (see CLAUDE.md's ~70% hit-rate note for why unknowns don't
    all resolve no matter how many times they're retried).

    Films from FILM_META_SKIP_DISTRIBUTORS (concerts/festivals/sports/event
    cinema) are excluded before collecting actors — their "cast" is usually
    band members or event participants, not film actors, so asking for a
    fame_tier is a category error that will never resolve no matter how many
    times it's retried (see e.g. Trafalgar Releasing concert films).

    Reading both parquet and checkpoint means a partially-completed run
    (checkpoint written, parquet not yet flushed) doesn't get re-extracted
    from scratch.
    """
    films = df_films
    if 'dstbtr' in films.columns:
        films = films[~films['dstbtr'].isin(FILM_META_SKIP_DISTRIBUTORS)]

    all_actors: set[str] = set()
    for val in films.get('actor_list', pd.Series(dtype=str)).dropna():
        for a in str(val).split('|'):
            a = _clean_actor(a)
            if a:
                all_actors.add(a)

    done: set[str] = set()
    retry: set[str] = set()
    if CAST_ENRICHED_PATH.exists():
        cast_df = pd.read_parquet(CAST_ENRICHED_PATH, columns=['actor_name', 'fame_tier'])
        names = cast_df['actor_name'].astype(str).str.upper().str.strip()
        done |= set(names)
        retry |= set(names[cast_df['fame_tier'].astype(str).str.strip().str.lower() == 'unknown'])
    if CAST_CHECKPOINT_PATH.exists():
        with open(CAST_CHECKPOINT_PATH) as f:
            checkpoint = json.load(f)
        for k, v in checkpoint.items():
            name = str(k).upper().strip()
            done.add(name)
            if str(v.get('fame_tier', '')).strip().lower() == 'unknown':
                retry.add(name)

    resolved = done - retry
    retry_candidates = all_actors & retry

    # Only retry a candidate if some film they're in has fewer than
    # MIN_KNOWN_CAST_FOR_RETRY already-resolved co-stars.
    min_known_for_actor: dict[str, int] = {}
    if retry_candidates:
        for val in films.get('actor_list', pd.Series(dtype=str)).dropna():
            names = [_clean_actor(a) for a in str(val).split('|')]
            names = [n for n in names if n]
            if not names:
                continue
            n_known = sum(1 for n in names if n in resolved)
            for n in names:
                if n in retry_candidates and n_known < min_known_for_actor.get(n, n_known + 1):
                    min_known_for_actor[n] = n_known

    to_retry = {a for a in retry_candidates if min_known_for_actor.get(a, 0) < MIN_KNOWN_CAST_FOR_RETRY}
    skipped_low_value = retry_candidates - to_retry

    new = sorted((all_actors - done) | to_retry)
    log.info(f"Cast diff: {len(all_actors - done)} new + {len(to_retry)} unknown-retry actors "
             f"({len(skipped_low_value)} unknown actors skipped — every film they're in already has "
             f">={MIN_KNOWN_CAST_FOR_RETRY} known co-stars) "
             f"({len(done)} already done across parquet+checkpoint)")
    return new


def _diff_directors(df_films: pd.DataFrame) -> list[str]:
    """Directors not yet in director_enriched.parquet OR director_progress.json,
    PLUS already-done directors whose director_tier came back "unknown"
    (retried every run — see CLAUDE.md's ~70% hit-rate note; mirrors
    _diff_actors' unknown-retry logic).

    Films from FILM_META_SKIP_DISTRIBUTORS (concerts/festivals/sports/event
    cinema) are excluded first — see _diff_actors' docstring for why.
    """
    films = df_films
    if 'dstbtr' in films.columns:
        films = films[~films['dstbtr'].isin(FILM_META_SKIP_DISTRIBUTORS)]

    all_dirs: set[str] = set()
    for val in films.get('director', pd.Series(dtype=str)).dropna():
        # Snowflake pipes; raw parquets sometimes comma — handle both.
        parts = re.split(r'[|,]', str(val))
        for d in parts:
            d = d.strip()
            if d:
                all_dirs.add(d)

    done: set[str] = set()
    retry: set[str] = set()
    if DIRECTOR_ENRICHED_PATH.exists():
        dir_df = pd.read_parquet(DIRECTOR_ENRICHED_PATH, columns=['director_name', 'director_tier'])
        names = dir_df['director_name'].astype(str).str.strip()
        done |= set(names)
        retry |= set(names[dir_df['director_tier'].astype(str).str.strip().str.lower() == 'unknown'])
    if DIRECTOR_CHECKPOINT_PATH.exists():
        with open(DIRECTOR_CHECKPOINT_PATH) as f:
            checkpoint = json.load(f)
        for k, v in checkpoint.items():
            name = str(k).strip()
            done.add(name)
            if str(v.get('director_tier', '')).strip().lower() == 'unknown':
                retry.add(name)

    to_retry = all_dirs & retry
    new = sorted((all_dirs - done) | to_retry)
    log.info(f"Director diff: {len(all_dirs - done)} new + {len(to_retry)} unknown-retry directors "
             f"({len(done)} already done across parquet+checkpoint)")
    return new


def _persist_variant_map(variant_map: pd.DataFrame) -> None:
    """Merge freshly detected film_id variants into FILM_ID_VARIANTS_PATH.
    keep='last' so a re-run with better data (richer synopsis/cast) can
    flip which side is canonical without a stale row surviving."""
    if variant_map.empty:
        return
    # Force to string — this file is written by two different code paths
    # (this one and cleanup_film_meta.py's format-variant merge) and a raw
    # Timestamp mixed with a string in the same parquet column breaks
    # pyarrow on write. Normalizing here too (on top of the fix at the
    # source) means a future third writer can't reintroduce the same bug.
    for col in ('variant_rel_at', 'canonical_rel_at'):
        if col in variant_map.columns:
            variant_map[col] = variant_map[col].apply(lambda v: str(v) if pd.notna(v) else None)

    FILM_ID_VARIANTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    if FILM_ID_VARIANTS_PATH.exists():
        existing = pd.read_parquet(FILM_ID_VARIANTS_PATH)
        for col in ('variant_rel_at', 'canonical_rel_at'):
            if col in existing.columns:
                existing[col] = existing[col].apply(lambda v: str(v) if pd.notna(v) else None)
        out = (pd.concat([existing, variant_map], ignore_index=True)
               .drop_duplicates(subset='film_id', keep='last'))
    else:
        out = variant_map
    out.to_parquet(FILM_ID_VARIANTS_PATH, index=False)
    log.info(f"film_meta variants: {len(variant_map)} film_id(s) mapped onto a canonical "
             f"release this run ({len(out)} total) → {FILM_ID_VARIANTS_PATH}")


def _apply_variant_merge(df: pd.DataFrame, film_lookup: pd.DataFrame | None) -> pd.DataFrame:
    """Drops genuine re-releases (keyword/year/language title match, no specific
    pairing) and duplicate-booking variants (fuzzy-matched to another film_id
    that already represents the same release — see film_variant_merge.py),
    keeping only the canonical film_id. Persists the variant crosswalk so a
    variant film_id can be resolved back to whichever film_id actually holds
    the film_meta data.
    """
    if film_lookup is None or 'rel_at' not in film_lookup.columns:
        return df
    try:
        df_filtered, variant_map = filter_variants(df, film_lookup)
    except Exception as e:
        log.warning(f"Variant merge skipped ({e})")
        return df

    n_dropped = len(df) - len(df_filtered)
    if n_dropped:
        log.info(f"film_meta re-releases/variants filtered: -{n_dropped} "
                 f"({len(variant_map)} duplicate-booking, {n_dropped - len(variant_map)} keyword/year/language) "
                 f"→ {len(df_filtered)}")
    _persist_variant_map(variant_map)
    return df_filtered


def _diff_film_meta(df_films: pd.DataFrame, film_lookup: pd.DataFrame | None) -> pd.DataFrame:
    """Films not yet in checkpoint JSON or enriched parquet, with skip-distributor
    filter and re-release/variant merge applied.

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

    df = _apply_variant_merge(df, film_lookup)

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

def _sync_synopsis_checkpoint(df: pd.DataFrame) -> None:
    """Regenerate synopsis_progress.json from the just-written parquet.

    refresh.py's synopsis path never writes this checkpoint mid-run (see
    _extract_synopses — it only flushes once, at the end), but main.py's own
    synopsis path checks it (existence only, no diffing) to decide what's
    already done. Without this, a film refresh.py just extracted would look
    "not done" to main.py and get re-extracted from scratch — see CLAUDE.md's
    "refresh.py and main.py must agree on the work-set" note.
    """
    if df.empty or 'film_id' not in df.columns:
        return

    def _to_json_native(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        return None if pd.isna(value) else value

    checkpoint = {
        str(row['film_id']): {k: _to_json_native(v) for k, v in row.items()}
        for _, row in df.iterrows()
    }
    SYNOPSIS_CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SYNOPSIS_CHECKPOINT_PATH, 'w') as f:
        json.dump(checkpoint, f, default=str)
    log.info(f"Synopsis checkpoint synced from parquet → {SYNOPSIS_CHECKPOINT_PATH} ({len(checkpoint)} films)")


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

    # alt_synopsis (VISTA_SYNOPSIS) is only ever used as extractor prompt
    # context, never returned by the extraction tasks themselves — persist it
    # explicitly so _diff_synopsis_films can detect a VISTA-only change on a
    # future run even when the primary (IHUB) synopsis stays the same.
    if 'alt_synopsis' in df.columns and 'film_id' in df_new.columns:
        alt_lookup = df.set_index('film_id')['alt_synopsis']
        df_new['alt_synopsis'] = df_new['film_id'].map(alt_lookup)

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
    _sync_synopsis_checkpoint(out)

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
        # keep='last' so a freshly retried actor's new result (df_new) overrides
        # the stale row in existing — keep='first' would silently discard every
        # unknown-retry update at flush time.
        out = (pd.concat([existing, df_new], ignore_index=True)
               .drop_duplicates(subset='actor_name', keep='last'))
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
        # keep='last' so a freshly retried director's new result (df_new) overrides
        # the stale row in existing — keep='first' would silently discard every
        # unknown-retry update at flush time.
        out = (pd.concat([existing, df_new], ignore_index=True)
               .drop_duplicates(subset='director_name', keep='last'))
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
    interruptible — every BATCH_SIZE films, progress is persisted to disk.

    `df` is expected to already have re-releases and duplicate-booking variants
    filtered out by the caller (_diff_film_meta / refresh_film_meta's force
    branch, via _apply_variant_merge) — this function no longer applies that
    filter itself.
    """
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

    # Genre normalize/consolidate/rarity-filter + format-variant dedup on the
    # WHOLE accumulated corpus (existing + this batch) — cheap at this scale
    # (~4-5k rows) and keeps every run internally consistent rather than only
    # cleaning the newly extracted rows. See cleanup_film_meta.py.
    out, out_variant_map, out_stats = clean_film_meta_df(out)
    if out_stats['n_genre_changed'] or out_stats['n_variants_dropped']:
        log.info(f"film_meta cleanup — {out_stats['n_genre_changed']} rows genre-normalized, "
                 f"{out_stats['n_variants_dropped']} format-variant duplicates merged away")
    _persist_variant_map(out_variant_map)

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
    return {'path': 'synopsis', 'updated': True, 'films_extracted': len(to_extract)}


def refresh_cast(df_films: pd.DataFrame | None = None, force: bool = False) -> dict:
    df_films = _ensure_films(df_films)
    if df_films is None:
        return {'path': 'cast', 'updated': False, 'reason': 'snowflake_unavailable'}

    if force:
        films = df_films
        if 'dstbtr' in films.columns:
            films = films[~films['dstbtr'].isin(FILM_META_SKIP_DISTRIBUTORS)]
        actors: set[str] = set()
        for val in films.get('actor_list', pd.Series(dtype=str)).dropna():
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
        films = df_films
        if 'dstbtr' in films.columns:
            films = films[~films['dstbtr'].isin(FILM_META_SKIP_DISTRIBUTORS)]
        dirs: set[str] = set()
        for val in films.get('director', pd.Series(dtype=str)).dropna():
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

    # film_lookup for the re-release/variant filter — pulls the FULL Snowflake
    # catalogue, not just the curated work-set, so older releases outside the
    # parquet snapshot window are still available for title matching.
    film_lookup = load_full_film_catalogue()

    if force:
        df = df_films.copy()
        if 'dstbtr' in df.columns:
            df = df[~df['dstbtr'].isin(FILM_META_SKIP_DISTRIBUTORS)]
        df = _apply_variant_merge(df, film_lookup)
    else:
        df = _diff_film_meta(df_films, film_lookup)

    if df.empty:
        return {'path': 'film_meta', 'updated': False, 'reason': 'up_to_date'}

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
