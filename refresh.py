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
import s3_checkpoint
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

# Lookback window for load_films_from_snowflake()'s live catalogue pull.
# SQL_FILM_DETAILS itself has no date filter (see films/sql.py) — this is a
# Python-side restriction, same pattern as rematch_gower.py::GOWER_MIN_REL_DATE.
# Added after a live run surfaced 15,263 total films vs. the ~5,032 the old
# parquet-snapshot-based work-set ever covered — the extra ~10,231 turned out
# to be mostly genuine 2008-2017 releases the old system never captured.
# 2018-01-01 is not a guess: it's the box office model's actual live training
# window start — cinema_admits_models/helper_fucntions.py::
# return_train_calib_test_dates(train_start=datetime(2018, 1, 1), ...) is the
# real default driving db_merge_20260420.sql/bo_pred_build.sql's date params.
# Films released before this are outside what the model trains on at all, so
# there's no point paying web_search prices to extract their metadata. Set to
# None to disable (pull the full historical catalogue back to 1935).
WORK_SET_MIN_REL_DATE: pd.Timestamp | None = pd.Timestamp("2018-01-01", tz="UTC")

# Local paths — no longer written by refresh.py itself (see s3_checkpoint.py;
# all four extraction paths now read/write S3 directly, not local disk). Kept
# defined and importable for main.py's own local-disk checkpoint logic and
# diagnostics/post_refresh_check.py's local checks, which are unaffected by
# this module's own I/O. The one exception is SYNOPSIS_CHECKPOINT_PATH, which
# _sync_synopsis_checkpoint still writes locally on purpose — see its docstring.
SYNOPSIS_CHECKPOINT_PATH  = DATA_DIR / 'meta_data' / 'synopsis_v2'   / 'synopsis_progress.json'
FILM_META_CHECKPOINT_PATH = DATA_DIR / 'meta_data' / 'film_meta'     / 'film_meta_progress.json'
FILM_META_ERRORS_PATH     = DATA_DIR / 'meta_data' / 'film_meta'     / 'film_meta_errors.json'
CAST_CHECKPOINT_PATH      = DATA_DIR / 'meta_data' / 'cast_meta'     / 'cast_progress.json'
CAST_ERRORS_PATH          = DATA_DIR / 'meta_data' / 'cast_meta'     / 'cast_errors.json'
DIRECTOR_CHECKPOINT_PATH  = DATA_DIR / 'meta_data' / 'director_meta' / 'director_progress.json'
DIRECTOR_ERRORS_PATH      = DATA_DIR / 'meta_data' / 'director_meta' / 'director_errors.json'

# S3 checkpoint names/filenames — passed to s3_checkpoint.py's load_checkpoint/
# append_checkpoint/read_parquet/write_parquet. Names match s3_sync.py's old
# SYNC_SPECS folder naming (synopsis_v2 locally is "synopsis" on the S3 side).
SYNOPSIS_S3_NAME    = 'synopsis'
CAST_S3_NAME        = 'cast_meta'
DIRECTOR_S3_NAME    = 'director_meta'
FILM_META_S3_NAME   = 'film_meta'
SYNOPSIS_FILENAME    = 'synopses_extracted.parquet'
CAST_FILENAME        = 'cast_enriched.parquet'
DIRECTOR_FILENAME    = 'director_enriched.parquet'
FILM_META_FILENAME   = 'film_meta_enriched.parquet'
FILM_ID_VARIANTS_FILENAME = 'film_id_variants.parquet'

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
    # NZ festival equivalents — found missing 2026-08-12 while investigating why
    # the live-Snowflake work-set's "new films" backlog was so large: these
    # accounted for ~760 of it on their own (synopsis diff wasn't even applying
    # this list at all until the same investigation — see _diff_synopsis_films).
    "NZ Italian Film Festival", "NZ French Film Festival",
    "NZ NEW ZEALAND INT FILM FESTIVAL", "ZZ International Film Festival NZ",
    "NZ RESENE ARCHITECTURE AND DESIGN FF", "NZ British Film Festival NZ",
    "ZZ SHOW ME SHORTS FILM FESTIVAL", "ZZ Veterans Film Festival",
    "ZZ GREEK FESTIVAL OF SYDNEY",
}

# Placeholder/no-content synopsis values — built from a value_counts scan of
# the live work-set (2026-08-12), not guessed. "plot unknown" alone accounts
# for ~90 films; language names leaking into the synopsis field instead of
# describing plot is the same bug pattern cleanup_film_meta.py documents for
# the genre field. A blanket length cutoff would ALSO wrongly drop legitimate
# terse synopses ("Remake of Train to Busan." is 25 chars and real) — this is
# a denylist of specific known-junk values instead, not a length threshold.
_SYNOPSIS_PLACEHOLDER_VALUES = {
    'testing code', 'tba', 'n/a', 'none', 'unknown', 'coming soon',
    'no synopsis available', 'synopsis not available', 'gaming booking',
    'telugu', 'tamil', 'hindi', 'hindi language', 'telugu version',
    'kannada', 'malayalam', 'punjabi', 'marathi', 'bengali', 'urdu',
    'japanese', 'korean', 'mandarin', 'cantonese',
}
_BRACKETED_YEAR_RE = re.compile(r'^\[\d{4}\]$')


def _is_placeholder_text(series: pd.Series) -> pd.Series:
    norm = series.fillna('').astype(str).str.strip().str.lower().str.strip('*').str.strip()
    return (
        (norm == '')
        | norm.str.contains('plot unknown', regex=False)
        | norm.str.contains('plot is unknown', regex=False)
        | norm.isin(_SYNOPSIS_PLACEHOLDER_VALUES)
        | norm.str.match(_BRACKETED_YEAR_RE)
    )


def _drop_no_usable_synopsis(df: pd.DataFrame) -> pd.DataFrame:
    """Drops films where NEITHER synopsis nor alt_synopsis has usable content
    — alt_synopsis (Vista's own booking description) often has real text even
    when the primary (IHUB) synopsis is a "Plot unknown" placeholder (true for
    about half of the "plot unknown" rows observed 2026-08-12), so a film is
    only dropped if both are placeholder/empty, not just the primary one."""
    if 'synopsis' not in df.columns:
        return df
    title_echo = (
        df['synopsis'].fillna('').astype(str).str.strip().str.lower()
        == df.get('film_title', pd.Series('', index=df.index)).fillna('').astype(str).str.strip().str.lower()
    )
    syn_bad = _is_placeholder_text(df['synopsis']) | title_echo
    alt_bad = _is_placeholder_text(df['alt_synopsis']) if 'alt_synopsis' in df.columns else pd.Series(True, index=df.index)
    no_usable = syn_bad & alt_bad

    n0 = len(df)
    df = df[~no_usable]
    if n0 - len(df):
        log.info(f"synopsis placeholder/no-content filtered: -{n0 - len(df)} → {len(df)}")
    return df

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

def _snowflake_array_to_pipe_string(val):
    """SQL_FILM_DETAILS' ACTOR_LIST/DIRECTOR_LIST come back as Snowflake
    ARRAYs (built via STRTOK_TO_ARRAY) — downstream code (_clean_actor,
    _diff_directors, etc.) expects pipe-delimited strings, matching the
    format the old parquet snapshots used. Handles a few possible wire
    formats defensively since this hasn't been exercised against a live
    Snowflake connection: a real list/tuple/ndarray, a JSON-encoded string
    (some connector versions serialize ARRAY columns this way), or already
    a plain string/None."""
    if val is None:
        return None
    if isinstance(val, str):
        stripped = val.strip()
        if stripped.startswith('['):
            try:
                items = json.loads(stripped)
                return '|'.join(str(v) for v in items) if items else None
            except (ValueError, TypeError):
                return val
        return val or None
    if isinstance(val, (list, tuple, np.ndarray)):
        items = list(val)
        return '|'.join(str(v) for v in items) if items else None
    return val


def load_films_from_snowflake() -> pd.DataFrame | None:
    """Returns the curated film work-set used by all four extraction paths.

    Pulls directly from Snowflake via films/sql.py::SQL_FILM_DETAILS — that
    query has no lookback restriction of its own (its only filter is
    `FILM_NAT_OPEN_DATE IS NOT NULL`, going back to 1935 in practice), so the
    raw pull is much BIGGER than the old snapshot-glob approach ever
    accumulated, not just "at least as much" as originally assumed here —
    confirmed on this function's first live run: 18,913 raw films vs. the
    ~5,032 the old work-set had ever covered. WORK_SET_MIN_REL_DATE (module
    level, above) restricts it back down to a sane window — see that
    constant's comment for why 2018-01-01 isn't an arbitrary guess.

    Verified against a live Snowflake connection as of 2026-08-12 — the
    array-to-string handling for ACTOR_LIST/DIRECTOR_LIST worked as expected.

    Returns None if Snowflake is unreachable — there is no local parquet
    fallback (that's main.py's job, via its own separate loader).
    """
    try:
        from base_snowflake import SnowFlakeBase
        sb = SnowFlakeBase(warehouse=SF_WAREHOUSE, database=SF_DATABASE, schema=SF_SCHEMA)
        sb.create_snowflake_connection(SF_RSA_KEY)
        df = pd.read_sql(films_sql.SQL_FILM_DETAILS, sb.engine)
    except Exception as e:
        log.error(f"Snowflake unavailable ({e}) — cannot load film work-set")
        return None

    df['actor_list'] = df['actor_list'].apply(_snowflake_array_to_pipe_string)
    df['director'] = df['director_list'].apply(_snowflake_array_to_pipe_string)
    df['dstbtr'] = df['distributor_name']
    df['rel_at'] = pd.to_datetime(
        df['film_nat_open_date'].fillna(df['film_open_date']), utc=True, errors='coerce',
    )

    if WORK_SET_MIN_REL_DATE is not None:
        n_before = len(df)
        df = df[df['rel_at'] >= WORK_SET_MIN_REL_DATE].reset_index(drop=True)
        log.info(f"Restricted to rel_at >= {WORK_SET_MIN_REL_DATE.date()}: {n_before:,} → {len(df):,} films")

    # IHUB_SYNOPSIS is primary (matches the pre-split COALESCE(m.SYNOPSIS,
    # f.FILM_DESC) behaviour); VISTA_SYNOPSIS becomes alt_synopsis only when
    # IHUB was actually present AND differs from it — so both sources feed
    # the extractor as independent cross-checks (diffed for changes on the
    # next run — see _diff_synopsis_films). When IHUB is empty and VISTA is
    # used as the synopsis fallback instead, alt_synopsis stays empty rather
    # than duplicating the same text into both fields.
    ihub  = df['ihub_synopsis'].fillna('').str.strip()
    vista = df['vista_synopsis'].fillna('').str.strip()
    df['synopsis'] = df['ihub_synopsis'].where(ihub != '', df['vista_synopsis'])
    df['alt_synopsis'] = df['vista_synopsis'].where((ihub != '') & (vista != ihub), None)

    df['film_id'] = df['film_id'].astype(int)
    df = df.drop_duplicates('film_id')
    log.info(f"Films from Snowflake: {len(df)}")

    df = df[df['synopsis'].notna() & (df['synopsis'].astype(str).str.len() >= 5)].copy()
    log.info(f"Films with synopsis: {len(df)}")

    return df


def _load_session_film_ids(film_ids) -> set[int]:
    """Which of the given film_ids have ever had an actual theatrical session
    logged — EDW_ENT_PRD.SEMANTIC.VW_VHO_SESSION_SUMMARY joined via
    DIM_VH_FILM.FILM_HO_CODE, the same join cinema_admits_models/sql/
    db_merge_20260420.sql uses to build the box office model's real training
    data. Used by _diff_film_meta to drop PAST-dated films with zero sessions
    (investigation 2026-08-12: of a 2,442-film backlog, 1,818 past-dated films
    had zero sessions even 8+ weeks after release — almost certainly films
    that never had a meaningful theatrical run and will never be used in
    training). Future-dated films are never checked against this — they need
    film_meta features before they've opened at all (prediction use case).

    Fails open on any Snowflake error — returns every film_id passed in,
    so a transient connection issue never wrongly excludes real films."""
    film_ids = [int(f) for f in film_ids]
    if not film_ids:
        return set()
    try:
        from base_snowflake import SnowFlakeBase
        sb = SnowFlakeBase(warehouse=SF_WAREHOUSE, database=SF_DATABASE, schema=SF_SCHEMA)
        sb.create_snowflake_connection(SF_RSA_KEY)
        id_list = ','.join(str(f) for f in film_ids)
        sql = f'''
            select distinct vh.film_id
            from EDW_ENT_PRD.SEMANTIC.VW_VHO_SESSION_SUMMARY as sess
            join EDW_ENT_PRD.CURATED.DIM_VH_FILM as vh on vh.film_ho_code = sess.film_ho_code
            where vh.film_id in ({id_list})
        '''
        result = pd.read_sql(sql, sb.engine)
        return set(result.iloc[:, 0].astype(int))
    except Exception as e:
        log.warning(f"Session-data lookup skipped ({e}) — no films dropped for lack of session data")
        return set(film_ids)


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

def _diff_synopsis_films(df_films: pd.DataFrame, film_lookup: pd.DataFrame | None = None) -> pd.DataFrame:
    """Films that are new OR whose synopsis (IHUB_SYNOPSIS) OR alt_synopsis
    (VISTA_SYNOPSIS) text has changed — either source changing is enough to
    trigger re-extraction, not just the primary one, since Vista's own
    booking description can get corrected independently of IHUB's.

    Applies the same skip-distributor filter (FILM_META_SKIP_DISTRIBUTORS —
    festivals/events/sports, no extractable synopsis-worthy content) and
    duplicate-booking variant filter (_apply_variant_merge) film_meta uses,
    BEFORE diffing — neither ran on this path before, so every festival/event
    booking and every 3D/IMAX/rescreening variant was getting its own separate
    synopsis extraction for nothing downstream would use. No-op if film_lookup
    is None (variant filter only). Also drops films with no usable synopsis
    text at all (see _drop_no_usable_synopsis) — extracting from "Plot
    unknown" produces nothing but null/unknown classifications.
    """
    if 'dstbtr' in df_films.columns:
        n0 = len(df_films)
        df_films = df_films[~df_films['dstbtr'].isin(FILM_META_SKIP_DISTRIBUTORS)]
        if n0 - len(df_films):
            log.info(f"synopsis skip-distributors: -{n0 - len(df_films)} → {len(df_films)}")
    df_films = _drop_no_usable_synopsis(df_films)
    df_films = _apply_variant_merge(df_films, film_lookup, label="synopsis")

    existing = s3_checkpoint.read_parquet(SYNOPSIS_S3_NAME, SYNOPSIS_FILENAME)
    if existing is None:
        log.info("No existing synopsis parquet on S3 — all films are new")
        return df_films

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
    cast_df = s3_checkpoint.read_parquet(CAST_S3_NAME, CAST_FILENAME, columns=['actor_name', 'fame_tier'])
    if cast_df is not None:
        names = cast_df['actor_name'].astype(str).str.upper().str.strip()
        done |= set(names)
        retry |= set(names[cast_df['fame_tier'].astype(str).str.strip().str.lower() == 'unknown'])
    checkpoint = s3_checkpoint.load_checkpoint(CAST_S3_NAME)
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
    dir_df = s3_checkpoint.read_parquet(DIRECTOR_S3_NAME, DIRECTOR_FILENAME, columns=['director_name', 'director_tier'])
    if dir_df is not None:
        names = dir_df['director_name'].astype(str).str.strip()
        done |= set(names)
        retry |= set(names[dir_df['director_tier'].astype(str).str.strip().str.lower() == 'unknown'])
    checkpoint = s3_checkpoint.load_checkpoint(DIRECTOR_S3_NAME)
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
    """Merge freshly detected film_id variants into film_id_variants.parquet
    on S3. keep='last' so a re-run with better data (richer synopsis/cast)
    can flip which side is canonical without a stale row surviving.

    NOTE: cleanup_film_meta.py's own persist_variant_map() (used by its
    standalone local CLI) still writes the LOCAL FILM_ID_VARIANTS_PATH copy
    — the two are no longer the same file. See CLAUDE.md."""
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

    existing = s3_checkpoint.read_parquet(FILM_META_S3_NAME, FILM_ID_VARIANTS_FILENAME)
    if existing is not None:
        for col in ('variant_rel_at', 'canonical_rel_at'):
            if col in existing.columns:
                existing[col] = existing[col].apply(lambda v: str(v) if pd.notna(v) else None)
        out = (pd.concat([existing, variant_map], ignore_index=True)
               .drop_duplicates(subset='film_id', keep='last'))
    else:
        out = variant_map
    s3_checkpoint.write_parquet(FILM_META_S3_NAME, FILM_ID_VARIANTS_FILENAME, out)
    log.info(f"film_meta variants: {len(variant_map)} film_id(s) mapped onto a canonical "
             f"release this run ({len(out)} total) → "
             f"{s3_checkpoint.s3_uri(FILM_META_S3_NAME, FILM_ID_VARIANTS_FILENAME)}")


def _apply_variant_merge(df: pd.DataFrame, film_lookup: pd.DataFrame | None, label: str = "film_meta") -> pd.DataFrame:
    """Drops genuine re-releases (keyword/year/language title match, no specific
    pairing) and duplicate-booking variants (fuzzy-matched to another film_id
    that already represents the same release — see film_variant_merge.py),
    keeping only the canonical film_id. Persists the variant crosswalk (shared
    across every caller — a variant film_id resolves to the same canonical
    film_id regardless of which extraction path is asking) so a variant
    film_id can be resolved back to whichever film_id actually holds the data.

    Called from both the film_meta path and the synopsis path (_diff_synopsis_
    films) — `label` is just for the log line, which path called doesn't change
    the filtering logic itself.
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
        log.info(f"{label} re-releases/variants filtered: -{n_dropped} "
                 f"({len(variant_map)} duplicate-booking, {n_dropped - len(variant_map)} keyword/year/language) "
                 f"→ {len(df_filtered)}")
    try:
        _persist_variant_map(variant_map)
    except Exception as e:
        # Filtering already succeeded and df_filtered is good to use — an S3
        # error persisting the crosswalk (e.g. an expired Stax token) shouldn't
        # crash the whole diff, just mean this run's variant map isn't saved
        # (the next successful run will re-detect and persist the same map).
        log.warning(f"{label} variant map persist skipped ({e})")
    return df_filtered


def _drop_no_session_past_films(df: pd.DataFrame) -> pd.DataFrame:
    """Drops past-dated films with zero theatrical session data (see
    _load_session_film_ids's docstring for why). Future-dated films are
    always kept regardless — they need film_meta features before they've
    opened at all (prediction use case)."""
    today = pd.Timestamp.now(tz='UTC')
    is_future = df['rel_at'] > today
    past = df[~is_future]
    session_ids = _load_session_film_ids(past['film_id'])
    no_session_ids = set(past.loc[~past['film_id'].astype(int).isin(session_ids), 'film_id'].astype(int))
    if no_session_ids:
        df = df[is_future | ~df['film_id'].astype(int).isin(no_session_ids)]
        log.info(f"film_meta no-session-data filtered: -{len(no_session_ids)} "
                 f"(past-dated, zero theatrical sessions) → {len(df)}")
    return df


def _diff_film_meta(df_films: pd.DataFrame, film_lookup: pd.DataFrame | None) -> pd.DataFrame:
    """Films not yet in checkpoint JSON or enriched parquet, with skip-distributor
    filter and re-release/variant merge applied.

    Checks the checkpoint as well as the parquet so a partially-completed run (where
    the checkpoint has been written but the final parquet flush has not yet happened)
    isn't re-extracted.

    Also drops past-dated films with zero theatrical session data (see
    _load_session_film_ids) — applied LAST, after the checkpoint diff, so the
    live Snowflake session-data query only runs against the actual candidate
    set instead of the whole catalogue.
    """
    df = df_films.copy()
    n0 = len(df)
    if 'dstbtr' in df.columns:
        df = df[~df['dstbtr'].isin(FILM_META_SKIP_DISTRIBUTORS)]
        if n0 - len(df):
            log.info(f"film_meta skip-distributors: -{n0 - len(df)} → {len(df)}")

    df = _apply_variant_merge(df, film_lookup)

    done_ids: set[int] = set()
    checkpoint = s3_checkpoint.load_checkpoint(FILM_META_S3_NAME)
    done_ids |= {int(k) for k in checkpoint}
    existing = s3_checkpoint.read_parquet(FILM_META_S3_NAME, FILM_META_FILENAME, columns=['film_id'])
    if existing is not None:
        done_ids |= set(existing['film_id'].astype(int))

    df = df[~df['film_id'].astype(int).isin(done_ids)]
    log.info(f"film_meta diff: {len(done_ids)} already done — {len(df)} to extract")

    df = _drop_no_session_past_films(df)
    return df


# ── Extraction steps ──────────────────────────────────────────────────────────

def _sync_synopsis_checkpoint(df: pd.DataFrame) -> None:
    """Regenerate a LOCAL synopsis_progress.json from the just-written parquet.

    This is the one deliberate local-disk write left in this module —
    refresh.py's own diff/checkpoint logic never reads it (see
    _diff_synopsis_films, which diffs against the S3 parquet only). It exists
    purely so main.py's own (local-disk-only) synopsis path, if run on the
    same machine, sees these films as already done instead of re-extracting
    them from scratch. Wrapped defensively since a Dagster/Kubernetes pod may
    have no writable local disk at all — that's fine, it just means this
    compatibility shim silently does nothing there, which is harmless (main.py
    isn't running in that pod either). See CLAUDE.md's S3 checkpoint notes.
    """
    if df.empty or 'film_id' not in df.columns:
        return

    def _to_json_native(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        # Plain Python list/tuple (e.g. an empty [] from a list-valued
        # extraction field, per the "lists default to [] never null" prompt
        # convention) — must be checked BEFORE pd.isna(), which is vectorized
        # over list-like inputs and returns an array, not a bool, making
        # `if pd.isna(value)` raise "truth value of an array is ambiguous."
        if isinstance(value, (list, tuple)):
            return list(value)
        if isinstance(value, np.generic):
            return value.item()
        return None if pd.isna(value) else value

    # Best-effort local compatibility shim (see docstring) — any failure here,
    # not just a file-write error, should never take down the real S3-based
    # synopsis path with it.
    try:
        checkpoint = {
            str(row['film_id']): {k: _to_json_native(v) for k, v in row.items()}
            for _, row in df.iterrows()
        }
        SYNOPSIS_CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(SYNOPSIS_CHECKPOINT_PATH, 'w') as f:
            json.dump(checkpoint, f, default=str)
        log.info(f"Synopsis checkpoint synced from parquet → {SYNOPSIS_CHECKPOINT_PATH} ({len(checkpoint)} films)")
    except Exception as e:
        log.info(f"Skipping local synopsis checkpoint sync ({e})")


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

    existing = s3_checkpoint.read_parquet(SYNOPSIS_S3_NAME, SYNOPSIS_FILENAME)
    if existing is not None:
        out = (pd.concat([df_new, existing], ignore_index=True)
               .drop_duplicates(subset='film_id', keep='first'))
    else:
        out = df_new
    s3_checkpoint.write_parquet(SYNOPSIS_S3_NAME, SYNOPSIS_FILENAME, out)
    log.info(f"Synopsis parquet → {s3_checkpoint.s3_uri(SYNOPSIS_S3_NAME, SYNOPSIS_FILENAME)}  ({len(out)} total)")
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

    checkpoint: dict = s3_checkpoint.load_checkpoint(CAST_S3_NAME)
    if checkpoint:
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

        batch_success: dict[str, dict] = {}
        batch_errors: dict[str, dict] = {}
        batch_success_keys: set[str] = set()
        for actor_name, data in results.items():
            if not data.get('_error'):
                entry = {**data, 'actor_name': actor_name}
                checkpoint[str(actor_name)] = entry
                batch_success[str(actor_name)] = entry
                batch_success_keys.add(str(actor_name))
            else:
                batch_errors[str(actor_name)] = {
                    'actor_name':  actor_name,
                    '_error':      data.get('_error'),
                    '_raw_output': data.get('_raw_output'),
                }

        # One small delta object per batch — not a full rewrite of the whole
        # checkpoint (see s3_checkpoint.py's module docstring for why).
        try:
            s3_checkpoint.append_checkpoint(CAST_S3_NAME, batch_success)
            log.info(f"  Checkpoint delta saved: {len(batch_success)} actors "
                     f"({len(checkpoint)} total so far)")
        except Exception as e:
            # See _enrich_film_meta's identical guard — stop spending on new
            # batches the moment persistence breaks; this batch's results are
            # still in `checkpoint` for the final flush to try.
            log.error(f"  Checkpoint save failed ({e}) — stopping after batch {batch_num}/{len(chunks)}. "
                      f"{len(checkpoint)} actors extracted so far will be flushed to parquet below; "
                      f"re-authenticate (stax2aws login) and re-run to pick up any remainder.")
            break

        if batch_errors or batch_success_keys:
            try:
                existing_errors = s3_checkpoint.load_errors(CAST_S3_NAME)
                purged_n = sum(1 for k in batch_success_keys
                               if existing_errors.pop(k, None) is not None)
                existing_errors.update(batch_errors)
                if purged_n or batch_errors:
                    s3_checkpoint.save_errors(CAST_S3_NAME, existing_errors)
                msg = f"  Errors this batch: {len(batch_errors)}"
                if purged_n:
                    msg += f"  [purged {purged_n} now-recovered]"
                log.info(msg)
            except Exception as e:
                log.warning(f"  Errors-file save skipped ({e}) — successful extractions this batch are unaffected")

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

    existing = s3_checkpoint.read_parquet(CAST_S3_NAME, CAST_FILENAME)
    if existing is not None:
        # keep='last' so a freshly retried actor's new result (df_new) overrides
        # the stale row in existing — keep='first' would silently discard every
        # unknown-retry update at flush time.
        out = (pd.concat([existing, df_new], ignore_index=True)
               .drop_duplicates(subset='actor_name', keep='last'))
    else:
        out = df_new
    s3_checkpoint.write_parquet(CAST_S3_NAME, CAST_FILENAME, out)
    log.info(f"Cast parquet → {s3_checkpoint.s3_uri(CAST_S3_NAME, CAST_FILENAME)}  ({len(out)} actors)")

    n_compacted = s3_checkpoint.compact_checkpoint(CAST_S3_NAME)
    log.info(f"Cast checkpoint compacted: {n_compacted} deltas folded into snapshot")

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

    checkpoint: dict = s3_checkpoint.load_checkpoint(DIRECTOR_S3_NAME)
    if checkpoint:
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

        batch_success: dict[str, dict] = {}
        batch_errors: dict[str, dict] = {}
        batch_success_keys: set[str] = set()
        for name, data in results.items():
            if not data.get('_error'):
                entry = {**data, 'director_name': name}
                checkpoint[str(name)] = entry
                batch_success[str(name)] = entry
                batch_success_keys.add(str(name))
            else:
                batch_errors[str(name)] = {
                    'director_name': name,
                    '_error':        data.get('_error'),
                    '_raw_output':   data.get('_raw_output'),
                }

        s3_checkpoint.append_checkpoint(DIRECTOR_S3_NAME, batch_success)
        log.info(f"  Checkpoint delta saved: {len(batch_success)} directors "
                 f"({len(checkpoint)} total so far)")

        if batch_errors or batch_success_keys:
            existing_errors = s3_checkpoint.load_errors(DIRECTOR_S3_NAME)
            purged_n = sum(1 for k in batch_success_keys
                           if existing_errors.pop(k, None) is not None)
            existing_errors.update(batch_errors)
            if purged_n or batch_errors:
                s3_checkpoint.save_errors(DIRECTOR_S3_NAME, existing_errors)
            msg = f"  Errors this batch: {len(batch_errors)}"
            if purged_n:
                msg += f"  [purged {purged_n} now-recovered]"
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

    existing = s3_checkpoint.read_parquet(DIRECTOR_S3_NAME, DIRECTOR_FILENAME)
    if existing is not None:
        # keep='last' so a freshly retried director's new result (df_new) overrides
        # the stale row in existing — keep='first' would silently discard every
        # unknown-retry update at flush time.
        out = (pd.concat([existing, df_new], ignore_index=True)
               .drop_duplicates(subset='director_name', keep='last'))
    else:
        out = df_new
    s3_checkpoint.write_parquet(DIRECTOR_S3_NAME, DIRECTOR_FILENAME, out)
    log.info(f"Director parquet → {s3_checkpoint.s3_uri(DIRECTOR_S3_NAME, DIRECTOR_FILENAME)}  ({len(out)} directors)")

    n_compacted = s3_checkpoint.compact_checkpoint(DIRECTOR_S3_NAME)
    log.info(f"Director checkpoint compacted: {n_compacted} deltas folded into snapshot")

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

    checkpoint: dict = s3_checkpoint.load_checkpoint(FILM_META_S3_NAME)
    if checkpoint:
        log.info(f"Loaded film_meta checkpoint: {len(checkpoint)} films already done")

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
        batch_success: dict[str, dict] = {}
        batch_errors: dict[str, dict] = {}
        batch_success_ids: set[str] = set()
        for film_id, data in results.items():
            if not data.get('_error'):
                data['film_id'] = film_id
                pt = evt_passthrough.get(film_id, {})
                data['evt_dstbtr'] = str(pt.get('evt_dstbtr')) if pd.notna(pt.get('evt_dstbtr')) else None
                data['evt_rel_at'] = str(pt.get('evt_rel_at')) if pd.notna(pt.get('evt_rel_at')) else None
                checkpoint[str(film_id)] = data
                batch_success[str(film_id)] = data
                batch_success_ids.add(str(film_id))
            else:
                batch_errors[str(film_id)] = {
                    'film_id':     film_id,
                    'film_title':  title_lookup.get(film_id),
                    '_error':      data.get('_error'),
                    '_raw_output': data.get('_raw_output'),
                }

        try:
            s3_checkpoint.append_checkpoint(FILM_META_S3_NAME, batch_success)
            log.info(f"  Checkpoint delta saved: {len(batch_success)} films "
                     f"({len(checkpoint)} total so far)")
        except Exception as e:
            # S3 auth (Stax tokens expire hourly — a ~25-batch film_meta run
            # routinely outlasts that) or a transient network error. Stop
            # starting NEW batches immediately — every further batch would
            # spend real OpenAI money on results we already know we can't
            # persist. This batch's results are still in `checkpoint` (set
            # above, before this call), so the flush below will still try to
            # save them along with everything from earlier successful batches.
            log.error(f"  Checkpoint save failed ({e}) — stopping after batch {batch_num}/{len(chunks)}. "
                      f"{len(checkpoint)} films extracted so far will be flushed to parquet below; "
                      f"re-authenticate (stax2aws login) and re-run to pick up any remainder.")
            break

        # Errors: merge in new errors AND drop entries for films that just
        # succeeded (covers both this batch's wins and stale entries from prior
        # runs that have since recovered).
        if batch_errors or batch_success_ids:
            try:
                existing_errors = s3_checkpoint.load_errors(FILM_META_S3_NAME)
                purged_n = sum(1 for fid in batch_success_ids
                               if existing_errors.pop(fid, None) is not None)
                existing_errors.update(batch_errors)
                if purged_n or batch_errors:
                    s3_checkpoint.save_errors(FILM_META_S3_NAME, existing_errors)
                err_counts: dict[str, int] = {}
                for v in batch_errors.values():
                    key = str(v.get('_error', 'unknown')).split(':')[0][:40]
                    err_counts[key] = err_counts.get(key, 0) + 1
                msg = f"  Errors this batch: {len(batch_errors)}"
                if err_counts:
                    msg += f" ({', '.join(f'{k}={n}' for k, n in err_counts.items())})"
                if purged_n:
                    msg += f"  [purged {purged_n} now-recovered]"
                log.info(msg)
            except Exception as e:
                log.warning(f"  Errors-file save skipped ({e}) — successful extractions this batch are unaffected")

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

    # Final flush — deliberately NOT wrapped to swallow errors: if this fails
    # (e.g. the same expired token that stopped the loop above), everything in
    # `checkpoint` from batches whose delta-append also failed is genuinely
    # unsaved and the run SHOULD report failure, not a false success. Batches
    # that appended successfully above are already safe in S3 regardless.
    existing = s3_checkpoint.read_parquet(FILM_META_S3_NAME, FILM_META_FILENAME)
    if existing is not None:
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

    s3_checkpoint.write_parquet(FILM_META_S3_NAME, FILM_META_FILENAME, out)
    log.info(f"film_meta parquet → {s3_checkpoint.s3_uri(FILM_META_S3_NAME, FILM_META_FILENAME)}  ({len(out)} films)")

    n_compacted = s3_checkpoint.compact_checkpoint(FILM_META_S3_NAME)
    log.info(f"film_meta checkpoint compacted: {n_compacted} deltas folded into snapshot")

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

    film_lookup = load_full_film_catalogue()
    if force:
        to_extract = df_films
        if 'dstbtr' in to_extract.columns:
            to_extract = to_extract[~to_extract['dstbtr'].isin(FILM_META_SKIP_DISTRIBUTORS)]
        to_extract = _drop_no_usable_synopsis(to_extract)
        to_extract = _apply_variant_merge(to_extract, film_lookup, label="synopsis")
    else:
        to_extract = _diff_synopsis_films(df_films, film_lookup)
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
        df = _drop_no_session_past_films(df)
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
