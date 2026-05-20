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
    """
    Pull current film list from Snowflake via SQL_FILM_DETAILS.

    Returns a DataFrame with normalised column names:
        film_id, film_title, synopsis, alt_synopsis,
        actor_list, director, dstbtr, rel_at

    Returns None if Snowflake is unavailable.
    """
    try:
        from base_snowflake import SnowFlakeBase
        sb = SnowFlakeBase(warehouse=SF_WAREHOUSE, database=SF_DATABASE, schema=SF_SCHEMA)
        sb.create_snowflake_connection(SF_RSA_KEY)
        df = pd.read_sql(films_sql.SQL_FILM_DETAILS, sb.engine)
    except Exception as e:
        log.warning(f"Snowflake unavailable ({e})")
        return None

    df = df.rename(columns={
        'director_list':       'director',
        'distributor_name':    'dstbtr',
        'film_nat_open_date':  'rel_at',
    })
    df['film_id'] = df['film_id'].astype(int)
    df['rel_at']  = pd.to_datetime(df['rel_at'], utc=True, errors='coerce')

    keep = ['film_id', 'film_title', 'synopsis', 'alt_synopsis',
            'actor_list', 'director', 'dstbtr', 'rel_at']
    df = df[[c for c in keep if c in df.columns]].copy()
    log.info(f"Snowflake: {len(df)} films loaded")
    return df


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
    all_actors: set[str] = set()
    for val in df_films.get('actor_list', pd.Series(dtype=str)).dropna():
        for a in str(val).split('|'):
            a = _clean_actor(a)
            if a:
                all_actors.add(a)

    if not CAST_ENRICHED_PATH.exists():
        log.info(f"No cast parquet — {len(all_actors)} actors are new")
        return sorted(all_actors)

    existing = set(
        pd.read_parquet(CAST_ENRICHED_PATH, columns=['actor_name'])
        ['actor_name'].str.upper().str.strip()
    )
    new = sorted(all_actors - existing)
    log.info(f"Cast diff: {len(new)} new actors")
    return new


def _diff_directors(df_films: pd.DataFrame) -> list[str]:
    all_dirs: set[str] = set()
    for val in df_films.get('director', pd.Series(dtype=str)).dropna():
        # Snowflake pipes; raw parquets sometimes comma — handle both.
        parts = re.split(r'[|,]', str(val))
        for d in parts:
            d = d.strip()
            if d:
                all_dirs.add(d)

    if not DIRECTOR_ENRICHED_PATH.exists():
        log.info(f"No director parquet — {len(all_dirs)} directors are new")
        return sorted(all_dirs)

    existing = set(
        pd.read_parquet(DIRECTOR_ENRICHED_PATH, columns=['director_name'])
        ['director_name'].str.strip()
    )
    new = sorted(all_dirs - existing)
    log.info(f"Director diff: {len(new)} new directors")
    return new


def _diff_film_meta(df_films: pd.DataFrame) -> pd.DataFrame:
    """Film IDs not yet in film_meta_enriched.parquet, with skip-distributor filter applied."""
    df = df_films.copy()
    n0 = len(df)
    if 'dstbtr' in df.columns:
        df = df[~df['dstbtr'].isin(FILM_META_SKIP_DISTRIBUTORS)]
        if n0 - len(df):
            log.info(f"film_meta skip-distributors: -{n0 - len(df)} → {len(df)}")

    if not FILM_META_ENRICHED_PATH.exists():
        log.info(f"No film_meta parquet — {len(df)} films are new")
        return df

    existing_ids = set(
        pd.read_parquet(FILM_META_ENRICHED_PATH, columns=['film_id'])
        ['film_id'].astype(int)
    )
    df = df[~df['film_id'].astype(int).isin(existing_ids)]
    log.info(f"film_meta diff: {len(df)} films to extract")
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
    if not new_actors:
        return
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

    log.info(f"Enriching {len(new_actors)} actors")
    results = await extractor.arun(
        df=df_actors,
        name_col='actor_name',
        max_concurrency=META_MAX_CONCURRENCY,
    )

    rows = []
    for actor_name, data in results.items():
        if not data.get('_error'):
            rows.append({**data, 'actor_name': actor_name})
    df_new = pd.DataFrame(rows)
    if df_new.empty:
        log.warning("No cast results produced")
        return
    df_new = df_new.drop(columns=[c for c in
        ['_error', '_error_message', '_raw_output', 'synopsis', 'title']
        if c in df_new.columns], errors='ignore')
    for col in ['fame_tier', 'fame_source', 'primary_market', 'age_range']:
        if col in df_new.columns:
            df_new[col] = df_new[col].str.lower().str.strip()

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
                 f"cost: ${u.get('cost_usd', 0):.4f}")
    del extractor
    gc.collect()


async def _enrich_directors(new_directors: list[str]) -> None:
    if not new_directors:
        return
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

    log.info(f"Enriching {len(new_directors)} directors")
    results = await extractor.arun(
        df=df_dirs,
        name_col='director_name',
        max_concurrency=META_MAX_CONCURRENCY,
    )

    rows = []
    for name, data in results.items():
        if not data.get('_error'):
            rows.append({**data, 'director_name': name})
    df_new = pd.DataFrame(rows)
    if df_new.empty:
        log.warning("No director results produced")
        return
    df_new = df_new.drop(columns=[c for c in
        ['_error', '_error_message', '_raw_output', 'synopsis', 'title']
        if c in df_new.columns], errors='ignore')
    for col in ['director_tier', 'primary_market']:
        if col in df_new.columns:
            df_new[col] = df_new[col].str.lower().str.strip()

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
                 f"cost: ${u.get('cost_usd', 0):.4f}")
    del extractor
    gc.collect()


async def _enrich_film_meta(df: pd.DataFrame, film_lookup: pd.DataFrame | None) -> None:
    if df.empty:
        return

    # Optional re-release filter — non-fatal if cinema_admits_models isn't on path.
    if film_lookup is not None and 'rel_at' in film_lookup.columns:
        try:
            sys.path.insert(0, '/Users/jonathanchapman/Documents/git/cinema_admits_models')
            from re_release_filter import ReReleaseFilter
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
    )

    log.info(f"Extracting film_meta for {len(df)} films")
    results = await extractor.arun(
        df=df,
        id_col='film_id',
        title_col='film_title',
        rel_at_col='rel_at',
        director_col='director',
        synopsis_col='synopsis',
        max_concurrency=META_MAX_CONCURRENCY,
    )

    rows = []
    for film_id, data in results.items():
        if data.get('_error'):
            continue
        data = dict(data)
        data['film_id'] = film_id
        pt = evt_passthrough.get(film_id, {})
        data['evt_dstbtr'] = str(pt.get('evt_dstbtr')) if pd.notna(pt.get('evt_dstbtr')) else None
        data['evt_rel_at'] = str(pt.get('evt_rel_at')) if pd.notna(pt.get('evt_rel_at')) else None
        rows.append(data)
    df_new = pd.DataFrame(rows)
    if df_new.empty:
        log.warning("No film_meta results produced")
        return
    df_new = df_new.drop(columns=[c for c in ['_error', '_raw_output']
                                  if c in df_new.columns], errors='ignore')

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

    # film_lookup for the re-release filter — uses the full Snowflake set.
    film_lookup = df_films.rename(columns={'film_title': 'film'})[
        [c for c in ['film_id', 'film', 'rel_at', 'dstbtr', 'director'] if c in df_films.columns or c == 'film']
    ].copy()

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
