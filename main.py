"""
main.py — local extraction script for synopsis meta and cast enrichment.

Edit the CONFIG block at the top, then run:
    python main.py

For scheduled/Dagster runs use refresh.py instead.
"""

import asyncio
import gc
import os
import sys

import nest_asyncio
nest_asyncio.apply()

import litellm
import logging
litellm.success_callback = []
litellm.failure_callback = []
litellm.drop_params = True
logging.getLogger("LiteLLM").setLevel(logging.ERROR)

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from extractor import LlmJsonExtractor
from film_meta_extractor import FilmMetaExtractor, ActorMetaExtractor, DirectorMetaExtractor
from load_prompts import load_tasks_from_yaml
from models import DEFAULT_MODEL, DEFAULT_FALLBACKS, MODELS
from config import (
    SYNOPSES_EXTRACTED_PATH, CAST_ENRICHED_PATH, CAST_FEATURES_PATH,
    DIRECTOR_ENRICHED_PATH, DIRECTOR_FEATURES_PATH,
    FILM_META_ENRICHED_PATH,
    DATA_DIR, SF_WAREHOUSE, SF_DATABASE, SF_SCHEMA, SF_RSA_KEY,
)
from films import sql as films_sql

# ── CONFIG — edit here ────────────────────────────────────────────────────────

# Set to a list of film_ids to target specific films; None = all films in parquets
# Load from adhoc_updates parquet if present (generated externally for gap-fill runs)
# _adhoc_path = DATA_DIR / 'adhoc_updates' / 'film_ids_to_encode.parquet'
# if _adhoc_path.exists():
#     FILM_IDS = pd.read_parquet(_adhoc_path)['film_id'].astype(int).tolist()
#     print(f"Loaded {len(FILM_IDS)} film IDs from {_adhoc_path}")
# else:
FILM_IDS = None
# FILM_IDS = [57603, 57343, 59530, 60336, 60261, 60560]

# Quota-recovery sanity-check: uncomment to run only the first N films that
# previously failed with "exceeded your current quota". Lets you verify the
# OpenAI top-up worked before kicking off a full Dagster run.
# _qpath = DATA_DIR / 'film_meta' / 'film_meta_errors.json'
# if _qpath.exists():
#     import json as _json
#     _errs = _json.loads(_qpath.read_text())
#     FILM_IDS = [int(fid) for fid, e in _errs.items()
#                 if 'exceeded your current quota' in str(e.get('_error', '')).lower()][:10]
#     print(f"Quota-recovery mode: testing {len(FILM_IDS)} previously-failed films")

SAMPLE_SIZE   = 0          # 0 = all; N = first N films by release date

RUN_SYNOPSIS  = False       # extract synopsis features via LLM
RUN_CAST      = True      # enrich cast profiles via LLM
RUN_DIRECTOR  = True      # enrich director profiles via LLM
RUN_META      = False      # web-grounded film_meta extraction (studios/cast/genres/budget/trailers)
RUN_ENCODE    = False    # DEPRECATED 2026-05-19 — encoding moved to cinema_admits_models/build_data/encode_llm_features.py. The encode_*.py scripts have been moved to depreciated/encoding/. Leave False; flip True only for legacy reruns.

MAX_CONCURRENCY = 2   # 8 concurrent × 11 tasks = ~88 requests/burst — stays under org TPM limit
BATCH_PAUSE_SECS = 3   # pause between 100-film batches to let the rate window reset
MAX_COST_USD    = 20.00    # hard stop if cumulative spend exceeds this

# film_meta uses lower concurrency — each call invokes web_search (~5-10s latency)
# Bound by TPM, not RPM: org limit is 200k TPM, each call ≈ 15k tokens → max
# sustainable ≈ 13 req/min. 2 concurrent × ~7s/call ≈ 17 req/min — leaves headroom
# for SDK retries to claw back any transient 429s.
META_MAX_CONCURRENCY = 2

PROMPTS_PATH           = 'prompts/prompts_v2.yaml'
CAST_PROMPTS_PATH      = 'prompts/cast_prompts.yaml'
DIRECTOR_PROMPTS_PATH  = 'prompts/director_prompts.yaml'
FILM_META_PROMPTS_PATH = 'prompts/film_meta_prompts.yaml'
 
CHECKPOINT_PATH            = DATA_DIR / 'synopsis_v2'    / 'synopsis_progress.json'
DIRECTOR_CHECKPOINT_PATH   = DATA_DIR / 'director_meta'  / 'director_progress.json'
FILM_META_CHECKPOINT_PATH  = DATA_DIR / 'film_meta'      / 'film_meta_progress.json'
FILM_META_ERRORS_PATH      = DATA_DIR / 'film_meta'      / 'film_meta_errors.json'

# ── Load films from parquets (all available train/test/pred dates) ────────────

import glob as _glob

_raw_paths  = sorted(_glob.glob(str(DATA_DIR / 'raw_from_snowflake'        / '*' / 'train' / 'train_raw_ds.parquet')))
_raw_paths += sorted(_glob.glob(str(DATA_DIR / 'raw_from_snowflake'        / '*' / 'test'  / 'test_raw_ds.parquet')))
_pred_paths = sorted(_glob.glob(str(DATA_DIR / 'prediction_from_snowflake' / '*' / 'prediction_raw.parquet')))

_parts = []
for _p in _raw_paths + _pred_paths:
    _df = pd.read_parquet(_p, columns=['film_id', 'synopsis', 'actor_list', 'rel_at', 'director'])
    _df['rel_at'] = pd.to_datetime(_df['rel_at'], utc=True)
    _parts.append(_df)
    _label = '/'.join(_p.replace('\\', '/').split('/')[-3:])
    print(f"  {_label}: {_df['film_id'].nunique()} films")

df_films = (
    pd.concat(_parts, ignore_index=True)
    .drop_duplicates('film_id')
    .assign(film_id=lambda d: d['film_id'].astype(int))
)
print(f"Total unique films from parquets: {len(df_films)}")

# Filter to target IDs if specified
if FILM_IDS:
    df_films = df_films[df_films['film_id'].isin(FILM_IDS)].copy()
    print(f"Filtered to {len(df_films)} target films")

# Drop rows without a usable synopsis
df_films = df_films[df_films['synopsis'].notna() & (df_films['synopsis'].str.len() >= 5)].copy()

# Sample if requested
if SAMPLE_SIZE > 0:
    df_films = df_films.sort_values('rel_at', ascending=False).head(SAMPLE_SIZE).copy()

print(f"Films with synopsis: {len(df_films)}")

# ── Get film titles from Snowflake (used by LLM prompt) ──────────────────────

try:
    from base_snowflake import SnowFlakeBase
    _sb = SnowFlakeBase(warehouse=SF_WAREHOUSE, database=SF_DATABASE, schema=SF_SCHEMA)
    _sb.create_snowflake_connection(SF_RSA_KEY)
    _titles = pd.read_sql(films_sql.SQL_FILM_DETAILS, _sb.engine)[['film_id', 'film_title']]
    _titles['film_id'] = _titles['film_id'].astype(int)
    df_films = df_films.merge(_titles, on='film_id', how='left')
    print(f"Film titles joined from Snowflake")
except Exception as _e:
    print(f"Snowflake unavailable ({_e}) — using film_id as title fallback")
    df_films['film_title'] = df_films['film_id'].astype(str)

# ── Diff against existing parquet (skip already-extracted films) ──────────────

if RUN_SYNOPSIS and SYNOPSES_EXTRACTED_PATH.exists():
    _existing = pd.read_parquet(SYNOPSES_EXTRACTED_PATH, columns=['film_id'])
    _existing_ids = set(_existing['film_id'].astype(int))
    _before = len(df_films)
    df_films = df_films[~df_films['film_id'].isin(_existing_ids)].copy()
    print(f"Skipping {_before - len(df_films)} already-extracted films → {len(df_films)} to extract")

print(f"\nReady to process: {len(df_films)} films\n")
if not df_films.empty:
    for _, r in df_films.sort_values('rel_at', ascending=False).head(10).iterrows():
        print(f"  {r['film_id']:>6}  {r['film_title']}")
    if len(df_films) > 10:
        print(f"  ... and {len(df_films) - 10} more")


# ── Synopsis extraction ───────────────────────────────────────────────────────

BATCH_SIZE = 25    # checkpoint to JSON every N films

# Running spend, incremented per batch inside each enrich function via `global`.
# Survives KeyboardInterrupt — inspect from the REPL after a partial run.
_total_cost = 0.0

async def extract_synopses(df: pd.DataFrame) -> float:
    global _total_cost
    import json

    if df.empty:
        print("No films to extract.")
        return 0.0

    # ── Load checkpoint ───────────────────────────────────────────────────────
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH) as f:
            checkpoint: dict = json.load(f)
        print(f"Loaded checkpoint: {len(checkpoint)} films already done")
    else:
        checkpoint = {}

    # Skip films already in checkpoint
    done_ids = {int(k) for k in checkpoint}
    df = df[~df['film_id'].isin(done_ids)].copy()
    if df.empty:
        print("All films already in checkpoint — nothing to extract.")
        return 0.0
    print(f"Extracting synopses for {len(df)} films in batches of {BATCH_SIZE}...")

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

    chunks = [df.iloc[i:i + BATCH_SIZE] for i in range(0, len(df), BATCH_SIZE)]
    prev_cost = 0.0
    for batch_num, chunk in enumerate(chunks, 1):
        print(f"\nBatch {batch_num}/{len(chunks)}  ({len(chunk)} films)")
        results = await extractor.arun_multiple_synopses(
            df=chunk,
            id_col='film_id',
            title_col='film_title',
            synopsis_col='synopsis',
            alt_synopsis_col='alt_synopsis' if 'alt_synopsis' in chunk.columns else None,
            flatten=True,
            max_concurrency=MAX_CONCURRENCY,
        )

        # Add successful results to checkpoint and save JSON
        for film_id, data in results.items():
            if not data.get('_error'):
                checkpoint[str(film_id)] = data

        with open(CHECKPOINT_PATH, 'w') as f:
            json.dump(checkpoint, f)
        print(f"  Checkpoint saved: {len(checkpoint)} films → {CHECKPOINT_PATH}")

        if extractor.token_usage:
            curr = extractor.token_usage.get('cost_usd', 0.0)
            delta = curr - prev_cost
            _total_cost += delta
            prev_cost = curr
            print(f"  Batch cost: ${delta:.4f}  |  Run total: ${_total_cost:.4f} / ${MAX_COST_USD:.2f}")

        if batch_num < len(chunks):
            await asyncio.sleep(BATCH_PAUSE_SECS)

    # ── Flush checkpoint → parquet ────────────────────────────────────────────
    df_new = pd.DataFrame(checkpoint.values())
    if '_error' in df_new.columns:
        df_new = df_new[df_new['_error'].isna()].drop(columns=['_error'], errors='ignore')

    SYNOPSES_EXTRACTED_PATH.parent.mkdir(parents=True, exist_ok=True)
    if SYNOPSES_EXTRACTED_PATH.exists():
        df_existing = pd.read_parquet(SYNOPSES_EXTRACTED_PATH)
        out = (
            pd.concat([df_new, df_existing], ignore_index=True)
            .drop_duplicates(subset='film_id', keep='first')
        )
    else:
        out = df_new

    out.to_parquet(SYNOPSES_EXTRACTED_PATH, engine='pyarrow', index=False)
    print(f"\nParquet saved → {SYNOPSES_EXTRACTED_PATH}  ({len(out)} total films)")

    cost = 0.0
    if extractor.token_usage:
        u = extractor.token_usage
        cost = u.get('cost_usd', 0.0)
        print(f"Tokens — prompt: {u['prompt_tokens']:,}  completion: {u['completion_tokens']:,}  cost: ${cost:.4f}")

    del extractor
    gc.collect()
    return cost


# ── Cast enrichment ───────────────────────────────────────────────────────────

import re as _re
_AND_PREFIX = _re.compile(r'^AND\s+', _re.IGNORECASE)

def _clean_actor(raw: str) -> str:
    import json
    raw = str(raw).strip()
    items = json.loads(raw) if raw.startswith('[') else raw.split('|')
    out = []
    for a in items:
        name = _AND_PREFIX.sub('', str(a).strip()).upper().strip()
        if name and name not in ('AND', 'N/A') and len(name) > 1:
            out.append(name)
    return out


CAST_CHECKPOINT_PATH = DATA_DIR / 'cast_meta' / 'cast_progress.json'

async def enrich_cast(df_source: pd.DataFrame) -> float:
    global _total_cost
    import json

    all_actors: set[str] = set()
    for val in df_source['actor_list'].dropna():
        all_actors.update(_clean_actor(val))

    # Load checkpoint
    CAST_CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if CAST_CHECKPOINT_PATH.exists():
        with open(CAST_CHECKPOINT_PATH) as f:
            checkpoint: dict = json.load(f)
        print(f"Loaded cast checkpoint: {len(checkpoint)} actors already done")
    else:
        checkpoint = {}

    done_names = {k.upper() for k in checkpoint}
    new_actors = sorted(all_actors - done_names)
    print(f"\nCast: {len(all_actors)} actors total — {len(done_names)} in checkpoint — {len(new_actors)} to extract")

    if not new_actors:
        print("Cast enrichment up to date.")
        return 0.0

    df_actors = pd.DataFrame({'actor_name': new_actors})
    tasks         = load_tasks_from_yaml(CAST_PROMPTS_PATH)
    cast_model    = os.environ.get('FILM_META_MODEL', 'gpt-5.4-mini')
    cast_model_cfg = MODELS.get(cast_model, {})
    extractor = ActorMetaExtractor(
        task=tasks['actor_profile'],
        model=cast_model,
        api_key=os.getenv('OPENAI_KEY'),
        cost_per_1m_input=cast_model_cfg.get('cost_per_1m_input'),
        cost_per_1m_output=cast_model_cfg.get('cost_per_1m_output'),
    )

    chunks = [df_actors.iloc[i:i + BATCH_SIZE] for i in range(0, len(df_actors), BATCH_SIZE)]
    prev_cost = 0.0
    for batch_num, chunk in enumerate(chunks, 1):
        print(f"\nCast batch {batch_num}/{len(chunks)}  ({len(chunk)} actors)")
        results = await extractor.arun(
            df=chunk,
            name_col='actor_name',
            max_concurrency=META_MAX_CONCURRENCY,
        )

        for actor_name, data in results.items():
            if not data.get('_error'):
                data = {**data, 'actor_name': actor_name}
                checkpoint[str(actor_name)] = data

        with open(CAST_CHECKPOINT_PATH, 'w') as f:
            json.dump(checkpoint, f)
        print(f"  Checkpoint saved: {len(checkpoint)} actors → {CAST_CHECKPOINT_PATH}")

        if extractor.token_usage:
            curr = extractor.token_usage.get('cost_usd', 0.0)
            delta = curr - prev_cost
            _total_cost += delta
            prev_cost = curr
            print(f"  Batch cost: ${delta:.4f}  |  Run total: ${_total_cost:.4f} / ${MAX_COST_USD:.2f}")

        if batch_num < len(chunks):
            await asyncio.sleep(BATCH_PAUSE_SECS)

    # Flush checkpoint → parquet
    df_new = pd.DataFrame(checkpoint.values())
    if df_new.empty:
        print("  Warning: no cast results in checkpoint.")
        return 0.0

    if 'title' in df_new.columns:
        df_new = df_new.rename(columns={'title': 'actor_name'})
    if '_error' in df_new.columns:
        df_new = df_new[df_new['_error'].isna()].drop(
            columns=[c for c in ['_error', '_error_message', '_raw_output', 'synopsis']
                     if c in df_new.columns], errors='ignore'
        )

    # Normalise LLM string fields to lowercase in case of case variation (e.g. 'astAR')
    for _col in ['fame_tier', 'fame_source', 'primary_market', 'age_range']:
        if _col in df_new.columns:
            df_new[_col] = df_new[_col].str.lower().str.strip()

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
    print(f"\nCast parquet → {CAST_ENRICHED_PATH}  ({len(out)} actors)")

    cost = 0.0
    if extractor.token_usage:
        u = extractor.token_usage
        cost = u.get('cost_usd', 0.0)
        print(f"Tokens — prompt: {u['prompt_tokens']:,}  completion: {u['completion_tokens']:,}  cost: ${cost:.4f}")

    del extractor
    gc.collect()
    return cost


# ── Director enrichment ───────────────────────────────────────────────────────

async def enrich_directors(df_source: pd.DataFrame) -> float:
    global _total_cost
    import json

    all_directors: set[str] = set()
    for val in df_source['director'].dropna():
        for d in str(val).split(','):
            d = d.strip()
            if d:
                all_directors.add(d)

    DIRECTOR_CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DIRECTOR_CHECKPOINT_PATH.exists():
        with open(DIRECTOR_CHECKPOINT_PATH) as f:
            checkpoint: dict = json.load(f)
        print(f"Loaded director checkpoint: {len(checkpoint)} directors already done")
    else:
        checkpoint = {}

    done_names = {k for k in checkpoint}
    new_directors = sorted(all_directors - done_names)
    print(f"\nDirectors: {len(all_directors)} total — {len(done_names)} in checkpoint — {len(new_directors)} to extract")

    if not new_directors:
        print("Director enrichment up to date.")
        return 0.0

    df_directors = pd.DataFrame({'director_name': new_directors})
    tasks         = load_tasks_from_yaml(DIRECTOR_PROMPTS_PATH)
    dir_model     = os.environ.get('FILM_META_MODEL', 'gpt-5.4-mini')
    dir_model_cfg = MODELS.get(dir_model, {})
    extractor = DirectorMetaExtractor(
        task=tasks['director_profile'],
        model=dir_model,
        api_key=os.getenv('OPENAI_KEY'),
        cost_per_1m_input=dir_model_cfg.get('cost_per_1m_input'),
        cost_per_1m_output=dir_model_cfg.get('cost_per_1m_output'),
    )

    chunks = [df_directors.iloc[i:i + BATCH_SIZE] for i in range(0, len(df_directors), BATCH_SIZE)]
    prev_cost = 0.0
    for batch_num, chunk in enumerate(chunks, 1):
        print(f"\nDirector batch {batch_num}/{len(chunks)}  ({len(chunk)} directors)")
        results = await extractor.arun(
            df=chunk,
            name_col='director_name',
            max_concurrency=META_MAX_CONCURRENCY,
        )

        for name, data in results.items():
            if not data.get('_error'):
                data = {**data, 'director_name': name}
                checkpoint[str(name)] = data

        with open(DIRECTOR_CHECKPOINT_PATH, 'w') as f:
            json.dump(checkpoint, f)
        print(f"  Checkpoint saved: {len(checkpoint)} directors → {DIRECTOR_CHECKPOINT_PATH}")

        if extractor.token_usage:
            curr = extractor.token_usage.get('cost_usd', 0.0)
            delta = curr - prev_cost
            _total_cost += delta
            prev_cost = curr
            print(f"  Batch cost: ${delta:.4f}  |  Run total: ${_total_cost:.4f} / ${MAX_COST_USD:.2f}")

        if batch_num < len(chunks):
            await asyncio.sleep(BATCH_PAUSE_SECS)

    # Flush checkpoint → parquet
    df_new = pd.DataFrame(checkpoint.values())
    if df_new.empty:
        print("  Warning: no director results in checkpoint.")
        return 0.0

    if 'title' in df_new.columns:
        df_new = df_new.rename(columns={'title': 'director_name'})
    if '_error' in df_new.columns:
        df_new = df_new[df_new['_error'].isna()].drop(
            columns=[c for c in ['_error', '_error_message', '_raw_output', 'synopsis']
                     if c in df_new.columns], errors='ignore'
        )

    for _col in ['director_tier', 'primary_market']:
        if _col in df_new.columns:
            df_new[_col] = df_new[_col].str.lower().str.strip()

    DIRECTOR_ENRICHED_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DIRECTOR_ENRICHED_PATH.exists():
        df_existing = pd.read_parquet(DIRECTOR_ENRICHED_PATH)
        out = (
            pd.concat([df_existing, df_new], ignore_index=True)
            .drop_duplicates(subset='director_name', keep='first')
        )
    else:
        out = df_new

    out.to_parquet(DIRECTOR_ENRICHED_PATH, engine='pyarrow', index=False)
    print(f"\nDirector parquet → {DIRECTOR_ENRICHED_PATH}  ({len(out)} directors)")

    cost = 0.0
    if extractor.token_usage:
        u = extractor.token_usage
        cost = u.get('cost_usd', 0.0)
        print(f"Tokens — prompt: {u['prompt_tokens']:,}  completion: {u['completion_tokens']:,}  cost: ${cost:.4f}")

    del extractor
    gc.collect()
    return cost


# ── Film meta (web-grounded) enrichment ───────────────────────────────────────

# Distributors to skip — concerts, sports, film festivals, special events.
# Films from these distributors don't have GPT-extractable cast/budget/studio
# metadata in the standard sense; same skip list used by the deprecated TMDB
# pipeline (depreciated/tmdb/tmdb_studio_audit.py in the box office repo).
FILM_META_SKIP_DISTRIBUTORS = {
    # Film festivals
    "ZZ Japanese Film Festival", "ZZ JEWISH FILM FESTIVAL", "ZZ Russian Film Festival",
    "AU Sydney Science Fiction Film Festival", "AU KOREAN FILM FESTIVAL",
    "AU Taiwan Film Festival", "AU SciFi Film Festival",
    "ZZ Sydney Underground Film Festival Inc", "ZZ Iranian Film Festival Australia",
    "ZZ SOUTH AFRICAN FILM FESTIVAL", "ZZ UKRANIAN FILM FESTIVAL",
    "ZZ JIFF Distribution", "ZZ Flickerfest", "ZZ Gold Coast Film Fantastic Ltd",
    "ZZ SF3 - SMARTFONE FLICK FEST", "ZZ GOETHE INSTITUTE", "ZZ FOR FILMS SAKE",
    # Concerts / live events
    "AU Trafalgar Releasing Ltd", "AU Cinema Live", "AU PATHE LIVE",
    "ZZ THE WIGGLES INTERNATIONAL", "AU FATHOM EVENTS",
    # Sports
    "ZZ Fox Sports Venues", "ZZ BeIN SPORTS",
    "ZZ Queensland Cricket Association Ltd", "ZZ ESPN Australia Pty Ltd",
    # Other non-theatrical
    "AU IMAX THEATRES INTL", "ZZ Nickelodeon Australia Management",
    "ZZ CRUNCHYROLL PTY LTD", "ZZ SBS-ALTERNATE CONTENT",
}


async def enrich_film_meta(df_source: pd.DataFrame, film_lookup: pd.DataFrame | None = None) -> float:
    """Per-film extraction of studios / cast (top-5 billing order) / genres /
    budget / description via OpenAI Responses API + web_search. Replaces the
    deprecated TMDB pipeline.

    Drops concerts/festivals/sports (FILM_META_SKIP_DISTRIBUTORS) and re-releases
    (via ReReleaseFilter) before calling the API.

    Each output row also carries `evt_dstbtr` (EVT's authoritative AU
    distributor) so downstream consumers can override GPT's distribution entry
    with the local value if needed.

    Checkpoint format: { "<film_id>": { ...schema..., "_error": ... } }
    Output parquet: FILM_META_ENRICHED_PATH (one row per film_id).
    """
    global _total_cost
    import json

    if df_source.empty:
        print("No films to enrich for film_meta.")
        return 0.0

    # Need film_id + title + rel_at + dstbtr; supply title fallback if missing.
    needed = ['film_id', 'film_title', 'rel_at']
    missing = [c for c in needed if c not in df_source.columns]
    if missing:
        raise KeyError(f"enrich_film_meta requires columns: {missing}")
    if 'dstbtr' not in df_source.columns:
        df_source = df_source.assign(dstbtr=None)

    # ── Filter: skip-distributors ─────────────────────────────────────────────
    n0 = len(df_source)
    df_source = df_source[~df_source['dstbtr'].isin(FILM_META_SKIP_DISTRIBUTORS)].copy()
    n_skip = n0 - len(df_source)
    if n_skip:
        print(f"film_meta filter — skip distributors: -{n_skip} → {len(df_source)} remaining")

    # ── Filter: re-releases ───────────────────────────────────────────────────
    if film_lookup is not None and 'rel_at' in film_lookup.columns:
        try:
            from vendored.cinema_admits_models.re_release_filter import ReReleaseFilter
            rr = ReReleaseFilter()
            flagged = rr.flag(df_source.rename(columns={'film_title': 'film'}), film_lookup, title_col='film')
            n_rr = int(flagged['rerelease_flag'].sum())
            df_source = flagged[flagged['rerelease_flag'] == 0].rename(columns={'film': 'film_title'}).copy()
            if n_rr:
                print(f"film_meta filter — re-releases:       -{n_rr} → {len(df_source)} remaining")
        except Exception as e:
            print(f"film_meta filter — re-release skip ({e}); proceeding without re-release filter")

    FILM_META_CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if FILM_META_CHECKPOINT_PATH.exists():
        with open(FILM_META_CHECKPOINT_PATH) as f:
            checkpoint: dict = json.load(f)
        print(f"Loaded film_meta checkpoint: {len(checkpoint)} films already done")
    else:
        checkpoint = {}

    done_ids = {int(k) for k in checkpoint}
    df = df_source[~df_source['film_id'].astype(int).isin(done_ids)].copy()
    print(f"film_meta: {len(df_source)} films total — {len(done_ids)} in checkpoint — {len(df)} to extract")
    if df.empty:
        print("film_meta enrichment up to date.")
        return 0.0

    # Lookup map for EVT passthrough on save
    evt_passthrough = (
        df_source.set_index('film_id')[['dstbtr', 'rel_at']]
        .rename(columns={'dstbtr': 'evt_dstbtr', 'rel_at': 'evt_rel_at'})
        .to_dict('index')
    )

    tasks     = load_tasks_from_yaml(FILM_META_PROMPTS_PATH)
    task      = tasks['film_meta']
    model_cfg = MODELS.get(os.environ.get('FILM_META_MODEL', 'gpt-5.4-mini'), {})
    extractor = FilmMetaExtractor(
        task=task,
        api_key=os.getenv('OPENAI_KEY'),
        cost_per_1m_input=model_cfg.get('cost_per_1m_input'),
        cost_per_1m_output=model_cfg.get('cost_per_1m_output'),
    )

    chunks = [df.iloc[i:i + BATCH_SIZE] for i in range(0, len(df), BATCH_SIZE)]
    prev_cost = 0.0
    for batch_num, chunk in enumerate(chunks, 1):
        print(f"\nfilm_meta batch {batch_num}/{len(chunks)}  ({len(chunk)} films)")
        results = await extractor.arun(
            df=chunk,
            id_col='film_id',
            title_col='film_title',
            rel_at_col='rel_at',
            director_col='director',
            synopsis_col='synopsis',
            max_concurrency=META_MAX_CONCURRENCY,
        )

        # Title lookup for error log enrichment
        title_lookup = chunk.set_index('film_id')['film_title'].to_dict()

        batch_errors: dict[str, dict] = {}
        batch_success_ids: set[str] = set()
        for film_id, data in results.items():
            if not data.get('_error'):
                data['film_id'] = film_id
                # EVT passthrough — lets downstream override GPT's distribution
                # entry with the authoritative AU distributor we already have.
                passthrough = evt_passthrough.get(film_id, {})
                data['evt_dstbtr'] = str(passthrough.get('evt_dstbtr')) if pd.notna(passthrough.get('evt_dstbtr')) else None
                data['evt_rel_at'] = str(passthrough.get('evt_rel_at')) if pd.notna(passthrough.get('evt_rel_at')) else None
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
        print(f"  Checkpoint saved: {len(checkpoint)} films → {FILM_META_CHECKPOINT_PATH}")

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
            print(msg)

        if extractor.token_usage:
            curr = extractor.token_usage.get('cost_usd', 0.0)
            delta = curr - prev_cost
            _total_cost += delta
            prev_cost = curr
            print(f"  Batch cost: ${delta:.4f}  |  Run total: ${_total_cost:.4f} / ${MAX_COST_USD:.2f}")

        if batch_num < len(chunks):
            await asyncio.sleep(BATCH_PAUSE_SECS)

    # Flush checkpoint → parquet
    df_new = pd.DataFrame(checkpoint.values())
    if '_error' in df_new.columns:
        df_new = df_new[df_new['_error'].isna()].drop(
            columns=[c for c in ['_error', '_raw_output'] if c in df_new.columns],
            errors='ignore',
        )

    FILM_META_ENRICHED_PATH.parent.mkdir(parents=True, exist_ok=True)
    if FILM_META_ENRICHED_PATH.exists():
        df_existing = pd.read_parquet(FILM_META_ENRICHED_PATH)
        out = (
            pd.concat([df_existing, df_new], ignore_index=True)
            .drop_duplicates(subset='film_id', keep='last')
        )
    else:
        out = df_new

    out.to_parquet(FILM_META_ENRICHED_PATH, engine='pyarrow', index=False)
    print(f"\nfilm_meta parquet → {FILM_META_ENRICHED_PATH}  ({len(out)} films)")

    cost = 0.0
    if extractor.token_usage:
        u = extractor.token_usage
        cost = u.get('cost_usd', 0.0)
        print(f"Tokens — prompt: {u['prompt_tokens']:,}  completion: {u['completion_tokens']:,}  cost: ${cost:.4f}")

    del extractor
    gc.collect()
    return cost


# ── Run ───────────────────────────────────────────────────────────────────────
# _total_cost is updated per-batch inside each enrich function via `global`,
# so it survives KeyboardInterrupt — inspect from the REPL after a partial run.

if RUN_SYNOPSIS:
    asyncio.run(extract_synopses(df_films))
    print(f"\nCumulative cost so far: ${_total_cost:.4f} / ${MAX_COST_USD:.2f}")

if RUN_CAST:
    if _total_cost >= MAX_COST_USD:
        print(f"Cost cap reached (${_total_cost:.4f}) — skipping cast enrichment")
    else:
        _all_films = pd.concat(_parts, ignore_index=True).drop_duplicates('film_id')
        asyncio.run(enrich_cast(_all_films))
        print(f"\nCumulative cost so far: ${_total_cost:.4f} / ${MAX_COST_USD:.2f}")

if RUN_DIRECTOR:
    if _total_cost >= MAX_COST_USD:
        print(f"Cost cap reached (${_total_cost:.4f}) — skipping director enrichment")
    else:
        _all_films = pd.concat(_parts, ignore_index=True).drop_duplicates('film_id')
        asyncio.run(enrich_directors(_all_films))
        print(f"\nCumulative cost so far: ${_total_cost:.4f} / ${MAX_COST_USD:.2f}")

if RUN_META:
    if _total_cost >= MAX_COST_USD:
        print(f"Cost cap reached (${_total_cost:.4f}) — skipping film_meta enrichment")
    else:
        # df_films already has film_title joined from Snowflake.
        # Pull dstbtr from raw parquets — join on film_id.
        _dstbtr_lookup = None
        for _p in _raw_paths + _pred_paths:
            _df = pd.read_parquet(_p, columns=['film_id', 'dstbtr'])
            _dstbtr_lookup = _df if _dstbtr_lookup is None else pd.concat([_dstbtr_lookup, _df])
        _dstbtr_lookup = _dstbtr_lookup.drop_duplicates('film_id') if _dstbtr_lookup is not None else pd.DataFrame()

        _meta_df = df_films.copy()
        if 'dstbtr' not in _meta_df.columns and not _dstbtr_lookup.empty:
            _meta_df = _meta_df.merge(_dstbtr_lookup, on='film_id', how='left')

        # film_lookup with rel_at + dstbtr + director + film for ReReleaseFilter
        try:
            _flu_full = pd.read_sql(films_sql.SQL_FILM_DETAILS, _sb.engine)
            _flu_full = _flu_full.rename(columns={'film_title': 'film'})
            _flu_full['film_id'] = _flu_full['film_id'].astype(int)
        except Exception as _e:
            print(f"film_lookup unavailable for re-release filter ({_e}) — proceeding without")
            _flu_full = None

        asyncio.run(enrich_film_meta(_meta_df, film_lookup=_flu_full))
        print(f"\nCumulative cost so far: ${_total_cost:.4f} / ${MAX_COST_USD:.2f}")

print(f"\nTotal spend: ${_total_cost:.4f}")
if _total_cost > MAX_COST_USD:
    print(f"WARNING: exceeded MAX_COST_USD cap of ${MAX_COST_USD:.2f}")

if RUN_ENCODE:
    # cast/director encoders moved to cinema_admits_models/build_data/
    # (encode_cast_features.py, encode_director_features.py) on 2026-05-19.
    # encode_synopsis.py remains in depreciated/encoding/ — superseded by
    # cinema_admits_models/build_data/encode_llm_features.py. Run those scripts
    # from the box office model project instead.
    raise RuntimeError(
        "RUN_ENCODE is no longer supported in film_synopsis_meta. "
        "Run cinema_admits_models/build_data/encode_llm_features.py, "
        "encode_cast_features.py, and encode_director_features.py instead."
    )
