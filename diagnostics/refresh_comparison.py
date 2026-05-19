"""
refresh_comparison.py
---------------------
Validate the new mini+search cast / director / film_meta extractors on a
100-film sample before full rollout. Writes everything to a timestamped
folder under data/_compare/ — production parquets are NOT touched.

Sample:
  30 from prediction parquets (most-recent rel_at across all 8 prediction dirs)
  70 from test parquets       (most-recent rel_at — currently 2026-01-30 to 2026-04-30)

Outputs (data/_compare/<ts>/):
  cast_enriched_new.parquet         mini+search extraction
  cast_enriched_old.parquet         existing production parquet sliced to matching actors
  director_enriched_new.parquet     "
  director_enriched_old.parquet
  film_meta_enriched_new.parquet
  film_meta_enriched_old.parquet
  comparison.xlsx                   side-by-side rows per entity
  *_progress.json                   checkpoints (live in _compare folder, not production)

Cost: ~$15-20 / wall-clock ~20-30 min at META_MAX_CONCURRENCY=4.

Usage:
  python refresh_comparison.py
"""

import asyncio
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

import nest_asyncio
nest_asyncio.apply()

import pandas as pd
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, '/Users/jonathanchapman/Documents/git/evt_back_up/base')

from film_meta_extractor import (
    FilmMetaExtractor, ActorMetaExtractor, DirectorMetaExtractor,
)
from load_prompts import load_tasks_from_yaml
from models import MODELS
from config import (
    DATA_DIR, CAST_ENRICHED_PATH, DIRECTOR_ENRICHED_PATH, FILM_META_ENRICHED_PATH,
    SF_WAREHOUSE, SF_DATABASE, SF_SCHEMA, SF_RSA_KEY,
)
from films import sql as films_sql

# ── Config ────────────────────────────────────────────────────────────────────

N_FUTURE  = 30
N_RECENT  = 0    # cast + director already validated on the first run; default to future-only for film_meta diagnostic
MODEL_NAME = os.environ.get('FILM_META_MODEL', 'gpt-5.4-mini')
MAX_CONCURRENCY      = 4   # actors + directors
FILM_META_CONCURRENCY = 2  # one at a time — gives us clean error signal before scaling back up
INTER_BATCH_SLEEP    = 10

# Which paths to run. cast/director validated on the 20260519_120824 run; default off.
RUN_ACTOR     = False
RUN_DIRECTOR  = False
RUN_FILM_META = True

# High-stakes / TMDB-misfuzzy films — always added to the sample (deduped).
# Drives the next round of mini+search-vs-TMDB comparison.
PROBLEM_FILM_IDS = [
    60335,  # THE MAGIC FARAWAY TREE — 0.07× under-pred; Blyton novel, family adventure
    60154,  # SONG SUNG BLUE       — 0.27×; Neil Diamond biopic (Hugh Jackman, Kate Hudson)
    60560,  # MARTY SUPREME        — 0.28×; Timothée Chalamet ping-pong biopic, A24
    60680,  # CRIME 101            — 0.30×; Mahershala Ali / Chris Hemsworth, Sony
    61145,  # IRON LUNG            — 0.22×; horror, Rialto
    59532,  # REMINDERS OF HIM     — 0.24×; Colleen Hoover novel romance, Universal
    61032,  # THE DRAMA            — 0.26×; Zendaya / Pattinson drama, VVS
    59967,  # SEND HELP            — 0.22×; horror, Sam Richardson, Disney
    59531,  # SCREAM 7             — calibrated 0.81× reference
    57343,  # MICHAEL              — under-predicted vs Gower
    60261,  # THE DEVIL WEARS PRADA 2
    59538,  # THE ODYSSEY
    60092,  # SPIDER-MAN: BRAND NEW DAY
]

TS      = datetime.now().strftime('%Y%m%d_%H%M%S')
OUT_DIR = DATA_DIR / '_compare' / TS
OUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Output dir: {OUT_DIR}")

CAST_CP     = OUT_DIR / 'cast_progress.json'
DIRECTOR_CP = OUT_DIR / 'director_progress.json'
FILM_META_CP = OUT_DIR / 'film_meta_progress.json'

CAST_PROMPTS_PATH      = str(_REPO / 'prompts' / 'cast_prompts.yaml')
DIRECTOR_PROMPTS_PATH  = str(_REPO / 'prompts' / 'director_prompts.yaml')
FILM_META_PROMPTS_PATH = str(_REPO / 'prompts' / 'film_meta_prompts.yaml')

_AND_RX = re.compile(r'^AND\s+', re.IGNORECASE)
def _clean_actor(raw: str) -> list[str]:
    items = json.loads(raw) if str(raw).startswith('[') else str(raw).split('|')
    out = []
    for a in items:
        n = _AND_RX.sub('', str(a).strip()).upper().strip()
        if n and n not in ('AND', 'N/A') and len(n) > 1:
            out.append(n)
    return out


# ── Sample selection ──────────────────────────────────────────────────────────

import glob

_test_paths = sorted(glob.glob(str(DATA_DIR / 'raw_from_snowflake' / '*' / 'test' / 'test_raw_ds.parquet')))
_pred_paths = sorted(glob.glob(str(DATA_DIR / 'prediction_from_snowflake' / '*' / 'prediction_raw.parquet')))

# Future films (prediction parquets) — no film_title column, use Snowflake later
_pred_cols = ['film_id', 'rel_at', 'dstbtr', 'director', 'actor_list', 'synopsis']
_pred_df = (
    pd.concat([pd.read_parquet(p, columns=_pred_cols) for p in _pred_paths], ignore_index=True)
    .drop_duplicates('film_id')
)
_pred_df['rel_at'] = pd.to_datetime(_pred_df['rel_at'], utc=True)
pred_sample = _pred_df.sort_values('rel_at', ascending=False).head(N_FUTURE).copy()

# Recent test films (have film_title in `film` column)
_test_cols = ['film_id', 'film', 'rel_at', 'dstbtr', 'director', 'actor_list', 'synopsis']
_test_df = (
    pd.concat([pd.read_parquet(p, columns=_test_cols) for p in _test_paths], ignore_index=True)
    .drop_duplicates('film_id')
    .rename(columns={'film': 'film_title'})
)
_test_df['rel_at'] = pd.to_datetime(_test_df['rel_at'], utc=True)
test_sample = _test_df.sort_values('rel_at', ascending=False).head(N_RECENT).copy()

# Title lookup from Snowflake for prediction films
try:
    from base_snowflake import SnowFlakeBase
    _sb = SnowFlakeBase(warehouse=SF_WAREHOUSE, database=SF_DATABASE, schema=SF_SCHEMA)
    _sb.create_snowflake_connection(SF_RSA_KEY)
    _titles = pd.read_sql(films_sql.SQL_FILM_DETAILS, _sb.engine)[['film_id', 'film_title']]
    _titles['film_id'] = _titles['film_id'].astype(int)
    pred_sample = pred_sample.merge(_titles, on='film_id', how='left')
    print(f"Snowflake titles joined for {pred_sample['film_title'].notna().sum()}/{len(pred_sample)} pred films")
except Exception as _e:
    print(f"Snowflake unavailable ({_e}) — using film_id as title fallback for pred films")
    pred_sample['film_title'] = pred_sample['film_id'].astype(str)

sample = pd.concat([pred_sample, test_sample], ignore_index=True)
sample['cohort'] = ['future'] * len(pred_sample) + ['recent'] * len(test_sample)

# Append PROBLEM_FILM_IDS — fetch their data from any parquet that has them.
if PROBLEM_FILM_IDS:
    _all_films_cols = _test_cols
    _all_paths = sorted(glob.glob(str(DATA_DIR / 'raw_from_snowflake' / '*' / '*' / '*_raw_ds.parquet')))
    _all_pred_paths = _pred_paths
    _problem_parts = []
    for p in _all_paths:
        try:
            df = pd.read_parquet(p, columns=_all_films_cols)
            df = df[df['film_id'].isin(PROBLEM_FILM_IDS)].drop_duplicates('film_id')
            if not df.empty:
                _problem_parts.append(df)
        except Exception:
            pass
    for p in _all_pred_paths:
        try:
            df = pd.read_parquet(p, columns=_pred_cols)
            df = df[df['film_id'].isin(PROBLEM_FILM_IDS)].drop_duplicates('film_id')
            if not df.empty:
                _problem_parts.append(df)
        except Exception:
            pass
    if _problem_parts:
        _problem_df = pd.concat(_problem_parts, ignore_index=True).drop_duplicates('film_id')
        _problem_df['rel_at'] = pd.to_datetime(_problem_df['rel_at'], utc=True)
        if 'film' in _problem_df.columns:
            _problem_df = _problem_df.rename(columns={'film': 'film_title'})
        if 'film_title' not in _problem_df.columns:
            _problem_df['film_title'] = _problem_df['film_id'].astype(str)
        # Title backfill from Snowflake when missing
        if _problem_df['film_title'].isna().any():
            try:
                if '_titles' in dir():
                    _problem_df = _problem_df.merge(_titles, on='film_id', how='left', suffixes=('', '_sf'))
                    _problem_df['film_title'] = _problem_df['film_title'].fillna(_problem_df.get('film_title_sf'))
            except Exception:
                pass
        _problem_df['cohort'] = 'problem'
        # Dedupe against the existing sample (keep problem rows on conflict)
        sample = pd.concat([sample[~sample['film_id'].isin(_problem_df['film_id'])], _problem_df], ignore_index=True)
        missing = [fid for fid in PROBLEM_FILM_IDS if fid not in sample['film_id'].values]
        print(f"Added {len(_problem_df)} PROBLEM_FILM_IDS; missing from parquets: {missing}")
    else:
        print(f"WARNING: none of PROBLEM_FILM_IDS found in raw parquets")

print(f"\nSample: {len(sample)} films  (cohorts: {sample['cohort'].value_counts().to_dict()})")

# Unique cast + director sets
all_actors: set[str] = set()
for v in sample['actor_list'].dropna():
    all_actors.update(_clean_actor(v))

all_directors: set[str] = set()
for v in sample['director'].dropna():
    for d in str(v).split(','):
        d = d.strip()
        if d:
            all_directors.add(d)

print(f"Unique actors:   {len(all_actors):,}")
print(f"Unique directors:{len(all_directors):,}")
print(f"Estimated cost:  ~${(len(sample) + len(all_actors) + len(all_directors)) * 0.02:.2f}\n")


# ── Extractors ────────────────────────────────────────────────────────────────

api_key   = os.getenv('OPENAI_KEY')
model_cfg = MODELS.get(MODEL_NAME, {})
cost_kw   = dict(
    cost_per_1m_input=model_cfg.get('cost_per_1m_input'),
    cost_per_1m_output=model_cfg.get('cost_per_1m_output'),
    api_key=api_key,
    model=MODEL_NAME,
)

cast_tasks     = load_tasks_from_yaml(CAST_PROMPTS_PATH)
director_tasks = load_tasks_from_yaml(DIRECTOR_PROMPTS_PATH)
film_tasks     = load_tasks_from_yaml(FILM_META_PROMPTS_PATH)

cast_extractor     = ActorMetaExtractor(task=cast_tasks['actor_profile'], **cost_kw)
director_extractor = DirectorMetaExtractor(task=director_tasks['director_profile'], **cost_kw)
film_extractor     = FilmMetaExtractor(task=film_tasks['film_meta'], **cost_kw)


# ── Run + save (no checkpoint reuse — fresh extractions for clean comparison) ─

def _summarise_errors(label: str, results: dict) -> None:
    """Bucket errors by type so we can spot 429s / parse failures / timeouts."""
    from collections import Counter
    n_total = len(results)
    n_err   = sum(1 for d in results.values() if d.get('_error'))
    if n_err == 0:
        print(f"      {label}: {n_total} ok, 0 errors")
        return
    by_type = Counter()
    samples = {}
    for k, d in results.items():
        e = d.get('_error')
        if not e:
            continue
        head = str(e).split(':', 1)[0]
        by_type[head] += 1
        if head not in samples:
            samples[head] = (k, str(e)[:200])
    print(f"      {label}: {n_total - n_err} ok, {n_err} errors ({100*n_err/n_total:.0f}%)")
    for etype, count in by_type.most_common():
        ex_key, ex_msg = samples[etype]
        print(f"        {etype:<28} ×{count:<4}  e.g. {ex_key}: {ex_msg}")


def _save_raw_results(label: str, results: dict) -> None:
    """Persist the FULL results dict (successes + errors) so we can diagnose
    failures without re-running. Errors are dropped from the parquet later."""
    path = OUT_DIR / f'{label}_raw_results.json'
    serialisable = {str(k): {kk: (str(vv) if hasattr(vv, 'isoformat') else vv) for kk, vv in d.items()}
                     for k, d in results.items()}
    with open(path, 'w') as f:
        json.dump(serialisable, f, indent=2, default=str)
    print(f"      raw results (incl. errors) → {path}")


async def run_all() -> dict:
    df_films = sample[['film_id', 'film_title', 'rel_at', 'dstbtr']].copy()

    actor_results: dict    = {}
    director_results: dict = {}
    film_results: dict     = {}

    if RUN_ACTOR:
        df_actors = pd.DataFrame({'actor_name': sorted(all_actors)})
        print(f"[actor] extraction ({len(df_actors)} actors, concurrency {MAX_CONCURRENCY}) ...")
        actor_results = await cast_extractor.arun(df_actors, name_col='actor_name', max_concurrency=MAX_CONCURRENCY)
        print(f"        done  cost ${cast_extractor.token_usage['cost_usd']:.4f}")
        _save_raw_results('actor', actor_results)
        _summarise_errors('actors', actor_results)
        await asyncio.sleep(INTER_BATCH_SLEEP)

    if RUN_DIRECTOR:
        df_directors = pd.DataFrame({'director_name': sorted(all_directors)})
        print(f"\n[director] extraction ({len(df_directors)} directors, concurrency {MAX_CONCURRENCY}) ...")
        director_results = await director_extractor.arun(df_directors, name_col='director_name', max_concurrency=MAX_CONCURRENCY)
        print(f"        done  cost ${director_extractor.token_usage['cost_usd']:.4f}")
        _save_raw_results('director', director_results)
        _summarise_errors('directors', director_results)
        await asyncio.sleep(INTER_BATCH_SLEEP)

    if RUN_FILM_META:
        print(f"\n[film] extraction ({len(df_films)} films, concurrency {FILM_META_CONCURRENCY}) ...")
        film_results = await film_extractor.arun(
            df_films, id_col='film_id', title_col='film_title',
            rel_at_col='rel_at', dstbtr_col='dstbtr',
            max_concurrency=FILM_META_CONCURRENCY,
        )
        print(f"        done  cost ${film_extractor.token_usage['cost_usd']:.4f}")
        _save_raw_results('film', film_results)
        _summarise_errors('films', film_results)

    return {
        'actor':    actor_results,
        'director': director_results,
        'film':     film_results,
    }


def _results_to_df(results: dict, key_name: str) -> pd.DataFrame:
    rows = []
    for k, d in results.items():
        row = {**d, key_name: k}
        rows.append(row)
    df = pd.DataFrame(rows)
    if '_error' in df.columns:
        df = df[df['_error'].isna()].drop(columns=['_error', '_raw_output'], errors='ignore')
    return df


def save_new(results: dict):
    actor_new    = _results_to_df(results['actor'], 'actor_name')      if results.get('actor')    else pd.DataFrame()
    director_new = _results_to_df(results['director'], 'director_name') if results.get('director') else pd.DataFrame()
    film_new     = _results_to_df(results['film'], 'film_id')          if results.get('film')     else pd.DataFrame()

    if not actor_new.empty:
        actor_new.to_parquet(OUT_DIR / 'cast_enriched_new.parquet', index=False)
    if not director_new.empty:
        director_new.to_parquet(OUT_DIR / 'director_enriched_new.parquet', index=False)
    if not film_new.empty:
        film_new.to_parquet(OUT_DIR / 'film_meta_enriched_new.parquet', index=False)
    print(f"\nWrote new parquets (non-empty only) → {OUT_DIR}")
    return actor_new, director_new, film_new


def save_old() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Slice production parquets to the entities in the sample."""
    def _load(p: Path) -> pd.DataFrame:
        return pd.read_parquet(p) if p.exists() else pd.DataFrame()

    actor_old = director_old = film_old = pd.DataFrame()

    if RUN_ACTOR:
        cast_full = _load(CAST_ENRICHED_PATH)
        if not cast_full.empty:
            actor_old = cast_full[cast_full['actor_name'].str.upper().isin(all_actors)].copy()
            actor_old.to_parquet(OUT_DIR / 'cast_enriched_old.parquet', index=False)
    if RUN_DIRECTOR:
        director_full = _load(DIRECTOR_ENRICHED_PATH)
        if not director_full.empty:
            director_old = director_full[director_full['director_name'].isin(all_directors)].copy()
            director_old.to_parquet(OUT_DIR / 'director_enriched_old.parquet', index=False)
    if RUN_FILM_META:
        film_full = _load(FILM_META_ENRICHED_PATH)
        if not film_full.empty:
            film_old = film_full[film_full['film_id'].astype(int).isin(sample['film_id'].astype(int))].copy()
            film_old.to_parquet(OUT_DIR / 'film_meta_enriched_old.parquet', index=False)

    print(f"Wrote old parquets:  actors {len(actor_old)}, directors {len(director_old)}, films {len(film_old)}")
    return actor_old, director_old, film_old


# ── Comparison Excel ──────────────────────────────────────────────────────────

def _agree(a, b) -> bool:
    if pd.isna(a) and pd.isna(b): return True
    if pd.isna(a) or pd.isna(b):  return False
    return str(a).strip().lower() == str(b).strip().lower()


def build_excel(actor_new, actor_old, director_new, director_old, film_new, film_old):
    out_path = OUT_DIR / 'comparison.xlsx'

    actor_fields = ['fame_source', 'fame_tier', 'primary_market', 'age_range']
    director_fields = ['director_tier', 'primary_market']
    film_fields = ['ip_strength', 'adaptation_type', 'budget_usd', 'budget_source', 'director', 'release_date']

    def _side_by_side(new_df, old_df, key, fields):
        if old_df.empty:
            return new_df.copy().assign(**{f'{f}__old': pd.NA for f in fields})
        old_df = old_df.copy()
        if key == 'actor_name':
            old_df['actor_name'] = old_df['actor_name'].str.upper()
        old_df = old_df.drop_duplicates(key).set_index(key)
        new_df = new_df.copy()
        if key == 'actor_name':
            new_df['actor_name'] = new_df['actor_name'].str.upper()
        rows = []
        for _, r in new_df.iterrows():
            k = r[key]
            o = old_df.loc[k].to_dict() if k in old_df.index else {}
            row = {key: k}
            for f in fields:
                row[f'{f}__new'] = r.get(f)
                row[f'{f}__old'] = o.get(f)
                row[f'{f}__agree'] = _agree(r.get(f), o.get(f))
            rows.append(row)
        return pd.DataFrame(rows)

    sheets = []
    if not actor_new.empty:
        sheets.append(('actor',    _side_by_side(actor_new,    actor_old,    'actor_name',    actor_fields)))
    if not director_new.empty:
        sheets.append(('director', _side_by_side(director_new, director_old, 'director_name', director_fields)))
    if not film_new.empty:
        sheets.append(('film',     _side_by_side(film_new,     film_old,     'film_id',       film_fields)))

    if not sheets:
        print("\nNo new rows to compare — Excel not written.")
        return

    with pd.ExcelWriter(out_path, engine='openpyxl') as xl:
        for sheet_name, df in sheets:
            df.to_excel(xl, sheet_name=sheet_name, index=False)
    print(f"\nComparison Excel → {out_path}")

    actor_cmp    = next((df for n, df in sheets if n == 'actor'),    pd.DataFrame())
    director_cmp = next((df for n, df in sheets if n == 'director'), pd.DataFrame())
    film_cmp     = next((df for n, df in sheets if n == 'film'),     pd.DataFrame())

    print("\n── Summary ─────────────────────────────────────────────────────")
    for name, df, fields in [('actor', actor_cmp, actor_fields),
                              ('director', director_cmp, director_fields),
                              ('film', film_cmp, film_fields)]:
        if df.empty:
            print(f"  {name}: no new rows")
            continue
        n = len(df)
        print(f"  {name}: {n} new rows")
        for f in fields:
            col = f'{f}__agree'
            if col in df.columns:
                matched = df[df[f'{f}__old'].notna()]
                if len(matched) > 0:
                    pct = matched[col].mean() * 100
                    print(f"    {f:<20}  agree {pct:5.1f}%  (n={len(matched)})")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    results = asyncio.run(run_all())
    actor_new, director_new, film_new = save_new(results)
    actor_old, director_old, film_old = save_old()
    build_excel(actor_new, actor_old, director_new, director_old, film_new, film_old)
    print(f"\nTotal cost: ${cast_extractor.token_usage['cost_usd'] + director_extractor.token_usage['cost_usd'] + film_extractor.token_usage['cost_usd']:.4f}")
