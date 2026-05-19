"""
compare_film_meta_search.py
---------------------------
A/B test the film_meta extractor with web_search ON vs OFF on the same 20-film
sample. Model is held constant so the diff isolates the value of web grounding.

Sample mix:
  10 historic (5 from 2022, 5 from 2024) — model should know these from training
  10 recent   (Feb-Apr 2026 test-set films) — beyond knowledge cutoff for most models

Output:
  data/film_meta/compare_search/comparison_<timestamp>.xlsx
      One row per film. Side-by-side columns: <field>__search vs <field>__nosearch.
  Console: cost, wall-clock, budget hit-rate, per-field agreement summary.

Usage:
  FILM_META_MODEL=gpt-5.4-mini python compare_film_meta_search.py
"""
import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

import nest_asyncio
nest_asyncio.apply()

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from film_meta_extractor import FilmMetaExtractor
from load_prompts import load_tasks_from_yaml
from models import MODELS
from config import DATA_DIR

# ── Sample (20 films) ─────────────────────────────────────────────────────────

HISTORIC_IDS = [40671, 32971, 47138, 46677, 47064,    # 2022
                55986, 57353, 57360, 56584, 55702]    # 2024
RECENT_IDS   = [60592, 60336, 57343, 59534, 60261,    # 2026 H1
                59041, 59042, 60335, 61212, 59531]
SAMPLE_IDS = HISTORIC_IDS + RECENT_IDS

TRAIN_PARQUET = DATA_DIR / 'raw_from_snowflake' / '20260129' / 'train' / 'train_raw_ds.parquet'
TEST_PARQUET  = DATA_DIR / 'raw_from_snowflake' / '20260129' / 'test'  / 'test_raw_ds.parquet'

MODEL_NAME = os.environ.get('FILM_META_MODEL', 'gpt-5.4-mini')
MAX_CONCURRENCY = 4
PROMPTS_PATH = str(_REPO / 'prompts' / 'film_meta_prompts.yaml')

OUTPUT_DIR  = DATA_DIR / 'film_meta' / 'compare_search'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Load sample ───────────────────────────────────────────────────────────────

def _load_sample() -> pd.DataFrame:
    parts = []
    for p in [TRAIN_PARQUET, TEST_PARQUET]:
        df = pd.read_parquet(p, columns=['film_id', 'film', 'rel_at', 'dstbtr'])
        df['rel_at'] = pd.to_datetime(df['rel_at'], utc=True)
        parts.append(df)
    df = pd.concat(parts).drop_duplicates('film_id')
    df = df[df['film_id'].isin(SAMPLE_IDS)].copy()
    df = df.rename(columns={'film': 'film_title'})
    # Keep the desired order — historic first, then recent
    order = {fid: i for i, fid in enumerate(SAMPLE_IDS)}
    df['_ord'] = df['film_id'].map(order)
    df = df.sort_values('_ord').drop(columns='_ord').reset_index(drop=True)
    df['cohort'] = df['film_id'].apply(lambda x: 'historic' if x in HISTORIC_IDS else 'recent')
    return df


# ── Run one mode ──────────────────────────────────────────────────────────────

async def _run_mode(df: pd.DataFrame, use_web_search: bool) -> tuple[dict[int, dict], float, float]:
    tasks      = load_tasks_from_yaml(PROMPTS_PATH)
    task       = tasks['film_meta']
    model_cfg  = MODELS.get(MODEL_NAME, {})
    extractor  = FilmMetaExtractor(
        task=task,
        model=MODEL_NAME,
        api_key=os.getenv('OPENAI_KEY'),
        cost_per_1m_input=model_cfg.get('cost_per_1m_input'),
        cost_per_1m_output=model_cfg.get('cost_per_1m_output'),
        use_web_search=use_web_search,
    )
    t0 = time.time()
    results = await extractor.arun(
        df=df, id_col='film_id', title_col='film_title',
        rel_at_col='rel_at', dstbtr_col='dstbtr',
        max_concurrency=MAX_CONCURRENCY,
    )
    elapsed = time.time() - t0
    cost = extractor.token_usage.get('cost_usd', 0.0)
    return results, elapsed, cost


# ── Comparison output ─────────────────────────────────────────────────────────

SCALAR_FIELDS = ['title', 'release_date', 'director', 'description',
                 'budget_usd', 'budget_source']
LIST_FIELDS   = ['writers', 'genres']
STRUCT_FIELDS = ['studios', 'cast']   # list-of-dict

def _fmt_struct(v) -> str:
    if not v: return ''
    if isinstance(v, list):
        if v and isinstance(v[0], dict):
            return ' | '.join(d.get('name') or d.get('actor') or '' for d in v)
        return ' | '.join(str(x) for x in v)
    return str(v)

def _agree_scalar(a, b) -> bool:
    if a is None and b is None: return True
    if a is None or b is None:  return False
    return str(a).strip().lower() == str(b).strip().lower()

def _agree_list(a, b) -> float:
    """Jaccard on lowercased string set."""
    sa = {str(x).strip().lower() for x in (a or [])}
    sb = {str(x).strip().lower() for x in (b or [])}
    if not sa and not sb: return 1.0
    if not sa or  not sb: return 0.0
    return len(sa & sb) / len(sa | sb)

def _agree_struct_names(a, b) -> float:
    """Jaccard on the 'name'/'actor' keys."""
    def _names(items):
        out = set()
        for d in (items or []):
            if isinstance(d, dict):
                v = d.get('name') or d.get('actor')
                if v: out.add(str(v).strip().lower())
        return out
    sa, sb = _names(a), _names(b)
    if not sa and not sb: return 1.0
    if not sa or  not sb: return 0.0
    return len(sa & sb) / len(sa | sb)


def _build_excel(df_sample: pd.DataFrame, res_search: dict, res_nosearch: dict, ts: str):
    rows = []
    for _, f in df_sample.iterrows():
        fid = int(f['film_id'])
        s   = res_search.get(fid, {})
        n   = res_nosearch.get(fid, {})
        row = {
            'film_id':    fid,
            'film_title': f['film_title'],
            'cohort':     f['cohort'],
            'rel_at':     f['rel_at'].strftime('%Y-%m-%d'),
            '_err_search':   s.get('_error'),
            '_err_nosearch': n.get('_error'),
        }
        for fld in SCALAR_FIELDS:
            row[f'{fld}__search']   = s.get(fld)
            row[f'{fld}__nosearch'] = n.get(fld)
            row[f'{fld}__agree']    = _agree_scalar(s.get(fld), n.get(fld))
        for fld in LIST_FIELDS:
            row[f'{fld}__search']   = ', '.join(s.get(fld) or [])
            row[f'{fld}__nosearch'] = ', '.join(n.get(fld) or [])
            row[f'{fld}__jaccard']  = round(_agree_list(s.get(fld), n.get(fld)), 2)
        for fld in STRUCT_FIELDS:
            row[f'{fld}__search']   = _fmt_struct(s.get(fld))
            row[f'{fld}__nosearch'] = _fmt_struct(n.get(fld))
            row[f'{fld}__jaccard']  = round(_agree_struct_names(s.get(fld), n.get(fld)), 2)
        rows.append(row)

    df_out = pd.DataFrame(rows)
    out_path = OUTPUT_DIR / f'comparison_{ts}.xlsx'
    with pd.ExcelWriter(out_path, engine='openpyxl') as xl:
        df_out.to_excel(xl, sheet_name='comparison', index=False)
        # Raw JSON dumps (debugging)
        pd.DataFrame([
            {'film_id': fid, 'mode': 'search',   'payload': json.dumps(res_search.get(fid, {}),   default=str)}
            for fid in SAMPLE_IDS
        ] + [
            {'film_id': fid, 'mode': 'nosearch', 'payload': json.dumps(res_nosearch.get(fid, {}), default=str)}
            for fid in SAMPLE_IDS
        ]).to_excel(xl, sheet_name='raw', index=False)
    return out_path, df_out


def _summarise(df_out: pd.DataFrame, t_search: float, t_nosearch: float, c_search: float, c_nosearch: float):
    n = len(df_out)
    print("\n" + "=" * 70)
    print(f"COMPARISON SUMMARY  ({n} films, model={MODEL_NAME})")
    print("=" * 70)
    print(f"\nWall-clock:  search={t_search:6.1f}s   nosearch={t_nosearch:6.1f}s   ratio={t_search/max(t_nosearch,1):.1f}x")
    print(f"Cost (USD):  search=${c_search:.4f}   nosearch=${c_nosearch:.4f}")

    err_s = df_out['_err_search'].notna().sum()
    err_n = df_out['_err_nosearch'].notna().sum()
    print(f"Errors:      search={err_s}   nosearch={err_n}")

    print("\nPer-field agreement (search ↔ nosearch):")
    for fld in SCALAR_FIELDS:
        agree = df_out[f'{fld}__agree'].mean()
        print(f"  {fld:<16}  agree {agree*100:5.1f}%")
    for fld in LIST_FIELDS + STRUCT_FIELDS:
        j = df_out[f'{fld}__jaccard'].mean()
        print(f"  {fld:<16}  jaccard {j:.2f}")

    print("\nBudget hit-rate (non-null budget_usd):")
    for cohort in ['historic', 'recent']:
        sub = df_out[df_out['cohort'] == cohort]
        hit_s = sub['budget_usd__search'].notna().sum()
        hit_n = sub['budget_usd__nosearch'].notna().sum()
        print(f"  {cohort:<8}  search {hit_s}/{len(sub)}   nosearch {hit_n}/{len(sub)}")


# ── Main ──────────────────────────────────────────────────────────────────────

async def _main():
    df = _load_sample()
    print(f"Loaded {len(df)} films from sample (expected {len(SAMPLE_IDS)})")
    if len(df) != len(SAMPLE_IDS):
        missing = set(SAMPLE_IDS) - set(df['film_id'])
        print(f"  WARNING: {len(missing)} film_ids missing from parquets: {missing}")
    print(f"\nRunning film_meta on {len(df)} films  ×  2 modes  (mini = {MODEL_NAME})")

    print("\n[1/2] WITH web_search ...")
    res_search,   t_search,   c_search   = await _run_mode(df, use_web_search=True)
    print(f"      done in {t_search:.1f}s  cost ${c_search:.4f}")

    print("\n[2/2] WITHOUT web_search ...")
    res_nosearch, t_nosearch, c_nosearch = await _run_mode(df, use_web_search=False)
    print(f"      done in {t_nosearch:.1f}s  cost ${c_nosearch:.4f}")

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path, df_out = _build_excel(df, res_search, res_nosearch, ts)
    print(f"\nWrote → {out_path}")
    _summarise(df_out, t_search, t_nosearch, c_search, c_nosearch)


if __name__ == '__main__':
    asyncio.run(_main())
