"""
inspect_film_meta.py — eyeball + summarise a film_meta refresh-comparison run.
Reads film_meta_enriched_new.parquet from the latest data/_compare/<ts>/ dir.

Usage:
    python inspect_film_meta.py                  # summary table
    python inspect_film_meta.py --detail         # per-film cards
    python inspect_film_meta.py --film-id 60592  # drill into one
    python inspect_film_meta.py --ts 20260519_124543   # pin a run
    python inspect_film_meta.py --no-budget      # only show films missing budget
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from config import DATA_DIR

ap = argparse.ArgumentParser()
ap.add_argument('--ts',       type=str,  help='Specific run timestamp')
ap.add_argument('--detail',   action='store_true', help='Per-film cards (slower to read but full picture)')
ap.add_argument('--film-id',  type=int,  help='Drill into a single film_id')
ap.add_argument('--no-budget', action='store_true', help='Only show films missing budget_usd')
ap.add_argument('--vs-tmdb',  action='store_true', help='Side-by-side vs old TMDB studio_mapping data')
args = ap.parse_args()

TMDB_MAPPING_PATH = DATA_DIR / 'tmdb' / 'studio_mapping_20260129.parquet'

COMPARE_ROOT = DATA_DIR / '_compare'
if args.ts:
    run_dir = COMPARE_ROOT / args.ts
else:
    runs = sorted([d for d in COMPARE_ROOT.iterdir() if d.is_dir()])
    if not runs:
        sys.exit(f"No runs found under {COMPARE_ROOT}")
    run_dir = runs[-1]

path = run_dir / 'film_meta_enriched_new.parquet'
print(f"Reading: {path}\n")
if not path.exists():
    sys.exit(f"film_meta_enriched_new.parquet not found in {run_dir}")

df = pd.read_parquet(path)
print(f"Films: {len(df)}\n")


# ── Per-film card ─────────────────────────────────────────────────────────────

def _fmt_list_of_dicts(items, key1, key2):
    if not isinstance(items, (list, tuple)) or len(items) == 0:
        return '(none)'
    rows = []
    for d in items[:8]:
        if not isinstance(d, dict): continue
        a = d.get(key1, '?')
        b = d.get(key2, '?')
        rows.append(f"    {a} — {b}")
    return '\n'.join(rows) if rows else '(none)'


def _print_card(row):
    fid = int(row['film_id'])
    print(f"── {row.get('title') or row.get('film_title') or '(no title)'} (film_id={fid}) ─────────────")
    print(f"  release_date:    {row.get('release_date')}")
    print(f"  evt_rel_at:      {row.get('evt_rel_at')}")
    print(f"  director:        {row.get('director')}")
    print(f"  writers:         {row.get('writers')}")
    print(f"  genres:          {row.get('genres')}")
    print(f"  ip_strength:     {row.get('ip_strength')}")
    print(f"  adaptation_type: {row.get('adaptation_type')}")
    budget = row.get('budget_usd')
    if budget is None or pd.isna(budget):
        print(f"  budget_usd:      (null)")
    else:
        print(f"  budget_usd:      ${int(budget):,}")
    print(f"  budget_source:   {row.get('budget_source')}")
    print(f"  evt_dstbtr:      {row.get('evt_dstbtr')}")
    print(f"  studios:")
    print(_fmt_list_of_dicts(row.get('studios'), 'name', 'role'))
    print(f"  cast (top-5):")
    print(_fmt_list_of_dicts(row.get('cast'), 'actor', 'character'))
    desc = str(row.get('description', '') or '')
    if desc:
        print(f"  description:     {desc[:180]}{'...' if len(desc) > 180 else ''}")
    print()


# ── Summary ───────────────────────────────────────────────────────────────────

def _summary(df: pd.DataFrame):
    print("── Coverage ─────────────────────────────────────────────────────")
    n = len(df)
    import numpy as np

    def _non_empty(v) -> bool:
        # Handles scalars, lists, numpy arrays (which parquet returns for list cols).
        if v is None:
            return False
        if isinstance(v, (list, tuple, np.ndarray)):
            return len(v) > 0
        try:
            if pd.isna(v):
                return False
        except (TypeError, ValueError):
            pass
        if isinstance(v, str):
            return len(v.strip()) > 0
        return True

    for col in ['title', 'release_date', 'director', 'writers', 'genres',
                'description', 'budget_usd', 'budget_source', 'studios', 'cast',
                'ip_strength', 'adaptation_type']:
        if col not in df.columns:
            print(f"  {col:<20}   column missing")
            continue
        non_empty = df[col].apply(_non_empty).sum()
        print(f"  {col:<20}   {non_empty:>3}/{n}  ({100*non_empty/n:5.1f}%)")

    print("\n── Budget detail ────────────────────────────────────────────────")
    if 'budget_usd' in df.columns:
        with_b = df[df['budget_usd'].notna()]
        print(f"  films with budget:   {len(with_b)}/{n}")
        if len(with_b):
            print(f"  median:              ${int(with_b['budget_usd'].median()):,}")
            print(f"  range:               ${int(with_b['budget_usd'].min()):,}  →  ${int(with_b['budget_usd'].max()):,}")
        if 'budget_source' in df.columns:
            src_counts = with_b['budget_source'].value_counts().head(15)
            print("  budget_source distribution:")
            for src, cnt in src_counts.items():
                print(f"    {cnt:>3}  {src}")

    print("\n── ip_strength ──────────────────────────────────────────────────")
    if 'ip_strength' in df.columns:
        for k, v in df['ip_strength'].value_counts(dropna=False).items():
            print(f"  {str(k):<12}  {v}")

    print("\n── adaptation_type ──────────────────────────────────────────────")
    if 'adaptation_type' in df.columns:
        for k, v in df['adaptation_type'].value_counts(dropna=False).items():
            print(f"  {str(k):<20}  {v}")

    print("\n── Films missing budget ─────────────────────────────────────────")
    miss = df[df['budget_usd'].isna()] if 'budget_usd' in df.columns else df.iloc[:0]
    if miss.empty:
        print("  (none — every film got a budget)")
    else:
        title_col = 'title' if 'title' in miss.columns else 'film_title'
        for _, r in miss.iterrows():
            print(f"  {int(r['film_id']):>6}  {r.get(title_col, '?')}")


# ── Run ───────────────────────────────────────────────────────────────────────

# ── vs-TMDB side-by-side ──────────────────────────────────────────────────────

def _fmt_studios_new(items) -> str:
    if not isinstance(items, (list, tuple)) or len(items) == 0:
        return '(none)'
    return ' | '.join(d.get('name', '?') for d in items[:6] if isinstance(d, dict))


def _fmt_studios_tmdb(items) -> str:
    if items is None:
        return '(none)'
    try:
        if len(items) == 0:
            return '(none)'
    except TypeError:
        return str(items)
    return ' | '.join(str(s) for s in list(items)[:6])


def _fmt_cast_new(items) -> str:
    if not isinstance(items, (list, tuple)) or len(items) == 0:
        return '(none)'
    return ' | '.join(d.get('actor', '?') for d in items[:5] if isinstance(d, dict))


def _print_vs_tmdb():
    if not TMDB_MAPPING_PATH.exists():
        sys.exit(f"TMDB mapping not found at {TMDB_MAPPING_PATH}")
    tmdb = pd.read_parquet(TMDB_MAPPING_PATH)
    tmdb['film_id'] = tmdb['film_id'].astype(int)
    df['film_id_int'] = df['film_id'].astype(int)

    matched = df.merge(
        tmdb[['film_id', 'matched_tmdb_title', 'match_score', 'status',
              'studios', 'budget', 'cast', 'director']],
        left_on='film_id_int', right_on='film_id', how='left', suffixes=('', '_tmdb'),
    )
    matched_n = matched['matched_tmdb_title'].notna().sum()
    print(f"── vs-TMDB ({matched_n}/{len(matched)} films have TMDB rows) ─────────────────────────────\n")

    for _, r in matched.iterrows():
        fid    = int(r['film_id_int'])
        title  = r.get('title') or '?'
        tmdb_t = r.get('matched_tmdb_title')
        score  = r.get('match_score')
        status = r.get('status')

        print(f"── {title}  (film_id={fid}) ─────")
        if pd.isna(tmdb_t):
            print(f"  TMDB:               (no match in studio_mapping)")
        else:
            print(f"  TMDB match:         '{tmdb_t}'  score={score}  status={status}")

        b_new = r.get('budget_usd')
        b_old = r.get('budget')
        bn = f"${int(b_new):,}" if pd.notna(b_new) else '(null)'
        bo = f"${int(b_old):,}" if pd.notna(b_old) else '(null)'
        print(f"  budget   new:  {bn:<20}  old TMDB: {bo}")
        print(f"  src      new:  {r.get('budget_source')}")
        print(f"  studios  new:  {_fmt_studios_new(r.get('studios'))}")
        print(f"           old:  {_fmt_studios_tmdb(r.get('studios_tmdb'))}")
        print(f"  cast     new:  {_fmt_cast_new(r.get('cast'))}")
        print(f"           old:  {_fmt_studios_tmdb(r.get('cast_tmdb'))}")
        print(f"  director new:  {r.get('director')}")
        print(f"           old:  {_fmt_studios_tmdb(r.get('director_tmdb'))}")
        print()


# ── Run ───────────────────────────────────────────────────────────────────────

if args.film_id:
    row = df[df['film_id'].astype(int) == args.film_id]
    if row.empty:
        sys.exit(f"film_id {args.film_id} not in {path}")
    _print_card(row.iloc[0])
    sys.exit(0)

if args.no_budget:
    df = df[df['budget_usd'].isna()] if 'budget_usd' in df.columns else df.iloc[:0]
    print(f"Filtered to {len(df)} films missing budget.\n")

if args.vs_tmdb:
    _print_vs_tmdb()
    sys.exit(0)

if args.detail:
    for _, r in df.iterrows():
        _print_card(r)
else:
    _summary(df)
