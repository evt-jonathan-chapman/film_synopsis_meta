"""
compare_film_meta_vs_tmdb.py — side-by-side comparison of the new film_meta
(web_search) extract vs the old TMDB cache, focused on **budget** and
**studios**. Merges on film_id, so films where TMDB matched the wrong title
also surface as "complete match errors".

Inputs (read-only):
    NEW_JSON_PROGRESS — film_meta checkpoint produced by main.py::RUN_META
    OLD_TMDB_DATA     — legacy TMDB cache parquet

Output:
    DATA_DIR/_compare/film_meta_vs_tmdb_<ts>.parquet  (full join, for follow-up)
    Console: budget summary, studios summary, worst disagreements.

Usage:
    python diagnostics/compare_film_meta_vs_tmdb.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

import numpy as np
import pandas as pd

from config import DATA_DIR

NEW_JSON_PROGRESS = "/Users/jonathanchapman/Documents/data/film_meta/film_meta_progress.json"
OLD_TMDB_DATA     = "/Users/jonathanchapman/Documents/data/_old_meta/tmdb/tmdb_cache.parquet"

OUT_DIR = DATA_DIR / '_compare'
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / f"film_meta_vs_tmdb_{datetime.now():%Y%m%d_%H%M%S}.parquet"

pd.set_option('display.max_rows',     200)
pd.set_option('display.max_columns',  None)
pd.set_option('display.width',        220)
pd.set_option('display.max_colwidth', 60)


# ── Load ──────────────────────────────────────────────────────────────────────

with open(NEW_JSON_PROGRESS) as f:
    checkpoint: dict = json.load(f)

new = pd.DataFrame.from_dict(checkpoint, orient='index')
new['film_id'] = new['film_id'].astype(int)

tmdb = pd.read_parquet(OLD_TMDB_DATA)
tmdb['film_id'] = tmdb['film_id'].astype(int)

print(f"new film_meta rows : {len(new):>5}")
print(f"tmdb cache rows    : {len(tmdb):>5}")


# ── Normalise to comparable shape ─────────────────────────────────────────────
# Studios: TMDB → list[str]; new → list[{name, order, role}]. Reduce both to a
# lowercase set of name strings for set comparison.

def _as_iterable(v) -> list:
    """Parquet round-trips lists as np.ndarray; JSON gives Python lists. Treat
    both — plus tuples — as iterables. Anything else (None, NaN, scalar) → []."""
    if isinstance(v, (list, tuple)):
        return list(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    return []


def _new_studio_names(studios) -> list[str]:
    out = []
    for s in _as_iterable(studios):
        if isinstance(s, dict) and s.get('name'):
            out.append(str(s['name']).strip())
        elif isinstance(s, str) and s.strip():
            out.append(s.strip())
    return out


def _tmdb_studio_names(studios) -> list[str]:
    return [str(s).strip() for s in _as_iterable(studios) if str(s).strip()]


def _norm_set(names) -> set[str]:
    return {n.lower() for n in names}


new_norm = pd.DataFrame({
    'film_id':        new['film_id'],
    'new_title':      new['title'],
    'new_budget':     pd.to_numeric(new['budget_usd'],   errors='coerce'),
    'new_budget_src': new.get('budget_source'),
    'new_studios':    new['studios'].apply(_new_studio_names),
})

tmdb_norm = pd.DataFrame({
    'film_id':         tmdb['film_id'],
    'evt_title':       tmdb['film'],
    'tmdb_title':      tmdb['tmdb_title'],
    'tmdb_match_conf': tmdb['match_confidence'],
    'tmdb_match_score':tmdb['match_score'],
    'tmdb_budget':     pd.to_numeric(tmdb['budget'], errors='coerce'),
    'tmdb_studios':    tmdb['studios'].apply(_tmdb_studio_names),
})

df = new_norm.merge(tmdb_norm, on='film_id', how='inner')
print(f"overlap on film_id : {len(df):>5}")


# ── Budget comparison ─────────────────────────────────────────────────────────

df['both_have_budget'] = df['new_budget'].notna() & df['tmdb_budget'].gt(0)
df['budget_delta']     = df['new_budget'] - df['tmdb_budget']
df['budget_ratio']     = (df['new_budget'] / df['tmdb_budget']).where(df['both_have_budget'])

both = df[df['both_have_budget']].copy()

print("\n── Budget coverage ──")
print(f"  new has budget       : {df['new_budget'].notna().sum():>5} / {len(df)}")
print(f"  tmdb has budget (>0) : {df['tmdb_budget'].gt(0).sum():>5} / {len(df)}")
print(f"  both                 : {len(both):>5}")

if not both.empty:
    print("\n── Budget agreement (where both populated) ──")
    print(f"  median ratio new/tmdb : {both['budget_ratio'].median():.2f}")
    print(f"  within 10%            : {(both['budget_ratio'].between(0.9, 1.1)).sum():>5} / {len(both)}")
    print(f"  within 25%            : {(both['budget_ratio'].between(0.75, 1.25)).sum():>5} / {len(both)}")
    print(f"  off by >2x            : {((both['budget_ratio'] > 2) | (both['budget_ratio'] < 0.5)).sum():>5} / {len(both)}")


# ── Studios comparison ────────────────────────────────────────────────────────

df['new_studio_set']  = df['new_studios'].apply(_norm_set)
df['tmdb_studio_set'] = df['tmdb_studios'].apply(_norm_set)

def _jaccard(a: set, b: set) -> float | None:
    if not a and not b:
        return None
    return len(a & b) / len(a | b) if (a | b) else None

df['studios_overlap']  = df.apply(lambda r: sorted(r['new_studio_set'] & r['tmdb_studio_set']), axis=1)
df['studios_jaccard']  = df.apply(lambda r: _jaccard(r['new_studio_set'], r['tmdb_studio_set']), axis=1)
df['studios_any_match']= df['studios_overlap'].apply(bool)

both_studios = df[df['new_studio_set'].apply(bool) & df['tmdb_studio_set'].apply(bool)]

print("\n── Studio coverage ──")
print(f"  new has studios      : {df['new_studio_set'].apply(bool).sum():>5} / {len(df)}")
print(f"  tmdb has studios     : {df['tmdb_studio_set'].apply(bool).sum():>5} / {len(df)}")
print(f"  both                 : {len(both_studios):>5}")

if not both_studios.empty:
    print("\n── Studio agreement (where both populated) ──")
    print(f"  any-name overlap     : {both_studios['studios_any_match'].sum():>5} / {len(both_studios)}")
    print(f"  median jaccard       : {both_studios['studios_jaccard'].median():.2f}")
    print(f"  zero overlap         : {(~both_studios['studios_any_match']).sum():>5} / {len(both_studios)}")


# ── TMDB match-confidence cross-tab ───────────────────────────────────────────
# If TMDB matched the wrong title (match_confidence ∈ {'borderline','poor'}),
# its budget/studios are suspect — those should be the rows where the new
# extract diverges most.

print("\n── TMDB match_confidence distribution (overlap set) ──")
print(df['tmdb_match_conf'].value_counts(dropna=False).to_string())

if not both_studios.empty:
    print("\n── Studio jaccard by TMDB match_confidence ──")
    print(
        both_studios.groupby('tmdb_match_conf', dropna=False)['studios_jaccard']
                    .agg(['count', 'mean', 'median'])
                    .round(2)
                    .to_string()
    )


# ── Worst disagreements (likely TMDB match errors) ────────────────────────────

print("\n── Top 20 zero-overlap studios cases (likely TMDB mismatch) ──")
suspect = df[
    df['new_studio_set'].apply(bool)
    & df['tmdb_studio_set'].apply(bool)
    & (~df['studios_any_match'])
].copy()
print(
    suspect[['film_id', 'evt_title', 'new_title', 'tmdb_title',
             'tmdb_match_conf', 'new_studios', 'tmdb_studios']]
    .head(20)
    .to_string(index=False)
)

print("\n── Top 20 budget blow-ups (>2x apart) ──")
blowups = both[(both['budget_ratio'] > 2) | (both['budget_ratio'] < 0.5)].copy()
blowups = blowups.reindex(blowups['budget_ratio'].sub(1).abs().sort_values(ascending=False).index)
print(
    blowups[['film_id', 'evt_title', 'new_title', 'tmdb_title',
             'tmdb_match_conf', 'new_budget', 'tmdb_budget', 'budget_ratio']]
    .head(20)
    .to_string(index=False)
)


# ── Save full join ────────────────────────────────────────────────────────────

save = df.copy()
for col in ('new_studios', 'tmdb_studios', 'studios_overlap'):
    save[col] = save[col].apply(lambda v: '|'.join(v) if isinstance(v, list) else '')
save = save.drop(columns=['new_studio_set', 'tmdb_studio_set'])
save.to_parquet(OUT_PATH, index=False)
print(f"\nSaved full join → {OUT_PATH}")
