"""
print_compare.py — print the most recent refresh_comparison.py run as readable
side-by-side tables. Reads comparison.xlsx from the newest data/_compare/<ts>/.

Usage:
    python print_compare.py                   # all rows
    python print_compare.py --disagree-only   # only rows where new != old
    python print_compare.py --ts 20260519_142530   # pin a specific run
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from config import DATA_DIR

ap = argparse.ArgumentParser()
ap.add_argument('--ts',             type=str,            help='Specific run timestamp')
ap.add_argument('--disagree-only',  action='store_true', help='Only show rows where any field disagrees')
ap.add_argument('--limit',          type=int, default=0, help='Max rows per sheet (0 = all)')
args = ap.parse_args()

COMPARE_ROOT = DATA_DIR / '_compare'
if args.ts:
    run_dir = COMPARE_ROOT / args.ts
else:
    runs = sorted([d for d in COMPARE_ROOT.iterdir() if d.is_dir()])
    if not runs:
        sys.exit(f"No runs found under {COMPARE_ROOT}")
    run_dir = runs[-1]

xlsx_path = run_dir / 'comparison.xlsx'
print(f"Reading: {xlsx_path}\n")
if not xlsx_path.exists():
    sys.exit(f"comparison.xlsx not found in {run_dir}")

pd.set_option('display.max_rows',     None)
pd.set_option('display.max_columns',  None)
pd.set_option('display.width',        220)
pd.set_option('display.max_colwidth', 50)


def _filter_disagree(df: pd.DataFrame) -> pd.DataFrame:
    agree_cols = [c for c in df.columns if c.endswith('__agree')]
    if not agree_cols:
        return df
    mask = df[agree_cols].apply(lambda row: not all(row.fillna(True)), axis=1)
    return df[mask].copy()


def _print_sheet(name: str, key: str, fields: list[str]):
    try:
        df = pd.read_excel(xlsx_path, sheet_name=name)
    except Exception as e:
        print(f"[{name}] sheet missing or unreadable: {e}\n")
        return

    print(f"── {name.upper()}  ({len(df)} rows) ───────────────────────────────────────────────")
    if df.empty:
        print("  (empty)\n")
        return

    # Build display dataframe — key, then for each field: new | old (with disagreement marker)
    rows = []
    for _, r in df.iterrows():
        row = {key: r[key]}
        any_disagree = False
        for f in fields:
            new = r.get(f'{f}__new')
            old = r.get(f'{f}__old')
            agree = r.get(f'{f}__agree')
            mark = '' if (agree is True or pd.isna(old)) else ' *'
            row[f'{f}'] = f"{new}  ←→  {old}{mark}"
            if agree is False:
                any_disagree = True
        row['_disagree_any'] = any_disagree
        rows.append(row)
    out = pd.DataFrame(rows)

    if args.disagree_only:
        out = out[out['_disagree_any']].copy()
    out = out.drop(columns='_disagree_any')

    if args.limit:
        out = out.head(args.limit)

    if out.empty:
        print("  (no rows after filter)\n")
        return

    print(out.to_string(index=False))
    print()


_print_sheet('actor',    'actor_name',    ['fame_tier', 'fame_source', 'primary_market', 'age_range'])
_print_sheet('director', 'director_name', ['director_tier', 'primary_market'])
