"""
inspect_film_meta_progress.py — summarise the live film_meta checkpoint + errors JSONs.

Reads:
  ~/Documents/data/film_meta/film_meta_progress.json   (successful extractions)
  ~/Documents/data/film_meta/film_meta_errors.json     (failed extractions)

Usage:
    python diagnostics/inspect_film_meta_progress.py                # coverage + error summary
    python diagnostics/inspect_film_meta_progress.py --detail       # per-film cards (successes)
    python diagnostics/inspect_film_meta_progress.py --errors       # full error dump
    python diagnostics/inspect_film_meta_progress.py --film-id N    # drill into one film (either file)
    python diagnostics/inspect_film_meta_progress.py --no-budget    # only successes missing budget
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path
import requests
from datetime import datetime
import os
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

from config import DATA_DIR

load_dotenv()

api_key=os.getenv('OPENAI_KEY'),
headers = {"Authorization": f"Bearer {api_key}"}

def get_openai_balance():
    # 1. Get Subscription Info (Hard Limit)
    sub_url = "https://api.openai.com/dashboard/billing/subscription"
    sub_res = requests.get(sub_url, headers=headers).json()
    hard_limit = sub_res.get("hard_limit_usd", 0)

    # 2. Get Usage Info (Current Spend)
    # Start date should be the beginning of your billing cycle or a safe early date
    start_date = "2024-01-01" 
    end_date = datetime.now().strftime('%Y-%m-%d')
    usage_url = f"https://api.openai.com/dashboard/billing/usage?start_date={start_date}&end_date={end_date}"
    usage_res = requests.get(usage_url, headers=headers).json()
    total_usage = usage_res.get("total_usage", 0) / 100  # Convert cents to dollars

    # 3. Calculate Remaining Balance
    remaining = hard_limit - total_usage
    
    print(f"Total Limit: ${hard_limit:.2f}")
    print(f"Total Usage: ${total_usage:.2f}")
    print(f"Remaining:   ${remaining:.2f}")


PROGRESS_PATH = DATA_DIR / 'film_meta' / 'film_meta_progress.json'
ERRORS_PATH   = DATA_DIR / 'film_meta' / 'film_meta_errors.json'

# Spyder default — applied only when no CLI args are passed (e.g. running via
# `%runfile ...` without `--args`). Edit this list to change the default view.
SPYDER_DEFAULT_ARGS = ['--errors', '--errors-type', 'json_parse_failed', '--no-budget']
if len(sys.argv) == 1:
    sys.argv += SPYDER_DEFAULT_ARGS

ap = argparse.ArgumentParser()
ap.add_argument('--detail',    action='store_true', help='Print a card per successful film')
ap.add_argument('--errors',    action='store_true', help='Print full error dump instead of summary')
ap.add_argument('--errors-type', type=str, help='With --errors, filter to one bucket (e.g. json_parse_failed, RateLimitError, ambiguous)')
ap.add_argument('--film-id',   type=int, help='Drill into a single film_id (checks both files)')
ap.add_argument('--no-budget', action='store_true', help='Only show successes missing budget_usd')
args = ap.parse_args()


def _load(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


progress = _load(PROGRESS_PATH)
errors   = _load(ERRORS_PATH)

print(f"Progress: {len(progress)} films  ({PROGRESS_PATH})")
print(f"Errors:   {len(errors)} films  ({ERRORS_PATH})")
total = len(progress) + len(errors)
if total:
    print(f"Success rate: {len(progress)}/{total} ({100*len(progress)/total:.1f}%)\n")


def _non_empty(v) -> bool:
    if v is None:
        return False
    if isinstance(v, (list, tuple, dict)):
        return len(v) > 0
    if isinstance(v, str):
        return len(v.strip()) > 0
    return True


def _fmt_list_of_dicts(items, key1, key2, limit=8):
    if not isinstance(items, list) or not items:
        return '    (none)'
    rows = []
    for d in items[:limit]:
        if not isinstance(d, dict):
            continue
        rows.append(f"    {d.get(key1, '?')} — {d.get(key2, '?')}")
    return '\n'.join(rows) if rows else '    (none)'


def _print_card(fid, row):
    print(f"── {row.get('title') or '(no title)'} (film_id={fid}) ─────────────")
    print(f"  release_date:    {row.get('release_date')}")
    print(f"  evt_rel_at:      {row.get('evt_rel_at')}")
    print(f"  director:        {row.get('director')}")
    print(f"  writers:         {row.get('writers')}")
    print(f"  genres:          {row.get('genres')}")
    print(f"  ip_strength:     {row.get('ip_strength')}")
    print(f"  adaptation_type: {row.get('adaptation_type')}")
    b = row.get('budget_usd')
    print(f"  budget_usd:      {'(null)' if b in (None, '') else f'${int(b):,}'}")
    print(f"  budget_source:   {row.get('budget_source')}")
    print(f"  evt_dstbtr:      {row.get('evt_dstbtr')}")
    print(f"  studios:")
    print(_fmt_list_of_dicts(row.get('studios'), 'name', 'role'))
    print(f"  cast (top-5):")
    print(_fmt_list_of_dicts(row.get('cast'), 'actor', 'character', limit=5))
    desc = str(row.get('description', '') or '')
    if desc:
        print(f"  description:     {desc[:200]}{'...' if len(desc) > 200 else ''}")
    print()


def _print_error(fid, e):
    print(f"── film_id={fid}  {e.get('film_title') or '(no title)'} ─────────────")
    print(f"  _error:      {e.get('_error')}")
    raw = e.get('_raw_output')
    if raw:
        s = str(raw)
        print(f"  _raw_output: {s[:400]}{'...' if len(s) > 400 else ''}")
    print()


def _summary_progress(prog: dict):
    print("── Coverage (successful films) ──────────────────────────────────")
    n = len(prog)
    if not n:
        print("  (empty)\n")
        return
    cols = ['title', 'release_date', 'director', 'writers', 'genres',
            'description', 'budget_usd', 'budget_source', 'studios', 'cast',
            'ip_strength', 'adaptation_type', 'evt_dstbtr', 'evt_rel_at']
    for col in cols:
        hit = sum(1 for v in prog.values() if _non_empty(v.get(col)))
        print(f"  {col:<20} {hit:>5}/{n}  ({100*hit/n:5.1f}%)")

    print("\n── Budget detail ────────────────────────────────────────────────")
    budgets = [v.get('budget_usd') for v in prog.values() if v.get('budget_usd') not in (None, '')]
    budgets = [float(b) for b in budgets]
    if budgets:
        budgets.sort()
        med = budgets[len(budgets) // 2]
        print(f"  films with budget:   {len(budgets)}/{n}")
        print(f"  median:              ${int(med):,}")
        print(f"  range:               ${int(min(budgets)):,}  →  ${int(max(budgets)):,}")
    else:
        print("  (no budgets populated)")

    src = Counter(v.get('budget_source') for v in prog.values() if v.get('budget_source'))
    if src:
        print("  budget_source distribution:")
        for s, c in src.most_common(15):
            print(f"    {c:>4}  {s}")

    print("\n── ip_strength ──────────────────────────────────────────────────")
    for k, v in Counter(r.get('ip_strength') for r in prog.values()).most_common():
        print(f"  {str(k):<14}  {v}")

    print("\n── adaptation_type ──────────────────────────────────────────────")
    for k, v in Counter(r.get('adaptation_type') for r in prog.values()).most_common():
        print(f"  {str(k):<22}  {v}")


def _summary_errors(errs: dict):
    print("\n── Error summary ────────────────────────────────────────────────")
    if not errs:
        print("  (no errors logged)")
        return
    bucket = Counter()
    for e in errs.values():
        key = str(e.get('_error', 'unknown')).split(':')[0].strip()[:60]
        bucket[key] += 1
    for k, v in bucket.most_common():
        print(f"  {v:>4}  {k}")
    print("\n  (run with --errors for full dump, or --film-id N for one film)")


# ── Run ───────────────────────────────────────────────────────────────────────

if args.film_id is not None:
    key = str(args.film_id)
    if key in progress:
        print("Found in progress (successful extraction):\n")
        _print_card(key, progress[key])
    elif key in errors:
        print("Found in errors:\n")
        _print_error(key, errors[key])
    else:
        sys.exit(f"film_id {args.film_id} not in progress or errors JSON")
    sys.exit(0)

if args.errors:
    if not errors:
        print("(no errors logged)")
        sys.exit(0)
    items = errors.items()
    if args.errors_type:
        needle = args.errors_type.lower()
        items = [(fid, e) for fid, e in items
                 if needle in str(e.get('_error', '')).lower()]
        print(f"── {len(items)} errors matching '{args.errors_type}' ─────────────\n")
    for fid, e in items:
        _print_error(fid, e)
    sys.exit(0)

if args.no_budget:
    miss = {k: v for k, v in progress.items() if v.get('budget_usd') in (None, '')}
    print(f"── {len(miss)} successful films missing budget_usd ────────────────\n")
    for fid, v in miss.items():
        print(f"  {fid:>6}  {v.get('title') or '(no title)'}")
    sys.exit(0)

if args.detail:
    for fid, v in progress.items():
        _print_card(fid, v)
    sys.exit(0)

_summary_progress(progress)
_summary_errors(errors)
