"""
inspect_cast_progress.py — summarise the live cast_meta checkpoint + errors JSONs.

Mirrors inspect_film_meta_progress.py.

Reads:
  ~/Documents/data/cast_meta/cast_progress.json    (successful enrichments)
  ~/Documents/data/cast_meta/cast_errors.json      (failed enrichments)

Usage:
    python diagnostics/inspect_cast_progress.py                    # coverage + distributions
    python diagnostics/inspect_cast_progress.py --detail           # per-actor cards
    python diagnostics/inspect_cast_progress.py --errors           # full error dump
    python diagnostics/inspect_cast_progress.py --errors-type json_parse_failed
    python diagnostics/inspect_cast_progress.py --actor "TOM HOLLAND"
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DATA_DIR

PROGRESS_PATH = DATA_DIR / 'cast_meta' / 'cast_progress.json'
ERRORS_PATH   = DATA_DIR / 'cast_meta' / 'cast_errors.json'

# Spyder default — applied only when no CLI args are passed.
SPYDER_DEFAULT_ARGS: list[str] = []
if len(sys.argv) == 1:
    sys.argv += SPYDER_DEFAULT_ARGS

ap = argparse.ArgumentParser()
ap.add_argument('--detail',      action='store_true', help='Print a card per actor')
ap.add_argument('--errors',      action='store_true', help='Print full error dump')
ap.add_argument('--errors-type', type=str, help='With --errors, filter to one bucket')
ap.add_argument('--actor',       type=str, help='Drill into one actor by name (case-insensitive)')
ap.add_argument('--no-fame',     action='store_true', help='Only show actors missing fame_tier (or fame_tier=unknown)')
args = ap.parse_args()


def _load(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


progress = _load(PROGRESS_PATH)
errors   = _load(ERRORS_PATH)

print(f"Progress: {len(progress)} actors  ({PROGRESS_PATH})")
print(f"Errors:   {len(errors)} actors    ({ERRORS_PATH})")
total = len(progress) + len(errors)
if total:
    print(f"Success rate: {len(progress)}/{total} ({100*len(progress)/total:.1f}%)\n")


def _non_empty(v) -> bool:
    if v is None:
        return False
    if isinstance(v, (list, tuple, dict)):
        return len(v) > 0
    if isinstance(v, str):
        return len(v.strip()) > 0 and v.strip().lower() != 'unknown'
    return True


def _print_card(actor_name, row):
    print(f"── {row.get('actor_name') or actor_name} ─────────────")
    print(f"  fame_tier:       {row.get('fame_tier')}")
    print(f"  fame_source:     {row.get('fame_source')}")
    print(f"  primary_market:  {row.get('primary_market')}")
    print(f"  age_range:       {row.get('age_range')}")
    print(f"  cross_media:     {row.get('cross_media')}")
    print()


def _print_error(actor_name, e):
    print(f"── {e.get('actor_name') or actor_name} ─────────────")
    print(f"  _error:      {e.get('_error')}")
    raw = e.get('_raw_output')
    if raw:
        s = str(raw)
        print(f"  _raw_output: {s[:400]}{'...' if len(s) > 400 else ''}")
    print()


def _summary(prog: dict):
    print("── Coverage (non-empty, non-'unknown') ──────────────────────────")
    n = len(prog)
    if not n:
        print("  (empty)\n")
        return
    cols = ['fame_tier', 'fame_source', 'primary_market', 'age_range', 'cross_media']
    for col in cols:
        hit = sum(1 for v in prog.values() if _non_empty(v.get(col)))
        print(f"  {col:<18} {hit:>5}/{n}  ({100*hit/n:5.1f}%)")

    for col in ['fame_tier', 'primary_market', 'age_range']:
        print(f"\n── {col} distribution ──")
        for k, v in Counter(r.get(col) for r in prog.values()).most_common(20):
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
    print("\n  (run with --errors for full dump, or --actor NAME for one row)")


# ── Run ───────────────────────────────────────────────────────────────────────

if args.actor:
    needle = args.actor.upper().strip()
    matches = [(k, v) for k, v in progress.items() if needle in str(k).upper()]
    if matches:
        print(f"Found {len(matches)} match(es) in progress:\n")
        for k, v in matches:
            _print_card(k, v)
    err_matches = [(k, v) for k, v in errors.items() if needle in str(k).upper()]
    if err_matches:
        print(f"Found {len(err_matches)} match(es) in errors:\n")
        for k, v in err_matches:
            _print_error(k, v)
    if not matches and not err_matches:
        sys.exit(f"No actor matching '{args.actor}' in progress or errors")
    sys.exit(0)

if args.errors:
    if not errors:
        print("(no errors logged)")
        sys.exit(0)
    items = list(errors.items())
    if args.errors_type:
        needle = args.errors_type.lower()
        items = [(k, e) for k, e in items if needle in str(e.get('_error', '')).lower()]
        print(f"── {len(items)} errors matching '{args.errors_type}' ─────────────\n")
    for k, e in items:
        _print_error(k, e)
    sys.exit(0)

if args.no_fame:
    miss = {k: v for k, v in progress.items()
            if not v.get('fame_tier') or str(v.get('fame_tier')).lower() == 'unknown'}
    print(f"── {len(miss)} actors with fame_tier missing/unknown ──\n")
    for k in list(miss)[:200]:
        print(f"  {k}")
    if len(miss) > 200:
        print(f"  ... and {len(miss) - 200} more")
    sys.exit(0)

if args.detail:
    for k, v in progress.items():
        _print_card(k, v)
    sys.exit(0)

_summary(progress)
_summary_errors(errors)
