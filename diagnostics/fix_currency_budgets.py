"""
fix_currency_budgets.py — patch film_meta_progress.json in place to fix rows
where the LLM dropped a local-currency figure into `budget_usd` without
converting (typically Indian films with the figure in rupees / crore).

Detection
    1. Parse `budget_source` for explicit currency tells (crore, ₹, won, yuan,
       Box Office India, Filmibeat, etc.) — high-confidence wins.
    2. Fall back to studio nationality (T-Series → INR, CJ Entertainment → KRW,
       Toho → JPY, …) when the source text is silent but the studios betray
       the origin.
    3. Anything with budget_usd > $400M (physically impossible — the most
       expensive real production was ~$400M) with no detected currency is
       flagged for manual review, not auto-converted.

Idempotent — rows already carrying `budget_usd_raw` are skipped, so re-running
is safe. Backs up the checkpoint before writing.

Usage:
    python diagnostics/fix_currency_budgets.py            # dry-run
    python diagnostics/fix_currency_budgets.py --apply    # write changes
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

CHECKPOINT = Path('/Users/jonathanchapman/Documents/data/film_meta/film_meta_progress.json')

# Mid-period reference rates (per USD). Films span 2018-2026 so rates have
# drifted; we use a single per-currency rate because budgets are ball-park
# anyway. Error from rate drift is well inside the inherent budget noise.
RATES = {
    'INR': 80.0,
    'KRW': 1300.0,
    'CNY': 7.0,
    'JPY': 145.0,
    'EUR': 0.92,
    'GBP': 0.80,
}

# Strong textual tells in budget_source — high-confidence detection.
SOURCE_TELLS: list[tuple[re.Pattern, str]] = [
    (re.compile(r'\bcrore\b|\blakh\b|₹|rupees?|\bRs\.?\s|\bINR\b|Box Office India|Bollywood|Filmibeat|Indian trade', re.I), 'INR'),
    (re.compile(r'\bwon\b|₩|\bKRW\b|Korean Film Council', re.I), 'KRW'),
    (re.compile(r'\byuan\b|\bRMB\b|\bCNY\b|China Box Office', re.I), 'CNY'),
    (re.compile(r'\byen\b|\bJPY\b|Japanese box office', re.I), 'JPY'),
    (re.compile(r'€|\bEUR\b|\beuros?\b', re.I), 'EUR'),
    (re.compile(r'£|\bGBP\b|\bpounds? sterling\b', re.I), 'GBP'),
]

# Studio-nationality fallback — only used if budget_source has no tell.
STUDIO_NATIONALITY: dict[str, list[str]] = {
    'INR': [
        'T-Series', 'Yash Raj', 'UV Creations', 'Balaji Motion',
        'Bhansali Productions', 'Viacom18', 'Dharma Productions',
        'Excel Entertainment', 'Red Chillies', 'AA Films', 'Eros',
        'Reliance Entertainment', 'Anil Kapoor Films', 'Zee Studios',
        'PVR Pictures', 'Saffron Broadcast',
    ],
    'KRW': ['CJ Entertainment', 'Lotte Entertainment', 'Showbox',
            'Next Entertainment World'],
    'CNY': ['Huayi Brothers', 'Bona Film', 'Wanda Pictures',
            'Enlight Media', 'Edko Films'],
    'JPY': ['Toho', 'Studio Ghibli', 'Shochiku', 'Studio Ponoc'],
}

USD_HARD_CEILING = 400_000_000  # biggest real production budget ever was ~$400M


def detect_currency(source: str, studios: list) -> tuple[str | None, str]:
    src = source or ''
    for rx, ccy in SOURCE_TELLS:
        m = rx.search(src)
        if m:
            return ccy, f"source-tell: {m.group(0)!r}"

    studio_blob = ' | '.join(
        (s.get('name', '') if isinstance(s, dict) else str(s))
        for s in (studios or [])
    ).lower()
    for ccy, markers in STUDIO_NATIONALITY.items():
        for marker in markers:
            if marker.lower() in studio_blob:
                return ccy, f"studio-tell: {marker!r}"

    return None, ''


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--apply', action='store_true', help='Write changes (default is dry-run)')
    args = ap.parse_args()

    with open(CHECKPOINT) as f:
        cp: dict = json.load(f)

    patches: list[tuple] = []
    review:  list[tuple] = []
    skipped_already = 0

    for fid, row in cp.items():
        b = row.get('budget_usd')
        if not isinstance(b, (int, float)) or b <= 0:
            continue
        if 'budget_usd_raw' in row:
            skipped_already += 1
            continue
        # Cheap escape: anything well within USD-plausible range stays put
        if b < 100_000_000:
            continue

        ccy, evidence = detect_currency(row.get('budget_source', ''), row.get('studios', []))

        if ccy:
            rate = RATES[ccy]
            converted = int(round(b / rate))
            patches.append((fid, row.get('title'), b, ccy, converted, evidence))
        elif b > USD_HARD_CEILING:
            review.append((fid, row.get('title'), b, row.get('budget_source')))

    print(f"Checkpoint:       {CHECKPOINT}")
    print(f"Rows scanned:     {len(cp)}")
    print(f"Already patched:  {skipped_already}")
    print(f"To patch:         {len(patches)}")
    print(f"Needs review:     {len(review)}")
    print()

    if patches:
        print(f"{'film_id':>8}  {'title':35}  {'raw':>15}  {'ccy':4}  {'usd':>12}  evidence")
        print('-' * 120)
        for fid, title, raw, ccy, usd, ev in patches:
            print(f"{fid:>8}  {(title or '')[:35]:35}  {raw:>15,.0f}  {ccy:4}  {usd:>12,}  {ev}")
        print()

    if review:
        print("Above the USD hard ceiling but no currency tell — eyeball these:")
        for fid, title, raw, src in review:
            print(f"  {fid:>8}  {(title or '')[:35]:35}  {raw:>15,.0f}  source={src!r}")
        print()

    if not args.apply:
        print("Dry-run only. Pass --apply to write changes.")
        return

    if not patches:
        print("Nothing to patch.")
        return

    backup = CHECKPOINT.with_name(
        f"{CHECKPOINT.stem}.pre-currency-fix-{datetime.now():%Y%m%d_%H%M%S}.json"
    )
    shutil.copy2(CHECKPOINT, backup)
    print(f"Backup written:   {backup}")

    for fid, _, raw, ccy, usd, ev in patches:
        row = cp[fid]
        row['budget_usd_raw']    = int(raw)
        row['budget_local']      = int(raw)
        row['budget_currency']   = ccy
        row['budget_usd']        = usd
        row['_patched_evidence'] = ev

    with open(CHECKPOINT, 'w') as f:
        json.dump(cp, f, default=str)
    print(f"Checkpoint updated: {len(patches)} rows patched")


if __name__ == '__main__':
    main()
