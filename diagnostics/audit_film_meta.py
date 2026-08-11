"""
audit_film_meta.py — data-quality audit for film_meta_enriched.parquet.

Two independent checks, both stemming from the fact that film_meta has no
postprocessing step (see post_process.py — only `clean_names` exists) and
each film_id is extracted independently via a non-deterministic web_search
LLM call, so nothing normalises output across films or catches malformed
values before they land in the parquet:

  1. Genre format anomalies — a `genres` entry that looks like two genres
     joined into one string (e.g. "Comedy-Drama", "Action-Adventure",
     "Sci-Fi/Fantasy") instead of being split into separate list items.
     Legitimate single-concept hyphenated genres (Sci-Fi, Coming-of-Age,
     Neo-Noir, ...) are allowlisted so they aren't flagged.

  2. Duplicate titles — the same `title` string appearing under more than
     one film_id. This is NOT the 3D/IMAX/festival-prefix "variant" case
     (title_cleaner.py strips those before extraction ever runs, and they'd
     still be one film_id). It means the EVT source catalogue
     (DIM_VH_FILM) has two distinct film_id rows for what looks like the
     same real-world release — most often a rescheduled release date that
     was booked as a new listing rather than an edit to the existing one.
     re_release_filter.py doesn't catch this because it requires a
     >=180-day gap (min_gap_days) between showings before flagging a
     re-release; a reschedule a few weeks or months out is under that bar.
     Each duplicate film_id runs through film_meta as a fully independent
     web_search call, so the two rows can (and do) disagree — different
     genre formatting, different adaptation_type, different description —
     because nothing in the pipeline is aware the two film_ids refer to the
     same movie.

Usage:
    python diagnostics/audit_film_meta.py                # both checks, printed summary
    python diagnostics/audit_film_meta.py --write-csv     # also write the two reports to disk
    python diagnostics/audit_film_meta.py --genres-only
    python diagnostics/audit_film_meta.py --duplicates-only
"""
import argparse
import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from config import DATA_DIR, FILM_META_ENRICHED_PATH

AUDIT_DIR = DATA_DIR / 'meta_data' / 'film_meta' / 'audit'

# Single-concept genres that legitimately contain a hyphen, slash, or "&" —
# anything else matching the hard-compound/hyphen checks below is assumed to
# be two genres concatenated into one. Kept in sync with cleanup_film_meta.py's
# _GENRE_SYNONYMS (that module owns the canonical casing/spelling for these).
LEGIT_HYPHENATED_GENRES = {
    "sci-fi", "coming-of-age", "neo-noir", "film-noir", "neo-western",
    "anti-war", "b-horror", "talk-show", "game-show", "reality-tv",
    "k-drama", "rom-com", "mockumentary",
}
LEGIT_AMPERSAND_GENRES = {
    "kids & family", "faith & spirituality",
}

# Tokens containing any of these are near-certain concatenations unless
# allowlisted above (a genre name never legitimately contains "," or " and ").
_HARD_COMPOUND_RE = re.compile(r',| and |/')


def _looks_like_compound(token: str) -> bool:
    t = token.strip()
    low = t.lower()
    if low in LEGIT_AMPERSAND_GENRES:
        return False
    if _HARD_COMPOUND_RE.search(t) or ' & ' in t:
        return True
    if '-' in t and low not in LEGIT_HYPHENATED_GENRES:
        return True
    return False


def audit_genres(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for film_id, title, genres in zip(df['film_id'], df['title'], df['genres']):
        if genres is None:
            continue
        for tok in genres:
            if _looks_like_compound(str(tok)):
                rows.append({
                    'film_id': film_id,
                    'title': title,
                    'offending_token': tok,
                    'full_genres': list(genres),
                })
    return pd.DataFrame(rows)


def audit_duplicate_titles(df: pd.DataFrame) -> pd.DataFrame:
    norm = df['title'].astype(str).str.strip().str.upper()
    dupe_titles = norm[norm.duplicated(keep=False)]
    if dupe_titles.empty:
        return pd.DataFrame()

    cols = [c for c in [
        'film_id', 'title', 'release_date', 'evt_rel_at', 'evt_dstbtr',
        'genres', 'ip_strength', 'adaptation_type', 'budget_usd',
    ] if c in df.columns]
    out = df.loc[dupe_titles.index, cols].copy()
    out['_title_norm'] = norm.loc[dupe_titles.index]
    return out.sort_values(['_title_norm', 'film_id']).drop(columns='_title_norm')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--write-csv', action='store_true', help='write reports to disk under meta_data/film_meta/audit/')
    parser.add_argument('--genres-only', action='store_true')
    parser.add_argument('--duplicates-only', action='store_true')
    args = parser.parse_args()

    df = pd.read_parquet(FILM_META_ENRICHED_PATH)
    print(f"Loaded {len(df):,} films from {FILM_META_ENRICHED_PATH}\n")

    run_genres = not args.duplicates_only
    run_dupes  = not args.genres_only

    if run_genres:
        genre_issues = audit_genres(df)
        print(f"=== Genre format anomalies: {len(genre_issues):,} entries "
              f"across {genre_issues['film_id'].nunique() if not genre_issues.empty else 0} films ===")
        if not genre_issues.empty:
            print(genre_issues['offending_token'].value_counts().to_string())
            print()
            print(genre_issues.head(20).to_string(index=False))
            if args.write_csv:
                AUDIT_DIR.mkdir(parents=True, exist_ok=True)
                path = AUDIT_DIR / 'genre_format_issues.csv'
                genre_issues.to_csv(path, index=False)
                print(f"\n→ {path}")
        print()

    if run_dupes:
        dupes = audit_duplicate_titles(df)
        n_titles = dupes['title'].astype(str).str.upper().nunique() if not dupes.empty else 0
        print(f"=== Duplicate titles: {n_titles:,} titles across {len(dupes):,} film_id rows ===")
        if not dupes.empty:
            print(dupes.to_string(index=False))
            if args.write_csv:
                AUDIT_DIR.mkdir(parents=True, exist_ok=True)
                path = AUDIT_DIR / 'duplicate_titles.csv'
                dupes.to_csv(path, index=False)
                print(f"\n→ {path}")


if __name__ == '__main__':
    main()
