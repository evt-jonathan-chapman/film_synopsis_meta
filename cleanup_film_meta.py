"""
cleanup_film_meta.py
---------------------
Genre normalization/consolidation/rarity-filter + format-variant dedup for
film_meta_enriched.parquet. `clean_film_meta_df()` is the reusable core —
it's called automatically after every extraction flush in refresh.py's
_enrich_film_meta and main.py's enrich_film_meta, so newly extracted films
get the same treatment without needing a manual pass. This module's CLI
(`python cleanup_film_meta.py`) is the same logic run standalone, for
retroactively cleaning rows extracted before this was wired in, or for
re-running after tuning the thresholds/maps below.

1. Genre normalization — splits mashed-together compound genre tokens
   ("Comedy-Drama" -> "Comedy" + "Drama", "Mystery & Thriller" -> "Mystery"
   + "Thriller") and merges spelling/casing synonyms ("Science Fiction" ->
   "Sci-Fi", "Sports" -> "Sport", "Coming of Age" -> "Coming-of-Age", ...).
   See _GENRE_SYNONYMS below — built from a full value_counts scan of every
   genre token seen in the parquet, not guessed.

2. Genre consolidation — rolls specific subgenre variants up into a broader
   parent ("Folk Horror" -> "Horror", "Black Comedy" -> "Comedy"). See
   _GENRE_PARENTS / _GENRE_PARENT_OVERRIDES below.

3. Rare-genre filter — drops any genre tag appearing on fewer than
   MIN_GENRE_FILM_COUNT films corpus-wide (catches one-off LLM idiosyncrasies
   and, in several observed cases, language names that leaked into the
   genre field instead of describing genre — Tamil, Punjabi, Hindi, Telugu,
   Japanese, Korean all showed up at count <=3).

4. Format-variant dedup — collapses rows that share an identical
   (title, evt_rel_at, evt_dstbtr) onto one canonical row, keeping whichever
   extraction is more complete. This is the post-extraction equivalent of
   film_variant_merge.py's format-variant merge (3D/IMAX/special-screening
   duplicate bookings) — safe to apply here because the LLM already
   normalizes the 3D/IMAX/festival prefix OUT of `title` per its own prompt
   instructions, so two format-variant bookings of the same release collapse
   to an identical title naturally; matching on evt_rel_at + evt_dstbtr too
   is what makes this high-confidence (see film_variant_merge.py's docstring
   for why the fuzzy-reschedule path is NOT used here — repeated
   franchise-sequel false positives, e.g. Mad Max / Mad Max 2). Note
   film_variant_merge.py's filter_variants() already runs BEFORE extraction
   in the live pipeline, so this step is mainly a backstop for cases that
   only become identical after the LLM's own title-cleaning.

5. Remaining duplicate titles (reschedules, language variants, anything not
   sharing evt_rel_at+evt_dstbtr) are reported but NOT touched — cross-check
   against diagnostics/audit_film_meta.py for the full picture.

Usage:
    python cleanup_film_meta.py              # dry run, prints impact only
    python cleanup_film_meta.py --apply       # writes the cleaned parquet
                                               # (backs up the original first)
"""
import argparse
import re

import pandas as pd

from config import FILM_META_ENRICHED_PATH, FILM_ID_VARIANTS_PATH

# Built from `Counter(tok for g in df['genres'] for tok in g)` over the full
# parquet — see the session's audit for the raw counts. Keys are lowercased;
# values are the canonical form to emit (whole-token match takes priority
# over splitting, so hyphenated single-concept genres like "Sci-Fi" are
# listed here to protect them from being split on their hyphen below).
_GENRE_SYNONYMS = {
    'sci-fi': 'Sci-Fi',
    'science fiction': 'Sci-Fi',
    'science-fiction': 'Sci-Fi',
    'space sci-fi': 'Sci-Fi',
    'coming-of-age': 'Coming-of-Age',
    'coming of age': 'Coming-of-Age',
    'neo-noir': 'Neo-Noir',
    'film-noir': 'Film-Noir',
    'neo-western': 'Neo-Western',
    'anti-war': 'Anti-War',
    'b-horror': 'B-Horror',
    'talk-show': 'Talk-Show',
    'game-show': 'Game-Show',
    'reality-tv': 'Reality-TV',
    'k-drama': 'K-Drama',
    'rom-com': 'Rom-Com',
    'mockumentary': 'Mockumentary',
    'kids & family': 'Kids & Family',
    'faith & spirituality': 'Faith & Spirituality',
    'dark-comedy': 'Dark Comedy',
    'sports': 'Sport',
    'musicals': 'Musical',
    'biopic': 'Biography',
    'biographical': 'Biography',
    'historical': 'History',
    'romantic': 'Romance',
    'animated': 'Animation',
    'espionage': 'Spy',
    'surrealist': 'Surreal',
    'road': 'Road Movie',
    'concert film': 'Concert',
}

# Splits on these ONLY when the token isn't a whole-token synonym above —
# handles "Mystery & Thriller", "Action and Adventure", "Sci-Fi/Fantasy"
# (the "/" splits it, then "Sci-Fi" survives the per-piece synonym check).
_SOFT_SEP_RE = re.compile(r'\s*(?:,|&|/| and )\s*', re.IGNORECASE)


def _split_genre_token(tok) -> list:
    stripped = str(tok).strip()
    low = stripped.lower()
    if low in _GENRE_SYNONYMS:
        return [_GENRE_SYNONYMS[low]]

    pieces = [p.strip() for p in _SOFT_SEP_RE.split(stripped) if p.strip()]
    out = []
    for piece in pieces:
        p_low = piece.lower()
        if p_low in _GENRE_SYNONYMS:
            out.append(_GENRE_SYNONYMS[p_low])
        elif '-' in piece:
            # Not an allowlisted hyphenated genre (Sci-Fi, Coming-of-Age, ...)
            # and not caught by the soft separators above — treat the hyphen
            # itself as two genres mashed together (Comedy-Drama, Action-
            # Adventure, Horror-Comedy).
            sub_parts = [sp.strip() for sp in piece.split('-') if sp.strip()]
            out.extend(_GENRE_SYNONYMS.get(sp.lower(), sp) for sp in sub_parts)
        else:
            out.append(piece)
    return out


# ── Consolidation: roll specific subgenre variants up into a broader parent ──
# "Folk Horror" / "B-Horror" / "Comedy Horror" are all flavors of Horror;
# "Black Comedy" / "Slapstick" / "Parody" are all flavors of Comedy. Losing
# the flavor is the point — this shrinks the long tail of one-off variants
# down to a small, consistent set of buckets.
#
# Rule: a multi-word (or hyphenated) token whose LAST word matches one of
# _GENRE_PARENTS rolls up to that parent (Political Thriller -> Thriller,
# Body Horror -> Horror, B-Horror -> Horror). A handful of tokens don't end
# in a recognizable parent word but are still obviously one — those get an
# explicit override (Slasher -> Horror, Farce -> Comedy).
#
# NOT rolled up: standalone genres that are a distinct category in their own
# right rather than "a flavor of X" — Superhero, Western, Anime, Spy, Epic,
# Martial Arts, Coming-of-Age, Concert, LGBTQ+, Survival, Period, Suspense,
# Short stay as-is even though they're specific, because collapsing them
# loses their own identity rather than a modifier's.
_GENRE_PARENTS = [
    'Horror', 'Comedy', 'Drama', 'Thriller', 'Fantasy', 'Romance',
    'Documentary', 'Mystery', 'Action', 'Satire', 'Animation', 'Family',
]

# Ends in a parent word but the modifier is a distinct cultural/format
# identity, not just "a flavor of the parent" — e.g. K-Drama denotes Korean
# television drama as a category, not merely "Drama". "Kids & Family" also
# splits (on the spaces around "&") to a last word of "Family", which would
# otherwise wrongly collapse it to plain "Family".
_GENRE_CONSOLIDATE_PROTECTED = {'k-drama', 'kids & family'}

# Doesn't end in the parent word, but is unambiguously one.
_GENRE_PARENT_OVERRIDES = {
    'slasher': 'Horror',
    'creature feature': 'Horror',
    'exorcism': 'Horror',
    'vampire': 'Horror',
    'monster': 'Horror',
    'zombie': 'Horror',
    'gothic': 'Horror',
    'slapstick': 'Comedy',
    'parody': 'Comedy',
    'spoof': 'Comedy',
    'tragicomedy': 'Comedy',
    'farce': 'Comedy',
}

_WORD_SPLIT_RE = re.compile(r'[\s\-]+')


def _consolidate_genre(token: str) -> str:
    stripped = str(token).strip()
    low = stripped.lower()
    if low in _GENRE_CONSOLIDATE_PROTECTED:
        return stripped
    if low in _GENRE_PARENT_OVERRIDES:
        return _GENRE_PARENT_OVERRIDES[low]

    words = _WORD_SPLIT_RE.split(stripped)
    if len(words) > 1:
        last = words[-1].lower()
        for parent in _GENRE_PARENTS:
            if last == parent.lower():
                return parent
    return stripped


def normalize_genre_list(genres) -> list:
    if genres is None or not hasattr(genres, '__iter__') or isinstance(genres, str):
        return []
    out, seen = [], set()
    for tok in genres:
        for piece in _split_genre_token(tok):
            piece = _consolidate_genre(piece)
            key = piece.lower()
            if piece and key not in seen:
                seen.add(key)
                out.append(piece)
    return out


def drop_documentary_biography_overlap(genres: list) -> list:
    """film_meta-only rule (NOT applied to synopsis's genres — see
    normalize_synopsis_genre_list): drop "Biography" whenever "Documentary"
    is also present.

    Standard genre taxonomies (IMDb etc.) tag these together deliberately —
    a "biographical documentary" (Diego Maradona, Whitney, RBG) is
    legitimately both Biography AND Documentary, not a contradiction. But for
    feature engineering we want a clean binary "is this dramatized fiction or
    not" signal, and Biography co-occurring with Documentary muddies that —
    so it's dropped here specifically for downstream feature use, not because
    the co-tagging is factually wrong.
    """
    if 'Documentary' in genres and 'Biography' in genres:
        return [g for g in genres if g != 'Biography']
    return genres


def normalize_synopsis_genre_list(genres) -> list:
    """Same normalization as normalize_genre_list, but for synopses_extracted.
    parquet's genres — which the nano synopsis extractor emits all-lowercase
    ("sci-fi", "sport"), unlike film_meta's Title Case ("Sci-Fi", "Sport").

    _split_genre_token/_consolidate_genre assume Title Case input (their
    synonym/parent tables are keyed and valued that way) — fed lowercase
    tokens directly, two things go wrong: (1) a token that only has a
    lowercase-hyphen form and no matching synonym key falls through to the
    "two genres mashed together" hyphen-split path instead of being
    recognized as one genre (this bit "science-fiction", which shattered into
    "science" + "fiction" before "science-fiction" was added to
    _GENRE_SYNONYMS), and (2) tokens that never hit a synonym/consolidation
    rule pass through unchanged and stay lowercase, so e.g. "sport" (already
    correct) and "sports" -> "Sport" (via the synonym map) end up as two
    different buckets instead of merging. Title-casing each token first
    fixes (2) outright and makes (1) survivable everywhere except genuinely
    novel lowercase-hyphen synonyms (still need an explicit _GENRE_SYNONYMS
    entry either way, same as the Title Case path).
    """
    if genres is None or not hasattr(genres, '__iter__') or isinstance(genres, str):
        return []
    title_cased = [str(g).strip().title() for g in genres if str(g).strip()]
    return normalize_genre_list(title_cased)


# ── Rarity filter: drop genre tags that are almost never used corpus-wide ────
# A genre appearing on only a handful of films out of thousands is either a
# one-off LLM idiosyncrasy or, in several observed cases, a language name
# that leaked into the genre field instead of describing genre (Tamil,
# Punjabi, Hindi, Telugu, Japanese, Korean all show up at count <=3 — see
# diagnostics/audit_film_meta.py). Checked across thresholds 1-10: the set of
# films left with zero genres afterward is identical (16, all already empty
# before this step) at every threshold in that range, so there's no real
# downside to being aggressive here. Raised from 5 to 10 (was 46 unique
# genres at 5; drops another 16 — Political, Holiday, Heist, Docudrama, Road
# Movie, Disaster, Social, Nature, Neo-Noir, Tragedy, Kids & Family, Anime,
# Period, LGBTQ+, Survival, Concert — down to 30).
MIN_GENRE_FILM_COUNT = 10


def compute_genre_counts(genre_lists) -> pd.Series:
    """Corpus-wide frequency of each (already normalized) genre token."""
    return pd.Series([g for lst in genre_lists for g in lst]).value_counts()


def drop_rare_genres(genres, counts: pd.Series, min_count: int = MIN_GENRE_FILM_COUNT) -> list:
    if genres is None or not hasattr(genres, '__iter__') or isinstance(genres, str):
        return []
    return [g for g in genres if counts.get(g, 0) >= min_count]


def _richness(row: pd.Series) -> int:
    """How complete is this film_meta row? Used to pick a keeper within a
    duplicate-booking group — favors the extraction with more populated
    fields, not simply the earlier release date."""
    score = 0
    if pd.notna(row.get('budget_usd')):
        score += 1
    studios = row.get('studios')
    if isinstance(studios, (list,)) and len(studios) > 0:
        score += 1
    cast = row.get('cast')
    if isinstance(cast, (list,)) and len(cast) > 0:
        score += 1
    if pd.notna(row.get('description')) and str(row.get('description')).strip():
        score += 1
    if pd.notna(row.get('director')) and str(row.get('director')).strip():
        score += 1
    return score


def find_format_variant_groups(df: pd.DataFrame) -> list:
    """film_ids sharing an identical (title, evt_rel_at, evt_dstbtr). The LLM
    already strips 3D/IMAX/special-screening/festival qualifiers out of
    `title` per its own prompt instructions, so two format-variant bookings
    of the same release collapse to the same title naturally — this is
    film_variant_merge.py's format-variant signal, observed post-extraction
    instead of pre-extraction."""
    key_cols = ['title', 'evt_rel_at', 'evt_dstbtr']
    if any(c not in df.columns for c in key_cols):
        return []
    tmp = df[['film_id'] + key_cols].dropna(subset=key_cols).copy()
    tmp['_title_norm'] = tmp['title'].astype(str).str.strip().str.upper()
    groups = []
    for _, g in tmp.groupby(['_title_norm', 'evt_rel_at', 'evt_dstbtr']):
        ids = g['film_id'].drop_duplicates().tolist()
        if len(ids) > 1:
            groups.append(ids)
    return groups


_VARIANT_MAP_COLUMNS = [
    'film_id', 'canonical_film_id', 'title', 'match_score',
    'confirmed_by', 'variant_rel_at', 'canonical_rel_at',
]


def clean_film_meta_df(df: pd.DataFrame) -> tuple:
    """Runs genre normalization/consolidation/rarity-filter, then
    format-variant dedup, on an in-memory film_meta dataframe.

    Returns (df_cleaned, variant_map, stats). Called both by this module's
    CLI (for reporting + backup on the full parquet) and by refresh.py's
    _enrich_film_meta / main.py's enrich_film_meta right after every
    extraction flush, so newly extracted films get the same treatment
    without a separate manual pass.
    """
    df = df.copy()

    old_genres = df['genres'].apply(lambda g: list(g) if g is not None and hasattr(g, '__iter__') else [])
    df['genres'] = df['genres'].apply(normalize_genre_list)
    df['genres'] = df['genres'].apply(drop_documentary_biography_overlap)

    # Captured before the rarity filter below, which would otherwise drop
    # "Concert" from the vocabulary entirely (9 films — under
    # MIN_GENRE_FILM_COUNT) and take this signal with it. Useful for
    # excluding concert screenings (not films) downstream regardless of
    # whether "Concert" survives as a genre label.
    df['is_concert'] = df['genres'].apply(lambda g: 'Concert' in g)

    counts = compute_genre_counts(df['genres'])
    rare_labels = sorted(counts[counts < MIN_GENRE_FILM_COUNT].index)
    df['genres'] = df['genres'].apply(lambda g: drop_rare_genres(g, counts))
    n_genre_changed = int(sum(a != b for a, b in zip(old_genres, df['genres'])))

    groups = find_format_variant_groups(df)
    df_indexed = df.set_index('film_id')
    variant_rows = []
    drop_ids = set()
    for ids in groups:
        scored = sorted(ids, key=lambda fid: (-_richness(df_indexed.loc[fid]), fid))
        canonical_id = scored[0]
        for variant_id in scored[1:]:
            drop_ids.add(variant_id)
            variant_rows.append({
                'film_id': variant_id,
                'canonical_film_id': canonical_id,
                'title': df_indexed.loc[variant_id, 'title'],
                'match_score': 100,
                'confirmed_by': 'format_variant',
                'variant_rel_at': df_indexed.loc[variant_id, 'evt_rel_at'],
                'canonical_rel_at': df_indexed.loc[canonical_id, 'evt_rel_at'],
            })
    variant_map = pd.DataFrame(variant_rows, columns=_VARIANT_MAP_COLUMNS)
    df_cleaned = df[~df['film_id'].isin(drop_ids)].copy()

    stats = {
        'n_genre_changed': n_genre_changed,
        'rare_labels_dropped': rare_labels,
        'n_variant_groups': len(groups),
        'n_variants_dropped': len(drop_ids),
    }
    return df_cleaned, variant_map, stats


def persist_variant_map(variant_map: pd.DataFrame) -> None:
    """Merge freshly detected format-variant rows into FILM_ID_VARIANTS_PATH
    (same convention as refresh.py's _persist_variant_map for the
    pre-extraction path — keep='last' so a re-run with better data can flip
    which side is canonical)."""
    if variant_map.empty:
        return
    # Force to string — this file is also written by film_variant_merge.py's
    # pre-extraction merge; a raw Timestamp mixed with a string in the same
    # parquet column breaks pyarrow on write.
    for col in ('variant_rel_at', 'canonical_rel_at'):
        if col in variant_map.columns:
            variant_map[col] = variant_map[col].apply(lambda v: str(v) if pd.notna(v) else None)

    FILM_ID_VARIANTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    if FILM_ID_VARIANTS_PATH.exists():
        existing = pd.read_parquet(FILM_ID_VARIANTS_PATH)
        for col in ('variant_rel_at', 'canonical_rel_at'):
            if col in existing.columns:
                existing[col] = existing[col].apply(lambda v: str(v) if pd.notna(v) else None)
        out = (pd.concat([existing, variant_map], ignore_index=True)
               .drop_duplicates(subset='film_id', keep='last'))
    else:
        out = variant_map
    out.to_parquet(FILM_ID_VARIANTS_PATH, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--apply', action='store_true',
                         help='write the cleaned parquet (backs up the original first)')
    args = parser.parse_args()

    df = pd.read_parquet(FILM_META_ENRICHED_PATH)
    print(f"Loaded {len(df):,} rows from {FILM_META_ENRICHED_PATH}")

    df_cleaned, variant_map, stats = clean_film_meta_df(df)

    print(f"\n=== Genre normalization + consolidation + rarity filter ===")
    print(f"  {stats['n_genre_changed']:,} of {len(df):,} rows have their genres list changed")
    print(f"  {len(stats['rare_labels_dropped'])} genre labels dropped from the vocabulary entirely "
          f"(< {MIN_GENRE_FILM_COUNT} films corpus-wide): {', '.join(stats['rare_labels_dropped'])}")

    print(f"\n=== Format-variant dedup ===")
    print(f"  {stats['n_variant_groups']} duplicate-booking groups found — "
          f"dropping {stats['n_variants_dropped']} variant rows, keeping canonicals")
    if not variant_map.empty:
        print(variant_map.to_string(index=False))

    remaining_norm = df_cleaned['title'].astype(str).str.strip().str.upper()
    remaining_dupes = df_cleaned[remaining_norm.duplicated(keep=False)]
    print(f"\n=== Remaining duplicate titles (NOT touched — reschedules / language variants / needs review) ===")
    print(f"  {remaining_dupes['title'].nunique()} titles across {len(remaining_dupes)} film_id rows")
    print(f"  See diagnostics/audit_film_meta.py for the full list with side-by-side diverging fields.")

    print(f"\n=== Summary ===")
    print(f"  {len(df):,} -> {len(df_cleaned):,} rows ({stats['n_variants_dropped']} removed)")
    print(f"  {stats['n_genre_changed']:,} rows had genres normalized")

    if not args.apply:
        print("\nDry run — no files written. Re-run with --apply to write the cleaned parquet.")
        return

    ts = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    backup_path = FILM_META_ENRICHED_PATH.with_name(f"film_meta_enriched.pre_cleanup_{ts}.parquet")
    df.to_parquet(backup_path, engine='pyarrow', index=False)
    print(f"\nBacked up pre-cleanup snapshot → {backup_path}")

    df_cleaned.to_parquet(FILM_META_ENRICHED_PATH, engine='pyarrow', index=False)
    print(f"Wrote cleaned parquet → {FILM_META_ENRICHED_PATH} ({len(df_cleaned):,} rows)")

    persist_variant_map(variant_map)
    if not variant_map.empty:
        print(f"Variant crosswalk → {FILM_ID_VARIANTS_PATH}")


if __name__ == '__main__':
    main()
