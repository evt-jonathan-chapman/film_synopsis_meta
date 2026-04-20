"""
Analyse film title prefix/suffix patterns across the full title catalogue.

Two passes:
  1. Known patterns — translated from SQL CASE/LIKE logic.
  2. Data-driven discovery — finds novel structured tokens at title boundaries
     that look like codes, festival names, or language tags not already covered.

Output: synopses/outputs/title_patterns.xlsx
  - Sheet 'flags'      : one row per film, boolean columns for each known tag
  - Sheet 'discovered' : candidate new tokens with count + example titles
  - Sheet 'summary'    : count of titles matched per known tag

Usage:
    python -m synopses.analyse_title_patterns
"""

import re
from collections import Counter
from pathlib import Path

import pandas as pd

from films.main import get_films_sources

OUTPUT_PATH = Path('synopses', 'outputs', 'title_patterns.xlsx')

# ── Known pattern definitions ─────────────────────────────────────────────────
# Each entry: (tag_name, list_of_regex_patterns)
# Patterns are matched case-insensitively against the full title string.
# A title is flagged if ANY pattern in the list matches.

KNOWN_PATTERNS: list[tuple[str, list[str]]] = [

    # ── 3D ───────────────────────────────────────────────────────────────────
    ('is_3d', [
        r'\b3d\b',                              # anywhere: "3D", "3D -", "(3D)"
    ]),

    # ── Presentation format (IMAX, 4DX, 70mm, 4K, ScreenX, etc.) ────────────
    ('is_format', [
        r'\bimax\b',
        r'\b4dx\b',
        r'\bscreenx\b',
        r'\bscrnx\b',
        r'\b70mm\b',
        r'\b4k\b',
        r'\bdolby\b',
        r'\batmos\b',
        r'\bhfr\b',
        r'\brestoration\b',                     # "4K Restoration", "Digital Restoration"
    ]),

    # ── Film festival — short code prefixes before " - " ─────────────────────
    # Festival codes are ALL-CAPS abbreviations. Using [A-Z] (no re.IGNORECASE
    # applied at match time via the helper) would be ideal, but since we apply
    # re.IGNORECASE globally, we instead require the prefix is short AND followed
    # by a dash separator — combined with the prose-word filter in discovery,
    # this avoids matching real film subtitle separators.
    # Named non-acronym series (Sci-Fi, Exhibition on Screen, etc.) listed explicitly.
    ('is_ff_prefix', [
        r'^[a-z]{2,5}\d{0,2}\s*[-–]\s(?!\s)',  # FFS-, AFI-, GC-, SQLD-, SF3-
        r'^[a-z]{2,5}\s+ff\s*[-–]\s',           # "Real FF - ", "Cans FF - "
        r'^sci[-\s]fi\s*[-–]\s',                 # "Sci-Fi - ", "SCIFI - "
        r'^(exhibition\s+on\s+screen|hw\s+classic|french\s+fridays|'
        r'sensory\s+screening|skycity|flickerfest|winda)\s*[-–]\s',
    ]),

    # ── Film festival — suffix: must end with a known festival-style code ─────
    # Avoid bare r'\s+ff$' — "ff" alone is too short and could appear in titles.
    # Require at least one letter prefix before "ff", or a dash separator.
    ('is_ff_suffix', [
        r'[-–][a-z]{2,4}ff$',                   # "-jff", "-nzff", "-rbff" at end
        r'\s[a-z]{2,4}ff$',                      # " jff", " nff" at end (space-separated)
    ]),

    # ── Film festival — named tokens anywhere in title ────────────────────────
    # "film fest" covers all long-form festival names.
    # Short abbreviations only listed where unambiguous (3+ chars, not common words).
    ('is_ff_named', [
        r'\bfilm\s+fest',                        # film festival, film fest
        r'\bfilm\s+tour\b',
        r'\bnziff\b',
        r'\bkoffia\b',
        r'\bgcff\b',
        r'\bstff\b',
        r'\bradff\b',
        r'\bslaff\b',
        r'\bssaff\b',
        r'\bsiffc\b',
        r'\bsmsff\b',
        r'\bniaff\b',
        r'\bcquff\b',
        r'\banimeff\b',
        r'\bflickerfest\b',
        r'\bwinda\b',
        r'\baff:',                               # "AFF: title"
        r'(?<![a-z])iff(?![a-z])',               # IFF not surrounded by letters (avoids "cliff", "biff")
        r'\brff\b',
        r'\bfff\b',                              # French Film Festival abbreviation
    ]),

    # ── Festival bracket tags — short codes and full names ────────────────────
    # e.g. [JFF], [NZIFF25], [French Film Festival], (Gold Coast Film Festival)
    # Short-code rule requires "ff" inside the brackets — every festival abbreviation
    # contains it (JFF, NZIFF, GCFF, IFF, FFF, RBFF). This avoids matching language
    # tags like (Hindi), (Telugu), (Japanese) or venue names like (Innaloo).
    ('is_ff_bracket', [
        r'[\[\(][a-z]{0,5}ff[a-z]{0,4}\d{0,2}[\]\)]',  # [JFF], [NZIFF], (GCFF), [FFF19]
        r'[\[\(][^\]\)]*\bfilm\s+fest[^\]\)]*[\]\)]',    # (Gold Coast Film Festival)
    ]),

    # ── Multi-film / marathon ─────────────────────────────────────────────────
    ('is_multi', [
        r'\+',
        r'\bplus\b',
        r'\bmarathon\b',
        r'\bdouble\s+feature\b',
        r'\beps?\s*\d',                         # "Eps 4-6", "Ep 1"
        r'\bepisodes?\s*\d',
    ]),

    # ── Language / version tags (hyphen-separated or in brackets) ────────────
    # Two general rules replace ~20 individual patterns:
    #   1. Language name at end of title, bare or after " -"
    #   2. Language name or "version" / "subbed" / "dubbed" inside parens
    ('is_alt_lang', [
        r'[-–\s](hindi|tamil|telugu|mandarin|malayalam|kannada|japanese|'
        r'cantonese|punjabi|gujarati|bengali|subtitled|dubbed)$',
        r'\((hindi|tamil|telugu|english|mandarin|malayalam|kannada|japanese|'
        r'cantonese|punjabi|gujarati|bengali|subbed|dubbed|subtitled)\)',
        r'\([a-z]+ version\)',                  # (English version), (Japanese version)
        r'\bsubtitles?\b',
    ]),

    # ── Re-release / anniversary / special edition ────────────────────────────
    ('is_rerelease', [
        r'\banniversary\b',
        r'\bremaster(ed)?\b',
        r"\bdirector'?s\s+cut\b",
        r'\bextended\s+edition\b',
        r'\bspecial\s+edition\b',
        r'\brestored\b',
        r'\(\d{4}\)',                           # (2004) year bracket
    ]),

    # ── Screening type (Q&A, sing-along, gala, sensory) ──────────────────────
    ('is_screening_type', [
        r'\bq\s*[&+]\s*a\b',
        r'\bsing.along\b',
        r'\bgala\b',
        r'\bopening\s+night\b',
        r'\bclosing\s+night\b',
        r'\bpreview\b',
    ]),

    # ── Event cinema (concerts, operas, sports broadcasts) ───────────────────
    ('is_event', [
        r'\bconcert\b',
        r'\blive\s+(at|from|in)\b',
        r'[-–]\s*live$',
        r'\bopera\b',
        r'\bballet\b',
        r'\borchestra\b',
        r'\bphilharmonic\b',
        r'^(nrl|fifa|world\s+cup|oscar\s+nominated)\b',
        r'\b(nrl|fifa)\b.*[-–]',               # "NRL - Broncos v Titans"
    ]),
]

# Convenience roll-up: any festival indicator
FF_TAGS = {'is_ff_prefix', 'is_ff_suffix', 'is_ff_named', 'is_ff_bracket'}


# ── Data-driven discovery ──────────────────────────────────────────────────────

# Pattern: titles of the form  TOKEN(S) - Real Title
# We capture the token(s) before the first " - " separator
PREFIX_SEP_RE = re.compile(r'^(.{2,25}?)\s*[-–]\s+\S', re.IGNORECASE)

# Pattern: Real Title [TOKEN] or Real Title (TOKEN) at end
BRACKET_SUFFIX_RE = re.compile(r'[\[\(]([^\]\)]{2,25})[\]\)]$', re.IGNORECASE)

# Pattern: standalone word(s) at title start that look like codes (all caps or alphanum ≤6 chars)
CODE_PREFIX_RE = re.compile(r'^([A-Z0-9]{2,6}(?:\s+[A-Z0-9]{1,4})?)\s+[-–\[]', re.IGNORECASE)


def _apply_known(title: str, patterns: list[str]) -> bool:
    for p in patterns:
        if re.search(p, title, re.IGNORECASE):
            return True
    return False


def _which_pattern(title: str) -> str:
    """Return 'tag:pattern' for the first match across all known patterns, or ''."""
    for tag, patterns in KNOWN_PATTERNS:
        for p in patterns:
            if re.search(p, title, re.IGNORECASE):
                return f'{tag}: {p}'
    return ''


# Common English prose words — a prefix containing these is likely a real film title
# subtitle (e.g. "Beauty and the Beast", "Mission: Impossible") not a code/event prefix.
_PROSE_WORDS = frozenset({
    'the', 'a', 'an', 'of', 'in', 'on', 'at', 'to', 'and', 'or', 'is', 'it',
    'by', 'be', 'as', 'do', 'for', 'up', 'so', 'if', 'my', 'me', 'we', 'he',
    'she', 'his', 'her', 'our', 'its', 'but', 'not', 'with', 'from', 'into',
})


def _looks_like_film_subtitle(token: str) -> bool:
    """Return True if the token before ' - ' looks like a real film title fragment."""
    words = re.split(r'[\s:]+', token.lower())
    words = [w.strip('.,!?') for w in words if w]
    prose_count = sum(1 for w in words if w in _PROSE_WORDS)
    # Multi-word with common prose words → film title (e.g. "Beauty and the Beast")
    if len(words) >= 2 and prose_count >= 1:
        return True
    # 4+ words regardless → too long to be a code (e.g. "Star Wars Episode I")
    if len(words) >= 4:
        return True
    return False


def _discover_prefixes(titles: pd.Series) -> pd.DataFrame:
    """Extract tokens that appear before ' - ' separators and rank by frequency."""
    prefix_counts: Counter = Counter()
    prefix_examples: dict[str, list[str]] = {}

    for title in titles:
        m = PREFIX_SEP_RE.match(str(title))
        if m:
            token = m.group(1).strip()
            if len(token) > 25 or token.isdigit():
                continue
            if _looks_like_film_subtitle(token):
                continue
            prefix_counts[token.lower()] += 1
            prefix_examples.setdefault(token.lower(), [])
            if len(prefix_examples[token.lower()]) < 3:
                prefix_examples[token.lower()].append(title)

    rows = [
        {
            'token': k,
            'count': v,
            'examples': ' | '.join(prefix_examples[k]),
        }
        for k, v in prefix_counts.most_common()
        if v >= 2  # only tokens seen at least twice
    ]
    return pd.DataFrame(rows)


def _discover_bracket_suffixes(titles: pd.Series) -> pd.DataFrame:
    """Extract bracket/paren tokens at end of title and rank by frequency."""
    token_counts: Counter = Counter()
    token_examples: dict[str, list[str]] = {}

    for title in titles:
        m = BRACKET_SUFFIX_RE.search(str(title))
        if m:
            token = m.group(1).strip().lower()
            token_counts[token] += 1
            token_examples.setdefault(token, [])
            if len(token_examples[token]) < 3:
                token_examples[token].append(title)

    rows = [
        {
            'token': k,
            'count': v,
            'type': 'bracket_suffix',
            'examples': ' | '.join(token_examples[k]),
        }
        for k, v in token_counts.most_common()
        if v >= 2
    ]
    return pd.DataFrame(rows)


def run_analysis() -> None:
    print('Loading films...')
    df = get_films_sources(persisted=True)
    titles = df['film_title'].dropna().astype(str)
    print(f'  {len(titles):,} titles loaded')

    # ── Known pattern flags ───────────────────────────────────────────────────
    print('Applying known patterns...')
    flag_rows = []
    for title in titles:
        row: dict = {'film_title': title}
        any_ff = False
        for tag, patterns in KNOWN_PATTERNS:
            matched = _apply_known(title, patterns)
            row[tag] = matched
            if tag in FF_TAGS and matched:
                any_ff = True
        row['is_ff_any'] = any_ff
        row['any_tag'] = any(row[t] for t, _ in KNOWN_PATTERNS)
        flag_rows.append(row)

    df_flags = pd.DataFrame(flag_rows)
    df_flags['matched_by'] = df_flags['film_title'].apply(_which_pattern)

    # ── Summary ───────────────────────────────────────────────────────────────
    tag_cols = [t for t, _ in KNOWN_PATTERNS] + ['is_ff_any', 'any_tag']
    summary_rows = []
    for col in tag_cols:
        matched = df_flags[col].sum()
        summary_rows.append({
            'tag': col,
            'n_matched': int(matched),
            'pct_of_titles': round(matched / len(df_flags) * 100, 2),
            'examples': ' | '.join(
                df_flags.loc[df_flags[col], 'film_title'].head(5).tolist()
            ),
        })
    df_summary = pd.DataFrame(summary_rows)

    # ── Discovery ─────────────────────────────────────────────────────────────
    print('Running discovery pass...')
    df_prefixes = _discover_prefixes(titles)
    df_prefixes['type'] = 'sep_prefix'

    df_brackets = _discover_bracket_suffixes(titles)

    df_discovered = pd.concat([df_prefixes, df_brackets], ignore_index=True)

    # Filter out tokens already covered by known patterns.
    # Must test against full example titles, not the bare token —
    # patterns like r'^[a-z]{1,6}ff\s*[-–]\s' only fire on the complete title string.
    def _already_known(example_titles_str: str) -> bool:
        examples = [e.strip() for e in example_titles_str.split('|') if e.strip()]
        for example in examples:
            for _, patterns in KNOWN_PATTERNS:
                if _apply_known(example, patterns):
                    return True
        return False

    if not df_discovered.empty:
        df_discovered['already_known'] = df_discovered['examples'].apply(_already_known)
        df_discovered = df_discovered.sort_values(['already_known', 'count'], ascending=[True, False])

    # ── Per-tag samples (20 random matched titles per tag) ────────────────────
    # Quick eyeball sheet: one column per tag, 20 rows of matched examples.
    # Makes false positives visible without reading the full flags sheet.
    tag_only_cols = [t for t, _ in KNOWN_PATTERNS]
    sample_dict: dict[str, pd.Series] = {}
    for col in tag_only_cols:
        matched_titles = df_flags.loc[df_flags[col], 'film_title']
        sample = matched_titles.sample(n=min(20, len(matched_titles)), random_state=1).reset_index(drop=True)
        sample_dict[col] = sample
    df_samples = pd.DataFrame(sample_dict)

    # ── False-positive candidates: matched by only one broad tag ─────────────
    # Titles that fired exactly one tag are the most borderline — worth a manual check.
    df_flags['n_tags_matched'] = df_flags[tag_only_cols].sum(axis=1)
    df_fp_candidates = (
        df_flags.loc[df_flags['n_tags_matched'] == 1, ['film_title', 'matched_by']]
        .sort_values('matched_by')
        .reset_index(drop=True)
    )

    # ── Write Excel ───────────────────────────────────────────────────────────
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f'Writing → {OUTPUT_PATH}')
    with pd.ExcelWriter(OUTPUT_PATH, engine='openpyxl') as writer:
        df_summary.to_excel(writer, sheet_name='summary', index=False)
        df_samples.to_excel(writer, sheet_name='samples_by_tag', index=False)
        df_fp_candidates.to_excel(writer, sheet_name='fp_candidates', index=False)
        df_flags.to_excel(writer, sheet_name='flags', index=False)
        df_discovered.to_excel(writer, sheet_name='discovered', index=False)

    # ── Console summary ───────────────────────────────────────────────────────
    print(f'\n{"═" * 55}')
    print(f'KNOWN PATTERN MATCHES  ({len(df_flags):,} titles total)')
    print(f'{"═" * 55}')
    for _, row in df_summary.iterrows():
        if row['n_matched'] > 0:
            print(f"  {row['tag']:25s} {row['n_matched']:5d}  ({row['pct_of_titles']:.1f}%)")

    print(f'\n{"─" * 55}')
    print(f'DISCOVERED PREFIXES/SUFFIXES (≥2 occurrences, not yet in known patterns)')
    print(f'{"─" * 55}')
    novel = df_discovered.loc[~df_discovered['already_known']] if not df_discovered.empty else pd.DataFrame()
    if novel.empty:
        print('  None found.')
    else:
        for _, row in novel.head(40).iterrows():
            print(f"  [{row['type']:14s}] {row['token']:30s} × {int(row['count'])}")
            print(f"    e.g. {row['examples'][:100]}")

    print(f'\nSaved → {OUTPUT_PATH}')


if __name__ == '__main__':
    run_analysis()
