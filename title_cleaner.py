"""
title_cleaner.py — strip event/festival/format/language noise from film titles
before passing them to LLM extraction prompts.

The goal is NOT to produce a canonical "clean" title for display — it is to
remove tokens that confuse LLMs into making wrong inferences:
  - Festival prefix codes (FFS, AFI, NZIFF) misread as sequel/franchise markers
  - Language suffixes (Hindi, Telugu) misread as part of the film concept
  - Format tags (3D, 70mm, IMAX) misread as genre or version signals
  - Year brackets (2004) misread as production date vs. release context

Public API:
    clean_title_for_llm(title: str) -> str
    tag_title(title: str) -> dict[str, str]   # returns which rules fired
"""

import re

# ── Prefix stripping rules ────────────────────────────────────────────────────
# Applied at the START of the title.
# Each entry: (compiled_regex, label)
# Matched portion is removed; what follows the match becomes the new title.
#
# IMPORTANT: uppercase-letter patterns (festival codes) are compiled WITHOUT
# re.IGNORECASE so that only genuine all-caps codes match — real film title
# words (e.g. "Rocketry", "Sarkar") are mixed-case and won't fire.

_PREFIX_RULES: list[tuple[re.Pattern, str]] = [
    # All-caps festival/event codes + separator: "FFS - ", "AFI - ", "GC - ",
    # "NZIFF25 - ", "SQLD - ", "LBS - ", "AMW - ", "NRL - ", "FIFA - "
    # Case-SENSITIVE: codes are uppercase; film title words are not.
    (re.compile(r'^[A-Z]{2,6}\d{0,2}\s*[-–]\s+'), 'festival_code_prefix'),

    # "3D Title" — 3D at start before a word (e.g. "3D Kanguva")
    (re.compile(r'^3D\s+', re.IGNORECASE), '3d_prefix'),

    # Named event series (explicit list — mixed case, can't use uppercase rule)
    (re.compile(r'^Sci-?\s*Fi\s*[-–]\s+', re.IGNORECASE),           'sci_fi_series'),
    (re.compile(r'^Exhibition\s+On\s+Screen\s*[-–]\s+', re.IGNORECASE), 'event_series'),
    (re.compile(r'^HW\s+Classic\s*[-–]\s+', re.IGNORECASE),          'event_series'),
    (re.compile(r'^French\s+Fridays\s*[-–]\s+', re.IGNORECASE),      'event_series'),
    (re.compile(r'^Sensory\s+Screening\s*[-–]\s+', re.IGNORECASE),   'event_series'),
    (re.compile(r'^Skycity\s*[-–]\s+', re.IGNORECASE),               'event_series'),
    (re.compile(r'^Flickerfest\s*[-–]\s+', re.IGNORECASE),           'event_series'),
    (re.compile(r'^Winda\s*[-–]\s+', re.IGNORECASE),                 'event_series'),
    (re.compile(r'^Real\s+FF\s*[-–]\s+', re.IGNORECASE),             'event_series'),
    (re.compile(r'^Cans\s+FF\s*[-–]\s+', re.IGNORECASE),             'event_series'),
    (re.compile(r'^\d{4}\s+Oscar\s+Nominated\s*[-–]\s+', re.IGNORECASE), 'oscar_shorts'),
]

# ── Bracket suffix stripping rules ────────────────────────────────────────────
# Applied at the END of the title.
# Matches bracket content including surrounding whitespace.

_LANG_NAMES = (
    r'hindi|tamil|telugu|english|mandarin|malayalam|kannada|japanese|'
    r'cantonese|punjabi|gujarati|bengali|subbed|dubbed|subtitled'
)
_FORMAT_NAMES = r'imax|4k|4dx|screenx|scrnx|70mm|hfr|dolby|atmos'

_BRACKET_SUFFIX_RULES: list[tuple[re.Pattern, str]] = [
    # Language/version in brackets: (Hindi), (Telugu), (English version), (Subbed)
    (re.compile(
        rf'\s*[\(\[](?:{_LANG_NAMES}|[a-z]+ version)[\)\]]\s*$', re.IGNORECASE
    ), 'language_bracket'),

    # Presentation format in brackets: (IMAX), (4K), (70mm)
    (re.compile(rf'\s*[\(\[](?:{_FORMAT_NAMES})[\)\]]\s*$', re.IGNORECASE), 'format_bracket'),

    # 3D in brackets: (3D)
    (re.compile(r'\s*\(3D\)\s*$', re.IGNORECASE), '3d_bracket'),

    # Year in brackets: (2004)
    (re.compile(r'\s*\(\d{4}\)\s*$'), 'year_bracket'),

    # Festival code in brackets: [JFF], [NZIFF25], (GCFF), (FFF)
    # Requires "ff" inside — avoids matching language/venue names (see analysis).
    (re.compile(r'\s*[\[\(][a-z]{0,5}ff[a-z]{0,4}\d{0,2}[\]\)]\s*$', re.IGNORECASE), 'festival_code_bracket'),

    # Full festival name in brackets: (Gold Coast Film Festival), [French Film Festival]
    (re.compile(r'\s*[\[\(][^\]\)]*\bfilm\s+fest[^\]\)]*[\]\)]\s*$', re.IGNORECASE), 'festival_name_bracket'),
]

# ── Trailing " - NOISE" stripping rules ───────────────────────────────────────
# Applied at the END of the title after a " - " separator.
# Only fires when the portion after the separator is clearly a tag, not a subtitle.
#
# Each entry: (compiled_regex, label, min_remaining)
# min_remaining: minimum character length the title before the match must have for
# the rule to fire. Use 0 for unambiguous technical terms; raise the bar for common
# words that could plausibly appear as legitimate subtitles (e.g. "gala", "preview").

_TRAILING_SEP_RULES: list[tuple[re.Pattern, str, int]] = [
    # Format: "Dunkirk - 70mm", "Dunkirk - 4K Restoration", "Title - IMAX"
    # Separator is mandatory to avoid stripping real film titles (e.g. "Restoration", "Dolby").
    (re.compile(
        rf'\s*[-–]\s*(?:{_FORMAT_NAMES}|\d{{1,2}}K?\s*(?:restoration|remaster(?:ed|ing)?))\s*$',
        re.IGNORECASE
    ), 'format_trailing', 0),

    # Format without separator but with explicit digit prefix: "Shiri 4K Remastering"
    # Requires the digit (2K/4K/8K) to anchor the match — prevents "Restoration" firing alone.
    (re.compile(r'\s+\d{1}K\s+(?:restoration|remaster(?:ed|ing)?)\s*$', re.IGNORECASE), 'format_trailing_prefixed', 0),

    # Unambiguous screening-type tags: Q&A, Sing-Along
    (re.compile(r'\s*[-–]\s*(?:q\s*[&+]\s*a|sing[- ]along)\s*$', re.IGNORECASE), 'screening_type_trailing', 0),

    # Common-word screening tags: "- Gala", "- Preview"
    # min_remaining=8 guards against stripping when the preceding title is very short,
    # reducing risk of misidentifying a legitimate subtitle on a short-titled film.
    (re.compile(r'\s*[-–]\s*(?:gala|preview)\s*$', re.IGNORECASE), 'screening_type_trailing', 8),

    # Festival code after separator: "Title - JFF", "Title - NZIFF", "Title - RBFF"
    # Requires "ff" in the code to avoid stripping real subtitles.
    (re.compile(r'\s*[-–]\s*[a-z]{0,5}ff[a-z]{0,3}\d{0,2}\s*$', re.IGNORECASE), 'festival_code_trailing', 0),

    # 3D suffix: "Title 3D" or "Title - 3D"
    (re.compile(r'\s*[-–]?\s*3D\s*$', re.IGNORECASE), '3d_trailing', 0),
]


# ── Public API ────────────────────────────────────────────────────────────────

def clean_title_for_llm(title: str) -> str:
    """
    Strip event/festival/format/language noise from a film title.

    Applies prefix and suffix rules iteratively until the title stabilises.
    The original title is not modified — callers should keep it separately.

    Examples:
        "FFS - All About E"                  → "All About E"
        "3D Kanguva (Hindi)"                 → "Kanguva"
        "Rocketry - The Nambi Effect (Tamil)" → "Rocketry - The Nambi Effect"
        "Dunkirk - 70mm"                     → "Dunkirk"
        "Title [French Film Festival]"        → "Title"
        "Skyfall"                             → "Skyfall"  (unchanged)
    """
    title = title.strip()
    prev = None
    while prev != title:
        prev = title
        for pattern, _ in _PREFIX_RULES:
            m = pattern.match(title)
            if m:
                title = title[m.end():].strip()
                break  # restart after any change
        for pattern, _ in _BRACKET_SUFFIX_RULES:
            title = pattern.sub('', title).strip()
        for pattern, _, min_remaining in _TRAILING_SEP_RULES:
            m = pattern.search(title)
            if m and len(title[:m.start()].strip()) >= min_remaining:
                title = title[:m.start()].strip()
    return title


def tag_title(title: str) -> dict[str, str]:
    """
    Return a dict of {label: matched_text} for every rule that fired on this title.
    Useful for debugging and for confirming which tags are present before cleaning.
    """
    tags: dict[str, str] = {}
    working = title.strip()

    for pattern, label in _PREFIX_RULES:
        m = pattern.match(working)
        if m:
            tags[label] = m.group(0).strip()

    for pattern, label in _BRACKET_SUFFIX_RULES:
        m = pattern.search(working)
        if m:
            tags[label] = m.group(0).strip()

    for pattern, label, min_remaining in _TRAILING_SEP_RULES:
        m = pattern.search(working)
        if m and len(working[:m.start()].strip()) >= min_remaining:
            tags[label] = m.group(0).strip()

    return tags
