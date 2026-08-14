"""
film_variant_merge.py
----------------------
Detects EVT film_id rows that represent the same real-world release and
produces a crosswalk mapping each "variant" film_id onto one canonical
"standard release" film_id (see diagnostics/audit_film_meta.py's
duplicate-title findings: ~20% of film_meta_enriched.parquet's rows are
duplicate titles).

Format variants (same day, same distributor) — Vista assigns a separate
film_id to each format (3D, IMAX), event (special screening, sing-along,
Q&A), or festival programme (TFF/FFF/MIFF/...) booking of the same
theatrical release, e.g. "3D ANT-MAN AND THE WASP" / "ANT-MAN AND THE WASP"
or "A BIG BOLD BEAUTIFUL JOURNEY - SPECIAL SCREENING" / the plain title, both
releasing the same day under the same distributor. Detected by stripping
those qualifiers with vendored/cinema_admits_models/encode_helper.py::
strip_format_variant (re-vendored from cinema_admits_models/encode_helper.py
— see vendored/cinema_admits_models/README.md) and grouping by (base_title,
rel_at, dstbtr). This is the same function build_train.py::BuildTrain.
create_merged_admits_column uses to consolidate format-variant *admits*
rows; here it consolidates film_meta *rows* instead, with a richness-based
keeper pick in place of highest-admits.

Deliberately NOT covered here: language-version variants (Hindi/Tamil/
Telugu/Malayalam/Kannada/Japanese/English dubs, e.g. "777 CHARLIE - HINDI" /
"- KANNADA"). encode_helper.py's own docstring says these are intentionally
left unstripped because they're distinct theatrical products for
box-office modelling — and for film_meta specifically, several of these
titles have NO plain-title booking at all (777 Charlie only ever exists as
"- HINDI"/"- KANNADA" rows), so merging or dropping either side would erase
the only extraction that film ever gets.

Genuine re-releases with no specific pairing (anniversary screenings,
"(2004)" year-in-title, "director's cut") are still dropped outright via
vendored/cinema_admits_models/re_release_filter.py::ReReleaseFilter's
keyword/year-in-title flagging, same as before this module existed.

Fuzzy-matched reschedules are INTENTIONALLY NOT merged (yet). ReReleaseFilter
also offers fuzzy title + director/cast/distributor-confirmed matching across
different rel_at dates (e.g. it correctly links a rescheduled release to its
earlier booking), which looked promising for merging near-term reschedules
like "Devil Wears Prada 2" being re-booked ~2 months apart. But testing it
against every duplicate title already in film_meta_enriched.parquet surfaced
repeated false positives, ALL involving franchise entries that legitimately
share a director and/or lead actor:
  - The Suicide Squad (2021, James Gunn) merged onto Suicide Squad (2016,
    David Ayer) on one shared top-billed actor (Margot Robbie) — confirmed
    by cast alone.
  - Halloween (2018, David Gordon Green) merged onto a 2016 revival screening
    of the 1978 original on shared lead Jamie Lee Curtis — confirmed by cast
    alone.
  - Requiring BOTH director AND cast to confirm (not just one) does not fix
    this — franchises are exactly where dir+cast overlap is most likely to
    false-positive: Happy Death Day 2U (2019) merged onto Happy Death Day
    (2017) on shared director (Christopher Landon) + returning lead (Jessica
    Rothe); Mad Max (1979) merged onto Mad Max 2 (1981) on shared director
    (George Miller) + returning lead (Mel Gibson).
  - Root cause is ReReleaseFilter's own sequel-guard (_extract_seq_number /
    _is_word_sequel): it only recognises a plain trailing digit or Roman
    numeral (misses "2U"-style suffixes and non-English numbering like
    Hindi "Do" in "Pati Patni Aur Woh Do"), and it's asymmetric — it only
    blocks a higher-numbered candidate from claiming an earlier-booked
    match, not the reverse, so an unnumbered title (cand_num=0) checked
    against a numbered "earlier" EVT booking (possible since EVT booking
    dates don't follow real-world release chronology — revival screenings,
    festival re-runs) is never guarded.
Given three distinct false-positive mechanisms surfaced in a small sample,
fuzzy-matched reschedule merging is disabled until the sequel-guard gets a
proper fix. Same-title reschedules (no format-variant qualifier) still get
independently re-extracted for now — duplicate extraction cost, but no risk
of misattributing a real film's metadata to a different movie.

Canonical selection (format-variant path) has two separate parts, added
2026-08-13 after finding real cases (Alita: Battle Angel, Aquaman, Lightyear,
Jurassic World: Fallen Kingdom, Spider-Man: Brand New Day, The LEGO Movie 2,
The Lion King) where they'd disagreed:
  1. Extraction SOURCE — whichever film_id has more complete Vista-sourced
     data (director, cast, synopsis populated) wins; a format variant can go
     either direction, the general release isn't always the one with richer
     metadata attached.
  2. Output IDENTITY — always the group's general-release film_id (title
     needs no 3D/GC/IMAX/etc. stripping at all) when one exists in the
     group, regardless of which side won #1. It's fine for the extracted
     content to come from the 3D booking; it must always be possible to look
     the film up by its plain-release film_id. Only falls back to the
     extraction-source winner when no general release exists in the group at
     all (e.g. a 3D + GC pair with no plain booking).
Both are handled in filter_variants() — see its docstring for the concrete
mechanics (richness_winner vs. output_id, relabel_map).
"""
import pandas as pd

from vendored.cinema_admits_models.encode_helper import strip_format_variant
from vendored.cinema_admits_models.re_release_filter import ReReleaseFilter

# Rows flagged for these reasons are dropped outright (no specific film_id to
# merge onto). language_variant is deliberately NOT in this set — see module
# docstring on why blanket-dropping dubbed-language versions is unsafe here.
_KEYWORD_DROP_REASONS = {'title_keyword', 'year_in_title'}

_VARIANT_MAP_COLUMNS = [
    'film_id', 'canonical_film_id', 'title', 'match_score',
    'confirmed_by', 'variant_rel_at', 'canonical_rel_at',
]


def _richness(row: pd.Series) -> int:
    """0-3: how many of {director, actor_list, synopsis} are populated."""
    score = 0
    for col in ('director', 'actor_list', 'synopsis'):
        val = row.get(col)
        if pd.notna(val) and str(val).strip():
            score += 1
    return score


def _build_richness_lookup(df: pd.DataFrame, film_lookup: pd.DataFrame) -> pd.DataFrame:
    """film_id -> {director, actor_list, synopsis, rel_at}, preferring df (has
    synopsis, since it's the current film_meta work-set) and falling back to
    film_lookup (director/actor_list only — no synopsis column) for film_ids
    outside the current work-set."""
    cols = ['film_id', 'director', 'actor_list', 'rel_at']
    primary = df[[c for c in cols if c in df.columns]].copy()
    if 'synopsis' in df.columns:
        primary['synopsis'] = df['synopsis']
    else:
        primary['synopsis'] = None

    fallback = film_lookup[[c for c in cols if c in film_lookup.columns]].copy()
    fallback['synopsis'] = None

    combined = pd.concat([primary, fallback], ignore_index=True)
    combined['film_id'] = combined['film_id'].astype(int)
    combined['rel_at'] = pd.to_datetime(combined['rel_at'], utc=True, errors='coerce').dt.tz_localize(None)
    return combined.drop_duplicates('film_id', keep='first').set_index('film_id')


def _resolve_groups(pairs: list[tuple]) -> dict[int, list[int]]:
    """Union-find over pairwise format-variant matches so transitive chains
    (A~B, B~C) collapse into one group sharing a single canonical member."""
    parent: dict[int, int] = {}

    def find(x: int) -> int:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for film_id, matched_id, *_ in pairs:
        union(film_id, matched_id)

    groups: dict[int, list[int]] = {}
    for node in parent:
        groups.setdefault(find(node), []).append(node)
    return groups


def _format_variant_pairs(df: pd.DataFrame, title_col: str = 'film') -> tuple[list[tuple], set[int]]:
    """Same-day, same-distributor bookings that differ only by a 3D/IMAX/GC/
    special-screening/festival-programme qualifier (encode_helper.py's
    _VARIANT_STRIP) — Vista assigns each of these its own film_id even though
    it's the same theatrical release. Emits one pair per non-anchor member of
    each (base_title, rel_at, dstbtr) group with >1 film_id.

    Also returns the set of film_ids that ARE the general release within
    their group — i.e. their own raw title needed no stripping at all (same
    "prefer the row whose title already equals its base_title" rule
    encode_helper.py::consolidate_all_admits uses for admits consolidation).
    filter_variants() uses this to force the general release's film_id as the
    output identity even when a richer-data 3D/format booking is what's
    actually extracted from — see its docstring."""
    needed = ['film_id', title_col, 'rel_at', 'dstbtr']
    if any(c not in df.columns for c in needed):
        return [], set()

    tmp = df[needed].dropna(subset=['rel_at']).copy()
    tmp['_base_title'] = tmp[title_col].apply(strip_format_variant)
    tmp['_is_general_release'] = tmp['_base_title'] == tmp[title_col].astype(str).str.upper().str.strip()

    pairs = []
    general_release_ids: set[int] = set()
    for (base_title, _rel_at, _dstbtr), group in tmp.groupby(['_base_title', 'rel_at', 'dstbtr']):
        ids = group['film_id'].astype(int).drop_duplicates().tolist()
        if len(ids) < 2:
            continue
        anchor = ids[0]
        for other_id in ids[1:]:
            pairs.append((other_id, anchor, base_title, 100, 'format_variant'))
        general_release_ids.update(
            int(fid) for fid in group.loc[group['_is_general_release'], 'film_id'].drop_duplicates()
        )
    return pairs, general_release_ids


def filter_variants(
    df: pd.DataFrame,
    film_lookup: pd.DataFrame | None,
    title_col: str = 'film_title',
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop genuine re-releases and merge format-variant duplicate bookings in
    `df`, keeping only one row per detected group. Fuzzy-matched reschedules
    are detected for the drop decision (ReReleaseFilter's keyword/year-in-
    title reasons) but NOT merged — see module docstring.

    Extraction source vs. output identity: the row that survives filtering
    (and so is what actually gets sent to the LLM) is whichever group member
    has the richest Vista data (_richness) — a 3D/format booking wins over
    the general release there if IT has the populated director/cast/synopsis
    and the general release doesn't. But the film_id that surviving row is
    stored under is always the group's general-release film_id when one
    exists in the group (see _format_variant_pairs' general_release_ids) —
    downstream consumers must always be able to look a film up by its plain
    film_id, regardless of which specific booking happened to have the
    richest metadata. When the richness winner IS the general release (the
    common case), this is a no-op — output_id == richness_winner.

    Returns (df_filtered, variant_map). variant_map has one row per film_id
    that does NOT have its own row in df_filtered — every other group member
    besides output_id, INCLUDING the richness winner itself when it differs
    from output_id (columns: film_id, canonical_film_id, title, match_score,
    confirmed_by, variant_rel_at, canonical_rel_at; confirmed_by is always
    "format_variant") — empty if none were found. Keyword/year-in-title-
    flagged re-releases are dropped from df_filtered too but have no specific
    film_id to map onto, so they never appear in variant_map.
    """
    df_renamed = df.rename(columns={title_col: 'film'})
    format_pairs, general_release_ids = _format_variant_pairs(df_renamed)

    keyword_drop_ids: set[int] = set()
    if film_lookup is not None and 'rel_at' in film_lookup.columns:
        flagged = ReReleaseFilter().flag(df_renamed, film_lookup, title_col='film')
        keyword_drop_ids = set(
            flagged.loc[flagged['rerelease_reason'].isin(_KEYWORD_DROP_REASONS), 'film_id']
            .astype(int)
        )

    df_drop_ids: set[int] = set()
    relabel_map: dict[int, int] = {}  # richness winner's film_id -> output_id, only when they differ

    if not format_pairs:
        variant_map = pd.DataFrame(columns=_VARIANT_MAP_COLUMNS)
    else:
        edge_info = {
            film_id: {'title': title, 'match_score': score, 'confirmed_by': confirmed_by}
            for film_id, _matched_id, title, score, confirmed_by in format_pairs
        }
        groups = _resolve_groups(format_pairs)
        richness = _build_richness_lookup(df, film_lookup) if film_lookup is not None else _build_richness_lookup(df, df.iloc[0:0])

        def sort_key(fid):
            r = _richness(richness.loc[fid]) if fid in richness.index else 0
            rel_at = richness.loc[fid, 'rel_at'] if fid in richness.index and pd.notna(richness.loc[fid, 'rel_at']) else pd.Timestamp.max
            return (-r, rel_at, fid)

        out_rows = []
        for members in groups.values():
            if len(members) < 2:
                continue
            ordered = sorted(members, key=sort_key)
            richness_winner = ordered[0]
            df_drop_ids.update(ordered[1:])  # only the richness winner's row survives in df

            general_candidates = [m for m in ordered if m in general_release_ids]
            output_id = general_candidates[0] if general_candidates else richness_winner
            if output_id != richness_winner:
                relabel_map[richness_winner] = output_id
            output_rel_at = richness.loc[output_id, 'rel_at'] if output_id in richness.index else None

            for member_id in ordered:
                if member_id == output_id:
                    continue
                info = edge_info.get(member_id) or edge_info.get(richness_winner) or {}
                member_rel_at = richness.loc[member_id, 'rel_at'] if member_id in richness.index else None
                out_rows.append({
                    'film_id': member_id,
                    'canonical_film_id': output_id,
                    'title': info.get('title'),
                    'match_score': info.get('match_score'),
                    'confirmed_by': info.get('confirmed_by'),
                    # Stringified, not a raw Timestamp — FILM_ID_VARIANTS_PATH
                    # is shared with cleanup_film_meta.py's format-variant
                    # merge, which stores these as strings (matching the
                    # evt_rel_at convention elsewhere); mixing the two dtypes
                    # in the same parquet column breaks pyarrow on write.
                    'variant_rel_at': str(member_rel_at) if pd.notna(member_rel_at) else None,
                    'canonical_rel_at': str(output_rel_at) if pd.notna(output_rel_at) else None,
                })
        variant_map = pd.DataFrame(out_rows, columns=_VARIANT_MAP_COLUMNS)

    drop_ids = keyword_drop_ids | df_drop_ids
    df_filtered = df[~df['film_id'].astype(int).isin(drop_ids)].copy() if drop_ids else df.copy()
    if relabel_map:
        df_filtered['film_id'] = df_filtered['film_id'].astype(int).replace(relabel_map)

    return df_filtered, variant_map
