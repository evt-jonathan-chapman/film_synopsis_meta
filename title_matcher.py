"""
title_matcher.py
----------------
Shared fuzzy title-matching engine behind comscore_matcher.py::ComscoreMatcher
and gower_matcher.py::GowerMatcher. Both map EVT film_id -> an external
source's own row identifier via noisy title matching; the only things that
differ between sources are column names, output paths, and thresholds, so
those are pulled out into subclass class-attributes and everything else
(normalisation, scoring, variant propagation, cache/review I/O) lives here
once.

FuzzyTitleMatcher subclasses must set:
  ID_COL, DATE_COL, TITLE_COLS   - raw column names on the source extract
  ID_FIELD, TITLE_FIELD,
  DATE_FIELD, MATCHED_TITLE_FIELD - cache column names (source-specific
                                     prefix, e.g. "cs_id" / "gw_id") so the
                                     on-disk cache schema for each source
                                     stays exactly what it was before this
                                     refactor (downstream repos read these
                                     column names directly)
  CACHE_PATH, REVIEW_PATH,
  MANUAL_OVERRIDES, MANUAL_OVERRIDES_CSV - output paths
  SOURCE_LABEL                   - used in print()s only

Subclasses may override ALT_CONTENT_COL (None = no alt-content filter),
HIGH_CONFIDENCE_SCORE, MAX_DAYS_HIGH_CONF, MIN_LENGTH_RATIO, YEAR_TOL.
"""

import os
import re
import unicodedata
import pandas as pd

import s3_checkpoint
from config import S3_BUCKET, S3_TITLE_MATCHING_PREFIX

try:
    from rapidfuzz import fuzz as _rf_fuzz
    _USE_RAPIDFUZZ = True
except ImportError:
    from difflib import SequenceMatcher
    _USE_RAPIDFUZZ = False

# Variant-strip patterns — mirrored from cinema_admits_models/encode_helper.py.
# Applied in order; each pattern is substituted once (not iteratively).
# Used to identify EVT film_ids that are format/event/festival variants of
# the same base film so one external-source ID can be inherited by all
# variants. EVT-side only, so shared verbatim across sources.
_VARIANT_STRIP = [
    # Format prefixes
    (re.compile(r"^3D[\s\-]+",                            re.I), ""),
    (re.compile(r"^GC\s+",                                re.I), ""),
    # Format suffixes
    (re.compile(r"\s*[-–]\s*3D$",                         re.I), ""),
    (re.compile(r"\s*[-–]\s*IMAX(\s+3D)?$",               re.I), ""),
    (re.compile(r"\s+\(3D\)$",                            re.I), ""),
    (re.compile(r"\s+\(IMAX\)$",                          re.I), ""),
    (re.compile(r"\s*[-–]\s*SCREEN\s*X\b.*$",             re.I), ""),
    (re.compile(r"\s*[-–]\s*70MM$",                       re.I), ""),
    # Special-screening / event suffixes
    (re.compile(r"\s*[-–]\s*SPECIAL\s+SCREENING[S]?$",    re.I), ""),
    (re.compile(r"\s*[-–]\s*SPECIAL\s+EVENT$",            re.I), ""),
    (re.compile(r"\s*[-–]\s*EVENT\s+CINEMA$",             re.I), ""),
    (re.compile(r"\s*[-–]\s*BONUS\s+CONTENT$",            re.I), ""),
    (re.compile(r"\s*[-–]\s*SING[\s\-]?ALONG$",           re.I), ""),
    (re.compile(r"\s*[-–]\s*SPECIAL\s+Q\s+AND\s+A.*$",    re.I), ""),
    (re.compile(r"\s*[-–]\s*BLOCK\s+PARTY\s+EDITION!?$",  re.I), ""),
    (re.compile(r":\s*THE\s+VALENTINE\s+ENCORE$",         re.I), ""),
    # Festival / distributor programme prefixes
    (re.compile(r"^(?:TFF|FFF|MIFF|CFF|MF)\s*[-–]?\s+",  re.I), ""),
    # Event/tour suffixes used by EVT that Comscore doesn't carry
    (re.compile(r"\s*[-–]\s*RE[\s\-]?RELEASE$",           re.I), ""),
    (re.compile(r"\s*[-–]\s*ROAD\s+TRIP$",                re.I), ""),
    # Language variants ("- HINDI VERSION", "- TAMIL DUBBED", "- HINDI", etc.)
    # TELEGU is a common EVT typo for TELUGU — both forms included.
    # Mirrors the _LANG_SUFFIX regex used in _normalise_title so that scoring
    # and variant propagation apply the same language stripping rules.
    (re.compile(
        r'[\s\-–]+(?:'
        r'(?:WITH\s+)?ENGLISH\s+SUBTITLES?|'
        r'HINDI|TAMIL|TELUGU|TELEGU|KANNADA|MALAYALAM|MARATHI|PUNJABI|BENGALI|'
        r'JAPANESE|KOREAN|MANDARIN|CANTONESE|FRENCH|SPANISH|ITALIAN|'
        r'PORTUGUESE|GERMAN|THAI|INDONESIAN|ARABIC|TURKISH|ENGLISH'
        r')(?:\s+(?:VERSION|DUBBED|SUBTITLED))?\s*$',
        re.I
    ), ""),
]


class FuzzyTitleMatcher:

    HIGH_CONFIDENCE_SCORE = 0.92
    MAX_DAYS_HIGH_CONF    = 365
    MIN_LENGTH_RATIO      = 0.5
    YEAR_TOL              = 1

    # Raw column name on the source extract that flags non-film content to
    # exclude before matching (Comscore: "is_alt_content"). None = no filter.
    ALT_CONTENT_COL = None

    SOURCE_LABEL = "source"

    # Variant/festival prefix regexes — applied to EVT titles before scoring.
    _VARIANT_PREFIX  = re.compile(
        r'^(?:GC|3D|IMAX|VMAX|XD|EVENT|GOLD|GOLDCLASS|VIP)\s+',
        re.IGNORECASE,
    )
    _FESTIVAL_PREFIX = re.compile(r'^[A-Z]{2,3}\s*-\s*')
    _VARIANT_SUFFIX  = re.compile(r'\s+(?:3D|IMAX|VMAX|XD)\s*$', re.IGNORECASE)
    _YEAR_PAREN      = re.compile(r'\s*\(\d{4}\)\s*$')

    # Language/version qualifiers appended by EVT — stripped before scoring so
    # "PATHAAN - HINDI" matches "PATHAAN", "DARBAR - TAMIL VERSION" matches
    # "DARBAR", "SUZUME JAPANESE" matches "SUZUME", etc.
    # Matches an optional VERSION / DUBBED / SUBTITLED suffix after the language
    # name so "- HINDI VERSION" and "- HINDI DUBBED" are both stripped.
    _LANG_SUFFIX = re.compile(
        r'[\s\-–]+(?:'
        r'(?:WITH\s+)?ENGLISH\s+SUBTITLES?|'   # longest form first
        r'HINDI|TAMIL|TELUGU|KANNADA|MALAYALAM|MARATHI|PUNJABI|BENGALI|'
        r'JAPANESE|KOREAN|MANDARIN|CANTONESE|FRENCH|SPANISH|ITALIAN|'
        r'PORTUGUESE|GERMAN|THAI|INDONESIAN|ARABIC|TURKISH|ENGLISH'
        r')(?:\s+(?:VERSION|DUBBED|SUBTITLED))?\s*$',
        re.IGNORECASE,
    )

    # Presentation/format suffixes appended by EVT that don't appear in the
    # external source.
    _FORMAT_SUFFIX = re.compile(
        r'[\s\-–:]+(?:'
        r'LIVE\s+ACTION|BONUS\s+CONTENT|RE[\s\-]?RELEASE|SING[\s\-]?ALONG|'
        r'EXTENDED\s+(?:EDITION|CUT|VERSION)|DIRECTOR\'?S?\s+CUT|'
        r'SPECIAL\s+EDITION|ANNIVERSARY\s+(?:EDITION)?|'
        r'\d+(?:ST|ND|RD|TH)\s+ANNIVERSARY|'
        r'4K|REMASTERED|RESTORED|UNCUT|THEATRICAL\s+(?:CUT|VERSION)|'
        r'EONE|ROAD\s+TRIP'
        r')\s*$',
        re.IGNORECASE,
    )

    def __init__(self, match_thresh: float = 0.80, carry_cols: list[str] | None = None):
        """
        carry_cols : source columns to copy into the cache alongside the
                     identity/match columns, prefixed f"{ID_FIELD.split('_')[0]}_".
                     Defaults to None (no extras — join back to the raw
                     source extract on the ID column when you need value
                     columns).
        """
        self.match_thresh  = match_thresh
        self.carry_cols    = carry_cols or []
        self.mapping_df    = None
        self._cache        = self._load_cache()
        self._last_review_df = None

    # ── Cache column schema (built from subclass field names) ────────────────

    @property
    def CACHE_COLS(self) -> list[str]:
        return [
            "film_id", "film",
            self.ID_FIELD, self.TITLE_FIELD, self.DATE_FIELD,
            "match_score", self.MATCHED_TITLE_FIELD,
            "matched_via_col",
            "match_confidence",
            "days_diff",
        ]

    def _carry_prefix(self) -> str:
        return self.ID_FIELD.rsplit("_", 1)[0]

    # ── Cache I/O ────────────────────────────────────────────────────────────

    def _load_cache(self):
        if os.path.exists(self.CACHE_PATH):
            df = pd.read_parquet(self.CACHE_PATH)
            print(f"{self.SOURCE_LABEL} cache: {len(df)} films from {self.CACHE_PATH}")
            return df
        return pd.DataFrame(columns=self.CACHE_COLS)

    def _save_cache(self):
        os.makedirs(os.path.dirname(self.CACHE_PATH), exist_ok=True)
        self._cache.to_parquet(self.CACHE_PATH, index=False)

    # ── Title normalisation + scoring (ported from StudioLookup) ─────────────

    @classmethod
    def _normalise_title(cls, s):
        if not s:
            return ""
        s = str(s)
        s = cls._LANG_SUFFIX.sub('', s)
        s = cls._FORMAT_SUFFIX.sub('', s)
        s = cls._VARIANT_PREFIX.sub('', s)
        s = cls._FESTIVAL_PREFIX.sub('', s)
        s = cls._VARIANT_SUFFIX.sub('', s)
        s = cls._YEAR_PAREN.sub('', s)
        s = unicodedata.normalize('NFKD', s)
        s = ''.join(c for c in s if not unicodedata.combining(c))
        # Normalise punctuation: collapse hyphens/colons/dashes to a space so
        # "DUNE - PART TWO" and "DUNE: PART TWO" score identically.
        s = re.sub(r'[\-–—:/\\|]+', ' ', s)
        s = re.sub(r'[^\w\s]', '', s)
        return ' '.join(s.split()).lower().strip()

    @classmethod
    def _score(cls, a, b, allow_subset: bool = True):
        a_n = cls._normalise_title(a)
        b_n = cls._normalise_title(b)
        if not a_n or not b_n:
            return 0.0
        la, lb = len(a_n), len(b_n)
        if min(la, lb) / max(la, lb) < cls.MIN_LENGTH_RATIO:
            return 0.0
        if _USE_RAPIDFUZZ:
            candidates = [
                _rf_fuzz.ratio(a_n, b_n),
                _rf_fuzz.token_sort_ratio(a_n, b_n),
            ]
            if allow_subset:
                # token_set_ratio handles "X presents Y" vs "X: Y" (Hobbs & Shaw
                # pattern) where one side has an extra connective word not in
                # the other — gives full credit when one title's words are a
                # subset of the other's. Only safe for two FULL titles being
                # compared as-is: allow_subset=False is passed for colon-split
                # fragments (see _article_variants), where the "subset" is a
                # deliberately truncated piece of the EVT title (e.g. "NT
                # LIVE" from "NT LIVE: VANYA") — subset-crediting that against
                # an untruncated source title like "NT Live: Hamlet" scores a
                # false 100: the extra words on the source side ("Hamlet") are
                # exactly what should have disqualified the match, not been
                # forgiven.
                candidates.append(_rf_fuzz.token_set_ratio(a_n, b_n))
            return max(candidates) / 100.0
        return SequenceMatcher(None, a_n, b_n).ratio()

    @staticmethod
    def _article_variants(title: str) -> list[tuple[str, bool]]:
        """Return (variant, is_truncated) pairs: article-transposed and
        colon-split forms to broaden matching coverage. is_truncated=True for
        the colon-split fragments, which drop part of the original title's
        content — callers should compare those with allow_subset=False (see
        _score) since they're missing real, potentially-distinguishing words
        rather than just being reworded/reordered."""
        variants = [(title, False)]
        t = title.strip()
        upper = t.upper()
        # "THE X" / "A X" / "AN X"  →  "X, THE" / "X, A" / "X, AN"
        for article in ('THE ', 'A ', 'AN '):
            if upper.startswith(article):
                rest = t[len(article):]
                variants.append((f"{rest}, {article.strip()}", False))
                break
        # "X, THE" / "X, A" / "X, AN"  →  "THE X" / "A X" / "AN X"
        for article in (', THE', ', A', ', AN'):
            if upper.endswith(article):
                rest = t[:-len(article)]
                variants.append((f"{article[2:]} {rest}", False))
                break
        # Colon-split variants: sources often omit either the subtitle
        # ("Peter Rabbit 2: The Runaway" → "Peter Rabbit 2") or the franchise
        # prefix ("Star Wars: The Mandalorian and Grogu" → "The Mandalorian and
        # Grogu"). Both sides are tried; MIN_LENGTH_RATIO in _score() suppresses
        # short fragments that can't meaningfully match. Marked truncated so
        # _score() won't give them token_set_ratio's subset forgiveness.
        if ': ' in t:
            pre, post = t.split(': ', 1)
            if pre.strip():
                variants.append((pre.strip(), True))
            if post.strip():
                variants.append((post.strip(), True))
        seen, out = set(), []
        for v, trunc in variants:
            if v not in seen:
                seen.add(v)
                out.append((v, trunc))
        return out

    # ── Source prep ───────────────────────────────────────────────────────────

    def _prep_source(self, src_df):
        """Drop alt-content rows (if configured), deduplicate to one row per
        title, parse dates. Subclasses with multiple rows per ID (e.g. Gower's
        snapshot rows) should override to pick the preferred row before
        calling super()."""
        src = src_df.copy()

        if self.ALT_CONTENT_COL and self.ALT_CONTENT_COL in src.columns:
            n_before = len(src)
            src = src[src[self.ALT_CONTENT_COL].fillna(False) == False].reset_index(drop=True)
            print(f"Dropped {n_before - len(src):,} alt-content rows "
                  f"({len(src):,} {self.SOURCE_LABEL} rows remain)")

        n_before = len(src)
        src = src.drop_duplicates(subset=[self.ID_COL], keep="first").reset_index(drop=True)
        if n_before != len(src):
            print(f"Deduplicated {self.SOURCE_LABEL}: {n_before:,} rows → "
                  f"{len(src):,} unique titles")

        src["_release_date"] = pd.to_datetime(src[self.DATE_COL], errors="coerce")
        # Strip tz so the days_diff arithmetic doesn't fight tz-naive EVT dates
        # (build_mapping strips tz from _evt_date the same way) — some Snowflake
        # extracts return tz-aware timestamps for this column, others don't.
        if hasattr(src["_release_date"], "dt") and src["_release_date"].dt.tz is not None:
            src["_release_date"] = src["_release_date"].dt.tz_convert(None)
        src["_year"]         = src["_release_date"].dt.year
        return src

    # ── Score a single EVT film against a source slice ───────────────────────

    def _score_candidates(self, evt_title, evt_date, src_slice):
        """Return top-K (id, score, matched_title_value, matched_via_col, year, days_diff, src_row)."""
        evt_variants = self._article_variants(evt_title)
        all_scored = []
        for _, cr in src_slice.iterrows():
            best_score = 0.0
            best_col   = None
            best_value = ""
            for col in self.TITLE_COLS:
                val = cr.get(col, "") or ""
                for variant, is_truncated in evt_variants:
                    s = self._score(variant, val, allow_subset=not is_truncated)
                    if s > best_score:
                        best_score = s
                        best_col   = col
                        best_value = val
            cr_date = cr["_release_date"]
            if pd.isna(evt_date) or pd.isna(cr_date):
                days_diff = 9999
            else:
                days_diff = abs((evt_date - cr_date).days)
            all_scored.append((
                cr[self.ID_COL], best_score, best_value, best_col,
                int(cr["_year"]) if pd.notna(cr["_year"]) else None,
                days_diff,
                cr,
            ))
        all_scored.sort(key=lambda x: (-x[1], x[5]))
        return all_scored

    def _build_record(self, film_id, film_title, scored, confidence_label=None):
        """Turn the top scored candidate (or an empty result) into a cache row."""
        record = {c: None for c in self.CACHE_COLS}
        for c in self.carry_cols:
            record[f"{self._carry_prefix()}_{c}"] = None
        record.update({"film_id": film_id, "film": film_title})

        if not scored or scored[0][1] < self.match_thresh:
            record["match_score"]      = round(scored[0][1], 3) if scored else 0.0
            record["match_confidence"] = confidence_label or "unmatched"
            return record

        src_id, best_score, matched_value, matched_col, _best_year, best_days, src_row = scored[0]

        # A title match this far from the EVT release date is very likely a
        # different release entirely (a sequel/remake/re-release sharing the
        # base title, e.g. FROZEN vs Frozen 3, THE WOLF MAN vs Wolf Man) —
        # reject outright rather than just demoting out of "high", so it
        # doesn't linger in the review file looking like a plausible
        # candidate regardless of text score. Manual overrides
        # (confidence_label="manual") are exempt — that's an explicit human
        # decision, not something this heuristic should second-guess.
        if confidence_label is None and best_days != 9999 and best_days > self.MAX_DAYS_HIGH_CONF:
            record["match_score"]      = round(best_score, 3)
            record["match_confidence"] = "unmatched"
            record["days_diff"]        = best_days
            return record

        is_high = best_score >= self.HIGH_CONFIDENCE_SCORE
        record.update({
            self.ID_FIELD:        src_id,
            self.TITLE_FIELD:     src_row.get(self.TITLE_COLS[0], ""),  # canonical title
            self.DATE_FIELD:      src_row["_release_date"] if pd.notna(src_row["_release_date"]) else None,
            "match_score":        round(best_score, 3),
            self.MATCHED_TITLE_FIELD: matched_value,
            "matched_via_col":    matched_col,
            "match_confidence":   confidence_label or ("high" if is_high else "borderline"),
            "days_diff":          None if best_days == 9999 else best_days,
        })
        for c in self.carry_cols:
            record[f"{self._carry_prefix()}_{c}"] = src_row.get(c)
        return record

    # ── Variant propagation ──────────────────────────────────────────────────

    @staticmethod
    def _strip_variant(title: str) -> str:
        """Strip format/event/festival variant tags; return uppercase base title."""
        t = str(title).upper().strip()
        for pat, repl in _VARIANT_STRIP:
            t = pat.sub(repl, t)
        return re.sub(r"\s+", " ", t).strip()

    def _propagate_variants(self, films: pd.DataFrame) -> int:
        """
        Inherit the matched ID from a matched EVT film to all unmatched variants.

        For each unmatched film whose title reduces to a different base (via
        _strip_variant), search matched films in scope for the same base title
        within ±1 year. The highest-confidence match wins (manual > high >
        borderline). Assigned confidence is 'variant', match_score=1.0 so the
        row is skipped on future incremental runs.

        Mirrors the consolidate_all_admits pattern from
        cinema_admits_models/encode_helper.py — one source ID covers all EVT
        film_id variants (3D, IMAX, GC, sing-along …) of the same base film.

        Returns the number of films newly propagated.
        """
        films = films.copy()
        films["_base"] = films["film"].apply(self._strip_variant)
        if "_evt_date" not in films.columns:
            films["_evt_date"] = pd.to_datetime(films.get("rel_at"), errors="coerce")
        if films["_evt_date"].dt.tz is not None:
            films["_evt_date"] = films["_evt_date"].dt.tz_convert(None)

        # Films eligible for propagation: unmatched true variants (base ≠ original)
        conf_map = self._cache.set_index("film_id")["match_confidence"].to_dict()
        to_propagate = films[
            films["film_id"].apply(
                lambda fid: conf_map.get(fid, "unmatched") in ("unmatched", "variant")
            ) &
            (films["_base"] != films["film"].str.upper().str.strip())
        ]
        if to_propagate.empty:
            return 0

        # Keeper pool: matched films in scope (any confidence ≥ threshold)
        matched_confs = {"high", "borderline", "manual"}
        matched_ids   = set(self._cache.loc[
            self._cache["match_confidence"].isin(matched_confs), "film_id"
        ])
        keepers = films[films["film_id"].isin(matched_ids)].copy()
        keepers["_base"] = keepers["film"].apply(self._strip_variant)
        cache_idx = self._cache.set_index("film_id")
        conf_prio = {"manual": 0, "high": 1, "borderline": 2}

        n_propagated = 0
        for _, vrow in to_propagate.iterrows():
            base  = vrow["_base"]
            vdate = vrow["_evt_date"]

            same_base = keepers[keepers["_base"] == base]
            if same_base.empty:
                continue

            # ±1 year date filter
            if pd.notna(vdate):
                same_base = same_base[same_base["_evt_date"].apply(
                    lambda d: abs((vdate - d).days) <= 365 if pd.notna(d) else True
                )]
            if same_base.empty:
                continue

            # Highest-confidence keeper
            best_keeper, best_prio = None, 9
            for _, kr in same_base.iterrows():
                if kr["film_id"] not in cache_idx.index:
                    continue
                crow = cache_idx.loc[kr["film_id"]]
                if isinstance(crow, pd.DataFrame):
                    crow = crow.iloc[0]
                prio = conf_prio.get(crow["match_confidence"], 9)
                if prio < best_prio:
                    best_prio, best_keeper = prio, crow

            if best_keeper is None:
                continue

            record = {c: None for c in self.CACHE_COLS}
            for c in self.carry_cols:
                record[f"{self._carry_prefix()}_{c}"] = None
            record.update({
                "film_id":              vrow["film_id"],
                "film":                 vrow["film"],
                self.ID_FIELD:          best_keeper[self.ID_FIELD],
                self.TITLE_FIELD:       best_keeper[self.TITLE_FIELD],
                self.DATE_FIELD:        best_keeper[self.DATE_FIELD],
                "match_score":          1.0,
                self.MATCHED_TITLE_FIELD: best_keeper[self.MATCHED_TITLE_FIELD],
                "matched_via_col":      "variant",
                "match_confidence":     "variant",
                "days_diff":            best_keeper["days_diff"],
            })
            for c in self.carry_cols:
                col = f"{self._carry_prefix()}_{c}"
                if col in best_keeper.index:
                    record[col] = best_keeper[col]

            self._cache = (
                pd.concat([self._cache, pd.DataFrame([record])], ignore_index=True)
                .drop_duplicates("film_id", keep="last")
            )
            n_propagated += 1
            print(f"  [variant] {vrow['film']!r} → {self.ID_FIELD}={best_keeper[self.ID_FIELD]} "
                  f"(base: {base!r})")

        return n_propagated

    # ── Exclusivity: one source row shouldn't back two unrelated EVT films ───

    def _resolve_duplicate_claims(self, films: pd.DataFrame, src: pd.DataFrame,
                                   max_rounds: int = 5) -> int:
        """
        Each EVT film independently picks its own best-scoring source row, so
        nothing stops two genuinely different films from both landing on the
        same source row (e.g. "PROJECT X" and "PROJECT HAIL MARY" both
        scoring against Gower's "Project Hail Mary" — the exact match should
        win and the impostor should look elsewhere). Rows that legitimately
        share a source ID on purpose (format/language variants of the same
        EVT film — matched independently here, not via _propagate_variants,
        e.g. 5 language-dub bookings of the same title) are left alone: they
        share a common _strip_variant base title with each other.

        For every source ID claimed by ≥2 films with *different* base
        titles, keep the best claim (confidence, then score, then days_diff)
        and re-score every other claimant against the source pool with that
        ID removed. Iterates (bounded by max_rounds) because bumping a loser
        to its next-best candidate can create a new collision with a
        different winner — this is what makes it "iterative": a chain of
        collisions resolves one link at a time.

        A film that loses a contested ID is permanently banned from
        reclaiming that same ID for the rest of this call (across rounds),
        not just excluded from it for the one re-score that just bumped it.
        Without that, two source rows that are themselves near-duplicates
        (e.g. Gower carrying two rows for what's really the same title) can
        make two films volley back and forth between them every round —
        each round's from-scratch conflict scan has no memory of what was
        already tried, so it can re-propose the exact swap that caused the
        previous round's collision. The per-film ban set makes forward
        progress monotonic: each bounce permanently shrinks that film's
        remaining candidate pool, so it can only cycle a finite number of
        times before settling (possibly on "unmatched") rather than
        oscillating for the full max_rounds.
        """
        evt_idx = films.set_index("film_id")
        conf_rank = {"manual": 0, "high": 1, "borderline": 2}
        total_reassigned = 0
        banned: dict = {}  # film_id -> set of source IDs it's already lost

        for _ in range(max_rounds):
            in_scope = self._cache[
                self._cache["film_id"].isin(films["film_id"])
                & self._cache["match_confidence"].isin(["manual", "high", "borderline"])
            ]
            any_conflict = False

            for src_id, group in in_scope.groupby(self.ID_FIELD):
                if pd.isna(src_id) or len(group) < 2:
                    continue
                bases = group["film"].apply(self._strip_variant).unique()
                if len(bases) <= 1:
                    continue  # same film's own format/language variants — fine to share

                any_conflict = True
                ranked = group.assign(
                    _conf_rank=group["match_confidence"].map(conf_rank).fillna(3)
                ).sort_values(
                    ["_conf_rank", "match_score", "days_diff"],
                    ascending=[True, False, True],
                )
                loser_ids = ranked["film_id"].iloc[1:]

                for fid in loser_ids:
                    if fid not in evt_idx.index:
                        continue
                    banned.setdefault(fid, set()).add(src_id)
                    title    = evt_idx.at[fid, "film"]
                    evt_date = evt_idx.at[fid, "_evt_date"]
                    remaining_src = src[~src[self.ID_COL].isin(banned[fid])]
                    rescored = self._score_candidates(title, evt_date, remaining_src)
                    new_record = self._build_record(fid, title, rescored)
                    self._cache = (
                        pd.concat([self._cache, pd.DataFrame([new_record])], ignore_index=True)
                        .drop_duplicates("film_id", keep="last")
                    )
                    total_reassigned += 1
                    print(f"  [conflict] {title!r} lost {self.ID_FIELD}={src_id} to "
                          f"{ranked.iloc[0]['film']!r} → "
                          f"re-matched to {new_record.get(self.ID_FIELD)!r} "
                          f"({new_record['match_confidence']})")

            if not any_conflict:
                break

        return total_reassigned

    # ── Manual overrides (highest precedence) ────────────────────────────────

    def _load_overrides(self) -> pd.DataFrame:
        """
        Load manual overrides from CSV (preferred) and parquet (legacy).
        Returns DataFrame[film_id, id, evt_film]. CSV wins on duplicate film_id.
        Rows with empty override id are skipped (treat as placeholders/TODOs).
        """
        rows: list[dict] = []
        override_col = f"manual_override_{self.ID_FIELD}"

        # Parquet (legacy — written by the old review-file flow)
        if os.path.exists(self.MANUAL_OVERRIDES):
            try:
                mo = pd.read_parquet(self.MANUAL_OVERRIDES).dropna(subset=[override_col])
                for _, r in mo.iterrows():
                    rows.append({"film_id": r["film_id"], "id": r[override_col],
                                 "evt_film": None})
                print(f"  Parquet overrides loaded: {len(mo)} row(s)")
            except Exception as e:
                print(f"  Parquet overrides skipped: {e}")

        # CSV (preferred — required columns: film_id, <ID_FIELD>
        #       optional columns: evt_film, <TITLE_FIELD>, note)
        if os.path.exists(self.MANUAL_OVERRIDES_CSV):
            try:
                csv_mo = pd.read_csv(self.MANUAL_OVERRIDES_CSV).dropna(subset=[self.ID_FIELD])
                for _, r in csv_mo.iterrows():
                    rows.append({"film_id": r["film_id"], "id": r[self.ID_FIELD],
                                 "evt_film": r.get("evt_film")})
                print(f"  CSV overrides loaded: {len(csv_mo)} row(s)")
            except Exception as e:
                print(f"  CSV overrides skipped: {e}")

        if not rows:
            return pd.DataFrame(columns=["film_id", "id", "evt_film"])
        return (
            pd.DataFrame(rows)
            .drop_duplicates("film_id", keep="last")  # CSV wins over parquet
            .reset_index(drop=True)
        )

    def _apply_manual_overrides(self, films, src_prepped):
        overrides = self._load_overrides()
        if overrides.empty:
            return 0

        film_ids = set(films["film_id"])
        overrides = overrides[overrides["film_id"].isin(film_ids)]
        if overrides.empty:
            return 0

        src_indexed = src_prepped.set_index(self.ID_COL)
        applied = 0
        for _, ov in overrides.iterrows():
            fid      = ov["film_id"]
            src_id   = ov["id"]
            expected = ov.get("evt_film")

            name_rows = films.loc[films["film_id"] == fid, "film"]
            if name_rows.empty:
                continue
            actual = name_rows.iloc[0]

            # Validate: if evt_film was recorded in the override, the film_id must
            # still point to the same title. film_ids can be recycled in EVT's DB.
            if pd.notna(expected) and str(expected).upper().strip() != actual.upper().strip():
                print(f"  [override SKIPPED] film_id={fid} — CSV has {expected!r} "
                      f"but current dataset has {actual!r}. Update the CSV with the correct film_id.")
                continue

            if src_id not in src_indexed.index:
                print(f"  [override] {actual} → {src_id} — NOT IN {self.SOURCE_LABEL.upper()} EXTRACT")
                continue

            src_row = src_indexed.loc[src_id]
            if isinstance(src_row, pd.DataFrame):
                src_row = src_row.iloc[0]
            fake_scored = [(
                src_id, 1.0, src_row.get(self.TITLE_COLS[0], ""), "manual",
                int(src_row["_year"]) if pd.notna(src_row["_year"]) else None,
                9999, src_row,
            )]
            record = self._build_record(fid, actual, fake_scored, confidence_label="manual")
            self._cache = (
                pd.concat([self._cache, pd.DataFrame([record])], ignore_index=True)
                .drop_duplicates("film_id", keep="last")
            )
            applied += 1
            print(f"  [override] {actual} → {src_id} ({src_row.get(self.TITLE_COLS[0])})")

        if applied:
            self._save_cache()
        return applied

    # ── Main entry point ─────────────────────────────────────────────────────

    def build_mapping(self, films_df: pd.DataFrame, src_df: pd.DataFrame):
        """
        Match EVT films to source rows.

        Parameters
        ----------
        films_df : DataFrame with [film_id, film, rel_at, dstbtr]
        src_df   : Output of this source's SQL extract (lowercase columns)
        """
        films = (
            films_df[["film_id", "film", "rel_at", "dstbtr"]]
            .drop_duplicates("film_id")
            .copy()
        )
        films["_evt_date"] = pd.to_datetime(films["rel_at"], errors="coerce")
        # Strip tz so the days_diff arithmetic doesn't fight tz-naive source dates
        if hasattr(films["_evt_date"], "dt") and films["_evt_date"].dt.tz is not None:
            films["_evt_date"] = films["_evt_date"].dt.tz_convert(None)
        films["_year"] = films["_evt_date"].dt.year

        src = self._prep_source(src_df)

        # 1. Apply manual overrides first.
        manual_applied = self._apply_manual_overrides(films, src)

        # 2. Skip films already matched at threshold (allow re-try of cached misses).
        cached_ok_set = set(
            self._cache.loc[
                self._cache["match_score"].fillna(0) >= self.match_thresh,
                "film_id",
            ].tolist()
        )
        to_match = films[~films["film_id"].isin(cached_ok_set)].reset_index(drop=True)
        n_retry = (
            films["film_id"].isin(self._cache["film_id"])
            & ~films["film_id"].isin(cached_ok_set)
        ).sum()
        print(f"Films to process: {len(to_match)}  "
              f"({len(cached_ok_set & set(films['film_id']))} already matched, "
              f"{n_retry} cached mismatches retrying, {manual_applied} manual)")

        # 3. Match loop.
        review_candidates = {}
        n_high = n_border = n_unmatched = 0
        for i, (_, row) in enumerate(to_match.iterrows(), 1):
            evt_year  = int(row["_year"]) if pd.notna(row["_year"]) else None
            evt_date  = row["_evt_date"]

            if evt_year is not None:
                sub = src[src["_year"].between(evt_year - self.YEAR_TOL,
                                                evt_year + self.YEAR_TOL)]
            else:
                sub = src

            all_scored = self._score_candidates(row["film"], evt_date, sub)
            review_candidates[row["film_id"]] = all_scored[:5]

            record = self._build_record(row["film_id"], row["film"], all_scored)
            if record["match_confidence"] == "unmatched":
                n_unmatched += 1
            elif record["match_confidence"] == "high":
                n_high += 1
            else:
                n_border += 1

            self._cache = (
                pd.concat([self._cache, pd.DataFrame([record])], ignore_index=True)
                .drop_duplicates("film_id", keep="last")
            )
            if i % 200 == 0:
                self._save_cache()
                print(f"  [{i:>4}/{len(to_match)}]  high={n_high} borderline={n_border} unmatched={n_unmatched}")

        self._save_cache()

        # 4. Propagate matched ID to format/event variants (3D, IMAX, GC, sing-along…)
        #    that share a base title with an already-matched film in scope.
        n_variant = self._propagate_variants(films)
        if n_variant:
            self._save_cache()
            print(f"Variant propagation: {n_variant} film(s) assigned from base title.")

        # 5. Resolve source rows independently claimed by unrelated EVT films
        #    (e.g. "PROJECT X" and "PROJECT HAIL MARY" both matching Gower's
        #    "Project Hail Mary") — the best claim wins, losers are re-matched
        #    against the remaining pool.
        n_reassigned = self._resolve_duplicate_claims(films, src)
        if n_reassigned:
            self._save_cache()
            print(f"Conflict resolution: {n_reassigned} film(s) re-matched off a contested source row.")

        self.mapping_df = (
            self._cache[self._cache["film_id"].isin(films["film_id"])]
            .copy()
            .reset_index(drop=True)
        )
        self._write_review_file(films, review_candidates)
        self._sync_to_s3()

        conf = self.mapping_df["match_confidence"].fillna("legacy").value_counts().to_dict()
        print(f"\nConfidence breakdown (this run's scope): {conf}")
        return self

    # ── Review file ──────────────────────────────────────────────────────────

    def _write_review_file(self, films_df, review_candidates):
        cache = self._cache.copy()
        needs_review = cache["match_confidence"].isin(["borderline", "unmatched"])
        in_scope     = cache["film_id"].isin(films_df["film_id"])
        review = cache[needs_review & in_scope].copy()
        if review.empty:
            print("Review file: nothing borderline/unmatched in scope — skipping write")
            self._last_review_df = None
            return

        override_col = f"manual_override_{self.ID_FIELD}"
        film_meta = films_df.set_index("film_id")[["rel_at", "dstbtr"]]
        rows = []
        for _, r in review.iterrows():
            fid = r["film_id"]
            base = {
                "film_id": fid,
                "film":    r["film"],
                "rel_at":  film_meta.at[fid, "rel_at"] if fid in film_meta.index else None,
                "dstbtr":  film_meta.at[fid, "dstbtr"] if fid in film_meta.index else None,
                "current_match_confidence": r["match_confidence"],
                f"current_{self.ID_FIELD}":  r[self.ID_FIELD],
                f"current_{self._carry_prefix()}_title": r[self.MATCHED_TITLE_FIELD],
                "current_matched_via":      r.get("matched_via_col"),
                "current_match_score":      r["match_score"],
                "current_days_diff":        r["days_diff"],
                override_col:               None,   # ← user fills this
            }
            cands = review_candidates.get(fid, [])
            for k in range(5):
                if k < len(cands):
                    src_id, score, title, via_col, year, days, _ = cands[k]
                    base[f"candidate_{k+1}_{self.ID_FIELD}"] = src_id
                    base[f"candidate_{k+1}_title"]    = title
                    base[f"candidate_{k+1}_via_col"]  = via_col
                    base[f"candidate_{k+1}_year"]     = year
                    base[f"candidate_{k+1}_score"]    = round(score, 3)
                    base[f"candidate_{k+1}_days"]     = None if days == 9999 else days
                else:
                    base[f"candidate_{k+1}_{self.ID_FIELD}"] = None
                    base[f"candidate_{k+1}_title"]    = None
                    base[f"candidate_{k+1}_via_col"]  = None
                    base[f"candidate_{k+1}_year"]     = None
                    base[f"candidate_{k+1}_score"]    = None
                    base[f"candidate_{k+1}_days"]     = None
            rows.append(base)

        out = pd.DataFrame(rows).sort_values(
            ["current_match_confidence", "current_match_score"],
            ascending=[True, True],
        )
        os.makedirs(os.path.dirname(self.REVIEW_PATH), exist_ok=True)
        out.to_parquet(self.REVIEW_PATH, index=False)
        n_un = (out["current_match_confidence"] == "unmatched").sum()
        n_bd = (out["current_match_confidence"] == "borderline").sum()
        print(f"Review file → {self.REVIEW_PATH}  ({n_un} unmatched, {n_bd} borderline). "
              f"Fill {override_col} and save as {self.MANUAL_OVERRIDES_CSV} to apply on next run.")
        self._last_review_df = out

    # ── Mirror local cache/review outputs to S3 ───────────────────────────────

    def _sync_to_s3(self):
        """Best-effort copy of the local cache (+ review file, if one was just
        written) to S3, under S3_TITLE_MATCHING_PREFIX/{name}/ — a sibling
        prefix to the LLM extraction paths' S3_PREFIX, not nested under it.
        The local parquet under CACHE_PATH/REVIEW_PATH stays authoritative
        (title_matcher's skip-if-already-matched logic reads it back on the
        next run); this is purely a mirror for other consumers. Never raises
        — a transient credentials/network issue here shouldn't undo a
        successful local matching run."""
        if not S3_BUCKET:
            return
        name = self.SOURCE_LABEL.lower()
        try:
            cache_filename = os.path.basename(self.CACHE_PATH)
            s3_checkpoint.write_parquet(name, cache_filename, self._cache,
                                         prefix=S3_TITLE_MATCHING_PREFIX)
            uris = [s3_checkpoint.s3_uri(name, cache_filename, prefix=S3_TITLE_MATCHING_PREFIX)]
            if self._last_review_df is not None:
                review_filename = os.path.basename(self.REVIEW_PATH)
                s3_checkpoint.write_parquet(name, review_filename, self._last_review_df,
                                             prefix=S3_TITLE_MATCHING_PREFIX)
                uris.append(s3_checkpoint.s3_uri(name, review_filename, prefix=S3_TITLE_MATCHING_PREFIX))
            print(f"Synced to S3: {', '.join(uris)}")
        except Exception as e:
            print(f"[warn] S3 sync skipped for {self.SOURCE_LABEL} ({e}) — "
                  f"local {self.CACHE_PATH} is unaffected and still authoritative")

    # ── Re-score the existing cache against the current algorithm/data ───────

    def re_score(self, src_df: pd.DataFrame, films_df: pd.DataFrame | None = None):
        """
        Re-evaluate every cached film against the (possibly updated) source
        extract. Manual overrides are left untouched. Equivalent of
        StudioLookup.re_score_cache but with no API refetch — source row
        values are read directly off the updated extract.
        """
        src = self._prep_source(src_df)
        n  = len(self._cache)
        print(f"Re-scoring {n:,} cached films against {len(src):,} {self.SOURCE_LABEL} rows")

        evt_dates = {}
        if films_df is not None:
            for _, r in films_df[["film_id", "rel_at"]].drop_duplicates("film_id").iterrows():
                d = pd.to_datetime(r["rel_at"], errors="coerce")
                if hasattr(d, "tz") and d is not pd.NaT and d.tz is not None:
                    d = d.tz_localize(None)
                evt_dates[r["film_id"]] = d

        new_rows, review_candidates = [], {}
        n_changed = n_unchanged = n_demoted = n_promoted = n_skipped_manual = n_skipped_variant = 0
        preserved_variants = []  # kept so we can re-add if films_df unavailable

        for _, r in self._cache.iterrows():
            row_dict = r.to_dict()
            if row_dict.get("match_confidence") == "manual":
                new_rows.append(row_dict)
                n_skipped_manual += 1
                continue
            if row_dict.get("match_confidence") == "variant":
                # Re-propagated after main loop based on updated scores; skip here.
                preserved_variants.append(row_dict)
                n_skipped_variant += 1
                continue

            fid      = row_dict["film_id"]
            title    = row_dict["film"]
            evt_date = evt_dates.get(fid)
            evt_year = evt_date.year if (evt_date is not None and pd.notna(evt_date)) else None

            if evt_year is not None:
                sub = src[src["_year"].between(evt_year - self.YEAR_TOL,
                                                evt_year + self.YEAR_TOL)]
            else:
                sub = src

            all_scored = self._score_candidates(title, evt_date, sub)
            review_candidates[fid] = all_scored[:5]
            new_record = self._build_record(fid, title, all_scored)

            cached_id = row_dict.get(self.ID_FIELD)
            new_id    = new_record.get(self.ID_FIELD)
            was_matched = pd.notna(cached_id) and (row_dict.get("match_score") or 0) >= self.match_thresh
            is_matched  = new_record["match_confidence"] in ("high", "borderline")

            if pd.notna(new_id) and pd.notna(cached_id) and new_id == cached_id:
                if was_matched and new_record["match_confidence"] == "high" and (row_dict.get("match_score") or 0) < self.HIGH_CONFIDENCE_SCORE:
                    n_promoted += 1
                else:
                    n_unchanged += 1
            elif is_matched and not was_matched:
                n_promoted += 1
            elif was_matched and not is_matched:
                n_demoted += 1
            elif is_matched and was_matched:
                n_changed += 1
            else:
                n_unchanged += 1
            new_rows.append(new_record)

        new_cache = pd.DataFrame(new_rows)
        for c in self.CACHE_COLS:
            if c not in new_cache.columns:
                new_cache[c] = None
        # Preserve carried columns even if they weren't in CACHE_COLS
        ordered = self.CACHE_COLS + [c for c in new_cache.columns if c not in self.CACHE_COLS]
        self._cache = new_cache[ordered].drop_duplicates("film_id", keep="last").reset_index(drop=True)
        self._save_cache()

        print(f"\nRe-score done — changed={n_changed} unchanged={n_unchanged} "
              f"promoted={n_promoted} demoted={n_demoted} "
              f"manual_skipped={n_skipped_manual} variant_skipped={n_skipped_variant}")
        print("Confidence breakdown:")
        print(self._cache["match_confidence"].fillna("legacy").value_counts().to_string())

        # Re-propagate variants against the freshly re-scored cache.
        if films_df is not None:
            n_variant = self._propagate_variants(films_df)
            if n_variant:
                self._save_cache()
                print(f"  Re-propagated {n_variant} variant film(s).")
        elif preserved_variants:
            # No date context to re-propagate — restore old variant rows as-is.
            var_df = pd.DataFrame(preserved_variants)
            self._cache = (
                pd.concat([self._cache, var_df], ignore_index=True)
                .drop_duplicates("film_id", keep="last")
                .reset_index(drop=True)
            )
            self._save_cache()
            print(f"  Preserved {len(preserved_variants)} variant row(s) "
                  f"(pass films_df to re_score() to re-propagate).")

        if films_df is None:
            films_df = self._cache[["film_id", "film"]].copy()
            films_df["rel_at"] = None
            films_df["dstbtr"] = None

        if "_evt_date" not in films_df.columns:
            films_df["_evt_date"] = pd.to_datetime(films_df.get("rel_at"), errors="coerce")
            if films_df["_evt_date"].dt.tz is not None:
                films_df["_evt_date"] = films_df["_evt_date"].dt.tz_convert(None)

        n_reassigned = self._resolve_duplicate_claims(films_df, src)
        if n_reassigned:
            self._save_cache()
            print(f"Conflict resolution: {n_reassigned} film(s) re-matched off a contested source row.")

        self._write_review_file(films_df, review_candidates)
        self._sync_to_s3()
        return self
