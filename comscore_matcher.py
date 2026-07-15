"""
comscore_matcher.py
-------------------
Match Comscore box-office rows (IBOE_TITLES + IBOE_FLASH_GROSS) → EVT film_id.

Matching logic mirrors the latest TMDB StudioLookup
(cinema_admits_models/depreciated/tmdb/studio_lookup.py):

  - rapidfuzz ratio + token_sort_ratio on NFKD-normalised titles
  - variant/festival prefix stripping ("GC", "3D", "IMAX", "TFF -", ...)
  - length-ratio guard (rejects short-title impostors like AVATAR→TÁR)
  - ±1-year window over Comscore release_date vs EVT rel_at
  - date-proximity tie-breaker when titles tie
  - confidence tiers: high (>=0.92 and ≤365 days off), borderline, unmatched, manual
  - review file + user-editable manual-override parquet

No API calls — Comscore data is pulled once via SQL and matched in-memory,
so the entire flow is CPU/regex work.

Outputs (under DATA_DIR/comscore/):
  - comscore_cache.parquet           keyed on EVT film_id
  - comscore_review_needed.parquet   borderline+unmatched, top-5 candidates
  - comscore_manual_overrides.parquet user fills manual_override_cs_id
"""

import os
import re
import unicodedata
import pandas as pd

try:
    from rapidfuzz import fuzz as _rf_fuzz
    _USE_RAPIDFUZZ = True
except ImportError:
    from difflib import SequenceMatcher
    _USE_RAPIDFUZZ = False

from config import DATA_DIR

# Variant-strip patterns — mirrored from cinema_admits_models/encode_helper.py.
# Applied in order; each pattern is substituted once (not iteratively).
# Used to identify EVT film_ids that are format/event/festival variants of
# the same base film so one Comscore cs_id can be inherited by all variants.
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


class ComscoreMatcher:

    COMSCORE_DIR         = DATA_DIR / "comscore"
    CACHE_PATH           = str(COMSCORE_DIR / "comscore_cache.parquet")
    REVIEW_PATH          = str(COMSCORE_DIR / "comscore_review_needed.parquet")
    MANUAL_OVERRIDES     = str(COMSCORE_DIR / "comscore_manual_overrides.parquet")  # legacy
    MANUAL_OVERRIDES_CSV = str(COMSCORE_DIR / "comscore_manual_overrides.csv")      # preferred

    HIGH_CONFIDENCE_SCORE = 0.92
    MAX_DAYS_HIGH_CONF    = 365
    MIN_LENGTH_RATIO      = 0.5
    YEAR_TOL              = 1

    # IBOE_TITLES columns we score against (max wins). Lower-cased by
    # SnowFlakeBase.return_query_output, so use lowercase here.
    # upper_name is a duplicate of film_name capitalised — _normalise_title
    # lowercases anyway, so the extra column doesn't change scores; kept for
    # symmetry with the SQL and in case Comscore ever diverges them.
    TITLE_COLS = ["film_name", "upper_name", "title_aka", "us_title_name", "short_name"]
    DATE_COL   = "release_date"
    ID_COL     = "title_global_id"

    # Variant/festival prefix regexes — same as StudioLookup.
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

    # Presentation/format suffixes appended by EVT that don't appear in Comscore.
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

    CACHE_COLS = [
        "film_id", "film",                          # EVT side
        "cs_id", "cs_title", "cs_release_date",     # Comscore identity (cs_title = film_name)
        "match_score", "matched_cs_title",          # the column-value that actually won the score
        "matched_via_col",                          # which TITLE_COL: film_name/title_aka/us_title_name/short_name/upper_name
        "match_confidence",                         # high|borderline|unmatched|manual
        "days_diff",
    ]

    def __init__(self, match_thresh: float = 0.80, carry_cols: list[str] | None = None):
        """
        carry_cols : Comscore columns to copy into the cache alongside the
                     identity/match columns. Defaults to None (no extras —
                     join back to the raw Comscore extract on cs_id when
                     you need the value columns).
        """
        self.match_thresh = match_thresh
        self.carry_cols   = carry_cols or []
        self.mapping_df   = None
        self._cache       = self._load_cache()

    # ── Cache I/O ────────────────────────────────────────────────────────────

    def _load_cache(self):
        if os.path.exists(self.CACHE_PATH):
            df = pd.read_parquet(self.CACHE_PATH)
            print(f"Comscore cache: {len(df)} films from {self.CACHE_PATH}")
            return df
        return pd.DataFrame(columns=self.CACHE_COLS)

    def _save_cache(self):
        os.makedirs(os.path.dirname(self.CACHE_PATH), exist_ok=True)
        self._cache.to_parquet(self.CACHE_PATH, index=False)

    # ── Title normalisation + scoring (ported from StudioLookup) ─────────────

    @staticmethod
    def _normalise_title(s):
        if not s:
            return ""
        s = str(s)
        s = ComscoreMatcher._LANG_SUFFIX.sub('', s)
        s = ComscoreMatcher._FORMAT_SUFFIX.sub('', s)
        s = ComscoreMatcher._VARIANT_PREFIX.sub('', s)
        s = ComscoreMatcher._FESTIVAL_PREFIX.sub('', s)
        s = ComscoreMatcher._VARIANT_SUFFIX.sub('', s)
        s = ComscoreMatcher._YEAR_PAREN.sub('', s)
        s = unicodedata.normalize('NFKD', s)
        s = ''.join(c for c in s if not unicodedata.combining(c))
        # Normalise punctuation: collapse hyphens/colons/dashes to a space so
        # "DUNE - PART TWO" and "DUNE: PART TWO" score identically.
        s = re.sub(r'[\-–—:/\\|]+', ' ', s)
        s = re.sub(r'[^\w\s]', '', s)
        return ' '.join(s.split()).lower().strip()

    @staticmethod
    def _score(a, b):
        a_n = ComscoreMatcher._normalise_title(a)
        b_n = ComscoreMatcher._normalise_title(b)
        if not a_n or not b_n:
            return 0.0
        la, lb = len(a_n), len(b_n)
        if min(la, lb) / max(la, lb) < ComscoreMatcher.MIN_LENGTH_RATIO:
            return 0.0
        if _USE_RAPIDFUZZ:
            return max(
                _rf_fuzz.ratio(a_n, b_n),
                _rf_fuzz.token_sort_ratio(a_n, b_n),
                # token_set_ratio handles "X presents Y" vs "X: Y" (Hobbs & Shaw pattern)
                # where one side has an extra connective word not in the other.
                _rf_fuzz.token_set_ratio(a_n, b_n),
            ) / 100.0
        return SequenceMatcher(None, a_n, b_n).ratio()

    @staticmethod
    def _article_variants(title: str) -> list[str]:
        """Return article-transposed and colon-split forms to broaden matching coverage."""
        variants = [title]
        t = title.strip()
        upper = t.upper()
        # "THE X" / "A X" / "AN X"  →  "X, THE" / "X, A" / "X, AN"
        for article in ('THE ', 'A ', 'AN '):
            if upper.startswith(article):
                rest = t[len(article):]
                variants.append(f"{rest}, {article.strip()}")
                break
        # "X, THE" / "X, A" / "X, AN"  →  "THE X" / "A X" / "AN X"
        for article in (', THE', ', A', ', AN'):
            if upper.endswith(article):
                rest = t[:-len(article)]
                variants.append(f"{article[2:]} {rest}")
                break
        # Colon-split variants: Comscore often omits either the subtitle
        # ("Peter Rabbit 2: The Runaway" → "Peter Rabbit 2") or the franchise
        # prefix ("Star Wars: The Mandalorian and Grogu" → "The Mandalorian and
        # Grogu"). Both sides are tried; MIN_LENGTH_RATIO in _score() suppresses
        # short fragments that can't meaningfully match.
        if ': ' in t:
            pre, post = t.split(': ', 1)
            if pre.strip():
                variants.append(pre.strip())
            if post.strip():
                variants.append(post.strip())
        return list(dict.fromkeys(variants))  # deduplicate, preserve order

    # ── Comscore prep ────────────────────────────────────────────────────────

    def _prep_comscore(self, cs_df):
        """Drop alt-content rows, deduplicate to one row per title, parse dates."""
        cs = cs_df.copy()

        # Hide concert/sports/etc. — EVT side doesn't have these and they're
        # the biggest source of name-collisions ("ABBA VOYAGE" concert vs film).
        if "is_alt_content" in cs.columns:
            n_before = len(cs)
            cs = cs[cs["is_alt_content"].fillna(False) == False].reset_index(drop=True)
            print(f"Dropped {n_before - len(cs):,} alt-content rows "
                  f"({len(cs):,} film rows remain)")

        # The Comscore SQL joins IBOE_TITLES to IBOE_FLASH_GROSS_STATE_TITLE,
        # producing one row per AU state/territory for the same title_global_id.
        # All title columns are identical across state rows — only state_global_id
        # differs. Deduplicate to one row per film before scoring so each title
        # is evaluated once rather than ~8 times.
        n_before = len(cs)
        cs = cs.drop_duplicates(subset=[self.ID_COL], keep="first").reset_index(drop=True)
        if n_before != len(cs):
            print(f"Deduplicated Comscore: {n_before:,} state-level rows → "
                  f"{len(cs):,} unique titles")

        cs["_release_date"] = pd.to_datetime(cs[self.DATE_COL], errors="coerce")
        cs["_year"]         = cs["_release_date"].dt.year
        return cs

    # ── Score a single EVT film against a Comscore slice ─────────────────────

    def _score_candidates(self, evt_title, evt_date, cs_slice):
        """Return top-K (cs_id, score, matched_title_value, matched_via_col, year, days_diff, cs_row)."""
        evt_variants = self._article_variants(evt_title)
        all_scored = []
        for _, cr in cs_slice.iterrows():
            best_score = 0.0
            best_col   = None
            best_value = ""
            for col in self.TITLE_COLS:
                val = cr.get(col, "") or ""
                for variant in evt_variants:
                    s = self._score(variant, val)
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
            record[f"cs_{c}"] = None
        record.update({"film_id": film_id, "film": film_title})

        if not scored or scored[0][1] < self.match_thresh:
            record["match_score"]      = round(scored[0][1], 3) if scored else 0.0
            record["match_confidence"] = confidence_label or "unmatched"
            return record

        cs_id, best_score, matched_value, matched_col, _best_year, best_days, cs_row = scored[0]
        is_high = (
            best_score >= self.HIGH_CONFIDENCE_SCORE
            and (best_days <= self.MAX_DAYS_HIGH_CONF or best_days == 9999)
        )
        record.update({
            "cs_id":              cs_id,
            "cs_title":           cs_row.get(self.TITLE_COLS[0], ""),  # canonical film_name
            "cs_release_date":    cs_row["_release_date"] if pd.notna(cs_row["_release_date"]) else None,
            "match_score":        round(best_score, 3),
            "matched_cs_title":   matched_value,
            "matched_via_col":    matched_col,
            "match_confidence":   confidence_label or ("high" if is_high else "borderline"),
            "days_diff":          None if best_days == 9999 else best_days,
        })
        for c in self.carry_cols:
            record[f"cs_{c}"] = cs_row.get(c)
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
        Inherit cs_id from a matched EVT film to all unmatched variants.

        For each unmatched film whose title reduces to a different base (via
        _strip_variant), search matched films in scope for the same base title
        within ±1 year. The highest-confidence match wins (manual > high >
        borderline). Assigned confidence is 'variant', match_score=1.0 so the
        row is skipped on future incremental runs.

        Mirrors the consolidate_all_admits pattern from
        cinema_admits_models/encode_helper.py — one cs_id covers all EVT
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
                record[f"cs_{c}"] = None
            record.update({
                "film_id":          vrow["film_id"],
                "film":             vrow["film"],
                "cs_id":            best_keeper["cs_id"],
                "cs_title":         best_keeper["cs_title"],
                "cs_release_date":  best_keeper["cs_release_date"],
                "match_score":      1.0,
                "matched_cs_title": best_keeper["matched_cs_title"],
                "matched_via_col":  "variant",
                "match_confidence": "variant",
                "days_diff":        best_keeper["days_diff"],
            })
            for c in self.carry_cols:
                col = f"cs_{c}"
                if col in best_keeper.index:
                    record[col] = best_keeper[col]

            self._cache = (
                pd.concat([self._cache, pd.DataFrame([record])], ignore_index=True)
                .drop_duplicates("film_id", keep="last")
            )
            n_propagated += 1
            print(f"  [variant] {vrow['film']!r} → cs_id={best_keeper['cs_id']} "
                  f"(base: {base!r})")

        return n_propagated

    # ── Manual overrides (highest precedence) ────────────────────────────────

    def _load_overrides(self) -> pd.DataFrame:
        """
        Load manual overrides from CSV (preferred) and parquet (legacy).
        Returns DataFrame[film_id, cs_id, evt_film]. CSV wins on duplicate film_id.
        Rows with empty cs_id are skipped (treat as placeholders/TODOs).
        """
        rows: list[dict] = []

        # Parquet (legacy — written by the old review-file flow)
        if os.path.exists(self.MANUAL_OVERRIDES):
            try:
                mo = pd.read_parquet(self.MANUAL_OVERRIDES).dropna(subset=["manual_override_cs_id"])
                for _, r in mo.iterrows():
                    rows.append({"film_id": r["film_id"], "cs_id": r["manual_override_cs_id"],
                                 "evt_film": None})
                print(f"  Parquet overrides loaded: {len(mo)} row(s)")
            except Exception as e:
                print(f"  Parquet overrides skipped: {e}")

        # CSV (preferred — required columns: film_id, cs_id
        #       optional columns: evt_film, cs_title, note)
        if os.path.exists(self.MANUAL_OVERRIDES_CSV):
            try:
                csv_mo = pd.read_csv(self.MANUAL_OVERRIDES_CSV).dropna(subset=["cs_id"])
                for _, r in csv_mo.iterrows():
                    rows.append({"film_id": r["film_id"], "cs_id": r["cs_id"],
                                 "evt_film": r.get("evt_film")})
                print(f"  CSV overrides loaded: {len(csv_mo)} row(s)")
            except Exception as e:
                print(f"  CSV overrides skipped: {e}")

        if not rows:
            return pd.DataFrame(columns=["film_id", "cs_id", "evt_film"])
        return (
            pd.DataFrame(rows)
            .drop_duplicates("film_id", keep="last")  # CSV wins over parquet
            .reset_index(drop=True)
        )

    def _apply_manual_overrides(self, films, cs_prepped):
        overrides = self._load_overrides()
        if overrides.empty:
            return 0

        film_ids = set(films["film_id"])
        overrides = overrides[overrides["film_id"].isin(film_ids)]
        if overrides.empty:
            return 0

        cs_indexed = cs_prepped.set_index(self.ID_COL)
        applied = 0
        for _, ov in overrides.iterrows():
            fid      = ov["film_id"]
            cs_id    = ov["cs_id"]
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

            if cs_id not in cs_indexed.index:
                print(f"  [override] {actual} → {cs_id} — NOT IN COMSCORE EXTRACT")
                continue

            cs_row = cs_indexed.loc[cs_id]
            if isinstance(cs_row, pd.DataFrame):
                cs_row = cs_row.iloc[0]
            fake_scored = [(
                cs_id, 1.0, cs_row.get(self.TITLE_COLS[0], ""), "manual",
                int(cs_row["_year"]) if pd.notna(cs_row["_year"]) else None,
                9999, cs_row,
            )]
            record = self._build_record(fid, actual, fake_scored, confidence_label="manual")
            self._cache = (
                pd.concat([self._cache, pd.DataFrame([record])], ignore_index=True)
                .drop_duplicates("film_id", keep="last")
            )
            applied += 1
            print(f"  [override] {actual} → {cs_id} ({cs_row.get(self.TITLE_COLS[0])})")

        if applied:
            self._save_cache()
        return applied

    # ── Main entry point ─────────────────────────────────────────────────────

    def build_mapping(self, films_df: pd.DataFrame, cs_df: pd.DataFrame):
        """
        Match EVT films to Comscore rows.

        Parameters
        ----------
        films_df : DataFrame with [film_id, film, rel_at, dstbtr]
        cs_df    : Output of sql/comscore_extract.sql (lowercase columns)
        """
        films = (
            films_df[["film_id", "film", "rel_at", "dstbtr"]]
            .drop_duplicates("film_id")
            .copy()
        )
        films["_evt_date"] = pd.to_datetime(films["rel_at"], errors="coerce")
        # Strip tz so the days_diff arithmetic doesn't fight tz-naive Comscore dates
        if hasattr(films["_evt_date"], "dt") and films["_evt_date"].dt.tz is not None:
            films["_evt_date"] = films["_evt_date"].dt.tz_convert(None)
        films["_year"] = films["_evt_date"].dt.year

        cs = self._prep_comscore(cs_df)

        # 1. Apply manual overrides first.
        manual_applied = self._apply_manual_overrides(films, cs)

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
                sub = cs[cs["_year"].between(evt_year - self.YEAR_TOL,
                                              evt_year + self.YEAR_TOL)]
            else:
                sub = cs

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

        # 4. Propagate cs_id to format/event variants (3D, IMAX, GC, sing-along…)
        #    that share a base title with an already-matched film in scope.
        n_variant = self._propagate_variants(films)
        if n_variant:
            self._save_cache()
            print(f"Variant propagation: {n_variant} film(s) assigned from base title.")

        self.mapping_df = (
            self._cache[self._cache["film_id"].isin(films["film_id"])]
            .copy()
            .reset_index(drop=True)
        )
        self._write_review_file(films, review_candidates)

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
            return

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
                "current_cs_id":            r["cs_id"],
                "current_cs_title":         r["matched_cs_title"],
                "current_matched_via":      r.get("matched_via_col"),
                "current_match_score":      r["match_score"],
                "current_days_diff":        r["days_diff"],
                "manual_override_cs_id":    None,   # ← user fills this
            }
            cands = review_candidates.get(fid, [])
            for k in range(5):
                if k < len(cands):
                    cs_id, score, title, via_col, year, days, _ = cands[k]
                    base[f"candidate_{k+1}_cs_id"]    = cs_id
                    base[f"candidate_{k+1}_title"]    = title
                    base[f"candidate_{k+1}_via_col"]  = via_col
                    base[f"candidate_{k+1}_year"]     = year
                    base[f"candidate_{k+1}_score"]    = round(score, 3)
                    base[f"candidate_{k+1}_days"]     = None if days == 9999 else days
                else:
                    base[f"candidate_{k+1}_cs_id"]    = None
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
              f"Fill manual_override_cs_id and save as {self.MANUAL_OVERRIDES} to apply on next run.")

    # ── Re-score the existing cache against the current algorithm/data ───────

    def re_score(self, cs_df: pd.DataFrame, films_df: pd.DataFrame | None = None):
        """
        Re-evaluate every cached film against the (possibly updated) Comscore
        extract. Manual overrides are left untouched. Equivalent of
        StudioLookup.re_score_cache but with no API refetch — Comscore row
        values are read directly off the updated extract.
        """
        cs = self._prep_comscore(cs_df)
        n  = len(self._cache)
        print(f"Re-scoring {n:,} cached films against {len(cs):,} Comscore rows")

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
                sub = cs[cs["_year"].between(evt_year - self.YEAR_TOL,
                                              evt_year + self.YEAR_TOL)]
            else:
                sub = cs

            all_scored = self._score_candidates(title, evt_date, sub)
            review_candidates[fid] = all_scored[:5]
            new_record = self._build_record(fid, title, all_scored)

            cached_id = row_dict.get("cs_id")
            new_id    = new_record.get("cs_id")
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
        # Preserve carried cs_* columns even if they weren't in CACHE_COLS
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
        self._write_review_file(films_df, review_candidates)
        return self
