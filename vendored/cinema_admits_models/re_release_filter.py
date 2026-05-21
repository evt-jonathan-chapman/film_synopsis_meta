import re
import pandas as pd
from rapidfuzz import fuzz, process
from .encode_helper import EncHelper  # vendored: upstream is `from encode_helper import EncHelper`


class ReReleaseFilter:
    """
    Detects re-releases in a film dataframe.

    Two detection strategies:
      1. Title keyword scan  — matches known re-release terms in the film title.
      2. Fuzzy title match   — compares against a historical film lookup and
                               confirms with director / cast / distributor signals.

    Sequel-guard logic prevents legitimate sequels from being mis-flagged.
    """

    ROMAN = {
        "I": 1, "II": 2, "III": 3, "IV": 4,
        "V": 5, "VI": 6, "VII": 7, "VIII": 8,
    }

    WORD_SEQUEL_TOKENS = [
        r'\bfreakier\b', r'\bfurther\b',   r'\bbeyond\b',    r'\bnewer\b',
        r'\bultimate\b', r'\bfinal\b',     r'\blast\b',      r'\breturn\b',
        r'\brise\b',     r'\blegacy\b',    r'\brewakens\b',  r'\bunleashed\b',
        r'\bbegins\b',   r'\bforever\b',   r'\breckoning\b',
        r'\bii\b', r'\biii\b', r'\biv\b', r'\bvi\b', r'\bvii\b', r'\bviii\b',
    ]

    RERELEASE_KEYWORDS = [
        "re-release", "rerelease", "re release",
        "double feature", "double bill",
        "anniversary", "anniversary edition",
        "director's cut", "directors cut", "extended cut",
        "special edition", "limited edition",
        "back by popular demand", "return to cinemas", "back in cinemas",
        "encore", "encore screening",
        "50th", "40th", "35th", "30th", "25th", "20th", "15th", "10th",
        "4k remaster", "4k re-release", "remastered",
        "sing along", "singalong", "sing-along",   # spaced, unspaced & hyphenated
        "classic",                               # classic re-release editions
        "test screening", "test screen",         # internal test screenings
    ]

    # Language variant suffixes — titles ending with these are alternative-language
    # versions of an existing release (not new films)
    LANGUAGE_VARIANT_SUFFIXES = [
        r"\s*[-–]\s*HINDI$",
        r"\s*[-–]\s*TAMIL$",
        r"\s*[-–]\s*TELUGU$",
        r"\s*[-–]\s*KANNADA$",
        r"\s*[-–]\s*MALAYALAM$",
        r"\s*[-–]\s*PUNJABI$",
        r"\s*[-–]\s*MAORI$",
        r"\bREO\s+MAORI$",
        r"\s*[-–]\s*MANDARIN$",
        r"\s*[-–]\s*CANTONESE$",
        r"\s*[-–]\s*JAPANESE$",
        r"\s*[-–]\s*KOREAN$",
        r"\s*[-–]\s*THAI$",
    ]

    # Titles containing a 4-digit year in parentheses are almost always re-releases
    YEAR_IN_TITLE_RE = re.compile(r"\(\d{4}\)")

    STRIP_PATTERNS = [
        r"\bRE-?RELEASE\b", r"\bDOUBLE FEATURE\b", r"\bDOUBLE BILL\b",
        r"\bANNIVERSARY\b", r"\bEDITION\b",         r"\bENCORE\b",
        r"\d{2}TH\b",       r"\d{2}RD\b",            r"\d{2}ND\b",
        r"\bSPECIAL\b",     r"\bLIMITED\b",          r"\bDIRECTOR.?S CUT\b",
        r"\b4K\b",          r"\bREMASTERED\b",        r"\bCLASSIC\b",
        r"\(\d{4}\)",       # strip year-in-parens before fuzzy matching
        r"\bSING.?ALONG\b",
    ]

    def __init__(self, fuzzy_threshold: int = 85, min_gap_days: int = 180, top_n_cast: int = 3):
        self.fuzzy_threshold = fuzzy_threshold
        self.min_gap_days    = min_gap_days
        self.top_n_cast      = top_n_cast

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def flag(
        self,
        df: pd.DataFrame,
        film_lookup: pd.DataFrame,
        title_col: str       = "title",
        film_id_col: str     = "film_id",
        rel_at_col: str      = "rel_at",
        director_col: str    = "director",
        distributor_col: str = "dstbtr",
        cast_col: str        = "actor_list",
    ) -> pd.DataFrame:
        """
        Return *df* with two new columns:
          - rerelease_flag   (int  0/1)
          - rerelease_reason (str)
        """
        df = df.copy()
        df[rel_at_col]         = pd.to_datetime(df[rel_at_col], utc=True).dt.tz_localize(None)
        df["rerelease_flag"]   = 0
        df["rerelease_reason"] = ""

        lk = self._prepare_lookup(film_lookup)
        df = self._flag_by_keyword(df, title_col)
        df = self._flag_by_fuzzy(df, lk, film_id_col, rel_at_col,
                                  director_col, distributor_col, cast_col)
        df.drop(columns=["_norm", "_dir", "_dist", "_cast"], errors="ignore", inplace=True)
        return df

    # ------------------------------------------------------------------
    # Step 1 — keyword scan
    # ------------------------------------------------------------------

    def _flag_by_keyword(self, df: pd.DataFrame, title_col: str) -> pd.DataFrame:
        title_upper = df[title_col].fillna("").str.upper().str.strip()

        # 1. Plain keyword match
        keyword_mask = title_upper.apply(
            lambda t: any(k.upper() in t for k in self.RERELEASE_KEYWORDS)
        )

        # 2. Year in parentheses — e.g. "SPIDER-MAN (2002)"
        year_mask = title_upper.apply(
            lambda t: bool(self.YEAR_IN_TITLE_RE.search(t))
        )

        # 3. Language variant suffix — e.g. "FILM - HINDI", "ENCANTO REO MAORI"
        lang_pat = re.compile(
            "|".join(self.LANGUAGE_VARIANT_SUFFIXES), flags=re.IGNORECASE
        )
        lang_mask = title_upper.apply(lambda t: bool(lang_pat.search(t)))

        for mask, reason in [
            (keyword_mask, "title_keyword"),
            (year_mask,    "year_in_title"),
            (lang_mask,    "language_variant"),
        ]:
            unflagged = mask & (df["rerelease_flag"] == 0)
            df.loc[unflagged, "rerelease_flag"]   = 1
            df.loc[unflagged, "rerelease_reason"] = reason

        return df

    # ------------------------------------------------------------------
    # Step 2 — fuzzy match
    # ------------------------------------------------------------------

    def _flag_by_fuzzy(
        self,
        df: pd.DataFrame,
        lk: pd.DataFrame,
        film_id_col: str,
        rel_at_col: str,
        director_col: str,
        distributor_col: str,
        cast_col: str,
    ) -> pd.DataFrame:
        film_col    = "film" if "film" in df.columns else film_id_col
        lk_cast_col = next((c for c in ("actor_list", "main_cast") if c in lk.columns), None)

        df["_norm"] = df[film_col].apply(self._normalise)
        lk["_norm"] = lk["film"].apply(self._normalise)

        df["_dir"]  = df[director_col].apply(self._normalise_person)  if director_col  in df.columns else ""
        df["_dist"] = df[distributor_col].apply(self._normalise_person) if distributor_col in df.columns else ""
        df["_cast"] = df[cast_col].apply(self._parse_cast)             if cast_col      in df.columns else [set()] * len(df)

        lk["_dir"]  = lk["director"].apply(self._normalise_person)
        lk["_dist"] = lk["dstbtr"].apply(self._normalise_person)
        lk["_cast"] = lk[lk_cast_col].apply(self._parse_cast) if lk_cast_col else [set()] * len(lk)

        film_level = (
            df[df["rerelease_flag"] == 0]
            [[film_id_col, rel_at_col, "_norm", "_dir", "_dist", "_cast"]]
            .drop_duplicates(film_id_col)
        )

        results = []
        for _, row in film_level.iterrows():
            if pd.isna(row[rel_at_col]):
                continue

            word_count = len(row["_norm"].split())
            cand_num   = self._extract_seq_number(row["_norm"])
            cutoff     = row[rel_at_col] - pd.Timedelta(days=self.min_gap_days)

            earlier = lk[
                (lk[film_id_col] != row[film_id_col]) &
                (lk["rel_at"] < cutoff)
            ]
            if earlier.empty:
                continue

            match = process.extractOne(
                row["_norm"],
                earlier["_norm"].tolist(),
                scorer=fuzz.token_sort_ratio,
                score_cutoff=self.fuzzy_threshold,
            )
            if not match:
                continue

            matched_score = match[1]
            matched_row   = earlier[earlier["_norm"] == match[0]].iloc[0]
            match_num     = self._extract_seq_number(matched_row["_norm"])

            if cand_num > match_num and cand_num > 0:
                continue
            if self._is_word_sequel(row["_norm"], matched_row["_norm"]):
                continue
            if not self._same_language_origin(row["_dist"], matched_row["_dist"]):
                continue

            dir_ok  = self._directors_match(row["_dir"], matched_row["_dir"])
            cast_ok = self._cast_match(row["_cast"], matched_row["_cast"])
            dist_ok = self._distributors_match(row["_dist"], matched_row["_dist"])

            conf_parts  = (["dir"]  if dir_ok  else []) + \
                          (["cast"] if cast_ok else []) + \
                          (["dist"] if dist_ok else [])
            conf        = "+".join(conf_parts) if conf_parts else "none"
            strong_conf = dir_ok or cast_ok

            reason = (
                f"fuzzy_match:{matched_row[film_id_col]}:{matched_row['film']}:"
                f"score={matched_score:.0f}:confirmed_by={conf}"
            )

            if matched_score == 100 and word_count > 3:
                if strong_conf or dist_ok:
                    results.append({film_id_col: row[film_id_col],
                                    "rerelease_flag": 1, "rerelease_reason": reason})
                continue

            if strong_conf:
                results.append({film_id_col: row[film_id_col],
                                "rerelease_flag": 1, "rerelease_reason": reason})

        if results:
            res_df   = pd.DataFrame(results).drop_duplicates(film_id_col)
            df       = df.merge(res_df, on=film_id_col, how="left", suffixes=("", "_new"))
            new_flag = df["rerelease_flag_new"].notna()
            df.loc[new_flag, "rerelease_flag"]   = df.loc[new_flag, "rerelease_flag_new"]
            df.loc[new_flag, "rerelease_reason"] = df.loc[new_flag, "rerelease_reason_new"]
            df.drop(columns=["rerelease_flag_new", "rerelease_reason_new"], inplace=True)

        return df

    # ------------------------------------------------------------------
    # Lookup preparation
    # ------------------------------------------------------------------

    def _prepare_lookup(self, film_lookup: pd.DataFrame) -> pd.DataFrame:
        lk = film_lookup.dropna(subset=["film", "rel_at"]).copy()
        lk["rel_at"] = pd.to_datetime(lk["rel_at"], utc=True).dt.tz_localize(None)
        lk_cast_col  = next((c for c in ("actor_list", "main_cast") if c in lk.columns), None)
        keep_cols    = ["film_id", "film", "rel_at", "director", "dstbtr"]
        if lk_cast_col:
            keep_cols.append(lk_cast_col)
        return lk[keep_cols].drop_duplicates("film_id").reset_index(drop=True)

    # ------------------------------------------------------------------
    # Text helpers
    # ------------------------------------------------------------------

    def _normalise(self, t: str) -> str:
        t = str(t).upper().strip()
        for pat in self.STRIP_PATTERNS:
            t = re.sub(pat, "", t)
        return re.sub(r"\s+", " ", t).strip()

    @staticmethod
    def _normalise_person(t) -> str:
        if pd.isna(t):
            return ""
        return re.sub(r"\s+", " ", str(t).upper().strip())

    def _parse_cast(self, raw) -> set:
        if pd.isna(raw) or not str(raw).strip():
            return set()
        sep   = "|" if "|" in str(raw) else ","
        names = [self._normalise_person(x) for x in str(raw).split(sep) if x.strip()]
        return set(names[:self.top_n_cast])

    def _extract_seq_number(self, t: str) -> int:
        t    = t.strip().upper()
        last = t.split()[-1] if t.split() else ""
        if last.isdigit():
            return int(last)
        return self.ROMAN.get(last, 0)

    def _is_word_sequel(self, candidate: str, matched: str) -> bool:
        c = candidate.lower()
        m = matched.lower()
        return any(
            re.search(pat, c) and not re.search(pat, m)
            for pat in self.WORD_SEQUEL_TOKENS
        )

    # ------------------------------------------------------------------
    # Confirmation signal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _directors_match(d1: str, d2: str) -> bool:
        if not d1 or not d2:
            return False
        return bool({x.strip() for x in d1.split(",")} & {x.strip() for x in d2.split(",")})

    @staticmethod
    def _cast_match(c1: set, c2: set) -> bool:
        return bool(c1 and c2 and c1 & c2)

    @staticmethod
    def _distributors_match(d1: str, d2: str) -> bool:
        if not d1 or not d2:
            return False
        def clean(s):
            return re.sub(r"^(AU|NZ|ZZ)\s+", "", s, flags=re.IGNORECASE).strip().upper()
        return fuzz.token_sort_ratio(clean(d1), clean(d2)) >= 85

    @staticmethod
    def _get_lang_origin(dist: str) -> str:
        if not dist or pd.isna(dist):
            return "unknown"
        d = re.sub(r"^(AU|NZ|ZZ)\s+", "", str(dist), flags=re.IGNORECASE).strip().lower()
        for keyword in EncHelper.NONENGLISH_DISTRIBUTOR_MAP_HARD:
            if keyword in d:
                return "nonenglish"
        for keyword in EncHelper.NONENGLISH_DISTRIBUTOR_MAP_SOFT:
            if keyword in d:
                return "nonenglish"
        for keyword in [k.lower() for k in EncHelper.ENGLISH_DISTRIBUTORS_HARD]:
            if keyword in d:
                return "english"
        for keyword in [k.lower() for k in EncHelper.ENGLISH_DISTRIBUTORS_SOFT]:
            if keyword in d:
                return "english"
        return "unknown"

    def _same_language_origin(self, dist1: str, dist2: str) -> bool:
        o1 = self._get_lang_origin(dist1)
        o2 = self._get_lang_origin(dist2)
        if o1 == "unknown" or o2 == "unknown":
            return True
        return not (
            (o1 == "english"    and o2 == "nonenglish") or
            (o1 == "nonenglish" and o2 == "english")
        )
