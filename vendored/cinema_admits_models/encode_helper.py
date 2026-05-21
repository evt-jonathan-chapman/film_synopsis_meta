import re
import numpy as np
import pandas as pd
from rapidfuzz import fuzz

class EncHelper:

    # -----------------------------------------------------------------------
    # Distributor lookup tables — two tiers
    # -----------------------------------------------------------------------

    # Hard override — these distribute almost exclusively English-language films
    # Distributor signal alone is sufficient to classify as english
    ENGLISH_DISTRIBUTORS_HARD = [
        "paramount",
        "warner bros",
        "studiocanal",
        "hoyts distribution",
        "united international pictures",
        "bbc studios",
        "mushroom pictures",
        "backlot films",
        "fourth wall distribution",
        "instinct distribution",
        "becker entertainment",
        "transmission films",
        "transmission too",
        "fathom events",
        "imax",
        "royal opera house",
        "more2screen",
        "a24",
    ]

    # Soft override — these regularly distribute foreign language films too
    # Only classify as english if language_cues also contains an English variant
    ENGLISH_DISTRIBUTORS_SOFT = [
        "walt disney", "disney",
        "sony pictures",
        "universal pictures",
        "roadshow",
        "madman",
        "umbrella entertainment",
        "sharmill",
        "hopscotch",
        "potential films",
        "palace entertainment",
        "icon film distribution",
        "greater union",
        "antidote films",
        "monster pictures",
        "vendetta films",
        "trafalgar releasing",
        "park circus",
        "pathe live",
        "arts alliance",
        "cinema live",
    ]

    # Non-English distributors — hard tier
    # These exclusively distribute non-English films; override even if language_cues says English
    NONENGLISH_DISTRIBUTOR_MAP_HARD = {
        # Indian / South Asian — dedicated Indian film distributors
        "eros":                  "hindi",
        "ayngaran":              "south_indian",
        "aascar":                "south_indian",
        "namaste entertainment": "hindi",
        "prathyangira":          "south_indian",
        "spectrum talkies":      "south_indian",
        "nanma cinemas":         "south_indian",   # Malayalam-focused
        "shree forum":           "hindi",
        "friends india":         "hindi",
        "white hill":            "hindi",
        "daricheh cinema":       "middle_eastern",
        "iranian film":          "middle_eastern",
        # Chinese / East Asian — dedicated Chinese film distributors
        "chinalion":             "east_asian",
        "china lion":            "east_asian",
        "sydney china":          "east_asian",
        "international chinese": "east_asian",
        "yu enterprises":        "east_asian",
        "seven colors":          "east_asian",
        "taiwan film":           "east_asian",
        "tangren":               "east_asian",
        # European / festival — language-specific festivals
        "alliance francaise":    "european",
        "italian film":          "european",
        "french film":           "european",
        "russian film":          "european",
        "japanese film":         "east_asian",
        "spanish film":          "european",
        "mexican film":          "european",
        "greek festival":        "european",
        "scandinavian film":     "european",
        "ukrainian film":        "european",
        "goethe":                "european",
        # Pacific
        "pasifika":              "pacific",
        # Other / catch-all festival
        "sbs-alternate":         "other",
        "sydney travelling":     "other",
        "travelling film":       "other",
        "melbourne international film": "other",
        "new zealand int film":  "other",
    }

    # Non-English distributors — soft tier
    # These primarily distribute non-English films but also release dubbed/English versions
    # Only classify as non-English if language_cues contains NO English cue
    NONENGLISH_DISTRIBUTOR_MAP_SOFT = {
        "mks talkies":   "hindi",       # mostly Hindi but some English releases
        "tolly movies":  "south_indian", # Telugu/South Indian but dubbed English releases
        "sugoi":         "east_asian",   # anime — both subbed and dubbed
        "k-movie":       "east_asian",   # Korean — both subbed and dubbed
        "crunchyroll":   "east_asian",   # anime — both subbed and dubbed
        "niu vision":    "east_asian",   # Chinese — mixed
        "cmc pictures":  "east_asian",   # Chinese — mixed English/Cantonese
        "korean film festival": "east_asian",
        "toei":          "east_asian",
        "cj 4dplex":     "east_asian",
        "serbian film":  "european",
    }

    # Keywords used to detect English in language_cues
    ENGLISH_LANG_KEYWORDS = [
        "english", "australian", "american", "canadian", "irish", "new zealand"
    ]

    # -----------------------------------------------------------------------
    # Language group inference
    # -----------------------------------------------------------------------

    @staticmethod
    def assign_language_group(lang_list):
        """Infer language group from LLM-returned language_cues list."""
        if lang_list is None:
            return "other"
        if isinstance(lang_list, np.ndarray):
            lang_list = lang_list.tolist()
        try:
            if len(lang_list) == 0:
                return "other"
        except TypeError:
            return "other"

        group_map = [
            ("english",           EncHelper.ENGLISH_LANG_KEYWORDS),
            ("hindi",             ["hindi", "punjabi", "urdu", "haryanvi", "sikh", "bollywood"]),
            ("south_indian",      ["tamil", "telugu", "malayalam", "kannada", "marathi", "tulu", "south indian"]),
            ("east_asian",        ["korean", "japanese", "chinese", "mandarin", "cantonese", "thai",
                                   "vietnamese", "filipino", "indonesian", "taiwanese"]),
            ("european",          ["french", "italian", "spanish", "german", "dutch", "russian",
                                   "portuguese", "greek", "swedish", "norwegian", "danish", "finnish",
                                   "polish", "hungarian", "ukrainian", "romanian", "croatian",
                                   "bulgarian", "albanian", "estonian", "icelandic", "scandinavian", "serbian"]),
            ("middle_eastern",    ["arabic", "persian", "farsi", "hebrew", "turkish", "egyptian",
                                   "middle eastern", "dari", "pashto"]),
            ("south_asian_other", ["nepali", "bengali", "gujarati", "sinhala", "kashmiri",
                                   "bhutanese", "dzongkha", "sri lankan"]),
            ("pacific",           ["māori", "maori", "samoan", "polynesian", "hawaiian", "tongan"]),
        ]

        for lang in lang_list:
            lang_lower = str(lang).strip().lower()
            for group, keywords in group_map:
                if any(kw in lang_lower for kw in keywords):
                    return group
        return "other"

    @staticmethod
    def infer_language_from_distributor(distributor: str, lang_list=None):
        """
        Returns a language_group string if the distributor is a strong signal,
        otherwise returns None (caller falls through to language_cues).

        Hard distributors: override regardless of language_cues.
        Soft distributors: only override if language_cues also contains an English cue
                           (prevents anime/arthouse foreign films from being mislabelled).
        """
        if distributor is None:
            return None
        d = re.sub(r"^(au|nz|zz)\s+", "", str(distributor).strip().lower()).strip()
        if not d or d in ("-", "?", "nan", "test", ""):
            return None

        # Normalise lang_list — may arrive as numpy array, list, or None
        if lang_list is None:
            lang_list = []
        elif isinstance(lang_list, np.ndarray):
            lang_list = lang_list.tolist()
        elif not isinstance(lang_list, list):
            try:
                lang_list = list(lang_list)
            except TypeError:
                lang_list = []

        # Hard override — distributor alone is sufficient
        for keyword in EncHelper.ENGLISH_DISTRIBUTORS_HARD:
            if keyword in d:
                return "english"

        # Compute English cue presence once — used by both soft tiers below
        has_english_cue = any(
            any(kw in str(l).lower() for kw in EncHelper.ENGLISH_LANG_KEYWORDS)
            for l in lang_list
        )

        # Soft English override — only if language_cues has at least one English cue
        for keyword in EncHelper.ENGLISH_DISTRIBUTORS_SOFT:
            if keyword in d and has_english_cue:
                return "english"

        # Non-English hard tier — override even if language_cues contains English
        for keyword, group in EncHelper.NONENGLISH_DISTRIBUTOR_MAP_HARD.items():
            if keyword in d:
                return group

        # Non-English soft tier — only override if language_cues has NO English cue
        if not has_english_cue:
            for keyword, group in EncHelper.NONENGLISH_DISTRIBUTOR_MAP_SOFT.items():
                if keyword in d:
                    return group

        return None  # no strong signal — fall through to language_cues

    @staticmethod
    def assign_language_group_with_distributor(row):
        """
        Primary:  infer from dstbtr distributor column.
        Fallback: infer from LLM language_cues.
        Passes language_cues to distributor check for soft-tier logic.
        """
        lang_list  = row.get("language_cues")
        dist_group = EncHelper.infer_language_from_distributor(
            row.get("dstbtr") or row.get("distr") or row.get("distributor"),
            lang_list=lang_list,
        )
        if dist_group is not None:
            return dist_group
        return EncHelper.assign_language_group(lang_list)

    # -----------------------------------------------------------------------
    # IP / adaptation inference
    # -----------------------------------------------------------------------

    
    @staticmethod
    def _to_list(x):
        """Safely convert any list-like field (list, ndarray, None, NaN) to a plain list."""
        if isinstance(x, list):        return x
        if isinstance(x, np.ndarray):  return x.tolist()
        if isinstance(x, (set, tuple)):return list(x)
        return []


    @staticmethod
    def infer_is_franchise(row):
        """
        Corrects / supplements the llama-extracted is_franchise flag.
        Returns True if the film belongs to a franchise or has iconic standalone IP
        that behaves like a franchise for box office purposes.
        If llama already said True, preserve it.
        """
        if row.is_franchise:
            return True
    
        ip_list  = EncHelper._to_list(row.intellectual_property)
        people   = EncHelper._to_list(row.people)
        ip_text  = " ".join(ip_list).lower()
        ppl_text = " ".join(people).lower()
    
        iconic_standalone = [
            # Music biopics / concert films — cultural icons with proven box office
            "michael jackson", "the jackson five",
            "elvis", "elvis presley",
            "freddie mercury", "bohemian rhapsody",
            "amy winehouse",
            "bob marley",
            "whitney houston",
            "david bowie",
            "taylor swift",
            "beyonce",
            "adele",
        
            # Sports biopics / event films
            "muhammad ali",
            "mike tyson",
            "formula 1", "f1", "formula one",
        
            # Historical / political figures
            "napoleon", "napoleon bonaparte",
            "cleopatra",
            "winston churchill",
            "abraham lincoln",
        
            # Literary / mythological prestige IP
            "the odyssey", "homer", "the iliad", "odysseus",
            "the great gatsby",
            "wuthering heights",
        
            # Cultural brand IP — not franchises but iconic draws
            "barbie",
            "oppenheimer",
            "the devil wears prada",
        
            # Space / historical event IP
            "apollo 13", "nasa",
        
            # Religious / faith-based event films with proven scale
            "jesus", "christ",
        ]
            
        iconic_directors = [
            "christopher nolan",
            "steven spielberg",
            "james cameron",
            "peter jackson",
            "martin scorsese",   # Killers of the Flower Moon — 37k in training
            "ridley scott",      # Napoleon — 69k in training
            "baz luhrmann"
        ]
    
        if any(k in ip_text  for k in iconic_standalone): return True
        if any(k in ppl_text for k in iconic_directors):  return True
        return False


    @staticmethod
    def infer_ip_strength(row):
        """
        v1-only fallback. v2-encoded films use LLM-direct ip_strength
        (applied in encode_llm_features.py after this function runs).

        Returns "iconic" / "strong" / "moderate" based on keyword match
        against iconic_ips, iconic_directors, then is_sequel. Defaults to
        "moderate" when none match — was previously "none" gated on
        is_franchise, but that gate destroyed signal for non-franchise
        biopics with iconic subjects (BR ip_strength=strong → forced
        to none). Gate removed so subject-fame can fire independently.
        """
        ip_list  = EncHelper._to_list(row.intellectual_property)
        people   = EncHelper._to_list(row.people)
        ip_text  = " ".join(ip_list).lower()
        ppl_text = " ".join(people).lower()
    
        iconic_ips = [
            # Studio / universe franchises
            "marvel", "star wars", "harry potter", "lord of the rings", "hobbit",
            "jurassic", "avatar", "frozen", "wicked", "dc", "disney", "pixar",
            "fast & furious", "despicable me", "minions", "spider-man", "the incredibles",
            "toy story", "hunger games", "lego", "how to train your dragon", "the batman",
            "james bond", "mission impossible", "mission: impossible",
            "guardians of the galaxy", "thor", "black panther",
            "the lion king", "aladdin", "nintendo", "super mario", "mario",
            "shrek", "kung fu panda", "moana", "encanto", "cars",
            "finding nemo", "the little mermaid", "beauty and the beast", "cinderella",
            "snow white", "transformers", "indiana jones", "alien", "predator",
            "terminator", "john wick", "top gun", "matrix", "planet of the apes",
            "pirates of the caribbean", "ghostbusters", "conjuring", "halloween",
            "scream", "sonic the hedgehog", "paddington", "asterix",
            "mandalorian", "star trek", "jumanji",
            # Cultural / biographical iconic IP
            "michael jackson", "the jackson five",
            "elvis", "freddie mercury", "bohemian rhapsody",
            "barbie", "oppenheimer",
            "the devil wears prada",
            # Classical / literary prestige IP
            "the odyssey", "homer", "the iliad",
        ]
    
        iconic_directors = [
            "christopher nolan",
            "steven spielberg",
            "james cameron",
            "peter jackson",
        ]
    
        if any(k in ip_text  for k in iconic_ips):       return "iconic"
        if any(k in ppl_text for k in iconic_directors): return "iconic"
        if row.is_sequel:                                 return "strong"
        return "moderate"

    @staticmethod
    def infer_adaptation_type(row):
        ip_text        = " ".join(EncHelper._to_list(row.intellectual_property)).lower()
        subgenres_text = " ".join(EncHelper._to_list(row.subgenres)).lower()

        stage_musicals = [
            "wicked", "les miserables", "hamilton", "cats", "phantom of the opera",
            "mamma mia", "chicago", "hairspray",
        ]
        if "musical" in subgenres_text:
            return "stage musical"
        if any(k in ip_text for k in stage_musicals):
            return "stage musical"
        if row.is_franchise:
            return "unknown"
        return "original"
    
    
    # -----------------------------------------------------------------------
    # GC enrichment
    # -----------------------------------------------------------------------

    @staticmethod
    def enrich_gc_and_standard(df, title_col="title", film_id_col="film_id",
                                synopsis_col="synopsis", fuzzy_threshold=60,
                                synopsis_prefix_chars=80):
        df = df.copy()
    
        def to_list_safe(x):
            if isinstance(x, list):                    return x
            if isinstance(x, np.ndarray):              return list(x)
            if isinstance(x, (set, tuple)):            return list(x)
            if x is None:                              return []
            if isinstance(x, float) and np.isnan(x):  return []
            return [x]
    
        def normalise_title(t):
            return (t.upper()
                     .replace("GC ", "", 1)
                     .replace(" & ", " AND ")
                     .strip())
    
        def synopsis_prefix(s):
            if not isinstance(s, str):
                return ""
            return s[:synopsis_prefix_chars].upper().strip()
    
        # ── 1. Identify GC rows ──────────────────────────────────────────────
        is_gc  = df[title_col].str.upper().str.startswith("GC ")
        df_gc  = df[is_gc].copy()
        df_std = df[~is_gc].copy()
    
        df_gc["_norm_title"]  = df_gc[title_col].apply(normalise_title)
        df_std["_norm_title"] = df_std[title_col].apply(normalise_title)
        df_gc["_synop_pre"]   = df_gc[synopsis_col].apply(synopsis_prefix)
        df_std["_synop_pre"]  = df_std[synopsis_col].apply(synopsis_prefix)
    
        list_cols = [
            "people", "themes", "intellectual_property", "genres",
            "subgenres", "setting_types", "time_periods",
            "protagonist_archetypes", "tone", "language_cues", "secondary_audiences",
        ]
    
        gc_ids_to_drop = []
    
        for _, gc_row in df_gc.iterrows():
            # Title similarity (0-100)
            title_scores = df_std["_norm_title"].apply(
                lambda t: fuzz.token_sort_ratio(gc_row["_norm_title"], t)
            )
    
            # Synopsis prefix similarity (0-100) — strong signal if both non-empty
            gc_pre = gc_row["_synop_pre"]
            synop_scores = df_std["_synop_pre"].apply(
                lambda s: fuzz.ratio(gc_pre, s) if gc_pre and s else 0
            )
    
            # Combined score — synopsis prefix weighted heavily when available
            has_synop = synop_scores.max() > 0
            if has_synop:
                combined = 0.4 * title_scores + 0.6 * synop_scores
            else:
                combined = title_scores  # fall back to title only
    
            best_score = combined.max()
            if best_score < fuzzy_threshold:
                continue  # no confident match — leave GC row intact for inspection
    
            std_idx = combined.idxmax()
            gc_idx  = gc_row.name
    
            # ── 2. Propagate the longer synopsis to the standard row ─────────
            gc_syn  = df.at[gc_idx,  synopsis_col] if isinstance(df.at[gc_idx,  synopsis_col], str) else ""
            std_syn = df.at[std_idx, synopsis_col] if isinstance(df.at[std_idx, synopsis_col], str) else ""
            df.at[std_idx, synopsis_col] = gc_syn if len(gc_syn) > len(std_syn) else std_syn
    
            # ── 3. Merge list columns (union) ────────────────────────────────
            for col in list_cols:
                if col not in df.columns:
                    continue
                merged = list(set(
                    to_list_safe(df.at[gc_idx,  col]) +
                    to_list_safe(df.at[std_idx, col])
                ))
                df.at[std_idx, col] = merged
    
            gc_ids_to_drop.append(gc_idx)
    
        # ── 4. Drop matched GC rows ──────────────────────────────────────────
        df.drop(index=gc_ids_to_drop, inplace=True)
        df.drop(columns=["_norm_title", "_synop_pre"], errors="ignore", inplace=True)
        df.reset_index(drop=True, inplace=True)
    
        return df
    
    # -----------------------------------------------------------------------
    # 3D enrichment
    # -----------------------------------------------------------------------

    @staticmethod
    def enrich_3d_and_standard(df, title_col="title", is_3d_col="is_three_d", film_id_col="film_id"):
        df = df.copy()
        df[is_3d_col] = df[is_3d_col].astype(bool)

        def to_list_safe(x):
            if isinstance(x, list):                       return x
            if isinstance(x, np.ndarray):                 return list(x)
            if isinstance(x, (set, tuple)):               return list(x)
            if x is None:                                 return []
            if isinstance(x, float) and np.isnan(x):     return []
            return [x]

        df["base_title"] = (
            df[title_col]
            .str.replace(r"^3D[\s-]+", "", regex=True)
            .str.strip()
            .str.upper()
        )

        df_3d  = df[df[is_3d_col]]
        df_std = df[~df[is_3d_col]]

        mapping = df_3d.merge(
            df_std[[film_id_col, "base_title"]],
            on="base_title", how="left",
            suffixes=("_3d", "_std")
        ).dropna(subset=[f"{film_id_col}_std"])

        df["has_3d_version"] = 0
        matched_ids = pd.concat([
            mapping[f"{film_id_col}_3d"],
            mapping[f"{film_id_col}_std"]
        ]).unique()
        df.loc[df[film_id_col].isin(matched_ids), "has_3d_version"] = 1

        list_cols = [
            "people", "themes", "intellectual_property", "genres",
            "subgenres", "setting_types", "time_periods",
            "protagonist_archetypes", "tone", "language_cues", "secondary_audiences",
        ]

        for _, row in mapping.iterrows():
            id_3d  = row[f"{film_id_col}_3d"]
            id_std = row[f"{film_id_col}_std"]
            idx_3d  = df.index[df[film_id_col] == id_3d][0]
            idx_std = df.index[df[film_id_col] == id_std][0]
            for col in list_cols:
                merged = list(set(
                    to_list_safe(df.at[idx_3d, col]) +
                    to_list_safe(df.at[idx_std, col])
                ))
                df.at[idx_3d, col]  = merged
                df.at[idx_std, col] = merged

        df.drop(columns=["base_title"], inplace=True)
        return df

    # -----------------------------------------------------------------------
    # Corrections / adjustments
    # -----------------------------------------------------------------------

    @staticmethod
    def correct_missing_ips(df, missing_ip_ids):
        df["is_ip"] = df["is_ip"].fillna(0).astype(int)
        df.loc[df["film_id"].isin(missing_ip_ids), "is_ip"] = 1
        return df

    @staticmethod
    def append_genres(row, adjust_dict):
        film_id = row["film_id"]
        if film_id in adjust_dict:
            current = row["genres"] if isinstance(row["genres"], list) else []
            row["genres"] = list(set(current + adjust_dict[film_id]))
        return row

    @staticmethod
    def adjust_genres(film_df, genre_dict):
        for film_id, new_genres in genre_dict.items():
            mask = film_df["film_id"] == film_id
            if mask.any():
                idx     = film_df.index[mask][0]
                current = list(film_df.at[idx, "genres"])
                film_df.at[idx, "genres"] = list(set(current + new_genres))
        return film_df