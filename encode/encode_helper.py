import re
import numpy as np
import pandas as pd
from rapidfuzz import fuzz

class EncHelper:

    # -----------------------------------------------------------------------
    # Distributor lookup tables — two tiers
    # -----------------------------------------------------------------------

    # Hard override — these distribute almost exclusively English-language films
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

    # Soft override — also distribute foreign language films
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

    NONENGLISH_DISTRIBUTOR_MAP_HARD = {
        "eros":                  "hindi",
        "ayngaran":              "south_indian",
        "aascar":                "south_indian",
        "namaste entertainment": "hindi",
        "prathyangira":          "south_indian",
        "spectrum talkies":      "south_indian",
        "nanma cinemas":         "south_indian",
        "shree forum":           "hindi",
        "friends india":         "hindi",
        "white hill":            "hindi",
        "daricheh cinema":       "middle_eastern",
        "iranian film":          "middle_eastern",
        "chinalion":             "east_asian",
        "china lion":            "east_asian",
        "sydney china":          "east_asian",
        "international chinese": "east_asian",
        "yu enterprises":        "east_asian",
        "seven colors":          "east_asian",
        "taiwan film":           "east_asian",
        "tangren":               "east_asian",
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
        "pasifika":              "pacific",
        "sbs-alternate":         "other",
        "sydney travelling":     "other",
        "travelling film":       "other",
        "melbourne international film": "other",
        "new zealand int film":  "other",
    }

    NONENGLISH_DISTRIBUTOR_MAP_SOFT = {
        "mks talkies":   "hindi",
        "tolly movies":  "south_indian",
        "sugoi":         "east_asian",
        "k-movie":       "east_asian",
        "crunchyroll":   "east_asian",
        "niu vision":    "east_asian",
        "cmc pictures":  "east_asian",
        "korean film festival": "east_asian",
        "toei":          "east_asian",
        "cj 4dplex":     "east_asian",
        "serbian film":  "european",
    }

    ENGLISH_LANG_KEYWORDS = [
        "english", "australian", "american", "canadian", "irish", "new zealand"
    ]

    # -----------------------------------------------------------------------
    # Language group inference
    # -----------------------------------------------------------------------

    @staticmethod
    def assign_language_group(lang_list):
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
        if distributor is None:
            return None
        d = re.sub(r"^(au|nz|zz)\s+", "", str(distributor).strip().lower()).strip()
        if not d or d in ("-", "?", "nan", "test", ""):
            return None

        if lang_list is None:
            lang_list = []
        elif isinstance(lang_list, np.ndarray):
            lang_list = lang_list.tolist()
        elif not isinstance(lang_list, list):
            try:
                lang_list = list(lang_list)
            except TypeError:
                lang_list = []

        for keyword in EncHelper.ENGLISH_DISTRIBUTORS_HARD:
            if keyword in d:
                return "english"

        has_english_cue = any(
            any(kw in str(l).lower() for kw in EncHelper.ENGLISH_LANG_KEYWORDS)
            for l in lang_list
        )

        for keyword in EncHelper.ENGLISH_DISTRIBUTORS_SOFT:
            if keyword in d and has_english_cue:
                return "english"

        for keyword, group in EncHelper.NONENGLISH_DISTRIBUTOR_MAP_HARD.items():
            if keyword in d:
                return group

        if not has_english_cue:
            for keyword, group in EncHelper.NONENGLISH_DISTRIBUTOR_MAP_SOFT.items():
                if keyword in d:
                    return group

        return None

    @staticmethod
    def assign_language_group_with_distributor(row):
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
        if isinstance(x, list):         return x
        if isinstance(x, np.ndarray):   return x.tolist()
        if isinstance(x, (set, tuple)): return list(x)
        return []

    @staticmethod
    def infer_is_franchise(row):
        if row.is_franchise:
            return True

        ip_list  = EncHelper._to_list(row.intellectual_property)
        people   = EncHelper._to_list(row.people)
        ip_text  = " ".join(ip_list).lower()
        ppl_text = " ".join(people).lower()

        iconic_standalone = [
            "michael jackson", "the jackson five",
            "elvis", "elvis presley",
            "freddie mercury", "bohemian rhapsody",
            "amy winehouse", "bob marley", "whitney houston",
            "david bowie", "taylor swift", "beyonce", "adele",
            "muhammad ali", "mike tyson",
            "formula 1", "f1", "formula one",
            "napoleon", "napoleon bonaparte",
            "cleopatra", "winston churchill", "abraham lincoln",
            "the odyssey", "homer", "the iliad", "odysseus",
            "the great gatsby", "wuthering heights",
            "barbie", "oppenheimer", "the devil wears prada",
            "apollo 13", "nasa",
            "jesus", "christ",
        ]

        iconic_directors = [
            "christopher nolan", "steven spielberg", "james cameron",
            "peter jackson", "martin scorsese", "ridley scott", "baz luhrmann",
        ]

        if any(k in ip_text  for k in iconic_standalone): return True
        if any(k in ppl_text for k in iconic_directors):  return True
        return False

    @staticmethod
    def infer_ip_strength(row):
        if not row.is_franchise:
            return "none"

        ip_list  = EncHelper._to_list(row.intellectual_property)
        people   = EncHelper._to_list(row.people)
        ip_text  = " ".join(ip_list).lower()
        ppl_text = " ".join(people).lower()

        iconic_ips = [
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
            "michael jackson", "the jackson five",
            "elvis", "freddie mercury", "bohemian rhapsody",
            "barbie", "oppenheimer", "the devil wears prada",
            "the odyssey", "homer", "the iliad",
        ]

        iconic_directors = [
            "christopher nolan", "steven spielberg", "james cameron", "peter jackson",
        ]

        if any(k in ip_text  for k in iconic_ips):       return "iconic"
        if any(k in ppl_text for k in iconic_directors): return "iconic"
        if row.is_sequel:                                 return "strong"
        return "moderate"

    @staticmethod
    def infer_adaptation_type(row):
        ip_text = " ".join(row.intellectual_property).lower() \
            if row.intellectual_property is not None and len(row.intellectual_property) > 0 else ""
        subgenres_text = " ".join(row.subgenres).lower() \
            if row.subgenres is not None and len(row.subgenres) > 0 else ""

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
            return (t.upper().replace("GC ", "", 1).replace(" & ", " AND ").strip())

        def synopsis_prefix(s):
            if not isinstance(s, str):
                return ""
            return s[:synopsis_prefix_chars].upper().strip()

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
            title_scores = df_std["_norm_title"].apply(
                lambda t: fuzz.token_sort_ratio(gc_row["_norm_title"], t)
            )
            gc_pre = gc_row["_synop_pre"]
            synop_scores = df_std["_synop_pre"].apply(
                lambda s: fuzz.ratio(gc_pre, s) if gc_pre and s else 0
            )
            has_synop = synop_scores.max() > 0
            combined  = (0.4 * title_scores + 0.6 * synop_scores) if has_synop else title_scores

            if combined.max() < fuzzy_threshold:
                continue

            std_idx = combined.idxmax()
            gc_idx  = gc_row.name

            gc_syn  = df.at[gc_idx,  synopsis_col] if isinstance(df.at[gc_idx,  synopsis_col], str) else ""
            std_syn = df.at[std_idx, synopsis_col] if isinstance(df.at[std_idx, synopsis_col], str) else ""
            df.at[std_idx, synopsis_col] = gc_syn if len(gc_syn) > len(std_syn) else std_syn

            for col in list_cols:
                if col not in df.columns:
                    continue
                merged = list(set(
                    to_list_safe(df.at[gc_idx,  col]) +
                    to_list_safe(df.at[std_idx, col])
                ))
                df.at[std_idx, col] = merged

            gc_ids_to_drop.append(gc_idx)

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
            df[title_col].str.replace(r"^3D[\s-]+", "", regex=True).str.strip().str.upper()
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
            mapping[f"{film_id_col}_3d"], mapping[f"{film_id_col}_std"]
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
