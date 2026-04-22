"""
encode_synopsis.py

Encodes synopses_extracted.parquet into llama_features.parquet for the box office model.
Moved from evt_back_up/box_office/encode_llm_features.py — this is now the canonical location.

Box office model reads the output parquet directly; it does not import this code.

Usage:
    python encode_synopsis.py              # encodes to today's date folder
    encode_synopsis_features(out_date)     # callable from refresh.py
"""

import sys
sys.path.insert(0, '/Users/jonathanchapman/Documents/git/evt_back_up/base')

import datetime
import os
from functools import reduce
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from dotenv import load_dotenv

from encode.encode_helper import EncHelper
from encode.film_synop_encode import (
    FeatureDiagnostics, auto_top_n,
    DynamicTopNAndPCA, TopNTokenMapper,
    return_feature_names, TopNMultiHotWithOther,
)
from config import (
    SYNOPSES_EXTRACTED_PATH, ENCODED_META_DIR, DATA_DIR,
    SQL_PATH, THREE_D_SQL, SF_WAREHOUSE, SF_DATABASE, SF_SCHEMA, SF_RSA_KEY,
)

load_dotenv()

# ── Feature column groups ──────────────────────────────────────────────────────

ONE_HOT      = ["primary_audience", "protagonist_type", "adaptation_type",
                "ip_strength", "language_group"]
MULTI_HOT    = ["secondary_audiences", "genres"]
EMBED        = ["tone", "themes", "people"]
EMBED_MAPPED = [f"{x}_mapped" for x in EMBED]
PASSTHROUGH  = ["is_sequel", "is_franchise", "is_ip", "has_english"]

MULTI_VALUED_COLS = [
    "people", "themes", "intellectual_property",
    "subgenres", "protagonist_archetypes",
    "tone", "setting_types", "language_cues", "time_periods",
]

# ── Hard overrides ─────────────────────────────────────────────────────────────
# Edit these when the LLM mis-classifies a film — re-run encode_synopsis.py then
# rebuild training data for changes to take effect on retrain.

MISSING_FILM_IPS = [59530, 60336, 59534]

ADJUST_GENRE_IDS = {
    57601: ["musical", "pop musical"],
    55704: ["sci-fi"],
    59041: ["animation"],
}

IP_STRENGTH_OVERRIDES = {
    56744: "none",    # REINVENTING ELVIS — concert/doc, not a franchise IP
    60158: "none",    # HANS ZIMMER & FRIENDS — concert film
    60472: "none",    # KISS OF THE SPIDER WOMAN — unrelated to Spider-Man
    59529: "strong",  # ANACONDA — reboot of 1997 franchise
    59530: "strong",  # THE HOUSEMAID — remake of acclaimed 2010 Korean film
    60336: "strong",  # PROJECT HAIL MARY — Andy Weir novel; LLM missed IP
    59534: "iconic",  # WUTHERING HEIGHTS — Brontë classic
}

IS_SEQUEL_OVERRIDES = {
    60092: 1,   # SPIDER-MAN: BRAND NEW DAY
    57603: 1,   # THE MANDALORIAN & GROGU
    57343: 1,   # MICHAEL
    59529: 1,   # ANACONDA
    59530: 1,   # THE HOUSEMAID
}

GENRE_OVERRIDES = {
    57343: ["biography", "drama", "music"],
    57601: ["drama", "family", "fantasy", "musical", "romance"],
}

PRIMARY_AUDIENCE_OVERRIDES = {
    57343: "family",
    60092: "family",
}

SECONDARY_AUDIENCES_OVERRIDES = {
    60092: ["teen", "children"],
    57603: ["children"],
    59537: ["family", "children", "teen"],
}


# ── is_three_d loader ──────────────────────────────────────────────────────────

def _load_three_d() -> pd.DataFrame:
    """Load is_three_d flag — Snowflake first, raw parquet fallback."""
    try:
        from base_snowflake import SnowFlakeBase
        sb = SnowFlakeBase(warehouse=SF_WAREHOUSE, database=SF_DATABASE, schema=SF_SCHEMA)
        sb.create_snowflake_connection(SF_RSA_KEY)
        three_d = pd.read_sql((SQL_PATH / THREE_D_SQL).read_text(), sb.engine)
        print(f"three_d loaded from Snowflake: {len(three_d)} rows")
        return three_d
    except Exception as e:
        print(f"Snowflake unavailable ({e}), building three_d from raw parquets...")
        raw_base = DATA_DIR / 'raw_from_snowflake'
        dates    = sorted(os.listdir(raw_base))[-3:]
        parts    = []
        for d in dates:
            for split in ('train', 'test'):
                p = raw_base / d / split / f'{split}_raw_ds.parquet'
                if p.exists():
                    parts.append(pd.read_parquet(p, columns=['film_id', 'is_three_d']))
        three_d = (
            pd.concat(parts, ignore_index=True)
            .drop_duplicates('film_id')
            .reset_index(drop=True)
        )
        print(f"three_d built from raw parquets: {len(three_d)} unique films")
        return three_d


# ── Main encoding function ─────────────────────────────────────────────────────

def encode_synopsis_features(
    synopses_path: str | Path | None = None,
    out_date: str | None = None,
) -> Path:
    """
    Encode synopses_extracted.parquet → llama_features.parquet.

    Args:
        synopses_path: Override the default synopsis parquet path.
        out_date:      Date string for the output directory (YYYYMMDD).
                       Defaults to today.

    Returns:
        Path to the saved llama_features.parquet.
    """
    out_date = out_date or datetime.datetime.today().strftime('%Y%m%d')
    out_dir  = ENCODED_META_DIR / out_date
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'llama_features.parquet'

    # ── Load synopsis features ───────────────────────────────────────────────
    synopses_path = Path(synopses_path) if synopses_path is not None else SYNOPSES_EXTRACTED_PATH
    if not synopses_path.exists():
        raise FileNotFoundError(f"Synopsis parquet not found: {synopses_path}")

    three_d = _load_three_d()

    film_synopsis_features = (
        pd.read_parquet(synopses_path)
        .assign(film_id=lambda df: df['film_id'].astype(int))
        .merge(three_d, on='film_id')
    )
    print(f"Loaded {len(film_synopsis_features)} synopsis rows")

    # ── Stage 1: enrich lists ────────────────────────────────────────────────
    fsf = (
        film_synopsis_features
        .pipe(EncHelper.enrich_gc_and_standard)
        .pipe(EncHelper.enrich_3d_and_standard)
        .pipe(EncHelper.adjust_genres, ADJUST_GENRE_IDS)
    )

    # Genre full-replacement overrides
    for fid, genres in GENRE_OVERRIDES.items():
        mask = fsf['film_id'] == fid
        if mask.any():
            fsf.loc[mask, 'genres'] = fsf.loc[mask, 'genres'].apply(lambda _: genres)
            print(f"  Genre override: film_id={fid} → {genres}")

    # Primary audience overrides
    fsf['primary_audience'] = fsf['primary_audience'].astype(object)
    for fid, audience in PRIMARY_AUDIENCE_OVERRIDES.items():
        mask = fsf['film_id'] == fid
        if mask.any():
            fsf.loc[mask, 'primary_audience'] = audience
            print(f"  Audience override: film_id={fid} → {audience}")

    # Secondary audience overrides (appends, deduplicates)
    for fid, add_audiences in SECONDARY_AUDIENCES_OVERRIDES.items():
        mask = fsf['film_id'] == fid
        if mask.any():
            def _append(existing, to_add):
                existing = existing if isinstance(existing, list) else []
                return list(dict.fromkeys(existing + [a for a in to_add if a not in existing]))
            fsf.loc[mask, 'secondary_audiences'] = (
                fsf.loc[mask, 'secondary_audiences']
                .apply(lambda x: _append(x, add_audiences))
            )
            print(f"  Secondary audience override: film_id={fid} added {add_audiences}")

    # ── Stage 2: base flags ──────────────────────────────────────────────────
    fsf['is_ip'] = fsf['intellectual_property'].apply(
        lambda x: 0 if (x is None or len(x) == 0) else 1
    )

    # ── Stage 3: missing IP corrections ─────────────────────────────────────
    fsf = EncHelper.correct_missing_ips(fsf, MISSING_FILM_IPS)

    # ── Stage 4: feature engineering ────────────────────────────────────────
    fsf['is_franchise']    = fsf.apply(EncHelper.infer_is_franchise,    axis=1)
    fsf['ip_strength']     = fsf.apply(EncHelper.infer_ip_strength,     axis=1)
    fsf['adaptation_type'] = fsf.apply(EncHelper.infer_adaptation_type, axis=1)

    # Hard IP strength overrides (run after inference so these always win)
    for fid, strength in IP_STRENGTH_OVERRIDES.items():
        mask = fsf['film_id'] == fid
        if mask.any():
            fsf.loc[mask, 'ip_strength'] = strength
            print(f"  IP override: film_id={fid} → {strength}")

    # is_sequel overrides
    for fid, val in IS_SEQUEL_OVERRIDES.items():
        mask = fsf['film_id'] == fid
        if mask.any():
            cast_val = type(fsf['is_sequel'].iloc[0])(val)
            fsf.loc[mask, 'is_sequel'] = cast_val
            print(f"  is_sequel override: film_id={fid} → {cast_val}")

    fsf['has_english'] = fsf['language_cues'].apply(
        lambda langs: int(any(
            any(kw in str(l).lower() for kw in EncHelper.ENGLISH_LANG_KEYWORDS)
            for l in (langs if langs is not None else [])
        ))
    )
    fsf['language_group'] = fsf.apply(
        EncHelper.assign_language_group_with_distributor, axis=1
    )

    for col in PASSTHROUGH:
        fsf[col] = fsf[col].astype(int)

    # ── Token mapping ────────────────────────────────────────────────────────
    top_n_dict = auto_top_n(fsf, MULTI_VALUED_COLS)
    mapped_res = []
    for col in MULTI_VALUED_COLS:
        mapper = TopNTokenMapper(column=col, top_n=top_n_dict[col], max_tokens_per_row=5)
        mapped = (
            mapper.fit_transform(pd.DataFrame(fsf[['film_id', col]]))
            .drop(columns=[col])
            .drop_duplicates(subset=['film_id'])
        )
        mapped_res.append(mapped)

    all_mapped = reduce(
        lambda left, right: left.merge(right, on='film_id', how='left'),
        mapped_res,
    )
    fsf = fsf.merge(all_mapped, on='film_id', how='left')

    # Diagnostics
    print(FeatureDiagnostics(fsf).summary().to_string())
    print(f"\nLanguage group:\n{fsf['language_group'].value_counts()}")

    # ── Encoding ─────────────────────────────────────────────────────────────
    all_columns    = ONE_HOT + MULTI_HOT + EMBED_MAPPED + PASSTHROUGH
    df_for_process = fsf.drop_duplicates(subset=['film_id'])
    film_id_key    = df_for_process['film_id'].values
    df_for_process = df_for_process[all_columns]

    assert len(film_id_key) == len(df_for_process)

    transformers = []

    existing_onehot = [c for c in ONE_HOT if c in df_for_process.columns]
    if existing_onehot:
        transformers.append((
            'one_hot',
            OneHotEncoder(handle_unknown='ignore', sparse_output=False),
            existing_onehot,
        ))

    for col in MULTI_HOT:
        if col in df_for_process.columns:
            transformers.append((f'multi_hot_{col}', TopNMultiHotWithOther(top_n=32), col))

    for col in EMBED_MAPPED:
        if col in df_for_process.columns:
            transformers.append((
                f'dynamic_embed_{col}',
                DynamicTopNAndPCA(top_n_max=150, pca_max=32),
                col,
            ))

    existing_passthrough = [c for c in PASSTHROUGH if c in df_for_process.columns]
    if existing_passthrough:
        transformers.append(('passthrough', 'passthrough', existing_passthrough))

    feature_encoder = ColumnTransformer(transformers=transformers, remainder='drop')
    X_final         = feature_encoder.fit_transform(df_for_process)
    feature_names   = return_feature_names(feature_encoder=feature_encoder, X_final=X_final)

    X_final_df            = pd.DataFrame(X_final, columns=feature_names,
                                         index=df_for_process.index)
    X_final_df['film_id'] = film_id_key

    assert X_final_df['film_id'].nunique() == len(X_final_df), \
        f"Duplicate film_ids in output"

    X_final_df.to_parquet(out_path, engine='pyarrow')
    print(f"\nFinal shape: {X_final_df.shape}")
    print(f"Saved → {out_path}")
    return out_path


# ── Run ────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    encode_synopsis_features()
