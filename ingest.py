from collections import Counter
from pathlib import Path
import json
import re
from typing import Any

import numpy as np
import pandas as pd
from snowflake.connector.pandas_tools import write_pandas
from sqlalchemy import text

from tools.connections import SnowflakeDB
from synopses import sql


def to_ndarray(v: Any) -> np.ndarray:
    """
    Convert Snowflake-returned JSON array string (possibly pretty-printed)
    into a numpy ndarray of strings. Nulls -> empty array.
    """
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return np.array([], dtype=object)

    # Sometimes connectors may already return list for VARIANT; handle anyway
    if isinstance(v, (list, tuple, set, np.ndarray)):
        arr = list(v) if not isinstance(v, np.ndarray) else v.tolist()
        return np.array([str(x) for x in arr if str(x).strip()], dtype=object)

    s = str(v).strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return np.array([], dtype=object)

    # Parse JSON list
    parsed = json.loads(s)  # will handle newlines/indentation fine
    if not isinstance(parsed, list):
        # if someone stored a scalar, normalise to singleton array
        parsed = [parsed]

    return np.array([str(x) for x in parsed if str(x).strip()], dtype=object)


def to_variant_json(v: Any, null_if_empty: bool = False) -> str | None:
    """Return compact JSON array string suitable for PARSE_JSON()."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None

    if isinstance(v, np.ndarray):
        arr = [str(x).strip() for x in v.tolist() if str(x).strip()]
    elif isinstance(v, (list, tuple, set)):
        arr = [str(x).strip() for x in v if str(x).strip()]
    else:
        # fallback: scalar
        s = str(v).strip()
        arr = [] if not s else [s]

    if null_if_empty and not arr:
        return None

    return json.dumps(arr, ensure_ascii=False, separators=(",", ":"))


def detect_variant_columns(df: pd.DataFrame, sample: int = 2000, threshold: float = 0.2) -> list[str]:
    cols = []
    n = min(sample, len(df))
    if n == 0:
        return cols

    samp = df.sample(n=n, random_state=0) if len(df) > n else df

    for c in samp.columns:
        if samp[c].dtype == "O":
            s = samp[c].dropna().astype(str).str.strip()
            if len(s) == 0:
                continue
            rate = (s.str.startswith("[") & s.str.endswith("]")).mean()
            if rate >= threshold:
                cols.append(c)
    return cols


def get_synopses_differences(local_synopses_path: Path):
    key = 'film_id'
    snow_db = SnowflakeDB(role='ent_forecast_owner')

    snow_synopses = snow_db.sync_select(sql.SQL_CURRENT_SYNOPSES)
    local_synopses = pd.read_parquet(local_synopses_path)
    variant_cols = detect_variant_columns(local_synopses)

    for c in variant_cols:
        snow_synopses[c] = snow_synopses[c].map(to_ndarray)

    local_keys = pd.Index(local_synopses[key].dropna().unique())
    snow_keys = pd.Index(snow_synopses[key].dropna().unique())

    local_only_keys = local_keys.difference(snow_keys)
    snow_only_keys = snow_keys.difference(local_keys)

    ingest_df = local_synopses[local_synopses[key].isin(local_only_keys)].copy()
    persist_df = snow_synopses[snow_synopses[key].isin(snow_only_keys)].copy()

    for c in variant_cols:
        ingest_df[c] = ingest_df[c].map(to_variant_json)

    # NaN -> None so DB driver sends NULLs properly
    ingest_df = ingest_df.where(pd.notna(ingest_df), None)

    print(f'New records for Snowflake: {len(ingest_df)}')
    print(f'New records for Local: {len(persist_df)}')

    return ingest_df, persist_df, local_synopses


def ingest_local_to_snowflake(ingest_df: pd.DataFrame):
    if ingest_df.empty:
        return

    snow_db = SnowflakeDB(role='ent_forecast_owner')

    ingest_cols = [c for c in ingest_df.columns if c.lower() not in ['synopsis']]

    clean_ingest_df = ingest_df[ingest_cols].copy()
    clean_ingest_df = clean_ingest_df.rename(columns={k: k.upper() for k in clean_ingest_df.columns})

    with snow_db.db_engine.connect() as connection:
        database, schema, table = tuple(sql.SQL_STAGING_TABLE.split('.'))

        raw_connection = connection.connection

        # write_pandas handles the staging and copying internally
        success, nchunks, nrows, _ = write_pandas(
            conn=raw_connection,
            df=clean_ingest_df,
            table_name=table,
            database=database,
            schema=schema
        )

        raw_connection.commit()
        print(f"Successfully loaded {nrows} rows across {nchunks} chunks: {success}")

        if not success:
            raise Exception('Failed to write pandas to Staging')

        connection.execute(snow_db.str_to_sqltext(sql.SQL_DELETE_STAGING_DEDUPE))
        connection.execute(snow_db.str_to_sqltext(sql.SQL_UPDATE_STAGING_VARIANTS))
        connection.execute(snow_db.str_to_sqltext(sql.SQL_MERGE_STAGING_TO_CURATED))

        connection.commit()


def persist_snowflake_to_local(persist_df: pd.DataFrame, local_df: pd.DataFrame, synopses_path: Path):
    persist_cols = [c for c in persist_df.columns if c.lower() not in ['synopses_id', 'loaded_timestamp']]

    clean_persist_df = persist_df[persist_cols].copy()
    clean_persist_df = clean_persist_df.rename(columns={k: k.lower() for k in clean_persist_df.columns})

    new_local_df = pd.concat([local_df, clean_persist_df])

    new_local_df.to_parquet(synopses_path)


def sync_synopses_sources(synopses_path: Path):
    ingest_df, persist_df, local_df = get_synopses_differences(synopses_path)
    ingest_local_to_snowflake(ingest_df)
    persist_snowflake_to_local(persist_df, local_df, synopses_path)


if __name__ == '__main__':
    sync_synopses_sources(Path('synopses', 'outputs', 'synopses_extracted.parquet'))
