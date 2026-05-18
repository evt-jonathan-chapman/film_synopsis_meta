from pathlib import Path
import yaml

_CFG_PATH = Path(__file__).parent / 'config.yaml'
with open(_CFG_PATH) as _f:
    _cfg = yaml.safe_load(_f)

_data    = Path(_cfg['paths']['main_dir'])
DATA_DIR = _data

# ── Snowflake ──────────────────────────────────────────────────────────────────
SF_WAREHOUSE = _cfg['snowflake']['warehouse']
SF_DATABASE  = _cfg['snowflake']['database']
SF_SCHEMA    = _cfg['snowflake']['schema']
SF_RSA_KEY   = _cfg['snowflake']['rsa_key']

# ── SQL ────────────────────────────────────────────────────────────────────────
SQL_PATH     = Path(_cfg['paths']['sql_path'])
THREE_D_SQL  = _cfg['paths']['three_d_sql']

# ── Data paths (all under main_dir) ───────────────────────────────────────────
_synopsis_dir = _cfg['paths']['synopsis_dir']
_cast_dir     = _cfg['paths']['cast_dir']

_director_dir  = _cfg['paths'].get('director_dir',  'director_meta')
_film_meta_dir = _cfg['paths'].get('film_meta_dir', 'film_meta')

SYNOPSES_EXTRACTED_PATH  = _data / _synopsis_dir  / 'synopses_extracted.parquet'
CAST_ENRICHED_PATH       = _data / _cast_dir      / 'cast_enriched.parquet'
CAST_FEATURES_PATH       = _data / _cast_dir      / 'cast_features.parquet'
DIRECTOR_ENRICHED_PATH   = _data / _director_dir  / 'director_enriched.parquet'
DIRECTOR_FEATURES_PATH   = _data / _director_dir  / 'director_features.parquet'
FILM_META_ENRICHED_PATH  = _data / _film_meta_dir / 'film_meta_enriched.parquet'
ENCODED_META_DIR        = _data / 'encoded_film_meta'

# Raw parquet globs — combine all three for full film coverage
RAW_PARQUET_GLOB      = str(_data / 'raw_from_snowflake'        / '*' / 'train' / 'train_raw_ds.parquet')
RAW_PARQUET_GLOB_TEST = str(_data / 'raw_from_snowflake'        / '*' / 'test'  / 'test_raw_ds.parquet')
RAW_PARQUET_GLOB_PRED = str(_data / 'prediction_from_snowflake' / '*' / 'prediction_raw.parquet')

# Proc parquets — have merged admits (week_admits_merged_w1) used for per-actor admits computation
PROC_PARQUET_GLOB      = str(_data / 'raw_from_snowflake' / '*' / 'train' / 'train_proc_ds.parquet')
PROC_PARQUET_GLOB_TEST = str(_data / 'raw_from_snowflake' / '*' / 'test'  / 'test_proc_ds.parquet')

# All globs combined — use this as the default for encode functions so test/pred films are included
RAW_PARQUET_GLOBS_ALL  = [RAW_PARQUET_GLOB, RAW_PARQUET_GLOB_TEST, RAW_PARQUET_GLOB_PRED]
PROC_PARQUET_GLOBS_ALL = [PROC_PARQUET_GLOB, PROC_PARQUET_GLOB_TEST]
