import gc
import os
from pathlib import Path

import litellm
import pandas as pd
from dotenv import load_dotenv
import sys

sys.path.insert(0, "/Users/jonathanchapman/Documents/git/evt_back_up/base")

from base_snowflake import SnowFlakeBase

# Disable LiteLLM's async logging worker — it leaks unawaited coroutines in sync contexts
litellm.success_callback = list()
litellm.failure_callback = list()
# Suppress verbose stdout (provider list warnings for unrecognised models, etc.)
litellm.suppress_debug_info = True
litellm.set_sverbose = False

from extractor import LlmJsonExtractor
from models import MODELS, DEFAULT_MODEL, DEFAULT_FALLBACKS
from load_prompts import load_tasks_from_yaml
from config import SYNOPSES_EXTRACTED_PATH, SF_WAREHOUSE, SF_DATABASE, SF_SCHEMA, SF_RSA_KEY
from films.main import get_films_sources
from films import sql

sb = SnowFlakeBase(warehouse=SF_WAREHOUSE, database=SF_DATABASE, schema=SF_SCHEMA)
sb.create_snowflake_connection(SF_RSA_KEY)

df_films = sb.return_query_output(sql.SQL_FILM_DETAILS)

sample_size: int = 100
sample_head: bool = True

# def main(df_films: pd.DataFrame, sample_size: int = 100, sample_head: bool = True) -> None:
load_dotenv()
api_key = os.getenv('OPENAI_KEY')

model_cfg = MODELS.get(DEFAULT_MODEL)
tasks = load_tasks_from_yaml(Path("prompts/prompts.yaml"))

df_new = df_films.loc[df_films['synopsis'].str.len() >= 5].copy(deep=True)

if SYNOPSES_EXTRACTED_PATH.exists():
    df_existing = pd.read_parquet(SYNOPSES_EXTRACTED_PATH)
    existing_ids = df_existing['film_id'].tolist()
    df_new_diff = df_new.loc[~df_new['film_id'].isin(existing_ids)]
else:
    df_existing = pd.DataFrame()
    df_new_diff = df_new.copy()

if sample_size == 0:
    sample_kwarg = {'n': len(df_new_diff)}
elif 1 > sample_size > 0:
    sample_kwarg = {'n': round(len(df_new_diff) * sample_size, 0)}
else:
    sample_kwarg = {'n': sample_size}

if sample_head:
    df_sampled = df_new_diff.sort_values(by='film_open_date', ascending=False).head(**sample_kwarg)
else:
    df_sampled = df_new_diff.sample(**sample_kwarg)

print(f'Input | Imported: {len(df_films)} | Synopsis > 5: {len(df_new)} | New: {len(df_new_diff)} | Sampled {len(df_sampled)}')

# if df_sampled.empty:
#     print('No new titles with valid synopses found')
#     return

print(f'Extracting meta from {len(df_sampled)} titles using {model_cfg["model"]} (fallback: {DEFAULT_FALLBACKS[0]["model"]})')

extractor = LlmJsonExtractor(
    tasks=tasks,
    model=model_cfg['model'],
    fallbacks=DEFAULT_FALLBACKS,
    api_key=api_key,
)

multi_results = extractor.run_multiple_synopses(df_sampled, id_col="film_id", title_col="film_title", synopsis_col="synopsis")
df_extraction = pd.DataFrame.from_dict(multi_results, orient='index')

if '_error' in df_extraction.columns:
    df_extraction = df_extraction.loc[df_extraction['_error']]

if not df_existing.empty:
    df_result = pd.concat([df_existing, df_extraction]).drop_duplicates(subset='film_id', keep='first')
else:
    df_result = df_extraction

out_df = extractor.merge_genres(df_result, df_sampled)

out_df.to_parquet(SYNOPSES_EXTRACTED_PATH, index=False)
out_df.to_excel(SYNOPSES_EXTRACTED_PATH.with_suffix('.xlsx'), index=False)
print(f'Output saved | Before: {len(df_existing)} | After: {len(out_df)} | New {len(df_extraction)}')

del extractor
gc.collect()


# if __name__ == '__main__':
#     df_films = get_films_sources(persisted=False)
#     df_films = df_films.loc[df_films['film_nat_open_date'].between('2026-04-15', '2026-04-30', inclusive='both')]
#     main(df_films, sample_size=0)
