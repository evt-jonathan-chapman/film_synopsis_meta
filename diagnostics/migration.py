from pathlib import Path
import pandas as pd

# One-off migration — preserved for reference. Paths kept relative to repo root.
_REPO = Path(__file__).resolve().parent.parent

df_dst = pd.read_parquet(_REPO / 'synopses' / 'source_data' / 'synopses_extracted.parquet')


df_dst.rename(columns={'film_name': 'film_title'}, inplace=True)

df_dst.to_parquet(_REPO / 'synopses' / 'synopses_extracted.parquet', index=False)
df_dst.to_excel(_REPO / 'synopses' / 'synopses_extracted.xlsx', index=False)
