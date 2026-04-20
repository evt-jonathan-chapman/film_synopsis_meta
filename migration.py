from pathlib import Path
import pandas as pd


df_dst = pd.read_parquet('./synopses/source_data/synopses_extracted.parquet')


df_dst.rename(columns={'film_name': 'film_title'}, inplace=True)

df_dst.to_parquet(Path('synopses', 'synopses_extracted.parquet'), index=False)
df_dst.to_excel(Path('synopses', 'synopses_extracted.xlsx'), index=False)
