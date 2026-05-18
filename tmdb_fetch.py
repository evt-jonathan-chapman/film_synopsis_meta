"""
tmdb_fetch.py — fetch TMDB production company data and map to studio tier.
"""

import sys
import time
import os
import glob

import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, '/Users/jonathanchapman/Documents/git/evt_back_up/base')

from config import DATA_DIR, SF_WAREHOUSE, SF_DATABASE, SF_SCHEMA, SF_RSA_KEY, RAW_PARQUET_GLOB
from films import sql as films_sql

TMDB_KEY  = os.getenv('TMDB_KEY')
TMDB_BASE = 'https://api.themoviedb.org/3'

TARGET_IDS = [57603, 57343, 59530, 60336, 60261, 60560]
# Mandalorian and Grogu, Michael, Housemaid, Hail Mary, Devil Wears Prada 2, Marty Supreme

# ── Studio tier mapping ────────────────────────────────────────────────────────

_MAJOR = [
    'Walt Disney', 'Pixar', 'Marvel Studios', 'Lucasfilm',
    '20th Century Studios', '20th Century Fox', 'Searchlight',
    'Universal Pictures', 'DreamWorks', 'Focus Features', 'Working Title',
    'Warner Bros', 'New Line Cinema', 'DC Films', 'DC Studios',
    'Paramount Pictures', 'Paramount Animation',
    'Sony Pictures', 'Columbia Pictures', 'TriStar', 'Screen Gems',
    'Netflix', 'Apple Original', 'Apple Studios',
    'Amazon Studios', 'Amazon MGM', 'Prime Video',
]

_MINI_MAJOR = [
    'Lionsgate', 'A24', 'Metro-Goldwyn-Mayer', 'MGM',
    'STX Entertainment', 'Neon', 'Annapurna', 'FilmNation',
    'StudioCanal', 'Pathé', 'Pathe', 'Entertainment One', 'eOne',
    'Blumhouse', 'Legendary Entertainment', 'Village Roadshow',
    'Miramax', 'IFC Films', 'Magnolia', 'Samuel Goldwyn',
]


def _classify(companies: list[dict]) -> str:
    names = [c.get('name', '') for c in companies]
    for name in names:
        if any(s.lower() in name.lower() for s in _MAJOR):
            return 'major'
    for name in names:
        if any(s.lower() in name.lower() for s in _MINI_MAJOR):
            return 'mini_major'
    return 'indie' if names else 'unknown'


def _search(title: str, year: int) -> dict | None:
    r = requests.get(f'{TMDB_BASE}/search/movie',
                     params={'api_key': TMDB_KEY, 'query': title, 'year': year},
                     timeout=10)
    results = r.json().get('results', []) if r.ok else []
    if not results:
        r = requests.get(f'{TMDB_BASE}/search/movie',
                         params={'api_key': TMDB_KEY, 'query': title},
                         timeout=10)
        results = r.json().get('results', []) if r.ok else []
    if not results:
        return None
    yr_match = [x for x in results
                if abs(int((x.get('release_date') or '0')[:4] or 0) - year) <= 1]
    return yr_match[0] if yr_match else results[0]


def _companies(tmdb_id: int) -> list[dict]:
    r = requests.get(f'{TMDB_BASE}/movie/{tmdb_id}',
                     params={'api_key': TMDB_KEY}, timeout=10)
    return r.json().get('production_companies', []) if r.ok else []


# ── Load titles from Snowflake ────────────────────────────────────────────────

from base_snowflake import SnowFlakeBase
sb = SnowFlakeBase(warehouse=SF_WAREHOUSE, database=SF_DATABASE, schema=SF_SCHEMA)
sb.create_snowflake_connection(SF_RSA_KEY)
df_titles = pd.read_sql(films_sql.SQL_FILM_DETAILS, sb.engine)[['film_id', 'film_title', 'film_nat_open_date']]
df_titles['film_id'] = df_titles['film_id'].astype(int)
df_titles = df_titles[df_titles['film_id'].isin(TARGET_IDS)]
df_titles['year'] = pd.to_datetime(df_titles['film_nat_open_date'], utc=True).dt.year


print(f"Titles:\n{df_titles.to_string(index=False)}\n")

# ── Load release years from raw parquets ──────────────────────────────────────

# paths = sorted(glob.glob(RAW_PARQUET_GLOB))
# df_dates = pd.concat([pd.read_parquet(p, columns=['film_id', 'rel_at']) for p in paths])
# df_dates = df_dates.drop_duplicates('film_id')
# df_dates['film_id'] = df_dates['film_id'].astype(int)
# df_dates = df_dates[df_dates['film_id'].isin(TARGET_IDS)]

df = df_titles
print(f"Films to search:\n{df.to_string(index=False)}\n")

# ── Search TMDB ───────────────────────────────────────────────────────────────

rows = []
for row in df.itertuples():
    match = _search(row.film_title, int(row.year))
    time.sleep(0.05)

    if match:
        companies = _companies(match['id'])
        time.sleep(0.05)
        tier          = _classify(companies)
        company_names = ', '.join(c['name'] for c in companies)
        tmdb_title    = match.get('title', '')
        tmdb_year     = (match.get('release_date') or '')[:4]
    else:
        companies     = []
        tier          = 'unknown'
        company_names = ''
        tmdb_title    = ''
        tmdb_year     = ''

    print(f"{row.film_title}")
    print(f"  TMDB match:  {tmdb_title} ({tmdb_year})")
    print(f"  Companies:   {company_names}")
    print(f"  Studio tier: {tier}")
    print()

    rows.append({
        'film_id':     row.film_id,
        'film_title':  row.film_title,
        'year':        row.year,
        'tmdb_title':  tmdb_title,
        'tmdb_year':   tmdb_year,
        'companies':   company_names,
        'studio_tier': tier,
    })

df_results = pd.DataFrame(rows)
print(df_results.to_markdown(mode="table", index=False))