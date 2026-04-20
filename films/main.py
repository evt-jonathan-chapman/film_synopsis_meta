from typing import Any
import pandas as pd


from tools.connections import SnowflakeDB
from films import sql
from films.config import FILM_PARQUET_PATH


def _deduplicate_alt_synopsis(df: pd.DataFrame) -> pd.DataFrame:
    """Null out alt_synopsis where it matches synopsis after stripping all whitespace."""
    if 'alt_synopsis' not in df.columns or 'synopsis' not in df.columns:
        return df
    primary = df['synopsis'].fillna('').str.replace(r'\s+', '', regex=True)
    alt = df['alt_synopsis'].fillna('').str.replace(r'\s+', '', regex=True)
    df.loc[alt == primary, 'alt_synopsis'] = None
    return df


def get_films_sources(persisted: bool = True) -> pd.DataFrame:
    try:
        if persisted:
            df_films = pd.read_parquet(FILM_PARQUET_PATH)
            return _deduplicate_alt_synopsis(df_films)

    except FileNotFoundError:
        pass

    snow_db = SnowflakeDB()

    df_films = snow_db.sync_select(sql.SQL_FILM_DETAILS)

    # df_films['primary_genres'] = df_films[['genre_1', 'genre_2', 'genre_3']].apply(lambda x: '|'.join(x.dropna()), axis=1).str.lower()
    df_films.to_parquet(FILM_PARQUET_PATH)

    return _deduplicate_alt_synopsis(df_films)


def get_films_by_release(*release_dates: str) -> dict[str, list[dict[str, Any]]]:
    snow_db = SnowflakeDB()
    result: dict[str, list[dict[str, Any]]] = {}

    for r in release_dates:
        df = snow_db.sync_select(sql.SQL_FILMS_BY_RELEASE, params={'release_date': r})
        if df.empty:
            continue

        result[r] = df.to_dict('records')

    return result


if __name__ == '__main__':
    f = get_films_sources(persisted=False)

    films = {
        59531: 'SCREAM 7',
        61031: 'THE TESTAMENT OF ANN LEE',
        61162: 'SOLO MIO',
        61195: 'PEGASUS 3',
        61196: 'NIGHT KING',
    }

    print(f.loc[f['film_id'].isin(list(films))])
