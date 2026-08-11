import pandas as pd

from base_snowflake import SnowFlakeBase
from config import SF_WAREHOUSE, SF_DATABASE, SF_SCHEMA, SF_RSA_KEY
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

    sb = SnowFlakeBase(warehouse=SF_WAREHOUSE, database=SF_DATABASE, schema=SF_SCHEMA)
    sb.create_snowflake_connection(SF_RSA_KEY)
    df_films = pd.read_sql(sql.SQL_FILM_DETAILS, sb.engine)
    df_films.columns = df_films.columns.str.lower()

    # df_films['primary_genres'] = df_films[['genre_1', 'genre_2', 'genre_3']].apply(lambda x: '|'.join(x.dropna()), axis=1).str.lower()
    df_films.to_parquet(FILM_PARQUET_PATH)

    return _deduplicate_alt_synopsis(df_films)


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
