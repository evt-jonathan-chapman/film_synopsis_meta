"""
Dagster definitions for film_synopsis_meta.

One upstream asset (`films_source`) pulls the film list from Snowflake once.
Four downstream assets (`synopsis`, `cast`, `directors`, `film_meta`) each
consume it independently — so any one can be re-materialised on its own.

Schedules:
  - nightly_schedule       — synopsis + cast + directors (cheap, ~$5/night)
  - film_meta_schedule     — film_meta only, weekly (expensive, ~$100/run)
"""

from dagster import (
    asset, Definitions, define_asset_job, ScheduleDefinition, AssetSelection,
)
import pandas as pd

from refresh import (
    load_films_from_snowflake,
    refresh_synopsis,
    refresh_cast,
    refresh_directors,
    refresh_film_meta,
)
from rematch_comscore import refresh_comscore_match


@asset
def films_source() -> pd.DataFrame:
    """Snowflake film list — shared input for all four extraction paths."""
    df = load_films_from_snowflake()
    if df is None:
        raise RuntimeError("Snowflake unavailable")
    return df


@asset
def synopsis(films_source: pd.DataFrame) -> dict:
    """Per-film text classifications (nano, no web search)."""
    return refresh_synopsis(films_source)


@asset
def cast(films_source: pd.DataFrame) -> dict:
    """Per-actor profiles (mini + web search)."""
    return refresh_cast(films_source)


@asset
def directors(films_source: pd.DataFrame) -> dict:
    """Per-director profiles (mini + web search)."""
    return refresh_directors(films_source)


@asset
def film_meta(films_source: pd.DataFrame) -> dict:
    """Per-film studios / billing / budget / IP (mini + web search). Expensive."""
    return refresh_film_meta(films_source)


@asset(deps=[film_meta])
def comscore_match() -> dict:
    """Match EVT films → Comscore (IBOE_TITLES) rows.

    Self-contained: pulls Comscore directly from Snowflake and builds the EVT
    work-set from parquet snapshots + film_meta (needed for the concert-film
    filter). Declared dep on film_meta so a fresh film_meta run marks this
    stale.
    """
    return refresh_comscore_match()


# ── Jobs ──────────────────────────────────────────────────────────────────────

nightly_job = define_asset_job(
    "nightly_job",
    selection=AssetSelection.assets(films_source, synopsis, cast, directors),
)

film_meta_job = define_asset_job(
    "film_meta_job",
    selection=AssetSelection.assets(films_source, film_meta),
)

comscore_job = define_asset_job(
    "comscore_job",
    selection=AssetSelection.assets(comscore_match),
)

full_refresh_job = define_asset_job("full_refresh_job", selection="*")


# ── Schedules ─────────────────────────────────────────────────────────────────

nightly_schedule = ScheduleDefinition(
    job=nightly_job,
    cron_schedule="0 2 * * *",      # 02:00 daily
)

film_meta_schedule = ScheduleDefinition(
    job=film_meta_job,
    cron_schedule="0 3 * * 0",      # 03:00 Sundays
)


defs = Definitions(
    assets=[films_source, synopsis, cast, directors, film_meta, comscore_match],
    jobs=[nightly_job, film_meta_job, comscore_job, full_refresh_job],
    schedules=[nightly_schedule, film_meta_schedule],
)
