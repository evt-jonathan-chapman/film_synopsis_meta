"""
Dagster definitions for film_synopsis_meta — LLM extraction paths only.

One upstream asset (`films_source`) pulls the film list from Snowflake once.
Four downstream assets (`synopsis`, `cast`, `directors`, `film_meta`) each
consume it independently — so any one can be re-materialised on its own.

Comscore/Gower matching (no LLM, no API — pure CPU fuzzy matching) lives in
the separate `dagster_matching_defs.py` code location instead of here, so
`dagster dev -f dagster_matching_defs.py` never has to import `refresh.py`'s
`litellm`/`openai` dependencies. Run both processes side by side if you want
both sets of assets available (see DAGSTER.md).

A fifth asset, `s3_sync`, uploads the four checkpoints (parquet + progress
json) to S3 afterwards — kept as its own job (`s3_sync_job`) rather than
appended to the extraction jobs, since it needs its own AWS auth step (see
s3_sync.py) that the extraction assets don't. Trigger it manually once the
week's run is done and your AWS session is fresh.

Schedules:
  - nightly_schedule       — synopsis + cast + directors (cheap, ~$5/night)
  - film_meta_schedule     — film_meta only, weekly (expensive, ~$100/run)
  - s3_sync_job            — unscheduled, ad-hoc only (see s3_sync.py)
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
from s3_sync import sync_meta_outputs_to_s3


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


@asset(deps=[synopsis, cast, directors, film_meta])
def s3_sync() -> dict:
    """Uploads the four meta checkpoints (parquet + progress json) to S3.

    Requires a fresh AWS session for the profile in config.yaml's `s3.profile`
    (Stax SSO credentials expire hourly) — run `stax2aws login` first if this
    fails with a credentials error. See s3_sync.py.
    """
    return sync_meta_outputs_to_s3()


# ── Jobs ──────────────────────────────────────────────────────────────────────

nightly_job = define_asset_job(
    "nightly_job",
    selection=AssetSelection.assets(films_source, synopsis, cast, directors),
)

film_meta_job = define_asset_job(
    "film_meta_job",
    selection=AssetSelection.assets(films_source, film_meta),
)

s3_sync_job = define_asset_job(
    "s3_sync_job",
    selection=AssetSelection.assets(s3_sync),
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
    assets=[films_source, synopsis, cast, directors, film_meta, s3_sync],
    jobs=[nightly_job, film_meta_job, s3_sync_job, full_refresh_job],
    schedules=[nightly_schedule, film_meta_schedule],
)
