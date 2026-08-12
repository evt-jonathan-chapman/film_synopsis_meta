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

A fifth asset, `merged_film_data`, outer-joins `synopsis` + `film_meta` into
one comprehensive per-film parquet (see film_data_merge.py) — a real
Dagster-tracked dependency since both inputs live in this same code
location. Its own job (`merged_film_data_job`) since it only makes sense to
run after a `film_meta_job` materialisation, not on the cheap nightly cadence.

There is deliberately no `s3_sync` asset here anymore. All five assets above
now read/write S3 directly via s3_checkpoint.py — no local disk involved, no
separate sync step, no dependency on an interactively-refreshed Stax SSO
session (see s3_checkpoint.py's module docstring and CLAUDE.md). s3_sync.py
still exists but is now scoped to main.py's local-disk ad-hoc runs only.

Schedules:
  - nightly_schedule       — synopsis + cast + directors (cheap, ~$5/night)
  - film_meta_schedule     — film_meta only, weekly (expensive, ~$100/run)
  - merged_film_data_job   — unscheduled, ad-hoc only, run after film_meta_job
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
from film_data_merge import build_merged_film_data


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


@asset(deps=[synopsis, film_meta])
def merged_film_data() -> dict:
    """Outer-joins synopsis + film_meta into one comprehensive per-film parquet
    (biography/documentary genre carve-out applied — see film_data_merge.py).
    Stateless: re-derived fresh from the two source parquets on every run.
    """
    return build_merged_film_data()


# ── Jobs ──────────────────────────────────────────────────────────────────────

nightly_job = define_asset_job(
    "nightly_job",
    selection=AssetSelection.assets(films_source, synopsis, cast, directors),
)

film_meta_job = define_asset_job(
    "film_meta_job",
    selection=AssetSelection.assets(films_source, film_meta),
)

merged_film_data_job = define_asset_job(
    "merged_film_data_job",
    selection=AssetSelection.assets(merged_film_data),
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
    assets=[films_source, synopsis, cast, directors, film_meta, merged_film_data],
    jobs=[nightly_job, film_meta_job, merged_film_data_job, full_refresh_job],
    schedules=[nightly_schedule, film_meta_schedule],
)
