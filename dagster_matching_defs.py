"""
Dagster definitions for film_synopsis_meta — Comscore/Gower matching only.

Deliberately separate from `dagster_defs.py`: this code location only ever
imports `rematch_comscore.py` / `rematch_gower.py` / `id_bridge.py`, never
`refresh.py` — so `dagster dev -f dagster_matching_defs.py` never pulls in
`litellm`/`openai`. See requirements-matching.txt for the lean dependency
set this needs (no LLM SDKs).

`comscore_match` and `gower_match` do NOT consume `films_source` — each
loads EVT films from parquet snapshots itself (via the shared
`rematch_comscore.py::load_evt_films`) and reads `film_meta_enriched.parquet`
directly off disk for the concert-film filter. Because `film_meta` lives in
the *other* code location (dagster_defs.py), there's no Dagster-level
`deps=[film_meta]` declaration here — the functional dependency on
`film_meta_enriched.parquet` still exists (materialise `film_meta` at least
once via `dagster_defs.py`, or run `python refresh.py --only film_meta`,
before running this job for the first time), it just won't show up as
"stale" in this UI when film_meta re-runs elsewhere. See COMSCORE.md /
GOWER.md for the matching algorithm.
"""

from dagster import asset, Definitions, define_asset_job, AssetSelection

from rematch_comscore import refresh_comscore_match
from rematch_gower import refresh_gower_match
from id_bridge import build_id_bridge


@asset
def comscore_match() -> dict:
    """Match EVT films → Comscore (IBOE_TITLES) rows. CPU fuzzy match, no API."""
    return refresh_comscore_match()


@asset
def gower_match() -> dict:
    """Match EVT films → Gower (GW_LIFE_TIME) rows. CPU fuzzy match, no API."""
    return refresh_gower_match()


@asset(deps=[comscore_match, gower_match])
def id_bridge() -> dict:
    """Outer-join comscore_cache + gower_cache on film_id → film_id_bridge.parquet.

    Pure join, no matching logic of its own — reads both caches directly off
    disk rather than the upstream assets' in-memory outputs.
    """
    return build_id_bridge()


# ── Jobs ──────────────────────────────────────────────────────────────────────

comscore_job = define_asset_job(
    "comscore_job",
    selection=AssetSelection.assets(comscore_match, gower_match, id_bridge),
)


defs = Definitions(
    assets=[comscore_match, gower_match, id_bridge],
    jobs=[comscore_job],
)
