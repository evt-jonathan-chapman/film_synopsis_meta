"""
id_bridge.py
------------
Joins the independently-matched comscore_cache.parquet and gower_cache.parquet
into one lookup keyed on EVT film_id: film_id, cs_id, gower_id (+ each
source's matched title and confidence, for triage). Both matchers run
independently against EVT titles (see COMSCORE.md) — this is a pure
outer-join on film_id, no additional matching logic of its own.

Read film_id_bridge.parquet, not the two matcher caches directly, when all
you need is the film_id -> cs_id / gower_id crosswalk.

Output: DATA_DIR/id_bridge/film_id_bridge.parquet
"""

import os
import pandas as pd

from config import DATA_DIR
from comscore_matcher import ComscoreMatcher
from gower_matcher import GowerMatcher

ID_BRIDGE_DIR  = DATA_DIR / "id_bridge"
ID_BRIDGE_PATH = str(ID_BRIDGE_DIR / "film_id_bridge.parquet")

_CS_COLS = ["film_id", "film", "cs_id", "cs_title", "match_confidence"]
_GW_COLS = ["film_id", "film", "gower_id", "gower_title", "match_confidence"]


def _read_cache(path: str, cols: list[str]) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=cols)
    return pd.read_parquet(path, columns=cols)


def build_id_bridge() -> dict:
    """Outer-join comscore_cache + gower_cache on film_id. Returns summary dict for Dagster."""
    cs = _read_cache(ComscoreMatcher.CACHE_PATH, _CS_COLS).rename(
        columns={"match_confidence": "cs_match_confidence"})
    gw = _read_cache(GowerMatcher.CACHE_PATH, _GW_COLS).rename(
        columns={"match_confidence": "gower_match_confidence"})

    bridge = cs.merge(gw, on="film_id", how="outer", suffixes=("_cs", "_gw"))
    bridge["film"] = bridge["film_cs"].combine_first(bridge["film_gw"])
    bridge = bridge.drop(columns=["film_cs", "film_gw"])
    bridge = bridge[[
        "film_id", "film",
        "cs_id", "cs_title", "cs_match_confidence",
        "gower_id", "gower_title", "gower_match_confidence",
    ]]

    os.makedirs(ID_BRIDGE_DIR, exist_ok=True)
    bridge.to_parquet(ID_BRIDGE_PATH, index=False)

    n_both      = (bridge["cs_id"].notna() & bridge["gower_id"].notna()).sum()
    n_cs_only   = (bridge["cs_id"].notna() & bridge["gower_id"].isna()).sum()
    n_gw_only   = (bridge["cs_id"].isna() & bridge["gower_id"].notna()).sum()
    n_neither   = (bridge["cs_id"].isna() & bridge["gower_id"].isna()).sum()

    print(f"id_bridge → {ID_BRIDGE_PATH}  "
          f"({len(bridge):,} films: {n_both:,} both, {n_cs_only:,} cs-only, "
          f"{n_gw_only:,} gower-only, {n_neither:,} neither)")

    return {
        "total":     int(len(bridge)),
        "both":      int(n_both),
        "cs_only":   int(n_cs_only),
        "gower_only": int(n_gw_only),
        "neither":   int(n_neither),
        "path":      ID_BRIDGE_PATH,
    }


if __name__ == "__main__":
    summary = build_id_bridge()
    print(f"\nSummary: {summary}")
