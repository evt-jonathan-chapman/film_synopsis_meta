"""
film_data_merge.py
-------------------
Joins synopses_extracted.parquet (per-film nano-LLM text classification) with
film_meta_enriched.parquet (per-film mini+web_search knowledge-grounded fields)
into one comprehensive per-film parquet, so downstream consumers get a single
usable film metadata source instead of having to merge the two raw extraction
outputs themselves. Both source parquets are read directly from S3 (via
s3_checkpoint.py — no local disk involved) and untouched by this step; the
output is written straight back to S3 too. Pure, stateless join re-derived
fresh on every run — no checkpoint of its own (same convention as
id_bridge.py, just via s3_checkpoint.py instead of local paths).

Join: outer join on film_id — synopsis covers most films (cheap nano path,
run for everything) and film_meta covers fewer (slower, pricier mini+
web_search path, see CLAUDE.md's cost table), but neither side is a strict
subset of the other in practice (a handful of films have a film_meta row
with no matching synopsis row, e.g. added directly via a different work-set
pull). An outer join keeps every film either side has ever extracted.

Column collisions (`title`, `genres`, `ip_strength`, `adaptation_type` exist
on both sides):
- genres: replaces the synopsis genre list's `biography`/`documentary` tags
  wholesale with film_meta's, for any film with a film_meta row. The
  synopsis nano-LLM (title+synopsis text only, no outside knowledge)
  reliably confuses biography with documentary/drama; film_meta's
  mini+web_search extraction gets this right because it can look the film
  up. Mirrors the sibling repo cinema_admits_models/build_data/
  encode_synopsis.py's `_merge_bio_doc` — this is a full replace, not an
  additive union: if film_meta tags the film with neither biography nor
  documentary, the merged row ends up with neither, even if synopsis
  originally had one.
- title / ip_strength / adaptation_type: film_meta's value wins when
  present (knowledge-grounded; `ip_strength`/`adaptation_type` extraction
  was also migrated from synopsis to film_meta for this reason — see
  prompts/prompts_v2.yaml's MIGRATED notes), falling back to synopsis's
  value for films with no film_meta row.

Output: s3://{bucket}/{prefix}/film_data_merged/film_data_merged.parquet
"""
import numpy as np
import pandas as pd

import s3_checkpoint

SYNOPSIS_S3_NAME  = "synopsis"
FILM_META_S3_NAME = "film_meta"
MERGED_S3_NAME    = "film_data_merged"

SYNOPSIS_FILENAME  = "synopses_extracted.parquet"
FILM_META_FILENAME = "film_meta_enriched.parquet"
MERGED_FILENAME    = "film_data_merged.parquet"

_BIO_DOC = {"biography", "documentary"}
_COLLISION_COLS = ["title", "genres", "ip_strength", "adaptation_type"]


def _as_list(genres) -> list:
    if isinstance(genres, (list, np.ndarray)):
        return list(genres)
    return []


def _merge_bio_doc(synopsis_genres, film_meta_genres, has_film_meta: bool) -> list:
    """Replace biography/documentary tags in synopsis_genres with film_meta's
    (matching is case-insensitive; original casing from each source is kept).
    No-op if this film has no film_meta row at all. If there's no synopsis
    genre baseline to carve from (film_meta-only row, e.g. a film with no
    matching synopsis extraction), falls back to film_meta's full list
    rather than just its biography/documentary tags."""
    syn_genres = _as_list(synopsis_genres)
    if not has_film_meta:
        return syn_genres
    fm_genres = _as_list(film_meta_genres)
    if not syn_genres:
        return fm_genres
    fm_bio_doc = [g for g in fm_genres if str(g).lower() in _BIO_DOC]
    kept       = [g for g in syn_genres if str(g).lower() not in _BIO_DOC]
    return kept + fm_bio_doc


def build_merged_film_data() -> dict:
    """Left-join synopsis + film_meta on film_id, resolve column collisions
    (genres via the biography/documentary carve-out, others via film_meta-
    wins-else-synopsis), write the combined parquet. Returns a summary dict
    for Dagster/CLI reporting."""
    synopsis = s3_checkpoint.read_parquet(SYNOPSIS_S3_NAME, SYNOPSIS_FILENAME)
    if synopsis is None:
        raise FileNotFoundError(
            f"Synopsis parquet not found: {s3_checkpoint.s3_uri(SYNOPSIS_S3_NAME, SYNOPSIS_FILENAME)}")

    film_meta = s3_checkpoint.read_parquet(FILM_META_S3_NAME, FILM_META_FILENAME)
    if film_meta is None:
        film_meta = pd.DataFrame(columns=["film_id"])

    # Rename collision columns explicitly before merging, rather than relying
    # on pandas' automatic suffixing — that only kicks in when a column is
    # present on BOTH sides. If either parquet is ever missing one of these
    # (older schema, empty bootstrap run, etc.), auto-suffixing silently
    # skips it and a later `.pop(f"{col}_fm")` KeyErrors instead of just
    # treating that side as absent.
    synopsis  = synopsis.rename(columns={c: f"{c}_syn" for c in _COLLISION_COLS if c in synopsis.columns})
    film_meta = film_meta.rename(columns={c: f"{c}_fm"  for c in _COLLISION_COLS if c in film_meta.columns})

    merged = synopsis.merge(film_meta, on="film_id", how="outer", indicator="_merge_ind")
    has_fm  = merged["_merge_ind"] != "left_only"   # film_meta present ("both" or "right_only")
    has_syn = merged["_merge_ind"] != "right_only"  # synopsis present ("both" or "left_only")
    merged  = merged.drop(columns=["_merge_ind"])

    def _pop_or_none(col: str) -> pd.Series:
        return merged.pop(col) if col in merged.columns else pd.Series([None] * len(merged), index=merged.index)

    merged["genres"] = [
        _merge_bio_doc(syn_g, fm_g, has_fm_i)
        for syn_g, fm_g, has_fm_i in zip(_pop_or_none("genres_syn"), _pop_or_none("genres_fm"), has_fm)
    ]
    for col in ("title", "ip_strength", "adaptation_type"):
        merged[col] = _pop_or_none(f"{col}_fm").combine_first(_pop_or_none(f"{col}_syn"))

    merged["has_film_meta"] = has_fm
    merged["has_synopsis"]  = has_syn

    lead_cols  = ["film_id", "title", "has_film_meta", "has_synopsis", "genres"]
    other_cols = [c for c in merged.columns if c not in lead_cols]
    merged     = merged[lead_cols + other_cols]

    s3_checkpoint.write_parquet(MERGED_S3_NAME, MERGED_FILENAME, merged)

    n_both        = int((has_fm & has_syn).sum())
    n_syn_only    = int((has_syn & ~has_fm).sum())
    n_fm_only     = int((has_fm & ~has_syn).sum())
    merged_uri    = s3_checkpoint.s3_uri(MERGED_S3_NAME, MERGED_FILENAME)
    print(f"film_data_merge → {merged_uri}  "
          f"({len(merged):,} films: {n_both:,} both, {n_syn_only:,} synopsis-only, "
          f"{n_fm_only:,} film_meta-only)")

    return {
        "total":          int(len(merged)),
        "both":           n_both,
        "synopsis_only":  n_syn_only,
        "film_meta_only": n_fm_only,
        "path":           merged_uri,
    }


if __name__ == "__main__":
    summary = build_merged_film_data()
    print(f"\nSummary: {summary}")
