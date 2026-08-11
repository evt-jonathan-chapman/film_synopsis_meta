# Gower Matching

Maps EVT `film_id` → Gower `gower_id` (`prmry_title_no` on the Gower side)
so Gower box-office estimates (`ENT_FORECAST_PRD.CURATED.GW_LIFE_TIME`) can be
joined onto EVT films — same goal as `COMSCORE.md`, different source.

Gower matching is **independent** of Comscore matching: both run their own
fuzzy pass straight against the EVT film catalogue rather than chaining
through each other, so a bad Comscore match can't drag down a Gower match
(or vice versa) and a film missing from one source can still resolve on the
other. `id_bridge.py` joins the two results together afterwards — see
"Combining with Comscore" below.

---

## Key files

| File | Purpose |
|---|---|
| `sql/gower_export.sql` | One-off Snowflake pull of Gower titles + lifetime box-office estimates |
| `title_matcher.py::FuzzyTitleMatcher` | Shared matching engine — same code Comscore matching uses (see `COMSCORE.md`) |
| `gower_matcher.py::GowerMatcher` | Gower-specific column names/paths on top of `FuzzyTitleMatcher` |
| `rematch_gower.py` | Driver script — pulls SQL, loads EVT films, runs matcher |

Outputs under `~/Documents/data/gower/`:

| File | Description |
|---|---|
| `gower_cache.parquet` | All matched films keyed on EVT `film_id` — this is the checkpoint |
| `gower_review_needed.parquet` | Borderline + unmatched rows with top-5 candidates, for human triage |
| `gower_manual_overrides.csv` | **Not created yet** — deferred until the review file shows it's needed. The mechanism already works (see below); someone just has to create the file. |

---

## What's different from Comscore

Gower is a thinner data source than Comscore, so the matcher config in
`gower_matcher.py` differs from `comscore_matcher.py` in a few ways:

- **One title column, not five.** Comscore scores against `film_name` /
  `upper_name` / `title_aka` / `us_title_name` / `short_name`; Gower only has
  `title`. `TITLE_COLS = ["title"]`.
- **No stable numeric ID.** Comscore has `title_global_id`. Gower's closest
  equivalent is `prmry_title_no` — used as `ID_COL`, and exposed downstream
  as the cache column `gower_id` (matching the requested
  `film_id` / `cs_id` / `gower_id` bridge schema).
- **No alt-content flag.** Comscore drops concert/sports rows via
  `is_alt_content`; Gower's extract has no equivalent column, so
  `ALT_CONTENT_COL = None` and no such filter runs. (Concert films are still
  excluded on the EVT side via `adaptation_type == "concert_film"`, same as
  Comscore — see `rematch_comscore.py::load_evt_films`, reused by
  `rematch_gower.py`.)
- **Multiple rows per title.** `sql/gower_export.sql` returns up to three
  snapshot rows per title — `snapshot_type` in `latest` /
  `1m_pre_release` / `3m_pre_release` (the CTE window-function logic picks
  these three so you can compare a title's box-office estimate at different
  points before/after release). `GowerMatcher._prep_source()` picks the
  `latest` snapshot (falling back to `1m_pre_release` then
  `3m_pre_release`) for each `prmry_title_no` before the shared
  dedup-by-`ID_COL` logic runs, so scoring only ever sees one row per title.
  The winning snapshot's `life_time_base` is carried into the cache as
  `gower_life_time_base` (via `GowerMatcher(carry_cols=["life_time_base"])`
  in `rematch_gower.py`).
- **Manual overrides are supported but not yet used.** The override
  mechanism lives entirely in `title_matcher.py` and is column-name-driven
  (`ID_FIELD` → `manual_override_{ID_FIELD}` for the review column,
  `{ID_FIELD}` for the CSV column) — for Gower that's
  `manual_override_gower_id` / `gower_id`. It activates automatically the
  moment `gower_manual_overrides.csv` exists at
  `DATA_DIR/gower/gower_manual_overrides.csv`; no code change needed.

Everything else — title normalisation, fuzzy scoring (rapidfuzz
`ratio`/`token_sort_ratio`/`token_set_ratio`), the length-ratio guard,
±1-year window, date-proximity tie-break, confidence tiers
(`high`/`borderline`/`unmatched`/`manual`/`variant`), and variant propagation
(3D/IMAX/GC/sing-along… format variants inheriting a base film's match) — is
identical to Comscore, because it's literally the same shared code in
`title_matcher.py::FuzzyTitleMatcher`. See `COMSCORE.md` for the full
algorithm write-up; it applies here unchanged.

---

## The Gower extract is windowed — and so is the EVT work-set

`sql/gower_export.sql` only pulls titles with `rel_date >= '2025-01-01'`
(currently AU-only, no equivalent to Comscore's pre/post-COVID split).
EVT films released earlier than that have no possible candidate row in
Gower, so they'd always land as `unmatched` — pure noise in the review
file. `rematch_gower.py::GOWER_MIN_REL_DATE` (currently `2025-01-01`,
**must be kept in sync with the SQL's `params.start_date`**) filters the
EVT work-set to the same window before matching, in
`refresh_gower_match()`. On the full EVT catalogue (18,513 → 16,337 films
after the usual festival/concert exclusions) this cuts the work-set to
~1,562 films — the ones actually released in-window.

This is a deliberate, narrower scope than Comscore's (which covers full
history in two windows) — Gower matching is currently only needed for
2025-onwards films for another project. To extend Gower matching back to
full history later: widen `start_date` in `sql/gower_export.sql`, bump/remove
`GOWER_MIN_REL_DATE` in `rematch_gower.py` to match, and re-run —
already-matched films are skipped as usual, so this is incremental, not a
full re-match.

---

## Checkpoint / resume behaviour

Same rules as Comscore (see `COMSCORE.md`'s "Checkpoint / resume behaviour"):
`gower_cache.parquet` is the checkpoint, films scoring ≥ `match_thresh`
(0.80) are skipped on re-run, below-threshold films are retried, and a full
re-match requires deleting both `gower_cache.parquet` and
`gower_review_needed.parquet`.

---

## Running

```bash
# Full run (pull Gower from Snowflake, match all EVT films)
python rematch_gower.py                     # LIMIT_FILMS=None at top of file

# Sample run for testing
# Set LIMIT_FILMS = 200 at top of rematch_gower.py, then:
python rematch_gower.py

# Via Dagster (runs build_mapping — skips already-matched films)
# ./start_dagster_matching.sh, then Jobs → comscore_job → Launch Run
# (runs comscore_match, gower_match, id_bridge together — separate Dagster
# process from the LLM paths, see DAGSTER.md)
# or Assets → gower_match → Materialize selected
```

---

## Combining with Comscore

`id_bridge.py::build_id_bridge()` outer-joins `comscore_cache.parquet` and
`gower_cache.parquet` on `film_id` and writes
`DATA_DIR/id_bridge/film_id_bridge.parquet` with columns:

```
film_id, film, cs_id, cs_title, cs_match_confidence,
gower_id, gower_title, gower_match_confidence
```

It's a pure join — no fuzzy matching of its own — so a film with a
Comscore match but no Gower match (or vice versa) still gets a row, just
with nulls on the missing side. Wired into Dagster as the `id_bridge` asset,
which depends on both `comscore_match` and `gower_match` and runs as part of
`comscore_job`.
