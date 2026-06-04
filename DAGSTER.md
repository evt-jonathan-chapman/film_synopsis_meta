# Running the extractions in Dagster

How to run synopsis / cast / director / film_meta / comscore_match through `dagster_defs.py`. For the underlying extraction logic see `CLAUDE.md`.

---

## One-time setup

```bash
source .venv/bin/activate           # repo venv (Python 3.11, deps in requirements.txt)
export DAGSTER_HOME=~/dagster_home  # persistent home — otherwise Dagster uses a tmp dir per launch
mkdir -p "$DAGSTER_HOME"
```

`.env` in the repo root must contain `OPENAI_KEY=...` (and optionally `FILM_META_MODEL=gpt-5.4-mini`). `config.yaml` must have working Snowflake creds — `films_source` will fail fast with `RuntimeError: Snowflake unavailable` otherwise.

## Start the UI

```bash
dagster dev -f dagster_defs.py     # → http://127.0.0.1:3000
```

Leave it running. All commands below are launched from the UI unless noted.

---

## What's defined

**One upstream asset** — `films_source` — pulls the film list once (parquet snapshots + Snowflake `film_title` join, see `refresh.py::load_films_from_snowflake`).

**Five downstream assets**, each independently materialisable:

| Asset | Path | Nightly (new only) | Cold-start backfill |
|---|---|---|---|
| `synopsis` | nano text classifiers | ~$1–3, ~10–15 min | same |
| `cast` | actor profiles (mini + web_search) | <$10, minutes | **~$100, ~5–6 hr** |
| `directors` | director profiles (mini + web_search) | <$1, seconds | **~$50, ~1.5 hr** (code-printed; real bill lower) |
| `film_meta` | studios / billing / budget / IP (mini + web_search) | ~$100–150, ~3 hrs | same |
| `comscore_match` | EVT film_id → Comscore IBOE_TITLES (CPU fuzzy match, no API) | seconds (only new/borderline retried) | **~30–60 min**, $0 |

> Printed `Run total: $X` for web_search paths uses a hardcoded $0.025/search-call rate in `film_meta_extractor.py:43` that's now stale — real dashboard bill is meaningfully lower. Override via `WEB_SEARCH_COST_USD` env if you care about accuracy.

**Four jobs:**

| Job | Selection | Schedule |
|---|---|---|
| `nightly_job` | `films_source` + `synopsis` + `cast` + `directors` | 02:00 daily (`nightly_schedule`) |
| `film_meta_job` | `films_source` + `film_meta` | 03:00 Sundays (`film_meta_schedule`) |
| `comscore_job` | `comscore_match` | unscheduled — ad-hoc only |
| `full_refresh_job` | everything (`*`) | unscheduled — ad-hoc only |

`film_meta` is split off because it's the $100+/run path; everything else fits in a cheap nightly. `comscore_match` is split off because it doesn't share `films_source` (loads EVT films from parquet snapshots itself, plus `film_meta_enriched.parquet` for the concert-film filter).

---

## Running things from the UI

**To run all four paths (full refresh):**
Jobs → `full_refresh_job` → *Launch Run*.

**To run just the cheap nightly paths (synopsis + cast + directors):**
Jobs → `nightly_job` → *Launch Run*.

**To run just film_meta (the $100+ path):**
Jobs → `film_meta_job` → *Launch Run*.

**To run just comscore_match:**
Jobs → `comscore_job` → *Launch Run*, or Assets → `comscore_match` → *Materialize selected*. **Doesn't need `films_source`** — it loads EVT films from parquet snapshots itself. Reads `film_meta_enriched.parquet` for the concert-film filter; if that file doesn't exist yet, it'll raise `FileNotFoundError`.

**To run a single asset (e.g. just `cast`):**
Assets → click the asset → *Materialize selected*. **`films_source` must already be materialised** in this Dagster home — otherwise the downstream will fail with `FileNotFoundError: …/storage/films_source`. If it's missing, materialise `films_source` once first, or cmd-click both and materialise together.

**To re-run just the downstreams without re-pulling Snowflake:**
Assets → select the downstream assets → *Materialize selected*. Dagster reuses the existing `films_source` materialisation.

---

## Turning the schedules on/off

Schedules are **off by default** until enabled in the UI. Schedules → toggle `nightly_schedule` / `film_meta_schedule` on. Dagster's daemon must be running for schedules to fire — `dagster dev` runs both the webserver and the daemon, so as long as `dagster dev` is up, the schedules tick.

For production / headless, run the daemon separately: `dagster-daemon run` (with `DAGSTER_HOME` set).

---

## Bypassing Dagster (when you just want to run something)

The Dagster assets are thin wrappers around `refresh.py` (and `rematch_comscore.py` for comscore). To run the same logic from the command line:

```bash
python refresh.py                          # all four paths, diff-based
python refresh.py --only synopsis cast     # subset
python refresh.py --force-film-meta        # ignore diff for this path
python rematch_comscore.py                 # comscore match (uses LIMIT_FILMS at top of file for sampling)
```

This is the right choice for one-off / debugging runs — same checkpoints, same output parquets, no Dagster overhead.

`python main.py` is the interactive variant (edit the `CONFIG` block at the top to toggle `RUN_SYNOPSIS` / `RUN_CAST` / `RUN_DIRECTOR` / `RUN_META`).

---

## Checkpointing — Dagster runs are resumable

Each path writes a progress JSON under `~/Documents/data/<dir>/*_progress.json` after every batch. If a Dagster run dies mid-flight, the next run picks up where it left off — it does **not** re-extract films already in the checkpoint. To force a full re-extraction, delete the checkpoint **and** the output parquet for that path.

For `film_meta` specifically: there's also `film_meta_errors.json`. Failed films land there (not in the checkpoint), so they auto-retry on the next run. Use `python diagnostics/inspect_film_meta_progress.py` mid-run to see coverage and what's stuck on `_error: "ambiguous"` etc.

`comscore_match` uses `comscore_cache.parquet` as the checkpoint (no separate progress JSON). Already-matched films (score ≥ threshold) are skipped on re-run; below-threshold films are retried. **To force a full re-match, move/delete `comscore_cache.parquet` AND `comscore_review_needed.parquet` first** — otherwise the cache-skip logic preserves stale rows. Use `python diagnostics/inspect_comscore_unmatched.py` to bucketise unmatched films by best-candidate score.

---

## Troubleshooting

- **`RuntimeError: Snowflake unavailable`** — `config.yaml` creds, or VPN not connected.
- **Cascading 429s on `film_meta`** — concurrency is already tuned to 2 for the 200k TPM cap. Don't raise `META_MAX_CONCURRENCY` in `film_meta_extractor.py` without first raising the org's TPM tier.
- **Dagster forgets prior runs between sessions** — `DAGSTER_HOME` not set, so it's using a tmp dir (look for `.tmp_dagster_home_*` in the repo). Set `DAGSTER_HOME` permanently in your shell profile.
- **Schedule didn't fire overnight** — daemon wasn't running. `dagster dev` must be up, or run `dagster-daemon run` separately.
