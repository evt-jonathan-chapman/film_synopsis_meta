# Running the extractions in Dagster

How to run synopsis / cast / director / film_meta / comscore_match / gower_match / id_bridge through Dagster. For the underlying extraction logic see `CLAUDE.md`.

There are **two separate Dagster code locations**, run as two separate processes:

| Code location | File | Assets | Needs `litellm`/`openai`? |
|---|---|---|---|
| LLM extraction paths | `dagster_defs.py` | `films_source`, `synopsis`, `cast`, `directors`, `film_meta` | Yes |
| Comscore/Gower matching | `dagster_matching_defs.py` | `comscore_match`, `gower_match`, `id_bridge` | No — pure CPU fuzzy matching |

They're split because `dagster_defs.py` imports `refresh.py`, which imports `litellm` at module level — so just *loading* it (even to run matching-only) forces you to install `litellm`/`openai`, both large and (on flaky/corporate networks) prone to failing mid-download. `dagster_matching_defs.py` only imports `rematch_comscore.py` / `rematch_gower.py` / `id_bridge.py`, none of which touch `litellm`/`openai` — so it needs a much smaller dependency set (`requirements-matching.txt`). Run whichever process you actually need; run both if you want everything available.

---

## One-time setup

```bash
# Full setup — both code locations, same venv
uv venv --python 3.11
uv pip install --system-certs -r requirements.txt   # or without --system-certs if not on a corporate/proxied network
export DAGSTER_HOME=~/dagster_home  # persistent home — otherwise Dagster uses a tmp dir per launch
mkdir -p "$DAGSTER_HOME"
```

```bash
# Matching-only setup — skips litellm/openai entirely
uv venv --python 3.11
uv pip install --system-certs -r requirements-matching.txt
```

`.env` in the repo root must contain `OPENAI_KEY=...` (and optionally `FILM_META_MODEL=gpt-5.4-mini`) — only needed for the LLM code location. `config.yaml` must have working Snowflake creds either way — `films_source` (LLM side) and `comscore_match`/`gower_match` (matching side) will fail fast with `RuntimeError: Snowflake unavailable` / a Snowflake connection error otherwise.

## Start the UI(s)

```bash
./start_dagster.sh              # LLM paths            → http://127.0.0.1:3000
./start_dagster_matching.sh     # Comscore/Gower match  → http://127.0.0.1:3001
```

Both scripts handle venv activation, `DAGSTER_HOME`, and `.env` sourcing. Run one or both — they're independent processes on different ports and don't need to run together. (Equivalent manual commands: `dagster dev -f dagster_defs.py` and `dagster dev -f dagster_matching_defs.py -p 3001`.)

Leave whichever you need running. All commands below are launched from the relevant UI unless noted.

---

## What's defined

### `dagster_defs.py` (LLM paths)

**One upstream asset** — `films_source` — pulls the film list once (parquet snapshots + Snowflake `film_title` join, see `refresh.py::load_films_from_snowflake`).

**Four downstream assets**, each independently materialisable:

| Asset | Path | Nightly (new only) | Cold-start backfill |
|---|---|---|---|
| `synopsis` | nano text classifiers | ~$1–3, ~10–15 min | same |
| `cast` | actor profiles (mini + web_search) | <$10, minutes | **~$100, ~5–6 hr** |
| `directors` | director profiles (mini + web_search) | <$1, seconds | **~$50, ~1.5 hr** (code-printed; real bill lower) |
| `film_meta` | studios / billing / budget / IP (mini + web_search) | ~$100–150, ~3 hrs | same |
| `s3_sync` | uploads the four checkpoints (parquet + progress json) to S3 | seconds, $0 | same |

> Printed `Run total: $X` for web_search paths uses a hardcoded $0.025/search-call rate in `film_meta_extractor.py:43` that's now stale — real dashboard bill is meaningfully lower. Override via `WEB_SEARCH_COST_USD` env if you care about accuracy.

`s3_sync` declares `deps=[synopsis, cast, directors, film_meta]` (Dagster-tracked, same code location) but doesn't consume their in-memory outputs — it just re-reads the parquet/progress-json files each one already wrote to `DATA_DIR`, via `s3_sync.py::sync_meta_outputs_to_s3`. See that module's docstring and `CLAUDE.md` for the S3 target and auth (a Stax SSO profile whose credentials expire hourly — run `stax2aws login` before triggering this job, there's no automatic refresh).

**Four jobs:**

| Job | Selection | Schedule |
|---|---|---|
| `nightly_job` | `films_source` + `synopsis` + `cast` + `directors` | 02:00 daily (`nightly_schedule`) |
| `film_meta_job` | `films_source` + `film_meta` | 03:00 Sundays (`film_meta_schedule`) |
| `s3_sync_job` | `s3_sync` | unscheduled — ad-hoc only (needs a fresh manual AWS login first) |
| `full_refresh_job` | everything in this code location (`*`) | unscheduled — ad-hoc only |

`film_meta` is split off because it's the $100+/run path; everything else fits in a cheap nightly. `s3_sync` is split off from both because it needs its own auth step that the extraction jobs don't — trigger it manually once a week's run is done and your AWS session is fresh.

### `dagster_matching_defs.py` (Comscore/Gower matching)

**Three assets**, no shared upstream (each is self-contained):

| Asset | Path | Nightly (new only) | Cold-start backfill |
|---|---|---|---|
| `comscore_match` | EVT film_id → Comscore IBOE_TITLES (CPU fuzzy match, no API) | seconds (only new/borderline retried) | **~30–60 min**, $0 |
| `gower_match` | EVT film_id → Gower GW_LIFE_TIME (CPU fuzzy match, no API) | seconds (only new/borderline retried) | similar order to comscore_match, $0 |
| `id_bridge` | Outer-join `comscore_cache` + `gower_cache` on `film_id` (pure join, no API) | seconds | seconds |

**One job:**

| Job | Selection | Schedule |
|---|---|---|
| `comscore_job` | `comscore_match` + `gower_match` + `id_bridge` | unscheduled — ad-hoc only |

`comscore_match` and `gower_match` each load EVT films from parquet snapshots themselves via `rematch_comscore.py::load_evt_films` (shared by both), and both read `film_meta_enriched.parquet` directly off disk for the concert-film filter — **this file must already exist** (materialise `film_meta` at least once in the other code location, or run `python refresh.py --only film_meta`, before running this job for the first time on a fresh data dir). Because `film_meta` lives in the *other* code location, there's no Dagster-level `deps=[film_meta]` declaration on `comscore_match`/`gower_match` — the functional dependency still exists (you'll get `FileNotFoundError` if the parquet is missing), it just won't show as "stale" in this UI when `film_meta` re-runs elsewhere. `id_bridge` depends on both matchers (same code location, so this dependency *is* Dagster-tracked) and just joins their caches — see `COMSCORE.md` / `GOWER.md` for the matching algorithm, `GOWER.md`'s "Combining with Comscore" section for the join.

---

## Running things from the UI

**LLM paths (`./start_dagster.sh`, port 3000):**

- **All four paths (full refresh):** Jobs → `full_refresh_job` → *Launch Run*.
- **Cheap nightly paths only (synopsis + cast + directors):** Jobs → `nightly_job` → *Launch Run*.
- **film_meta only (the $100+ path):** Jobs → `film_meta_job` → *Launch Run*.
- **Sync the four meta checkpoints to S3:** run `stax2aws login` first, then Jobs → `s3_sync_job` → *Launch Run*.
- **A single asset (e.g. just `cast`):** Assets → click the asset → *Materialize selected*. **`films_source` must already be materialised** in this Dagster home — otherwise the downstream will fail with `FileNotFoundError: …/storage/films_source`. If it's missing, materialise `films_source` once first, or cmd-click both and materialise together.
- **Re-run just the downstreams without re-pulling Snowflake:** Assets → select the downstream assets → *Materialize selected*. Dagster reuses the existing `films_source` materialisation.

**Comscore/Gower matching (`./start_dagster_matching.sh`, port 3001):**

- **All three (comscore_match + gower_match + id_bridge):** Jobs → `comscore_job` → *Launch Run*.
- **Just one matcher:** Assets → click `comscore_match` or `gower_match` → *Materialize selected*.

---

## Turning the schedules on/off

Schedules are **off by default** until enabled in the UI (LLM code location only — matching has no schedules, it's ad-hoc). Schedules → toggle `nightly_schedule` / `film_meta_schedule` on. Dagster's daemon must be running for schedules to fire — `dagster dev` runs both the webserver and the daemon, so as long as `./start_dagster.sh` is up, the schedules tick.

For production / headless, run the daemon separately: `dagster-daemon run` (with `DAGSTER_HOME` set).

---

## Bypassing Dagster (when you just want to run something)

The Dagster assets are thin wrappers around `refresh.py` (LLM paths) and `rematch_comscore.py` / `rematch_gower.py` / `id_bridge.py` (matching paths). To run the same logic from the command line:

```bash
python refresh.py                          # all four LLM paths, diff-based
python refresh.py --only synopsis cast     # subset
python refresh.py --force-film-meta        # ignore diff for this path
python rematch_comscore.py                 # comscore match (uses LIMIT_FILMS at top of file for sampling)
python rematch_gower.py                    # gower match (same LIMIT_FILMS pattern)
python id_bridge.py                        # join comscore_cache + gower_cache on film_id
```

This is the right choice for one-off / debugging runs — same checkpoints, same output parquets, no Dagster overhead. It also sidesteps the two-code-location split entirely: `rematch_comscore.py`/`rematch_gower.py`/`id_bridge.py` never import `refresh.py`, so running them directly needs only `requirements-matching.txt`.

`python main.py` is the interactive variant of the LLM paths (edit the `CONFIG` block at the top to toggle `RUN_SYNOPSIS` / `RUN_CAST` / `RUN_DIRECTOR` / `RUN_META`).

---

## Checkpointing — Dagster runs are resumable

Each path writes a progress JSON under `~/Documents/data/<dir>/*_progress.json` after every batch. If a Dagster run dies mid-flight, the next run picks up where it left off — it does **not** re-extract films already in the checkpoint. To force a full re-extraction, delete the checkpoint **and** the output parquet for that path.

For `film_meta` specifically: there's also `film_meta_errors.json`. Failed films land there (not in the checkpoint), so they auto-retry on the next run. Use `python diagnostics/inspect_film_meta_progress.py` mid-run to see coverage and what's stuck on `_error: "ambiguous"` etc.

`comscore_match` uses `comscore_cache.parquet` as the checkpoint (no separate progress JSON); `gower_match` works identically with `gower_cache.parquet`. Already-matched films (score ≥ threshold) are skipped on re-run; below-threshold films are retried. **To force a full re-match, move/delete the cache AND the matching `*_review_needed.parquet` first** — otherwise the cache-skip logic preserves stale rows. Use `python diagnostics/inspect_comscore_unmatched.py` to bucketise unmatched films by best-candidate score (Comscore only — no Gower equivalent yet).

---

## Troubleshooting

- **`RuntimeError: Snowflake unavailable`** — `config.yaml` creds, or VPN not connected.
- **Cascading 429s on `film_meta`** — concurrency is already tuned to 2 for the 200k TPM cap. Don't raise `META_MAX_CONCURRENCY` in `film_meta_extractor.py` without first raising the org's TPM tier.
- **Dagster forgets prior runs between sessions** — `DAGSTER_HOME` not set, so it's using a tmp dir (look for `.tmp_dagster_home_*` in the repo). Set `DAGSTER_HOME` permanently in your shell profile.
- **Schedule didn't fire overnight** — daemon wasn't running. `./start_dagster.sh` must be up, or run `dagster-daemon run` separately.
- **`ModuleNotFoundError: litellm` when running `dagster_matching_defs.py`** — you're using the wrong requirements file (or a shared venv missing `requirements.txt`'s extras is fine — the matching code location never imports `litellm`, so this shouldn't happen; if it does, check nothing in `dagster_matching_defs.py`/`rematch_comscore.py`/`rematch_gower.py`/`id_bridge.py` accidentally started importing `refresh.py`).
- **`uv pip install` fails with `invalid peer certificate: UnknownIssuer`** — corporate/proxied network rejecting `uv`'s bundled CA list. Add `--system-certs` to the install command.
- **`uv pip install` fails mid-download on a large package (e.g. `litellm`) with a connection reset** — transient network flakiness, not a real failure. `uv cache clean <package>` then retry.