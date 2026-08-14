# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

LLM extraction of film metadata for the EVT box office model. **Raw extraction only** — encoding lives in the sibling repo `cinema_admits_models/build_data/`.

**Data root:** `~/Documents/data` (shared with `cinema_admits_models`) — used by `main.py`'s ad-hoc local path and by the Comscore/Gower/ID-bridge matchers. The four LLM extraction paths plus `film_data_merge.py`, run via `refresh.py`/Dagster, read and write **S3 directly** instead (see "Direct-to-S3 checkpointing" below) — there is no local copy of their output on a machine running `refresh.py`.

See `DAGSTER.md` for Dagster operational details. See `COMSCORE.md` for the Comscore matching deep-dive and `GOWER.md` for the (near-identical) Gower matching deep-dive.

---

## Eight extraction paths

| Path | Model | Cardinality | Output |
|---|---|---|---|
| **Synopsis** (LiteLLM batch) | `gpt-4.1-nano` (default) | per film | S3: `synopsis/synopses_extracted.parquet` |
| **Film meta** (Responses + `web_search`) | `gpt-5.4-mini` | per film | S3: `film_meta/film_meta_enriched.parquet` |
| **Actor** (Responses + `web_search`) | `gpt-5.4-mini` | per unique actor | S3: `cast_meta/cast_enriched.parquet` |
| **Director** (Responses + `web_search`) | `gpt-5.4-mini` | per unique director | S3: `director_meta/director_enriched.parquet` |
| **Comscore match** (rapidfuzz, no API) | — | per film | local: `comscore/comscore_cache.parquet` |
| **Gower match** (rapidfuzz, no API) | — | per film | local: `gower/gower_cache.parquet` |
| **ID bridge** (pure join, no API) | — | per film | local: `id_bridge/film_id_bridge.parquet` |
| **Merged film data** (pure join, no API) | — | per film | S3: `film_data_merged/film_data_merged.parquet` |

Synopsis uses `DEFAULT_MODEL` from `models.py` (currently `gpt-4.1-nano`, with `gpt-5.4-mini` as fallback). Nano = pure text classification (9 tasks read only title + synopsis). Mini + web_search = knowledge-grounded fields (budget, studios, fame_tier, director_tier, ip_strength, adaptation_type) where training memory is too brittle, especially on recent or upcoming films.

All LLM extractors are checkpoint-resumable. For `refresh.py`/Dagster runs this means an S3 write-ahead-log (see "Direct-to-S3 checkpointing" below); for `main.py`'s local ad-hoc path it's still local progress JSONs at `~/Documents/data/<dir>/*_progress.json` — delete those to force re-extraction there. The two checkpoint stores are independent (see the "refresh.py and main.py" divergence note further down).

### Direct-to-S3 checkpointing

`s3_checkpoint.py` gives `refresh.py`'s four extraction paths and `film_data_merge.py` checkpoint I/O with **no local disk involved at all** — needed because a Dagster run on Kubernetes may execute each asset in its own pod with no shared or persistent disk between them, so the old "write locally, sync to S3 as a separate step" pattern (`s3_sync.py`) can't work unattended in that environment.

The core problem it solves: S3 has no partial-write/append primitive, so a single mutable JSON checkpoint (the local pattern — `film_meta_progress.json` alone is 5.9MB for ~4,400 films) would mean re-uploading the *entire* file on every flush, and concurrent writers (film_meta runs at concurrency=2, synopsis at 8) would clobber each other's progress with last-write-wins. Instead:

```
s3://{bucket}/{prefix}/{name}/cache/snapshot.json       compacted state
s3://{bucket}/{prefix}/{name}/cache/deltas/{key}.json   one small object per flush batch
s3://{bucket}/{prefix}/{name}/cache/errors.json         small, rewritten wholesale
s3://{bucket}/{prefix}/{name}/{filename}.parquet        final output, rewritten wholesale
```

- `load_checkpoint(name)` — merges the snapshot with every delta (delta keys are timestamp-prefixed, so lexicographic order is also chronological order); a later delta wins on the same id.
- `append_checkpoint(name, new_items)` — writes ONE new small delta per flush batch (O(batch size), not O(corpus size)) — this is what each extraction path calls after every batch instead of rewriting the whole checkpoint.
- `compact_checkpoint(name)` — folds the snapshot + all current deltas into a fresh snapshot and deletes the deltas. Called once at the end of each `_enrich_cast`/`_enrich_directors`/`_enrich_film_meta` run so `load_checkpoint()`'s list+download cost stays bounded as runs accumulate. (Synopsis has no batches to compact — it flushes once at the end of a run.)
- `read_parquet(name, filename)` / `write_parquet(name, filename, df)` — the final output parquet, read/written directly via an in-memory buffer (no temp file).

Auth: `boto3.Session(profile_name=S3_PROFILE)` if `config.yaml`'s `s3.profile` is set (currently the Stax SSO profile, for local testing — run `stax2aws login` first), else boto3's default credential chain (picks up an IAM role/IRSA automatically in prod). Clear `s3.profile` in `config.yaml` for a production deployment.

`main.py`'s own checkpoint logic is untouched by any of this — it still reads/writes local JSON+parquet under `DATA_DIR`, independently of `refresh.py`'s S3 state.

### Comscore + Gower matching (fifth/sixth paths — no LLM, no API)

`comscore_matcher.py::ComscoreMatcher` maps EVT `film_id` → Comscore `title_global_id` so IBOE_TITLES + IBOE_FLASH_GROSS can join onto EVT films. Driver: `rematch_comscore.py`. Comscore data is pulled once via `sql/comscore_extract.sql`.

`gower_matcher.py::GowerMatcher` does the same for Gower's `GW_LIFE_TIME` box-office estimates, mapping EVT `film_id` → `gower_id` (`prmry_title_no` on the Gower side). Driver: `rematch_gower.py`. Gower data is pulled once via `sql/gower_export.sql`. Both matchers subclass the shared `title_matcher.py::FuzzyTitleMatcher` engine and run **independently** against the EVT catalogue (Gower doesn't chain through Comscore) — see `GOWER.md` for exactly what differs (single title column, `prmry_title_no` in place of `title_global_id`, no alt-content flag, multi-snapshot dedup).

`id_bridge.py::build_id_bridge()` outer-joins `comscore_cache.parquet` + `gower_cache.parquet` on `film_id` into `id_bridge/film_id_bridge.parquet` (`film_id`, `cs_id`, `gower_id`, + each side's matched title/confidence) — the one file to read when you just need the crosswalk rather than either matcher's full detail.

Matching: NFKD-normalised titles scored with `max(ratio, token_sort_ratio)` across five Comscore title columns (`film_name`/`upper_name`/`title_aka`/`us_title_name`/`short_name`), variant/festival prefix stripping (`GC`/`3D`/`IMAX`/`TFF -`…), a length-ratio guard (rejects short-title impostors like AVATAR→TÁR), a ±1-year release window, and a date-proximity tie-breaker. Each film lands in a confidence tier: **high** (≥0.92 and ≤365 days off), **borderline**, **unmatched**, or **manual**.

Outputs under `~/Documents/data/comscore/`:
- `comscore_cache.parquet` — keyed on EVT `film_id` (this is the checkpoint — no separate progress JSON)
- `comscore_review_needed.parquet` — borderline+unmatched rows with top-5 candidates, for human triage
- `comscore_manual_overrides.parquet` — **user-authored**: fill `manual_override_cs_id` in the review file, save under this name, and overrides win on the next run (forced to `confidence=manual`, `score=1.0`)

`ComscoreMatcher.re_score()` re-evaluates the whole cache against an updated extract/algorithm without re-pulling (manual overrides preserved). Concert films are dropped on both sides (`is_alt_content` in Comscore; `adaptation_type == "concert_film"` from `film_meta_enriched.parquet` on the EVT side).

### Merged film data (eighth path — no LLM, no API)

`film_data_merge.py::build_merged_film_data()` outer-joins `synopses_extracted.parquet` + `film_meta_enriched.parquet` on `film_id` into one comprehensive per-film parquet, so consumers get a single usable film metadata source instead of merging the two raw outputs themselves — both source parquets (read via `s3_checkpoint.py`, directly from S3) are untouched and still there if needed. Stateless: re-derived fresh from the two source parquets on every run, no checkpoint of its own (same convention as `id_bridge.py`, just via S3 instead of local paths). Outer join, not left, because a handful of films only have a film_meta row (no matching synopsis row) — a left join off synopsis would silently drop them.

Column collisions (`title`, `genres`, `ip_strength`, `adaptation_type` exist on both sides) resolve as:
- **`genres`** — mirrors the sibling repo `cinema_admits_models/build_data/encode_synopsis.py`'s `_merge_bio_doc`: wherever film_meta has a row, `biography`/`documentary` tags are stripped from synopsis's genre list and replaced **wholesale** with film_meta's (not additive — if film_meta tags neither, the film ends up with neither). Every other genre tag comes from synopsis untouched; film_meta's other genres (e.g. `Drama`, `Musical` on a biopic) are never pulled in. This fixes the nano synopsis-LLM's well-known biography/documentary confusion (it only reads title+synopsis text, no outside knowledge) using film_meta's knowledge-grounded tag instead. For a film_meta-only row (no synopsis genre baseline to carve from), the merge falls back to film_meta's full genre list rather than just its biography/documentary tags.
- **`title` / `ip_strength` / `adaptation_type`** — film_meta's value wins when present, falling back to synopsis's (these two fields were also migrated from synopsis to film_meta — see prompts_v2.yaml's `MIGRATED` notes above).

Two boolean flags — `has_film_meta` / `has_synopsis` — mark which side(s) actually matched, so partial coverage is visible rather than silently backfilled.

---

## Running

```bash
# Setup (one-time) — full (LLM + matching); use --system-certs if on a proxied/corporate network
uv venv --python 3.11 && source .venv/bin/activate
uv pip install -r requirements.txt
# Matching-only setup (skips litellm/openai entirely — see "Two Dagster code locations" below):
#   uv pip install -r requirements-matching.txt

# Ad-hoc / interactive
python main.py                    # edit CONFIG block (RUN_SYNOPSIS / RUN_CAST / RUN_DIRECTOR / RUN_META)

# Diff-based CLI (the four LLM paths) — needs a valid AWS session (see "Direct-to-S3
# checkpointing" above): run `stax2aws login -i stax-au1 -o event` first if config.yaml's
# s3.profile is still set to the Stax profile.
python refresh.py                              # all four paths
python refresh.py --only synopsis cast         # subset (choices: synopsis cast director film_meta)
python refresh.py --force-synopsis             # ignore diff for synopsis
python refresh.py --force-cast
python refresh.py --force-director
python refresh.py --force-film-meta

# Comscore + Gower matching (CPU only, no API)
python rematch_comscore.py                     # set LIMIT_FILMS at top of file (default None = full run)
python rematch_gower.py                        # same LIMIT_FILMS pattern
python id_bridge.py                            # join comscore_cache + gower_cache on film_id
python film_data_merge.py                      # join synopsis + film_meta into one comprehensive parquet
                                                # (reads/writes S3 directly — needs a valid AWS session, same as refresh.py)

# Retroactive film_meta cleanup (genre normalization + format-variant dedup) — runs
# automatically after every refresh.py/main.py extraction flush; this CLI is for
# re-running standalone (e.g. after tuning thresholds, or on rows extracted before
# this was wired in)
python cleanup_film_meta.py

# s3_sync.py is now main.py-only (see "S3 sync" below) — refresh.py and film_data_merge.py
# read/write S3 directly and need no separate sync step. Only run this after an ad-hoc
# main.py run, if you want to push its local output to S3 manually:
./run_s3_sync.sh                               # refreshes the Stax session (opens browser SSO login,
                                                # blocks until you complete it) then runs s3_sync.py
# ...or the two steps separately:
stax2aws login -i stax-au1 -o event            # refresh AWS session first (expires hourly)
python s3_sync.py

# Dagster (local) — two separate processes, see "Two Dagster code locations" below.
# Preferred: these scripts handle venv + DAGSTER_HOME + .env sourcing.
./start_dagster.sh                             # LLM paths            → http://localhost:3000
./start_dagster_matching.sh                    # Comscore/Gower match → http://localhost:3001
# ...or manually:
export DAGSTER_HOME=~/dagster_home && mkdir -p "$DAGSTER_HOME"
dagster dev -f dagster_defs.py                          # UI at http://localhost:3000
dagster dev -f dagster_matching_defs.py -p 3001         # UI at http://localhost:3001

# Diagnostics / validation (in diagnostics/ subfolder):
python diagnostics/test_film_meta.py            # single-film smoke test
python diagnostics/compare_film_meta_search.py  # A/B mini+search vs mini-no-search
python diagnostics/refresh_comparison.py        # full sample run vs production parquets
python diagnostics/inspect_film_meta.py [--detail|--vs-tmdb|--film-id N]
python diagnostics/inspect_film_meta_progress.py  # live coverage + error breakdown mid-run
python diagnostics/print_compare.py [--disagree-only]
python diagnostics/inspect_comscore_unmatched.py  # bucketise unmatched/borderline by candidate_1_score
python diagnostics/post_refresh_check.py          # consolidated post-run health check across all four
                                                   # outputs — row counts, checkpoint/parquet agreement,
                                                   # genre vocab sanity, film_id_variants.parquet dtypes
```

Required env (`.env`): `OPENAI_KEY=...`

Optional env:
| Variable | Default | Notes |
|---|---|---|
| `FILM_META_MODEL` | `gpt-5.4-mini` | Override model for all Responses extractors |
| `WEB_SEARCH_COST_USD` | `0.025` | Per-call web_search fee — verify against current OpenAI pricing; this default may be stale |
| `WEB_SEARCH_ALLOWED_DOMAINS` | (none) | Comma-separated allowlist; narrowing to wiki+imdb hurts budget coverage |
| `DAGSTER_HOME` | tmp dir | Set persistently in shell profile; without it Dagster forgets prior runs between sessions |

### Two Dagster code locations

`dagster_defs.py` and `dagster_matching_defs.py` are **separate code locations, run as separate processes** — they're not combined into one `Definitions` object. The reason is dependency weight: `dagster_defs.py` imports `refresh.py`, which imports `litellm` at module level, so merely *loading* it (even just to run matching) forces installing `litellm`/`openai` — both large and prone to failing mid-download on flaky/corporate networks. `dagster_matching_defs.py` only imports `rematch_comscore.py`/`rematch_gower.py`/`id_bridge.py`, none of which touch `litellm`/`openai`, so it needs only `requirements-matching.txt`. Run `./start_dagster.sh` and/or `./start_dagster_matching.sh` (see `DAGSTER.md`).

**`dagster_defs.py`** exposes one upstream `films_source` asset (single Snowflake pull), four downstream extraction assets — `synopsis`, `cast`, `directors`, `film_meta` — each callable independently, and a `merged_film_data` asset (`deps=[synopsis, film_meta]` — a real Dagster-tracked dependency, both being in this code location). There is no `s3_sync` asset — all five read/write S3 directly (see "Direct-to-S3 checkpointing" above), so there's nothing left to sync.

Jobs:
- `nightly_job` — synopsis + cast + directors, scheduled 02:00 daily
- `film_meta_job` — film_meta only, scheduled 03:00 Sundays (separated because it's the $100+/run path)
- `merged_film_data_job` — merged_film_data only, ad-hoc (run after a film_meta_job materialisation, so the merge picks up the latest film_meta rows)
- `full_refresh_job` — everything in this code location, ad-hoc

**`dagster_matching_defs.py`** exposes three self-contained assets — `comscore_match`, `gower_match`, `id_bridge` — with one job:
- `comscore_job` — comscore_match + gower_match + id_bridge, ad-hoc

`comscore_match` and `gower_match` do **not** consume `films_source` — each loads EVT films from parquet snapshots itself (via the shared `rematch_comscore.py::load_evt_films`), plus reads `film_meta_enriched.parquet` for the concert-film filter. Since `film_meta` lives in the *other* code location, neither declares a Dagster-level `deps=[film_meta]` — the functional dependency is still real, it just isn't tracked as a Dagster staleness link. `film_meta_enriched.parquet` now lives on S3 (`refresh.py` writes it directly via `s3_checkpoint.py`, no local copy) — `load_evt_films` reads it from there first via `s3_checkpoint.read_parquet`, falling back to the local `FILM_META_ENRICHED_PATH` copy if S3 has nothing (e.g. a `main.py`-driven local run). This is why `requirements-matching.txt` now also needs `boto3`, even though this code location has no other AWS usage. `id_bridge` declares `deps=[comscore_match, gower_match]` (same code location, so this one *is* Dagster-tracked) and just outer-joins their two caches on `film_id` — no matching logic of its own.

Schedules are **off by default** — toggle on in the UI. `dagster dev` runs both the webserver and the daemon; for headless, run `dagster-daemon run` separately (with `DAGSTER_HOME` set).

### S3 sync (main.py's local path only)

`s3_sync.py::sync_meta_outputs_to_s3()` is **no longer used by `refresh.py`, Dagster, or `film_data_merge.py`** — those read/write S3 directly via `s3_checkpoint.py` (see "Direct-to-S3 checkpointing" above), so there's no local checkpoint left to sync for them. This module still exists for `main.py`'s local-disk ad-hoc workflow: it uploads the four LLM meta checkpoints — parquet + progress json for synopsis, film_meta, cast_meta, director_meta — plus `film_data_merged.parquet`, to `s3://<s3.bucket>/<s3.prefix>/...` (config in `config.yaml`'s `s3:` block; `synopsis_v2` locally is renamed to `synopsis` on the S3 side).

**Caution:** the parquet files this uploads land at the *same* S3 keys `refresh.py`'s direct writes use — running `s3_sync.py` after a `main.py` run can overwrite the Dagster path's S3 output with `main.py`'s local (and possibly differently-scoped — see the "refresh.py and main.py" work-set divergence note below) version. The `*_progress.json` files it uploads are `main.py`'s local checkpoint format; they are **not** read by `s3_checkpoint.py`'s cache (that lives under `<name>/cache/`), so uploading them is just a backup/inspection artifact, not a live checkpoint for anything.

Auth is a Stax SSO profile (`stax-stax-au1-event` in `~/.aws/credentials`, generated by `stax2aws login`) whose session credentials **expire after 1 hour** (`~/stax2aws.yaml`'s `session-duration: 3600`). There's no automatic refresh — run `stax2aws login -i stax-au1 -o event` manually before running `s3_sync.py`, or it fails with a clear "no valid AWS credentials" error. `s3_checkpoint.py` (the path that actually matters in production) sidesteps this entirely by falling back to boto3's default credential chain (IAM role/IRSA) once `s3.profile` is cleared in `config.yaml` — see its module docstring.

**`run_s3_sync.sh`** chains the login and the sync: it calls `stax_login.sh` (runs `stax2aws login`, scrapes the SSO "Full URL:" out of its output, opens it in the browser, then blocks on `wait` until the login completes) and only then runs `python s3_sync.py`. Use this instead of the two manual steps above when running ad-hoc from a terminal.

---

## Prompts

YAML per file, one task per top-level key. Loaded via `load_prompts.py::load_tasks_from_yaml`.

| File | Used by | Tasks |
|---|---|---|
| `prompts/prompts_v2.yaml` | synopsis nano batch | 9 enabled text-only classifiers (`is_franchise`, `language_cues`, `is_sequel`, `intellectual_property`, `genres`, `protagonist_archetype`, `primary_audience`, `tone`, `narrative_scope`). `ip_strength` + `adaptation_type` are disabled with `MIGRATED` notes — they moved into film_meta. |
| `prompts/film_meta_prompts.yaml` | film_meta mini+search | `film_meta` — full film schema including ip_strength + adaptation_type |
| `prompts/cast_prompts.yaml` | actor mini+search | `actor_profile` |
| `prompts/director_prompts.yaml` | director mini+search | `director_profile` |
| `prompts/prompts.yaml` | not used | legacy v1 |

### Design rules

- **Nano if** the prompt forbids external knowledge (look for *"Base all decisions ONLY on the synopsis"*). Cheap, deterministic.
- **Mini + web_search if** correctness needs current facts (budget, billing order, who's a-star this year, which novel is hot).
- Always include an `"unknown"` enum value for categorical fields that can legitimately fail.
- Lists default to `[]` (never `null`) with explicit caps (`0-3 items`, `top 5 only`).
- Scalars default to `null` when unknown.
- Every prompt ends with *"Output ONLY valid JSON. No explanations. No backticks."*

---

## Extractor classes

- **`extractor.py::LlmJsonExtractor`** — LiteLLM (Chat Completions). Used by the synopsis nano batch only. Supports task batching per film and llama_cpp local inference. `fallbacks` deliberately excluded from async path because LiteLLM's `fallback_utils` intercepts 429s incorrectly.
- **`film_meta_extractor.py::ResponsesExtractor`** — OpenAI Responses API directly. Required for `web_search`, which isn't on Chat Completions. Three subclasses: `FilmMetaExtractor` (per film), `ActorMetaExtractor` (per actor), `DirectorMetaExtractor` (per director). All share `_call_api` + JSON parse + token accounting.

Concurrency: nano at 8 (`MAX_CONCURRENCY`), **mini+search at 2** (`META_MAX_CONCURRENCY`). The bound is **TPM, not RPM** — org limit is 200k TPM and each web_search call burns ~15k tokens, so sustainable throughput ≈ 13 req/min. With `AsyncOpenAI(max_retries=8)` the SDK absorbs transient 429s; with concurrency=2 the rolling window stays ~17 req/min and retries claw back the rest. Bumping concurrency above 3 reliably triggers cascading rate limits.

---

## Key modules

- **`config.py`** — single source of truth for all paths and constants. Reads `config.yaml` (Snowflake creds, data root) and calls `.expanduser()` on every path field, so `config.yaml` can use `~`-relative paths (`main_dir: "~/Documents/data"`) instead of hardcoding a specific machine's username. Import paths from here; never hardcode `~/Documents/data`.
- **`models.py`** — model registry for LiteLLM paths. `DEFAULT_MODEL` is the synopsis nano model; `DEFAULT_FALLBACKS` is the fallback chain. To switch the synopsis model, change `DEFAULT_MODEL` here.
- **`extraction.py::ExtractionTask`** — the core data structure that connects prompts to extractors. Each task holds the prompt text, JSON field schema, and output key. `load_prompts.py::load_tasks_from_yaml` converts YAML entries into `ExtractionTask` objects; both `LlmJsonExtractor` and `ResponsesExtractor` consume them. Also contains shared JSON parse + repair utilities used by all extractors.
- **`refresh.py`** — diff-based orchestrator for all four LLM paths. The Dagster assets are thin wrappers around the four `refresh_*` functions here. `load_films_from_snowflake` is the canonical work-set loader — pulls the full catalogue directly from `films/sql.py::SQL_FILM_DETAILS` in one live query, then restricts it to `WORK_SET_MIN_REL_DATE` (module constant, currently `2018-01-01`) — see "Live Snowflake pull is much bigger than the old snapshot-glob approach" below for why that filter exists and isn't a guess. Returns `None` if Snowflake is unreachable — there's no local fallback. `main.py`'s own loader still globs the local `raw_from_snowflake`/`prediction_from_snowflake` parquet snapshots (see the "must agree on the work-set" note below — this is now a real divergence between the two, not just a risk). All checkpoint/parquet I/O in this module goes through `s3_checkpoint.py` (direct to S3, no local disk) — the one exception is `_sync_synopsis_checkpoint`, a deliberate local-only compatibility shim for `main.py` (see its docstring, and its whole body is wrapped in try/except — a plain Python `list` value, e.g. an empty `[]` from a list-valued field, must be checked before `pd.isna()` in its `_to_json_native` helper, since `pd.isna()` on a list-like returns an array, not a bool, and raises "truth value of an array is ambiguous" in an `if`; this crashed a real Dagster run on 2026-08-13 — fixed, but any *other* unexpected error in this best-effort shim is now caught too rather than crashing the real S3-based path behind it). `SYNOPSIS_CHECKPOINT_PATH`/`CAST_CHECKPOINT_PATH`/etc. constants are still defined here for that shim and for `diagnostics/post_refresh_check.py`'s imports, but are otherwise unused by this module's own logic now. `_apply_variant_merge`'s call to `_persist_variant_map` is also wrapped in try/except — filtering already succeeded and `df_filtered` is fine to use even if persisting the crosswalk to S3 fails transiently; the next successful run just re-detects and re-persists the same map. Variant merge (`film_variant_merge.py::filter_variants`, see below) is applied on both the film_meta path (`_diff_film_meta`/`_apply_variant_merge`) and the synopsis path (`_diff_synopsis_films`, added 2026-08-13 — see `_apply_variant_merge`'s `label` param) — cast and director diffing still never call it, but that's fine there since actor/director sets dedupe by name regardless of how many duplicate-booking film_ids reference them. `_enrich_film_meta`/`_enrich_cast`/`_enrich_directors`'s per-batch checkpoint save (`s3_checkpoint.append_checkpoint`) is wrapped in try/except (added 2026-08-13) — on failure (expired Stax token, network blip) it logs and stops starting new batches immediately rather than burning further OpenAI spend on results it can't persist, but still proceeds to the final flush with everything already extracted (the in-memory `checkpoint` dict already has that data regardless of whether its own delta-append succeeded).
- **`s3_checkpoint.py`** — direct-to-S3 checkpoint I/O for `refresh.py`'s four paths and `film_data_merge.py`. See "Direct-to-S3 checkpointing" above for the write-ahead-log + compaction design and why a single mutable JSON checkpoint doesn't work on S3.
- **`films/main.py::get_films_sources(persisted=True)`** — alternative film loader used by the `films_source` Dagster asset. Tries `films/source_data/films.parquet` first; falls back to a full Snowflake pull via `base_snowflake.py::SnowFlakeBase` (same key-pair auth as `refresh.py::load_films_from_snowflake`). Distinct from `refresh.py::load_films_from_snowflake` — both must stay in sync with each other.
- **`films/sql.py`** — Snowflake SQL queries. `SQL_FILM_DETAILS` is the main join used by `load_films_from_snowflake` to fetch authoritative titles from `EDW_ENT_PRD.CURATED.DIM_VH_FILM`. The `SYNOPSIS`/`ALT_SYNOPSIS` columns are explicitly cast to `VARCHAR(16777216)` — without it, Snowflake infers a narrower width from `DIM_VH_FILM.FILM_DESC`'s declared column width and truncation-errors on longer synopses from the forecast table.
- **`base_snowflake.py::SnowFlakeBase`** — minimal vendored Snowflake helper. Hard-coded to EVT Snowflake account (`mm31132.ap-southeast-2`); uses key-pair auth (`SF_RSA_KEY` from `config.yaml`) with a key path passed directly. The one and only Snowflake auth path in this repo — used by both `refresh.py::load_films_from_snowflake` and `films/main.py::get_films_sources`.
- **`title_cleaner.py`** — strips variant prefixes (`3D`, `IMAX`, `GC`) from titles before LLM prompt construction. Used by both `LlmJsonExtractor` and `FilmMetaExtractor`.
- **`title_matcher.py::FuzzyTitleMatcher`** — shared fuzzy title-matching engine behind both `comscore_matcher.py::ComscoreMatcher` and `gower_matcher.py::GowerMatcher`: title normalisation/scoring, variant propagation, and cache/review/manual-override I/O. Subclasses set column names (`ID_COL`/`DATE_COL`/`TITLE_COLS` on the source side, `ID_FIELD`/`TITLE_FIELD`/`DATE_FIELD`/`MATCHED_TITLE_FIELD` on the cache side) and output paths; everything else is identical between sources. See `COMSCORE.md` / `GOWER.md` for the algorithm.
- **`comscore_matcher.py::ComscoreMatcher`** / **`rematch_comscore.py`** — Comscore's column config on `FuzzyTitleMatcher` + driver script. `rematch_comscore.py::load_evt_films` is the canonical EVT work-set loader for *both* matching paths — `rematch_gower.py` imports it directly so Comscore and Gower always score against the same film catalogue.
- **`gower_matcher.py::GowerMatcher`** / **`rematch_gower.py`** — Gower's column config on `FuzzyTitleMatcher` + driver script. See `GOWER.md` for what's different from Comscore (single title column, `prmry_title_no` as ID, no alt-content flag, multi-snapshot dedup).
- **`id_bridge.py::build_id_bridge()`** — outer-joins `comscore_cache.parquet` + `gower_cache.parquet` on `film_id` into `id_bridge/film_id_bridge.parquet`. Pure join, no matching logic.
- **`film_data_merge.py::build_merged_film_data()`** — outer-joins `synopses_extracted.parquet` + `film_meta_enriched.parquet` on `film_id` into `film_data_merged/film_data_merged.parquet`, applying the biography/documentary genre carve-out (see "Merged film data" above). Pure join, no checkpoint of its own — re-derived fresh every run, same convention as `id_bridge.py`, reading/writing via `s3_checkpoint.py` instead of local paths. Renames each side's overlapping columns (`title`/`genres`/`ip_strength`/`adaptation_type`) to explicit `_syn`/`_fm` suffixes *before* merging rather than relying on pandas' automatic collision suffixing — that only fires when a column exists on both sides, so if either parquet is ever missing one (older schema, empty bootstrap run) auto-suffixing silently skips it and a later lookup KeyErrors instead of just treating that side as absent.
- **`s3_sync.py::sync_meta_outputs_to_s3()`** — **main.py's local path only** now (see "S3 sync" above) — uploads the four meta checkpoints (parquet + progress json) plus `film_data_merged.parquet` to S3. `SYNC_SPECS` is the explicit (local dir, S3 folder, filenames) list — deliberately not a glob, so stray `.bak`/errors files in `DATA_DIR/film_meta/` etc. never get swept up.
- **`dagster_defs.py`** / **`dagster_matching_defs.py`** — two separate Dagster code locations (see "Two Dagster code locations" above). The matching one deliberately never imports `refresh.py`, so it needs only `requirements-matching.txt`.
- **`post_process.py`** — postprocessor registry (`POSTPROCESSORS` dict) that maps task names to cleanup functions. Currently only `clean_names` is live; hooked in `ExtractionTask.postprocess` if set.
- **`film_variant_merge.py::filter_variants()`** — pre-extraction dedup, called from both `refresh.py::_apply_variant_merge` and `main.py`. ~20% of raw film_meta rows are duplicate titles because Vista assigns a separate `film_id` to every format (3D/IMAX), event (special screening, Q&A), or festival-programme booking of the same theatrical release. `filter_variants()` groups these by `(base_title, rel_at, dstbtr)` (stripping format/festival qualifiers via the vendored `encode_helper.py::strip_format_variant`). Genuine re-releases (keyword/year-in-title, no specific pairing) are dropped outright via `vendored/cinema_admits_models/re_release_filter.py::ReReleaseFilter`. Deliberately NOT covered: language-version variants (Hindi/Tamil/Telugu dubs etc.) — these are distinct theatrical products, and some titles (e.g. 777 Charlie) only ever exist as the language-suffixed row, so merging would erase the only extraction that film gets (**known bug as of 2026-08-13**: `cleanup_film_meta.py`'s post-extraction backstop, `find_format_variant_groups`, matches on the LLM's *own* output title rather than the raw Vista title — if the LLM's title-cleaning step over-normalizes and drops a language qualifier, two genuinely distinct language versions collapse onto the same key and get wrongly merged as `confirmed_by='format_variant'`; found affecting at least 58 synopsis + 6 film_meta rows, e.g. `Padmaavat - Hindi` merged into `Padmaavat - Telugu`, several Tamil/Malayalam/Japanese dub titles. Not yet fixed — investigation paused, see the CLAUDE.md changelog note below). Fuzzy-matched reschedule merging (linking a rescheduled booking to its earlier one) is also disabled — it produced repeated franchise false-positives (The Suicide Squad merged onto Suicide Squad on shared cast, Mad Max onto Mad Max 2 on shared director+lead) — see the module docstring for the full root-cause analysis.

  **Extraction source vs. output identity (added 2026-08-13):** the row that survives filtering (and so is what's actually sent to the LLM) is whichever group member has the richest Vista data (`_richness` — director/cast/synopsis populated) — a 3D/GC booking can win here over the general release. But the film_id that survivor is *stored* under is always the group's general-release film_id when one exists (`_format_variant_pairs`' `general_release_ids` — title needs no stripping at all), regardless of which side won the richness contest; only falls back to the richness winner when no general release exists in the group at all. Real cases found: Alita: Battle Angel, Aquaman, Lightyear, Jurassic World: Fallen Kingdom, Spider-Man: Brand New Day, The LEGO Movie 2, The Lion King all had their extracted data sitting under the 3D booking's film_id before this fix. Mechanics: `richness_winner` (extraction source) vs. `output_id` (general release) are tracked separately in `filter_variants()`; when they differ, `relabel_map` renames the surviving row's `film_id` after the initial drop-filtering, and every other group member (including `richness_winner` itself, if it isn't also `output_id`) gets a `variant_map` row pointing at `output_id`.

  **The `^GC\s+` regex bug (fixed 2026-08-13, upstream in `cinema_admits_models/encode_helper.py`, re-vendored):** only stripped plain `"GC Title"` bookings, not the equally-common `"GC - Title"` / `"GC -Title"` forms, leaving a dangling `"- "` that never matched the base title — so these groups were never even detected as variants at all (not a canonical-selection problem like the above, a *detection* problem). Fixed to `^GC(?:\s*[-–—]\s*|\s+)` (two alternatives — whitespace-then-optional-dash, or a mandatory dash — so `"GCTITLE"` with zero separator is never stripped, no risk of matching a hypothetical real title starting with those letters). Also added a `BTQ` prefix (previously unrecognized entirely — one title, Spider-Man: Brand New Day). Found via a full-catalogue punctuation-mismatch scan: 62 → 7 pairwise mismatches after the fix; the remaining 7 are genuine internal-title punctuation inconsistencies (colon vs. hyphen, e.g. Spider-Man: Brand New Day's 4 booking variants split across two groups by "- BRAND NEW DAY" vs ": BRAND NEW DAY") unrelated to prefix stripping and deliberately not auto-fixed (blanket punctuation normalization risks merging genuinely distinct titles). This fix surfaced 151 groups with real duplicate extracted rows across synopsis/film_meta that had never been recognized as variants before (Deadpool and Wolverine, The Marvels, Indiana Jones and the Dial of Destiny, Avatar: The Way of Water, Guardians of the Galaxy Vol. 3, and ~145 others) — one-time remediated: `synopses_extracted.parquet` 7,231→7,065 rows (166 dropped, 5 renamed), `film_meta_enriched.parquet` 4,709→4,703 (6 dropped, 11 renamed), `film_id_variants.parquet` rebuilt to 686 rows covering all 504 detected groups (not just the ones with duplicates, so future lookups resolve correctly even for groups not yet extracted).
- **`cleanup_film_meta.py::clean_film_meta_df()`** — post-extraction cleanup for `film_meta_enriched.parquet`, called automatically after every flush in `refresh.py::_enrich_film_meta` and `main.py::enrich_film_meta`; the CLI (`python cleanup_film_meta.py`) reruns the same logic standalone. Does genre normalization (splits mashed compound tokens, merges spelling/casing synonyms), genre consolidation (subgenre → parent, e.g. Folk Horror → Horror), a rare-genre filter (drops tags seen on fewer than `MIN_GENRE_FILM_COUNT` films — catches LLM idiosyncrasies and leaked language names), and a second, backstop format-variant dedup (`find_format_variant_groups`, keyed on `(title, evt_rel_at, evt_dstbtr)` post-LLM-cleaning — mainly catches variants that only became identical after the LLM normalized the 3D/IMAX prefix out of `title`).
- **The two variant-merge paths no longer write to the same file** — `refresh.py::_persist_variant_map` now writes `film_id_variants.parquet` to S3 (via `s3_checkpoint.py`, `film_meta/film_id_variants.parquet`); `cleanup_film_meta.py::persist_variant_map` (used by its standalone local CLI) still writes the local `FILM_ID_VARIANTS_PATH` copy (`config.py`, `<film_meta_dir>/film_id_variants.parquet`). Before the S3 migration these were the same file (`keep='last'` on `drop_duplicates(subset='film_id')` so a re-run with richer data could flip which side is canonical) — that guarantee no longer holds across the two paths, only within each independently. Both writers still force `variant_rel_at`/`canonical_rel_at` to strings before writing — mixing a raw `Timestamp` with a string in the same parquet column breaks pyarrow on write (this exact bug broke a film_meta run once); `diagnostics/post_refresh_check.py::check_variant_map` exists to catch a dtype regression here, but only checks the local copy (see the diagnostics-scripts gap noted above).
- **`tmdb_fetch.py`** — fetches production company data from the TMDB API and maps companies to studio tiers. Used only by the `--vs-tmdb` flag in `diagnostics/inspect_film_meta.py`; not part of any extraction path.
- **`cast_main.py`** — legacy standalone cast enrichment script using `LlmJsonExtractor` (LiteLLM, no web_search). Predates `refresh.py`. Use `refresh.py` or Dagster instead; this file is kept for reference only.
- **`encode/`** — sklearn-based feature encoding transformers (`TopNTokenMapper`, `DynamicTopNAndPCA`, `EmbeddingPCA`, etc.). **Not part of the extraction pipeline** — this repo is raw-extraction only. These utilities are kept here for reference but encoding runs in `cinema_admits_models/build_data/`.

---

## Cost / volume estimate (full nightly run, ~4,100 films)

| Path | Wall-clock | Cost |
|---|---|---|
| Synopsis nano | ~10-15 min | ~$1-3 |
| Film meta mini+search | ~3 hrs | ~$100-150 |
| Actor mini+search (new only) | minutes | <$10 |
| Director mini+search (new only) | seconds | <$1 |
| Comscore match | ~30-60 min cold-start | $0 |
| Gower match | similar order to Comscore, cold-start | $0 |
| ID bridge | seconds | $0 |

The printed "Run total: $X" for web_search paths uses `WEB_SEARCH_COST_USD` which may be stale — verify against the actual dashboard bill.

---

## Non-obvious behaviours

- **Re-run dedup keeps `first`** (synopsis) or **`last`** (cast/director/film_meta) — see each `_enrich_*`'s comment for why (`last` lets a freshly retried unknown-tier actor/director, or a re-extracted film, override a stale row). For `main.py`'s local path, delete the local checkpoint + parquet to fully re-extract. For `refresh.py`'s S3 path there's no single file to delete — see the next bullet.
- **Forcing a full S3 re-extract** — delete every object under `<name>/` in S3 (both `<name>/cache/` — snapshot + deltas + errors — and `<name>/<filename>.parquet`) via the AWS console/CLI, since `s3_checkpoint.py` has no local file to remove. `python refresh.py --force-synopsis` etc. sidesteps the checkpoint diff entirely without deleting anything, if that's sufficient.
- **Actor names normalised to uppercase** before matching against `cast_enriched.parquet`. Directors kept as-is.
- **`|AND ` prefix in actor_list** — Snowflake artifact, stripped by `_clean_actor` in `main.py`.
- **LLM field clamping happens at encode time** in `cinema_admits_models`, not here. Raw checkpoint data (S3 snapshot+deltas, or local progress JSON for `main.py`) retains original LLM output (including malformed values).
- **Director hit rate ~70%** is expected — regional/indie directors return `unknown` because they're not in training data. Web_search helps the worst cases but not all of them.
- **Errored films get auto-retried** — failures go to an errors store (S3 `<name>/cache/errors.json` for `refresh.py`; local `film_meta_errors.json` for `main.py`), NOT the checkpoint. The next run's diff sees them as not-done and retries. There's no single "delete this one entry" fix for the S3 path (state is spread across the snapshot + any deltas) — the practical option is to let it keep retrying (harmless) or force a full re-extract as above.
- **TPM bound, not RPM** — film_meta concurrency is gated by tokens-per-minute (200k org cap, ~15k per web_search call). Don't bump `META_MAX_CONCURRENCY` past 2-3 without first raising the org's TPM tier.
- **Diagnostics scripts still read local paths — known gap, not yet fixed** — `diagnostics/audit_film_meta.py`, `refresh_comparison.py`, `compare_prompt_versions.py`, and `post_refresh_check.py` (plus `cleanup_film_meta.py`'s standalone CLI) all read `FILM_META_ENRICHED_PATH`/`SYNOPSES_EXTRACTED_PATH`/`CAST_ENRICHED_PATH`/`DIRECTOR_ENRICHED_PATH` straight off local disk. Since `refresh.py`'s Dagster/CLI path no longer writes those local files at all, these scripts will silently show stale, empty, or missing data after an S3-based run unless a copy happens to exist locally already (e.g. from a `main.py` run, or downloaded manually from S3 for inspection). `rematch_comscore.py`'s concert-film filter got the S3-aware fix (falls back to local); these diagnostics scripts did not — that's the next thing to fix if you rely on them after a Dagster run rather than a local `main.py` run.
- **`refresh.py` and `main.py` no longer agree on the work-set source** — `main.py` still globs the local `raw_from_snowflake`/`prediction_from_snowflake` parquet snapshots; `refresh.py::load_films_from_snowflake` now pulls live from Snowflake via `SQL_FILM_DETAILS` instead (see above — done so this doesn't keep re-reading an ever-growing pile of weekly snapshots when run against S3 in the dataplatform-hosted environment). This means Dagster/CLI `refresh.py` runs and ad-hoc `main.py` runs can now genuinely see different film sets (e.g. a film present in Snowflake's live catalogue but not yet in a local snapshot, or vice versa). Not yet reconciled — if `main.py` also needs to move off local snapshots, it should follow the same pattern.
- **Live Snowflake pull is much bigger than the old snapshot-glob approach — five layered filters bring it back down to a sane backlog (full investigation 2026-08-12)** — `SQL_FILM_DETAILS` has no date filter of its own (`WHERE FILM_NAT_OPEN_DATE IS NOT NULL`, going back to 1935). The first real run of `load_films_from_snowflake()` surfaced **18,913** raw films vs. the ~5,032 the old parquet-snapshot-based work-set had ever accumulated — a 100% checkpoint-match, zero-miss diff still showed **10,231** "new" synopsis films, because they genuinely were never processed before (mostly real 2008-2017 releases the old snapshot system never captured, not junk). Layered fixes, synopsis backlog shown at each step:
  1. **`WORK_SET_MIN_REL_DATE`** (`refresh.py`, currently `pd.Timestamp("2018-01-01", tz="UTC")`) — restricts the live pull to the box office model's actual training window. Not a guess: `cinema_admits_models/helper_fucntions.py::return_train_calib_test_dates`'s real default is `train_start=datetime(2018, 1, 1)`, which drives the date params substituted into `db_merge_20260420.sql`/`bo_pred_build.sql`. Films released before this are outside what the model trains on at all. → 10,231 → 3,338 new.
  2. **Variant merge extended to the synopsis path** — `_apply_variant_merge` was already applied to film_meta but not synopsis/cast/director; now also called from `_diff_synopsis_films` (and `refresh_synopsis`'s force branch). → 3,338 → 3,239.
  3. **Skip-distributor filter extended to the synopsis path**, plus 9 new NZ festival entries added to `FILM_META_SKIP_DISTRIBUTORS` (`NZ Italian Film Festival`, `NZ French Film Festival`, `NZ NEW ZEALAND INT FILM FESTIVAL`, `ZZ International Film Festival NZ`, `NZ RESENE ARCHITECTURE AND DESIGN FF`, `NZ British Film Festival NZ`, `ZZ SHOW ME SHORTS FILM FESTIVAL`, `ZZ Veterans Film Festival`, `ZZ GREEK FESTIVAL OF SYDNEY` — the list previously only had AU festival names). This was the single biggest lever. → 3,239 → 2,273.
  4. **`_drop_no_usable_synopsis`** (synopsis-path only) — drops films where *both* `synopsis` and `alt_synopsis` are placeholder/empty ("plot unknown" ~90 occurrences, leaked language names like "Telugu"/"Tamil"/"Hindi" — same bug pattern `cleanup_film_meta.py` documents for the genre field — bracketed years like `[1996]`, "Testing Code", synopsis-equals-title). Deliberately a denylist of known-junk values, not a length cutoff — legitimate synopses can be genuinely short (`"Remake of Train to Busan."` is 25 characters and real). Checks `alt_synopsis` too since it often has real content even when the primary (IHUB) synopsis is a placeholder (true for about half of the observed "plot unknown" rows). → 2,273 → **2,228** (final synopsis backlog).
  5. **`_drop_no_session_past_films`** (film_meta-path only, via `_load_session_film_ids`) — drops PAST-dated films with zero theatrical sessions ever logged in `EDW_ENT_PRD.SEMANTIC.VW_VHO_SESSION_SUMMARY` (joined via `DIM_VH_FILM.FILM_HO_CODE`, the same join `db_merge_20260420.sql` uses for real training data). Of film_meta's 2,442-film backlog (after steps 1+3, which also apply there — step 2/4 are synopsis-only), only 544 (22%) had any session data at all, even 8+ weeks after release (past any plausible reporting lag) — the rest were mostly small arthouse/niche-distributor bookings that likely never had a meaningful theatrical run. **Future-dated films are always kept regardless of session data** — film_meta's job includes covering upcoming releases before they've opened (`cinema_admits_models/predict_future_films.py`'s prediction use case), so this only filters past films that already had their chance to show up in the session data and didn't. Fails open (keeps every film) on a Snowflake error — a transient connection issue should never silently exclude real films. → 2,442 → **604** (final film_meta backlog).
  Set `WORK_SET_MIN_REL_DATE = None` to disable step 1 and pull the full historical catalogue, if that's ever actually wanted. **`main.py`'s own `enrich_film_meta` does not have the session-data filter (step 5)** — it's only wired into `refresh.py`'s path.
- **`vendored/cinema_admits_models/` is read-only** — these files are copies, not the source. Edit upstream in `cinema_admits_models/` and re-vendor (procedure in `vendored/cinema_admits_models/README.md`). The one in-repo modification — `from .encode_helper import EncHelper` relative-import patch — must be re-applied after re-vendoring. The key vendored file is `re_release_filter.py`, which filters re-release films from the work-set before extraction (used by `refresh.py`).
- **Comscore cache is the checkpoint** — already-matched films (score ≥ `match_thresh`, default 0.80) are skipped on re-run; below-threshold films are retried. To force a full re-match, delete **both** `comscore_cache.parquet` and `comscore_review_needed.parquet`, or call `re_score()` instead of `build_mapping()`.
- **Comscore manual overrides win and are never touched by `re_score()`** — they're applied first and forced to `confidence=manual`, `score=1.0`. The matcher writes the *review* file; the *overrides* file is created by a human.
- **`match_thresh` (0.80) vs `HIGH_CONFIDENCE_SCORE` (0.92) are different gates** — anything ≥0.80 is a match (cached, not retried); only ≥0.92 within 365 days is labelled `high`. The 0.80–0.92 band is `borderline` and lands in the review file.
- **`confidence='variant'` is a fifth tier** — assigned by `_propagate_variants()` after the fuzzy loop. Format/event variants (3D, IMAX, GC, sing-along…) that share a base title with a matched film inherit its cs_id with `match_score=1.0`. Patterns mirror `cinema_admits_models/encode_helper.py::_VARIANT_STRIP`. Variants are excluded from the review file and skipped on incremental re-runs.
- **Comscore SQL is windowed** — `sql/comscore_extract.sql` is AU-only, filtered to `RELEASE_DATE >= 2018-01-01` with no upper bound (the `params` CTE still carries vestigial `pre_covid_start`/`pre_covid_end`/`post_covid_start` columns that nothing selects on — don't read them as active filters). Films released before 2018 won't match because they're absent from the extract, not because the matcher failed.
- **`rematch_comscore.py` needs `film_lookup.parquet`** — at `DATA_DIR/look_ups/film_lookup.parquet`. The comscore driver joins this for the `film` title column. If it's missing the load will raise `FileNotFoundError`.
- **`diagnostics/inspect_comscore_unmatched.py` calls Snowflake at module level** — the top of the file runs `pull_comscore()` outside `main()`. Run as a script (`python diagnostics/inspect_comscore_unmatched.py`) rather than importing it; requires VPN + Snowflake creds.
- **`cast_encode.py` / `director_encode.py` do not exist here** — they moved to `cinema_admits_models/build_data/`. Don't recreate them.
- **`main.py::RUN_ENCODE` raises** with a pointer to the new encode locations — encoding is no longer done in this repo.
- **Gower matching mirrors Comscore's rules exactly** (same cache-skip threshold, `high`/`borderline`/`unmatched`/`variant` tiers, `±1` year window) because both subclass `title_matcher.py::FuzzyTitleMatcher` — this includes the same "delete both cache + review parquet to force a full re-match" rule, just with `gower_` filenames instead of `comscore_`.
- **Gower has no `gower_manual_overrides.csv` yet** — the override mechanism (column names driven by each matcher's `ID_FIELD`) is inherited from Comscore's and works the moment that file is created at `DATA_DIR/gower/gower_manual_overrides.csv` with a `gower_id` column; no code change needed. Comscore's `comscore_manual_overrides.csv` (with a `cs_id` column) already exists and is in active use.
- **Gower's `ID_COL` is `prmry_title_no`**, not a true global title ID like Comscore's `title_global_id` — it's the closest stable identifier the Gower extract has. Exposed downstream as the cache column `gower_id`.
- **`sql/gower_export.sql` returns up to 3 rows per title** (`snapshot_type` in `latest`/`1m_pre_release`/`3m_pre_release`) — `GowerMatcher._prep_source()` keeps only the `latest` snapshot (falling back to pre-release snapshots) before matching, so `gower_life_time_base` in the cache always reflects the most recent estimate, not an arbitrary snapshot.
- **Gower matching is windowed to `rel_at >= GOWER_MIN_REL_DATE`** (currently 2025-01-01, in `rematch_gower.py`) — narrower than Comscore's window, and must be kept in sync with `sql/gower_export.sql`'s own `params.start_date`. EVT films released earlier are filtered out of the work-set before matching (not just left as noisy `unmatched` rows), since Gower has no candidate row for them at all. Deliberate for now — Gower is only needed 2025-onwards for another project; see `GOWER.md` for how to widen it back to full history later.

---

## Troubleshooting

- **`RuntimeError: Snowflake unavailable`** — check `config.yaml` creds and that VPN is connected.
- **Cascading 429s on `film_meta`** — concurrency is already tuned to 2 for the 200k TPM cap. Don't raise `META_MAX_CONCURRENCY` without first raising the org's TPM tier.
- **Dagster forgets prior runs between sessions** — `DAGSTER_HOME` not set, so it used a tmp dir. Set `export DAGSTER_HOME=~/dagster_home` permanently in your shell profile.
- **Schedule didn't fire overnight** — daemon wasn't running. `dagster dev` must stay up, or run `dagster-daemon run` separately.
- **`comscore_match`/`gower_match`'s concert-film filter silently skips** (logs "not found, skipping") — `film_meta_enriched.parquet` doesn't exist on S3 or locally yet; materialise `film_meta` in `dagster_defs.py` first (or `python refresh.py --only film_meta`), or check `config.yaml`'s `s3.bucket`/`s3.prefix` are correct if it should exist on S3 already.
- **Single asset materialisation fails** — if running a downstream asset alone (e.g. `cast`), `films_source` must already be materialised in the current `DAGSTER_HOME`. Materialise it once first, or select both together.
- **`_error: "ambiguous"` in film_meta** — run `python diagnostics/inspect_film_meta_progress.py` to see which films are stuck and why. These stay in `film_meta_errors.json` and auto-retry on the next run.
- **`uv pip install` fails with `invalid peer certificate: UnknownIssuer`** — corporate/proxied network rejecting `uv`'s bundled CA bundle. Add `--system-certs` to the install command.
- **`uv pip install` fails mid-download on a large package (e.g. `litellm`) with a connection reset** — transient network flakiness, not a real failure. `uv cache clean <package>` then retry; if it only needs to run matching, `requirements-matching.txt` sidesteps `litellm`/`openai` entirely.
