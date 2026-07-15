# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

LLM extraction of film metadata for the EVT box office model. **Raw extraction only** — encoding lives in the sibling repo `cinema_admits_models/build_data/`.

**Data root:** `~/Documents/data` (shared with `cinema_admits_models`)

See `DAGSTER.md` for Dagster operational details. See `COMSCORE.md` for the Comscore matching deep-dive.

---

## Five extraction paths

| Path | Model | Cardinality | Output parquet |
|---|---|---|---|
| **Synopsis** (LiteLLM batch) | `gpt-4.1-nano` (default) | per film | `synopsis_v2/synopses_extracted.parquet` |
| **Film meta** (Responses + `web_search`) | `gpt-5.4-mini` | per film | `film_meta/film_meta_enriched.parquet` |
| **Actor** (Responses + `web_search`) | `gpt-5.4-mini` | per unique actor | `cast_meta/cast_enriched.parquet` |
| **Director** (Responses + `web_search`) | `gpt-5.4-mini` | per unique director | `director_meta/director_enriched.parquet` |
| **Comscore match** (rapidfuzz, no API) | — | per film | `comscore/comscore_cache.parquet` |

Synopsis uses `DEFAULT_MODEL` from `models.py` (currently `gpt-4.1-nano`, with `gpt-5.4-mini` as fallback). Nano = pure text classification (9 tasks read only title + synopsis). Mini + web_search = knowledge-grounded fields (budget, studios, fame_tier, director_tier, ip_strength, adaptation_type) where training memory is too brittle, especially on recent or upcoming films.

All LLM extractors are checkpoint-resumable — progress JSONs in `~/Documents/data/<dir>/*_progress.json`. Delete to force re-extraction.

### Comscore matching (fifth path — no LLM, no API)

`comscore_matcher.py::ComscoreMatcher` maps EVT `film_id` → Comscore `title_global_id` so IBOE_TITLES + IBOE_FLASH_GROSS can join onto EVT films. Driver: `rematch_comscore.py`. Comscore data is pulled once via `sql/comscore_extract.sql`.

Matching: NFKD-normalised titles scored with `max(ratio, token_sort_ratio)` across five Comscore title columns (`film_name`/`upper_name`/`title_aka`/`us_title_name`/`short_name`), variant/festival prefix stripping (`GC`/`3D`/`IMAX`/`TFF -`…), a length-ratio guard (rejects short-title impostors like AVATAR→TÁR), a ±1-year release window, and a date-proximity tie-breaker. Each film lands in a confidence tier: **high** (≥0.92 and ≤365 days off), **borderline**, **unmatched**, or **manual**.

Outputs under `~/Documents/data/comscore/`:
- `comscore_cache.parquet` — keyed on EVT `film_id` (this is the checkpoint — no separate progress JSON)
- `comscore_review_needed.parquet` — borderline+unmatched rows with top-5 candidates, for human triage
- `comscore_manual_overrides.parquet` — **user-authored**: fill `manual_override_cs_id` in the review file, save under this name, and overrides win on the next run (forced to `confidence=manual`, `score=1.0`)

`ComscoreMatcher.re_score()` re-evaluates the whole cache against an updated extract/algorithm without re-pulling (manual overrides preserved). Concert films are dropped on both sides (`is_alt_content` in Comscore; `adaptation_type == "concert_film"` from `film_meta_enriched.parquet` on the EVT side).

---

## Running

```bash
# Setup (one-time)
uv venv --python 3.11 && source .venv/bin/activate
uv pip install -r requirements.txt

# Ad-hoc / interactive
python main.py                    # edit CONFIG block (RUN_SYNOPSIS / RUN_CAST / RUN_DIRECTOR / RUN_META)

# Diff-based CLI (the four LLM paths)
python refresh.py                              # all four paths
python refresh.py --only synopsis cast         # subset (choices: synopsis cast director film_meta)
python refresh.py --force-synopsis             # ignore diff for synopsis
python refresh.py --force-cast
python refresh.py --force-director
python refresh.py --force-film-meta

# Comscore matching (CPU only, no API)
python rematch_comscore.py                     # set LIMIT_FILMS at top of file (default None = full run)

# Dagster (local)
export DAGSTER_HOME=~/dagster_home && mkdir -p "$DAGSTER_HOME"
dagster dev -f dagster_defs.py                 # UI at http://localhost:3000

# Diagnostics / validation (in diagnostics/ subfolder):
python diagnostics/test_film_meta.py            # single-film smoke test
python diagnostics/compare_film_meta_search.py  # A/B mini+search vs mini-no-search
python diagnostics/refresh_comparison.py        # full sample run vs production parquets
python diagnostics/inspect_film_meta.py [--detail|--vs-tmdb|--film-id N]
python diagnostics/inspect_film_meta_progress.py  # live coverage + error breakdown mid-run
python diagnostics/print_compare.py [--disagree-only]
python diagnostics/inspect_comscore_unmatched.py  # bucketise unmatched/borderline by candidate_1_score
```

Required env (`.env`): `OPENAI_KEY=...`

Optional env:
| Variable | Default | Notes |
|---|---|---|
| `FILM_META_MODEL` | `gpt-5.4-mini` | Override model for all Responses extractors |
| `WEB_SEARCH_COST_USD` | `0.025` | Per-call web_search fee — verify against current OpenAI pricing; this default may be stale |
| `WEB_SEARCH_ALLOWED_DOMAINS` | (none) | Comma-separated allowlist; narrowing to wiki+imdb hurts budget coverage |
| `DAGSTER_HOME` | tmp dir | Set persistently in shell profile; without it Dagster forgets prior runs between sessions |

### Dagster assets

`dagster_defs.py` exposes one upstream `films_source` asset (single Snowflake pull) and five downstream assets — `synopsis`, `cast`, `directors`, `film_meta`, `comscore_match` — each callable independently.

Jobs:
- `nightly_job` — synopsis + cast + directors, scheduled 02:00 daily
- `film_meta_job` — film_meta only, scheduled 03:00 Sundays (separated because it's the $100+/run path)
- `comscore_job` — comscore_match only, ad-hoc
- `full_refresh_job` — everything, ad-hoc

`comscore_match` does **not** consume `films_source` — it loads EVT films from parquet snapshots itself, plus reads `film_meta_enriched.parquet` for the concert-film filter. It declares `deps=[film_meta]` so a fresh film_meta run marks it stale.

Schedules are **off by default** — toggle on in the UI. `dagster dev` runs both the webserver and the daemon; for headless, run `dagster-daemon run` separately (with `DAGSTER_HOME` set).

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

- **`config.py`** — single source of truth for all paths and constants. Reads `config.yaml` (Snowflake creds, data root). Import paths from here; never hardcode `~/Documents/data`.
- **`models.py`** — model registry for LiteLLM paths. `DEFAULT_MODEL` is the synopsis nano model; `DEFAULT_FALLBACKS` is the fallback chain. To switch the synopsis model, change `DEFAULT_MODEL` here.
- **`extraction.py::ExtractionTask`** — the core data structure that connects prompts to extractors. Each task holds the prompt text, JSON field schema, and output key. `load_prompts.py::load_tasks_from_yaml` converts YAML entries into `ExtractionTask` objects; both `LlmJsonExtractor` and `ResponsesExtractor` consume them. Also contains shared JSON parse + repair utilities used by all extractors.
- **`refresh.py`** — diff-based orchestrator for all four LLM paths. The Dagster assets are thin wrappers around the four `refresh_*` functions here. `load_films_from_snowflake` is the canonical work-set loader (primary source is parquet snapshots, Snowflake join is for titles only).
- **`films/main.py::get_films_sources(persisted=True)`** — alternative film loader used by the `films_source` Dagster asset. Tries `films/source_data/films.parquet` first; falls back to a full Snowflake pull via `tools/connections.py::SnowflakeDB`. Distinct from `refresh.py::load_films_from_snowflake` — both must stay in sync with each other.
- **`films/sql.py`** — Snowflake SQL queries. `SQL_FILM_DETAILS` is the main join used by `load_films_from_snowflake` to fetch authoritative titles from `EDW_ENT_PRD.CURATED.DIM_VH_FILM`.
- **`base_snowflake.py::SnowFlakeBase`** — minimal vendored Snowflake helper. Hard-coded to EVT Snowflake account (`mm31132.ap-southeast-2`); uses key-pair auth with a key path passed directly. Used by the main extraction path.
- **`tools/connections.py::SnowflakeDB`** — richer Snowflake connection class with proxy detection, key-pair auth via env vars (`CABOODLE_SNOW_USER`, `CABOODLE_SNOW_ACCOUNT`, `SNOWFLAKE_KP_PATH`, `SNOWFLAKE_KP_AUTH`). Used by `films/main.py`. Also contains `CaboodleDB` (SQL Server) and `CaboodleProxy` for corporate proxy routing.
- **`title_cleaner.py`** — strips variant prefixes (`3D`, `IMAX`, `GC`) from titles before LLM prompt construction. Used by both `LlmJsonExtractor` and `FilmMetaExtractor`.
- **`ingest.py`** — `sync_synopses_sources` writes extracted synopses back to Snowflake after a synopsis run (called by `refresh_synopsis`, non-fatal if it fails).
- **`post_process.py`** — postprocessor registry (`POSTPROCESSORS` dict) that maps task names to cleanup functions. Currently only `clean_names` is live; hooked in `ExtractionTask.postprocess` if set.
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

The printed "Run total: $X" for web_search paths uses `WEB_SEARCH_COST_USD` which may be stale — verify against the actual dashboard bill.

---

## Non-obvious behaviours

- **Re-run dedup keeps `first`** — existing parquet rows win. Delete checkpoint + parquet to fully re-extract.
- **Actor names normalised to uppercase** before matching against `cast_enriched.parquet`. Directors kept as-is.
- **`|AND ` prefix in actor_list** — Snowflake artifact, stripped by `_clean_actor` in `main.py`.
- **LLM field clamping happens at encode time** in `cinema_admits_models`, not here. Raw checkpoint JSONs retain original LLM output (including malformed values).
- **Director hit rate ~70%** is expected — regional/indie directors return `unknown` because they're not in training data. Web_search helps the worst cases but not all of them.
- **Errored films get auto-retried** — `main.py` puts failures into `film_meta_errors.json`, NOT the checkpoint. The next run's diff sees them as not-done and retries. To force-retry a specific film, delete its entry from `film_meta_progress.json`.
- **TPM bound, not RPM** — film_meta concurrency is gated by tokens-per-minute (200k org cap, ~15k per web_search call). Don't bump `META_MAX_CONCURRENCY` past 2-3 without first raising the org's TPM tier.
- **`refresh.py` and `main.py` must agree on the work-set** — both drive off the same parquet snapshots. If `main.py`'s loader changes, `refresh.py::load_films_from_snowflake` has to track it, or Dagster runs will silently process a different film set than ad-hoc runs.
- **`vendored/cinema_admits_models/` is read-only** — these files are copies, not the source. Edit upstream in `cinema_admits_models/` and re-vendor (procedure in `vendored/cinema_admits_models/README.md`). The one in-repo modification — `from .encode_helper import EncHelper` relative-import patch — must be re-applied after re-vendoring. The key vendored file is `re_release_filter.py`, which filters re-release films from the work-set before extraction (used by `refresh.py`).
- **Comscore cache is the checkpoint** — already-matched films (score ≥ `match_thresh`, default 0.80) are skipped on re-run; below-threshold films are retried. To force a full re-match, delete **both** `comscore_cache.parquet` and `comscore_review_needed.parquet`, or call `re_score()` instead of `build_mapping()`.
- **Comscore manual overrides win and are never touched by `re_score()`** — they're applied first and forced to `confidence=manual`, `score=1.0`. The matcher writes the *review* file; the *overrides* file is created by a human.
- **`match_thresh` (0.80) vs `HIGH_CONFIDENCE_SCORE` (0.92) are different gates** — anything ≥0.80 is a match (cached, not retried); only ≥0.92 within 365 days is labelled `high`. The 0.80–0.92 band is `borderline` and lands in the review file.
- **`confidence='variant'` is a fifth tier** — assigned by `_propagate_variants()` after the fuzzy loop. Format/event variants (3D, IMAX, GC, sing-along…) that share a base title with a matched film inherit its cs_id with `match_score=1.0`. Patterns mirror `cinema_admits_models/encode_helper.py::_VARIANT_STRIP`. Variants are excluded from the review file and skipped on incremental re-runs.
- **Comscore SQL is windowed** — `sql/comscore_extract.sql` is AU-only and filtered to specific release-date windows (see `params` CTE). Films outside that window won't match because they're absent from the extract, not because the matcher failed.
- **`rematch_comscore.py` needs `film_lookup.parquet`** — at `DATA_DIR/look_ups/film_lookup.parquet`. The comscore driver joins this for the `film` title column. If it's missing the load will raise `FileNotFoundError`.
- **`diagnostics/inspect_comscore_unmatched.py` calls Snowflake at module level** — the top of the file runs `pull_comscore()` outside `main()`. Run as a script (`python diagnostics/inspect_comscore_unmatched.py`) rather than importing it; requires VPN + Snowflake creds.
- **`cast_encode.py` / `director_encode.py` do not exist here** — they moved to `cinema_admits_models/build_data/`. Don't recreate them.
- **`main.py::RUN_ENCODE` raises** with a pointer to the new encode locations — encoding is no longer done in this repo.

---

## Troubleshooting

- **`RuntimeError: Snowflake unavailable`** — check `config.yaml` creds and that VPN is connected.
- **Cascading 429s on `film_meta`** — concurrency is already tuned to 2 for the 200k TPM cap. Don't raise `META_MAX_CONCURRENCY` without first raising the org's TPM tier.
- **Dagster forgets prior runs between sessions** — `DAGSTER_HOME` not set, so it used a tmp dir. Set `export DAGSTER_HOME=~/dagster_home` permanently in your shell profile.
- **Schedule didn't fire overnight** — daemon wasn't running. `dagster dev` must stay up, or run `dagster-daemon run` separately.
- **`comscore_match` asset fails with `FileNotFoundError`** — `film_meta_enriched.parquet` doesn't exist yet; materialise `film_meta` first.
- **Single asset materialisation fails** — if running a downstream asset alone (e.g. `cast`), `films_source` must already be materialised in the current `DAGSTER_HOME`. Materialise it once first, or select both together.
- **`_error: "ambiguous"` in film_meta** — run `python diagnostics/inspect_film_meta_progress.py` to see which films are stuck and why. These stay in `film_meta_errors.json` and auto-retry on the next run.
