# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

LLM extraction of film metadata for the EVT box office model. **Raw extraction only** — encoding lives in the sibling repo `cinema_admits_models/build_data/`.

**Data root:** `~/Documents/data` (shared with `cinema_admits_models`)

See `DAGSTER.md` for Dagster operational details. See `COMSCORE.md` for the Comscore matching deep-dive and `GOWER.md` for the (near-identical) Gower matching deep-dive.

---

## Seven extraction paths

| Path | Model | Cardinality | Output parquet |
|---|---|---|---|
| **Synopsis** (LiteLLM batch) | `gpt-4.1-nano` (default) | per film | `synopsis_v2/synopses_extracted.parquet` |
| **Film meta** (Responses + `web_search`) | `gpt-5.4-mini` | per film | `film_meta/film_meta_enriched.parquet` |
| **Actor** (Responses + `web_search`) | `gpt-5.4-mini` | per unique actor | `cast_meta/cast_enriched.parquet` |
| **Director** (Responses + `web_search`) | `gpt-5.4-mini` | per unique director | `director_meta/director_enriched.parquet` |
| **Comscore match** (rapidfuzz, no API) | — | per film | `comscore/comscore_cache.parquet` |
| **Gower match** (rapidfuzz, no API) | — | per film | `gower/gower_cache.parquet` |
| **ID bridge** (pure join, no API) | — | per film | `id_bridge/film_id_bridge.parquet` |

Synopsis uses `DEFAULT_MODEL` from `models.py` (currently `gpt-4.1-nano`, with `gpt-5.4-mini` as fallback). Nano = pure text classification (9 tasks read only title + synopsis). Mini + web_search = knowledge-grounded fields (budget, studios, fame_tier, director_tier, ip_strength, adaptation_type) where training memory is too brittle, especially on recent or upcoming films.

All LLM extractors are checkpoint-resumable — progress JSONs in `~/Documents/data/<dir>/*_progress.json`. Delete to force re-extraction.

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

# Diff-based CLI (the four LLM paths)
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

# Sync the four meta checkpoints (parquet + progress json) to S3 — see "S3 sync" below
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

**`dagster_defs.py`** exposes one upstream `films_source` asset (single Snowflake pull), four downstream extraction assets — `synopsis`, `cast`, `directors`, `film_meta` — each callable independently, and one further downstream asset, `s3_sync` (see "S3 sync" below).

Jobs:
- `nightly_job` — synopsis + cast + directors, scheduled 02:00 daily
- `film_meta_job` — film_meta only, scheduled 03:00 Sundays (separated because it's the $100+/run path)
- `s3_sync_job` — s3_sync only, ad-hoc (separated because it needs its own AWS auth step)
- `full_refresh_job` — everything in this code location, ad-hoc

**`dagster_matching_defs.py`** exposes three self-contained assets — `comscore_match`, `gower_match`, `id_bridge` — with one job:
- `comscore_job` — comscore_match + gower_match + id_bridge, ad-hoc

`comscore_match` and `gower_match` do **not** consume `films_source` — each loads EVT films from parquet snapshots itself (via the shared `rematch_comscore.py::load_evt_films`), plus reads `film_meta_enriched.parquet` for the concert-film filter. Since `film_meta` lives in the *other* code location, neither declares a Dagster-level `deps=[film_meta]` — the functional dependency (the parquet must exist on disk) is still real, it just isn't tracked as a Dagster staleness link. `id_bridge` declares `deps=[comscore_match, gower_match]` (same code location, so this one *is* Dagster-tracked) and just outer-joins their two caches on `film_id` — no matching logic of its own.

Schedules are **off by default** — toggle on in the UI. `dagster dev` runs both the webserver and the daemon; for headless, run `dagster-daemon run` separately (with `DAGSTER_HOME` set).

### S3 sync

`s3_sync.py::sync_meta_outputs_to_s3()` uploads the four LLM meta checkpoints — parquet + progress json for synopsis, film_meta, cast_meta, director_meta — to `s3://<s3.bucket>/<s3.prefix>/...` (config in `config.yaml`'s `s3:` block; `synopsis_v2` locally is renamed to `synopsis` on the S3 side). Deliberately **not** appended to the end of `refresh_synopsis`/`refresh_cast`/`refresh_directors`/`refresh_film_meta` — it's its own step (`s3_sync` asset, `s3_sync_job`) so the local parquet/progress-json stay the fast, no-network checkpoint, and the S3 upload can be triggered independently once a run is done.

Auth is a Stax SSO profile (`stax-stax-au1-event` in `~/.aws/credentials`, generated by `stax2aws login`) whose session credentials **expire after 1 hour** (`~/stax2aws.yaml`'s `session-duration: 3600`). There's no automatic refresh — run `stax2aws login -i stax-au1 -o event` manually before triggering `s3_sync_job` (or the CLI), or it fails with a clear "no valid AWS credentials" error. This is a real gap for unattended weekly scheduling: as of now `s3_sync_job` has no cron schedule, precisely because nothing can refresh the SSO session on its own. When this moves into a real production environment, a long-lived service credential (IAM role/user scoped to this bucket prefix) should replace the Stax profile — swap `s3.profile` in `config.yaml` (or point `boto3.Session` at the default credential chain instead) and the rest of `s3_sync.py` is unaffected.

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
- **`refresh.py`** — diff-based orchestrator for all four LLM paths. The Dagster assets are thin wrappers around the four `refresh_*` functions here. `load_films_from_snowflake` is the canonical work-set loader (primary source is parquet snapshots, Snowflake join is for titles only).
- **`films/main.py::get_films_sources(persisted=True)`** — alternative film loader used by the `films_source` Dagster asset. Tries `films/source_data/films.parquet` first; falls back to a full Snowflake pull via `base_snowflake.py::SnowFlakeBase` (same key-pair auth as `refresh.py::load_films_from_snowflake`). Distinct from `refresh.py::load_films_from_snowflake` — both must stay in sync with each other.
- **`films/sql.py`** — Snowflake SQL queries. `SQL_FILM_DETAILS` is the main join used by `load_films_from_snowflake` to fetch authoritative titles from `EDW_ENT_PRD.CURATED.DIM_VH_FILM`. The `SYNOPSIS`/`ALT_SYNOPSIS` columns are explicitly cast to `VARCHAR(16777216)` — without it, Snowflake infers a narrower width from `DIM_VH_FILM.FILM_DESC`'s declared column width and truncation-errors on longer synopses from the forecast table.
- **`base_snowflake.py::SnowFlakeBase`** — minimal vendored Snowflake helper. Hard-coded to EVT Snowflake account (`mm31132.ap-southeast-2`); uses key-pair auth (`SF_RSA_KEY` from `config.yaml`) with a key path passed directly. The one and only Snowflake auth path in this repo — used by both `refresh.py::load_films_from_snowflake` and `films/main.py::get_films_sources`.
- **`title_cleaner.py`** — strips variant prefixes (`3D`, `IMAX`, `GC`) from titles before LLM prompt construction. Used by both `LlmJsonExtractor` and `FilmMetaExtractor`.
- **`title_matcher.py::FuzzyTitleMatcher`** — shared fuzzy title-matching engine behind both `comscore_matcher.py::ComscoreMatcher` and `gower_matcher.py::GowerMatcher`: title normalisation/scoring, variant propagation, and cache/review/manual-override I/O. Subclasses set column names (`ID_COL`/`DATE_COL`/`TITLE_COLS` on the source side, `ID_FIELD`/`TITLE_FIELD`/`DATE_FIELD`/`MATCHED_TITLE_FIELD` on the cache side) and output paths; everything else is identical between sources. See `COMSCORE.md` / `GOWER.md` for the algorithm.
- **`comscore_matcher.py::ComscoreMatcher`** / **`rematch_comscore.py`** — Comscore's column config on `FuzzyTitleMatcher` + driver script. `rematch_comscore.py::load_evt_films` is the canonical EVT work-set loader for *both* matching paths — `rematch_gower.py` imports it directly so Comscore and Gower always score against the same film catalogue.
- **`gower_matcher.py::GowerMatcher`** / **`rematch_gower.py`** — Gower's column config on `FuzzyTitleMatcher` + driver script. See `GOWER.md` for what's different from Comscore (single title column, `prmry_title_no` as ID, no alt-content flag, multi-snapshot dedup).
- **`id_bridge.py::build_id_bridge()`** — outer-joins `comscore_cache.parquet` + `gower_cache.parquet` on `film_id` into `id_bridge/film_id_bridge.parquet`. Pure join, no matching logic.
- **`s3_sync.py::sync_meta_outputs_to_s3()`** — uploads the four meta checkpoints (parquet + progress json) to S3. `SYNC_SPECS` is the explicit (local dir, S3 folder, filenames) list — deliberately not a glob, so stray `.bak`/errors files in `DATA_DIR/film_meta/` etc. never get swept up. See "S3 sync" above for auth.
- **`dagster_defs.py`** / **`dagster_matching_defs.py`** — two separate Dagster code locations (see "Two Dagster code locations" above). The matching one deliberately never imports `refresh.py`, so it needs only `requirements-matching.txt`.
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
| Gower match | similar order to Comscore, cold-start | $0 |
| ID bridge | seconds | $0 |

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
- **`comscore_match`/`gower_match` asset fails with `FileNotFoundError`** — `film_meta_enriched.parquet` doesn't exist yet; materialise `film_meta` in `dagster_defs.py` first (or `python refresh.py --only film_meta`).
- **Single asset materialisation fails** — if running a downstream asset alone (e.g. `cast`), `films_source` must already be materialised in the current `DAGSTER_HOME`. Materialise it once first, or select both together.
- **`_error: "ambiguous"` in film_meta** — run `python diagnostics/inspect_film_meta_progress.py` to see which films are stuck and why. These stay in `film_meta_errors.json` and auto-retry on the next run.
- **`uv pip install` fails with `invalid peer certificate: UnknownIssuer`** — corporate/proxied network rejecting `uv`'s bundled CA bundle. Add `--system-certs` to the install command.
- **`uv pip install` fails mid-download on a large package (e.g. `litellm`) with a connection reset** — transient network flakiness, not a real failure. `uv cache clean <package>` then retry; if it only needs to run matching, `requirements-matching.txt` sidesteps `litellm`/`openai` entirely.
