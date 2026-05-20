# CLAUDE.md

LLM extraction of film metadata for the EVT box office model. **Raw extraction only** — encoding moved to `cinema_admits_models/build_data/` (2026-05-19).

**Data root:** `~/Documents/data` (shared with `cinema_admits_models`)
**Last updated:** 2026-05-20 (standalone repo + Dagster setup + TPM-tuned concurrency)

---

## Four extraction paths

| Path | Model | Cardinality | Output parquet |
|---|---|---|---|
| **Synopsis** (LiteLLM batch) | `gpt-5.4-nano` | per film | `synopsis_v2/synopses_extracted.parquet` |
| **Film meta** (Responses + `web_search`) | `gpt-5.4-mini` | per film | `film_meta/film_meta_enriched.parquet` |
| **Actor** (Responses + `web_search`) | `gpt-5.4-mini` | per unique actor | `cast_meta/cast_enriched.parquet` |
| **Director** (Responses + `web_search`) | `gpt-5.4-mini` | per unique director | `director_meta/director_enriched.parquet` |

Nano = pure text classification (9 tasks read only title + synopsis). Mini + web_search = knowledge-grounded fields (budget, studios, fame_tier, director_tier, ip_strength, adaptation_type) where training memory is too brittle, especially on recent or upcoming films.

All extractors are checkpoint-resumable — progress JSONs in `~/Documents/data/<dir>/*_progress.json`. Delete to force re-extraction.

---

## Running

```bash
# Setup (one-time)
uv venv --python 3.11 && source .venv/bin/activate
uv pip install -r requirements.txt

# Ad-hoc / interactive
python main.py                    # edit CONFIG block (RUN_SYNOPSIS / RUN_CAST / RUN_DIRECTOR / RUN_META)

# Diff-based CLI (the four refresh paths)
python refresh.py                              # all four paths
python refresh.py --only synopsis cast         # subset
python refresh.py --force-film-meta            # ignore diff for one path

# Dagster (local)
dagster dev -f dagster_defs.py                 # UI at http://localhost:3000

# Diagnostics / validation (in diagnostics/ subfolder):
python diagnostics/test_film_meta.py            # single-film smoke test
python diagnostics/compare_film_meta_search.py  # A/B mini+search vs mini-no-search
python diagnostics/refresh_comparison.py        # full sample run vs production parquets
python diagnostics/inspect_film_meta.py [--detail|--vs-tmdb|--film-id N]
python diagnostics/print_compare.py [--disagree-only]
```

Required env (`.env`): `OPENAI_KEY=...`. Optional: `FILM_META_MODEL=gpt-5.4-mini` (default).

### Dagster assets

`dagster_defs.py` exposes one upstream `films_source` asset (single Snowflake pull) and four downstream extraction assets — `synopsis`, `cast`, `directors`, `film_meta` — each callable independently. Jobs: `nightly_job` (synopsis+cast+directors, 02:00 daily), `film_meta_job` (film_meta only, 03:00 Sundays — separated because it's the $100+/run path), `full_refresh_job` (all five, ad-hoc).

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

## Repo map

```
film_synopsis_meta/
├── main.py                       # ad-hoc / interactive runner
├── refresh.py                    # diff-based CLI + 4 public refresh_* functions for Dagster
├── dagster_defs.py               # films_source + 4 extraction assets + jobs/schedules
├── requirements.txt              # standalone-repo dependencies (incl. dagster)
├── base_snowflake.py             # vendored SnowFlakeBase (formerly evt_back_up/base)
├── extractor.py                  # LlmJsonExtractor (LiteLLM, nano)
├── film_meta_extractor.py        # ResponsesExtractor + 3 subclasses (mini+search)
├── extraction.py                 # ExtractionTask dataclass + JSON parse helpers
├── models.py                     # MODELS dict + default + pricing
├── load_prompts.py
├── config.py / config.yaml       # paths, Snowflake creds
├── prompts/*.yaml                # see Prompts table above
├── films/                        # Snowflake SQL queries (SQL_FILM_DETAILS in sql.py)
├── encode/                       # legacy helpers (EncHelper imported by depreciated synopsis encoder)
├── diagnostics/                  # validation + A/B utilities — non-production
│   ├── refresh_comparison.py     # sample run + side-by-side vs production
│   ├── inspect_film_meta.py      # eyeball film_meta extraction results
│   ├── print_compare.py          # console actor/director new-vs-old
│   ├── compare_film_meta_search.py  # mini+search vs mini-no-search A/B
│   ├── test_film_meta.py         # single-film smoke test
│   └── (compare_models, compare_prompt_versions, analyse_title_patterns, gpu_test, migration — legacy)
└── depreciated/encoding/
    └── encode_synopsis.py        # superseded by cinema_admits_models/build_data/encode_llm_features.py
```

`cast_encode.py` + `director_encode.py` moved to `cinema_admits_models/build_data/encode_cast_features.py` + `encode_director_features.py` on 2026-05-19. Don't recreate them here.

---

## Cost / volume estimate (full nightly run, ~4,100 films)

| Path | Wall-clock | Cost |
|---|---|---|
| Synopsis nano | ~10-15 min | ~$1-3 |
| Film meta mini+search | ~3 hrs | ~$100-150 |
| Actor mini+search (new only) | minutes | <$10 |
| Director mini+search (new only) | seconds | <$1 |

Check balance: https://platform.openai.com/settings/organization/billing/overview

---

## Recent changes

**2026-05-20 — standalone repo + Dagster setup + TPM-tuned concurrency**
- Vendored `base_snowflake.py` into the repo root (formerly imported via absolute path from `/Users/.../evt_back_up/base`). Removed `sys.path.insert(...)` lines from 6 callers (`main.py`, `refresh.py`, `cast_main.py`, `tmdb_fetch.py`, two `diagnostics/*` scripts, deprecated encoder). Repo is now standalone.
- `refresh.py` rewritten: was synopsis+cast only, now covers all four paths as separate `refresh_synopsis/cast/directors/film_meta` functions. Each takes optional pre-loaded `df_films` so Dagster shares one Snowflake pull. Snowflake loader fixed (was reading a non-existent SQL file).
- `dagster_defs.py` added with `films_source` upstream asset + 4 extraction assets. Three jobs: `nightly_job` (cheap paths), `film_meta_job` (weekly), `full_refresh_job`. Run via `dagster dev -f dagster_defs.py`.
- `requirements.txt` added (dagster, pandas, openai, litellm, snowflake-sqlalchemy, ...).
- **Rate-limit fix:** `META_MAX_CONCURRENCY` lowered from 4 → 2 and `AsyncOpenAI(max_retries=8)` set in `film_meta_extractor.py`. Symptom diagnosed: org TPM cap is 200k/min, web_search calls ≈15k tokens each → sustainable ~13 req/min. At concurrency=4 the rolling window saturates within 30s and ~70% of subsequent calls 429. At concurrency=2 with retries, error rate drops to ~4% (genuine `json_parse_failed`, not rate limits).
- `main.py` now writes errored film_meta entries to a side log `~/Documents/data/film_meta/film_meta_errors.json` instead of dropping silently. Checkpoint and error log are guaranteed disjoint, so failed films get automatically retried on the next run.

**2026-05-19 (later) — film_meta disambiguation inputs changed**
- `FilmMetaExtractor` now sends **title + release date + director + first sentence of EVT synopsis** to the LLM. AU distributor is no longer sent (it stays in `df_source` for the skip-filter and `evt_dstbtr` passthrough on save, but is hidden from the model).
- Motivation: two films can share title + year. Director and synopsis-opening are strong disambiguation anchors; distributor was weak.
- Prompt (`prompts/film_meta_prompts.yaml`) gained a **Disambiguation** section instructing the model to verify candidates against the EVT director + synopsis and to return `{"_error": "ambiguous", "_candidates": [...]}` instead of guessing when two films still match. Worth grepping checkpoints for `_error: "ambiguous"` after runs.
- `arun()` signature: `dstbtr_col` removed; `director_col='director'` and `synopsis_col='synopsis'` added. Both columns are already in `df_films` from the raw parquets — no new joins needed.

**2026-05-19 — encoding moved out + cast/director upgraded to web_search**
- Three encoders moved/deprecated: `encode_synopsis.py` → `depreciated/encoding/` (functionality in `cinema_admits_models/build_data/encode_llm_features.py`); `cast_encode.py` and `director_encode.py` moved to `cinema_admits_models/build_data/` as `encode_cast_features.py` and `encode_director_features.py`.
- `ip_strength` + `adaptation_type` migrated from synopsis nano batch into film_meta mini+search call (web grounding is more reliable on recent IP).
- Cast + director extractors switched from nano (LiteLLM) to mini + web_search (Responses API).
- Default model bumped from `gpt-4o-mini` (deprecating) to `gpt-5.4-mini`.
- `main.py::RUN_ENCODE` now raises with a pointer to the new locations. `refresh.py` no longer calls encoders.

---

## Non-obvious behaviours

- **Re-run dedup keeps `first`** — existing parquet rows win. Delete checkpoint + parquet to fully re-extract.
- **Actor names normalised to uppercase** before matching against `cast_enriched.parquet`. Directors kept as-is.
- **`|AND ` prefix in actor_list** — Snowflake artifact, stripped by `_clean_actor` in `main.py` and the moved cast encoder.
- **LLM field clamping happens at encode time** in `cinema_admits_models`, not here. Raw checkpoint JSONs retain original LLM output (including malformed values).
- **Director hit rate ~70%** is expected — regional/indie directors return `unknown` because they're not in training data. Web_search helps the worst cases but not all of them.
- **Errored films get auto-retried** — `main.py` puts failures into `film_meta_errors.json`, NOT the checkpoint. The next run's diff sees them as not-done and retries. To force-retry a specific film, delete its entry from `film_meta_progress.json`.
- **TPM bound, not RPM** — film_meta concurrency is gated by tokens-per-minute (200k org cap, ~15k per web_search call), not requests-per-minute. Don't bump `META_MAX_CONCURRENCY` past 2-3 without first raising the org's TPM tier.
- **`refresh.py` column normalisation** — Snowflake returns `director_list` / `distributor_name` / `film_nat_open_date`; the loader renames these to `director` / `dstbtr` / `rel_at` to match the parquet schema used by `main.py`. Don't be surprised that `SQL_FILM_DETAILS` columns and `df_films` columns don't line up 1:1.
