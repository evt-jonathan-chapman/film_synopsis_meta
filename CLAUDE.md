# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

LLM-based pipeline that extracts structured metadata from film synopses, cast lists, and directors, then encodes them into feature parquets for the EVT box office model and other downstream projects.

This is the **canonical home for all film metadata extraction and encoding**. The box office model (`evt_back_up/box_office`) reads output parquets only — it does not import code from here.

**Box office project context** (full notes in `/Users/jonathanchapman/Documents/data/claude_chats/CLAUDE.md`):
- Box office model: PyTorch MLP ensemble, predicts NZ cinema admissions weeks 1–4
- LLM features are merged in `build_train.py` — use `left` join + 0-fill to avoid silent film drops
- Hard overrides for specific film_ids live in `encode_synopsis.py` — change there and re-run encode before retraining

**Cast/director feature leakage caveat:**
- Actor and director profiles are classified based on their fame *today* (2025), not at each film's release date
- Acceptable for training data back to ~2018 — fame tiers shift slowly (years, not months)
- Training data excludes 2020–2021 (COVID closures — admissions not representative)
- Intentional: pipeline is primarily for forward prediction where current fame tier is correct

---

## Running the Pipeline

```bash
# One-off extraction (edit CONFIG block at top of main.py first)
python main.py                    # synopsis + cast + director + film_meta enrichment
# Flags: RUN_SYNOPSIS, RUN_CAST, RUN_DIRECTOR, RUN_META, RUN_ENCODE, SAMPLE_SIZE

# Encode individually
python encode_synopsis.py         # → llama_features.parquet
python cast_encode.py             # → cast_features.parquet
python director_encode.py         # → director_features.parquet

# TMDB studio tier lookup (test sample then full run)
python tmdb_fetch.py              # 50-film sample, prints match quality
python tmdb_fetch.py --all        # full run, saves tmdb_meta/studio_tier.parquet

# Weekly refresh (Dagster entry point)
python refresh.py
python refresh.py --force-synopsis / --force-cast / --force-encode

# Sync local parquet ↔ Snowflake
python ingest.py
```

There is no test suite, linting config, or `requirements.txt`. Dependencies are managed externally.

---

## Architecture

### Pipeline flow

```
Snowflake (AA_M_WH, EDW_ENT_PRD.CURATED)
  │
  ├─ Synopsis path ──────────────────────────────────────────
  │   main.py / refresh.py → LlmJsonExtractor.arun_multiple_synopses()
  │   prompts_v2.yaml → ExtractionTask objects
  │       ↓
  │   synopsis_v2/synopses_extracted.parquet
  │       ↓
  │   encode_synopsis.py → encode_synopsis_features()
  │   (hard overrides, EncHelper enrichment, OneHot/MultiHot/PCA encoding)
  │       ↓
  │   {data_dir}/encoded_film_meta/{date}/llama_features.parquet
  │
  ├─ Cast path ──────────────────────────────────────────────
  │   main.py (RUN_CAST) → enrich_cast()
  │   cast_prompts.yaml → actor_profile task
  │       ↓
  │   cast_meta/cast_enriched.parquet       # one row per actor (uppercase name)
  │       ↓
  │   cast_encode.py → encode_cast_features()
  │   (clamps invalid LLM values, joins on pipe-separated actor_list)
  │       ↓
  │   cast_meta/cast_features.parquet       # one row per film_id
  │
  ├─ Director path ──────────────────────────────────────────
  │   main.py (RUN_DIRECTOR) → enrich_directors()
  │   director_prompts.yaml → director_profile task
  │       ↓
  │   director_meta/director_enriched.parquet   # one row per director
  │       ↓
  │   director_encode.py → encode_director_features()
  │       ↓
  │   director_meta/director_features.parquet   # one row per film_id
  │
  └─ Film meta path (web-grounded, replaces TMDB) ───────────
      main.py (RUN_META) → enrich_film_meta()
      film_meta_prompts.yaml → film_meta task
      film_meta_extractor.py → OpenAI Responses API + web_search tool
          ↓
      film_meta/film_meta_enriched.parquet      # one row per film_id
      (cols: title, release_date, director, writers, genres, description,
             budget_usd, budget_source, studios[...], cast[...], trailers[...])
```

### Key modules

**Extraction:**
- **extractor.py** — `LlmJsonExtractor`; `arun_multiple_synopses` (async, semaphore-bounded); cost tracking. Fallbacks deliberately excluded from async path — LiteLLM's `fallback_utils` intercepts 429s and retries on same-org model, crashing before the custom retry loop runs.
- **main.py** — One-off local extraction. CONFIG block at top: `TRAIN_DATE`, `PRED_DATE`, `FILM_IDS`, `SAMPLE_SIZE`, `RUN_SYNOPSIS/CAST/DIRECTOR/ENCODE`. Reads from raw parquets, gets titles from Snowflake. Batch checkpoints every 25 items to JSON; resumes on restart.
- **refresh.py** — Weekly Dagster orchestrator. Diffs Snowflake vs local parquets, re-extracts only what changed.
- **models.py** — LLM registry. `DEFAULT_MODEL = 'gpt-4.1-nano'`, fallbacks = `gpt-4o-mini`.
- **extraction.py** — `ExtractionTask` dataclass; `extract_json()`; `flatten_extraction()`.
- **tmdb_fetch.py** — DEPRECATED May 2026 (TMDB commercial pricing didn't land). Studio data now comes from `film_meta_extractor.py` via web_search. Kept for reference; do not call from new code.
- **film_meta_extractor.py** — Web-grounded per-film extractor. Uses OpenAI's Responses API directly (NOT LiteLLM) because the `web_search` hosted tool is only available on the Responses endpoint, not Chat Completions. Async + semaphore-bounded like LlmJsonExtractor; lower default concurrency (4) because each web_search adds 5-10s latency. Output schema in `prompts/film_meta_prompts.yaml::film_meta`.

**Encoding:**
- **encode_synopsis.py** — `encode_synopsis_features()`. Hard overrides for IP strength, genres, audience, sequel flags live here. Outputs `llama_features.parquet` (105 features).
- **cast_encode.py** — `encode_cast_features()`. 6 film-level features: `cast_lead_tier`, `cast_avg_fame_score`, `cast_global_astar_count`, `cast_astar_plus_count`, `cast_known_ratio`, `cast_non_cinema_ratio`. Clamps out-of-schema LLM values on load.
- **director_encode.py** — `encode_director_features()`. 2 features per film: `director_tier` (ordinal 0–3), `director_market` (string). Takes first-billed director from comma-separated `director` column.
- **encode_helper.py** — `EncHelper`: distributor lookup, language group, franchise/IP/adaptation inference. Guards against missing columns (`people`, `subgenres`, `themes`) — these are disabled in `prompts_v2.yaml` but `encode_helper.py` was written against `prompts.yaml`.
- **film_synop_encode.py** — sklearn transformers: `DynamicTopNAndPCA`, `TopNTokenMapper`, `TopNMultiHotWithOther`.

**Prompts:**
- **prompts_v2.yaml** — Current prompt set. Disabled: `characters` (→`people`), `themes`, `setting_time`, `conflict_type`. Active tasks: `is_franchise`, `language_cues`, `is_sequel`, `intellectual_property`, `genres`, `protagonist_archetype`, `primary_audience`, `tone`, `ip_strength`, `adaptation_type`, `narrative_scope`.
- **cast_prompts.yaml** — Actor profile: `fame_source`, `fame_tier`, `primary_market`, `cross_media`, `age_range`.
- **director_prompts.yaml** — Director profile: `director_tier` (top_tier/established/emerging/unknown), `primary_market`.
- **film_meta_prompts.yaml** — Single `film_meta` task. Returns the full film bundle (title, release_date, director, writers, genres, description, budget_usd, budget_source, studios[order/name/role], cast[order/actor/character], trailers[type/channel/date/url]) in one call with web_search-grounded answers.
- **prompts.yaml** — Original prompts (retained for reference; used by old extracts).

---

## Output Files

| File | Contents |
|---|---|
| `synopsis_v2/synopses_extracted.parquet` | Raw LLM synopsis features, one row per film_id |
| `cast_meta/cast_enriched.parquet` | Actor profiles, one row per actor_name (uppercase) |
| `cast_meta/cast_features.parquet` | Film-level cast aggregates (6 features), one row per film_id |
| `director_meta/director_enriched.parquet` | Director profiles, one row per director_name |
| `director_meta/director_features.parquet` | Film-level director features (2 features), one row per film_id |
| `film_meta/film_meta_enriched.parquet` | Web-grounded film bundle (studios/cast/genres/budget/trailers), one row per film_id |
| `tmdb_meta/studio_tier.parquet` | DEPRECATED — TMDB-sourced studio tier (do not use for new runs) |
| `{data_dir}/encoded_film_meta/{date}/llama_features.parquet` | Encoded synopsis features (105 cols) for box office model |

---

## External Dependencies (not in this repo)

- **`films` module** — `films.sql.SQL_FILM_DETAILS` for title lookups
- **Snowflake table** — `ENT_FORECAST_PRD.CURATED.FILM_SYNOPSES_EXTRACTED`; role `ent_forecast_owner`
- **Raw parquets** — `{data_dir}/raw_from_snowflake/*/train/train_raw_ds.parquet` — required by cast/director encoders for actor_list/director joins

---

## Environment Variables

- `OPENAI_KEY` — OpenAI API key, loaded via `.env`
- `TMDB_KEY` — TMDB API key, loaded via `.env`
- Snowflake RSA key: `/Users/jonathanchapman/Documents/git/rsa_key.p8`
- Snowflake warehouse: `AA_M_WH`, database `EDW_ENT_PRD`, schema `CURATED`

---

## Non-Obvious Behaviours

- **Deduplication keeps `first`** — existing rows win on re-run. Use `--force-synopsis` / `--force-cast` to re-extract.
- **Actor/director matching is case-sensitive after normalisation** — `cast_enriched` stores names uppercase; `cast_encode.py` uppercases `actor_list` before lookup. `director_enriched` preserves original casing; `director_encode.py` matches as-is.
- **`|AND` artifact in actor_list** — SQL normalises commas→pipes but leaves "AND" as a name prefix (e.g. `AND YOON KYE-SANG`). Stripped with `re.compile(r'^AND\s+', re.IGNORECASE)` in both `main.py` and `cast_encode.py`.
- **LLM field clamping happens at encode time** — invalid values (e.g. `primary_market: "philippine"`, `fame_tier: "astAR"`) are clamped to valid sets in `cast_encode.py` and `director_encode.py`, not during extraction. The raw checkpoint JSONs retain original LLM output.
- **`encode_helper.py` guards against missing columns** — `people`, `subgenres`, `themes` are disabled in `prompts_v2.yaml` but still referenced in `EncHelper`. All access uses `getattr(row, col, [])` or `if col not in df.columns: continue`.
- **Director hit rate ~70%** — regional/indie directors not in LLM training data return `unknown` (tier=0). Expected; does not indicate a pipeline error.
- **TMDB studio tier has high unknown rate for recent films** — LLM training cutoff means films released close to the current date often return `unknown`. Use `tmdb_fetch.py` (TMDB API) for reliable studio data.
- **`flatten_extraction` collision handling** — if two tasks produce the same key, the second is namespaced as `{task_name}__{key}`.
- **LiteLLM async fallbacks are disabled** — passing `fallbacks` to `litellm.acompletion` causes `fallback_utils.py` to intercept 429s and retry on the fallback model (same org TPM limit), crashing before the custom retry loop. Fallbacks are omitted from async kwargs in `extractor.py`.
- **Hardcoded absolute paths** in `main.py` (`evt_back_up/base` sys.path insert) — update if the repo moves.
