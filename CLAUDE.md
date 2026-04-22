# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

LLM-based pipeline that extracts structured metadata from film synopses and cast lists, then encodes them into feature parquets for the EVT box office model and other downstream projects.

This is the **canonical home for all film metadata extraction and encoding**. The box office model (`evt_back_up/box_office`) reads output parquets only — it does not import code from here.

**Box office project context** (full notes in `/Users/jonathanchapman/Documents/data/claude_chats/CLAUDE.md`):
- Box office model: PyTorch MLP ensemble, predicts NZ cinema admissions weeks 1–4
- LLM features are merged via `inner` join in `build_train.py` — films with no synopsis entry are silently dropped from training. Prefer `left` join + 0-fill to avoid coverage gaps.
- Hard overrides for specific film_ids live in `encode_synopsis.py` — change there and re-run encode before retraining.

---

## Running the Pipeline

```bash
# Weekly refresh — checks Snowflake for changes, re-extracts and re-encodes only what changed
python refresh.py
python refresh.py --force-synopsis   # re-extract all synopses
python refresh.py --force-cast       # re-enrich all actors
python refresh.py --force-encode     # re-encode even if nothing changed

# One-off synopsis extraction (edit date range / sample params in main.py first)
python main.py

# Cast enrichment (run after raw parquets are available, validate with sample_size=50 first)
python cast_main.py

# Encode synopsis features to llama_features.parquet
python encode_synopsis.py

# Encode cast features to cast_features.parquet
python cast_encode.py

# Sync local parquet ↔ Snowflake
python ingest.py

# Benchmark multiple models on a sample
python compare_models.py
```

There is no test suite, linting config, or `requirements.txt`. Dependencies are managed externally.

---

## Architecture

### Pipeline flow

```
Snowflake (AA_M_WH, EDW_ENT_PRD.CURATED)
  │
  ├─ Synopsis path ─────────────────────────────────────────
  │   refresh.py → _find_new_synopsis_films()         # diff: new film_ids + changed synopsis text
  │       ↓
  │   LlmJsonExtractor.arun_multiple_synopses()       # async, max_concurrency=20
  │   prompts_v2.yaml → ExtractionTask objects
  │       ↓
  │   synopses/outputs/synopses_extracted.parquet     # merge: new rows win on dedupe
  │       ↓
  │   encode_synopsis.py → encode_synopsis_features()
  │   (applies hard overrides, EncHelper enrichment, OneHot/MultiHot/PCA encoding)
  │       ↓
  │   {data_dir}/encoded_film_meta/{date}/llama_features.parquet
  │
  └─ Cast path ─────────────────────────────────────────────
      refresh.py → _find_new_actors()                # diff: actors not in cast_enriched
          ↓
      LlmJsonExtractor.arun_multiple_synopses()      # actor_name as both id and synopsis
      cast_prompts.yaml → actor_profile task
          ↓
      cast_meta/outputs/cast_enriched.parquet        # one row per actor
          ↓
      cast_encode.py → encode_cast_features()
      (join to raw parquets on actor_list, aggregate film-level features)
          ↓
      cast_meta/outputs/cast_features.parquet        # one row per film_id
```

### Key modules

**Extraction:**
- **extractor.py** — `LlmJsonExtractor` class; `run_multiple_synopses` (sync) and `arun_multiple_synopses` (async, `max_concurrency=20`); cost tracking; genre merging.
- **main.py** — One-off synopsis extraction: loads config, filters films, calls extractor, merges with existing parquet, saves output.
- **refresh.py** — Weekly orchestrator. Diffs Snowflake vs local parquets (synopsis text + actor names), re-extracts and re-encodes only what changed. `run_refresh()` is the Dagster entry point.
- **cast_main.py** — One-off cast enrichment: loads unique actors from raw parquets (`min_film_count=3`), async extraction, checkpoint/resume support.
- **models.py** — LLM registry (`MODELS` dict). `DEFAULT_MODEL = 'gpt-4.1-nano'`, `DEFAULT_FALLBACKS = [{'model': 'gpt-4o-mini'}]`.
- **extraction.py** — `ExtractionTask` dataclass; `extract_json()` for robust JSON parsing; `flatten_extraction()` for merging task results.
- **load_prompts.py** — loads YAML prompt files into `ExtractionTask` objects.
- **title_cleaner.py** — Regex preprocessing to remove noise before LLM sees the title.
- **ingest.py** — Snowflake sync: `get_synopses_differences()`, `ingest_local_to_snowflake()`, `persist_snowflake_to_local()`. Staging table + MERGE for idempotent writes.
- **config.py** — `SYNOPSES_EXTRACTED_PATH` (output parquet location).

**Encoding:**
- **encode_synopsis.py** — `encode_synopsis_features(synopses_path, out_date)`. Applies all hard overrides (IP strength, genres, audience, sequel flags — edit here). Outputs `llama_features.parquet`.
- **cast_encode.py** — `encode_cast_features()`. Joins cast_enriched to raw parquets on pipe-separated `actor_list`, aggregates to film-level. Outputs `cast_features.parquet`.
- **encode_helper.py** — `EncHelper` class: distributor lookup tables, language group assignment, franchise/IP/adaptation inference, genre adjustment. Pure Python (re, numpy, pandas, rapidfuzz).
- **film_synop_encode.py** — sklearn custom transformers: `DynamicTopNAndPCA`, `TopNTokenMapper`, `TopNMultiHotWithOther`. Utilities: `auto_top_n()`, `return_feature_names()`, `FeatureDiagnostics`.

**Prompts:**
- **prompts.yaml** — Original prompt set (used by `main.py`). Tasks: genres, audiences, is_sequel, IP, protagonist, themes, tone, setting, characters.
- **prompts_v2.yaml** — Improved prompts (used by `refresh.py`). Disabled: characters, themes, setting_time, conflict_type. New tasks: ip_strength, adaptation_type, narrative_scope. Rewrites: is_sequel (covers remakes/reboots), intellectual_property (allows external knowledge), genres (animation + biography rules).
- **cast_prompts.yaml** — Actor enrichment prompts. Single task `actor_profile`: fame_source, fame_tier, primary_market, cross_media.

---

## Output Files

| File | Contents |
|---|---|
| `synopses/outputs/synopses_extracted.parquet` | Raw LLM-extracted synopsis features, one row per film_id |
| `cast_meta/outputs/cast_enriched.parquet` | Raw LLM-extracted actor profiles, one row per actor_name (uppercase) |
| `cast_meta/outputs/cast_features.parquet` | Film-level cast aggregates, one row per film_id |
| `{data_dir}/encoded_film_meta/{date}/llama_features.parquet` | Encoded synopsis features for box office model |

---

## External Dependencies (not in this repo)

- **`films` module** — `films.main.get_films_sources()` returns the input DataFrame
- **Snowflake table** — `ENT_FORECAST_PRD.CURATED.FILM_SYNOPSES_EXTRACTED`; role `ent_forecast_owner`
- **Raw parquets** — `{data_dir}/raw_from_snowflake/*/train/train_raw_ds.parquet` — required by `cast_encode.py` for the `actor_list` → film_id join

---

## Environment Variables

- `OPENAI_KEY` — loaded via `.env` (python-dotenv)
- Snowflake RSA key: `/Users/jonathanchapman/Documents/git/rsa_key.p8`
- Snowflake warehouse: `AA_M_WH`, database `EDW_ENT_PRD`, schema `CURATED`

---

## Controlling a Run

**Synopsis extraction (`main.py`)** — parameters are module-level (not inside `__main__`):
```python
df_films = df_films.loc[df_films['film_nat_open_date'].between('2026-04-15', '2026-04-30')]
sample_size = 0      # 0 = all, 0<x<1 = fraction, x≥1 = literal count
sample_head = True   # True = newest first; False = random
```

**Cast extraction (`cast_main.py`)** — same pattern:
```python
asyncio.run(main(actors_df, sample_size=50))  # sample_size=0 for all actors
```

---

## Non-Obvious Behaviours

- **Deduplication keeps `first`** — if a film_id already exists in the parquet, the existing (older) result wins on re-run. To force re-extraction, use `--force-synopsis`.
- **Actor matching is uppercase** — `cast_enriched.parquet` stores actor names in uppercase; `cast_encode.py` uppercases all `actor_list` values before lookup. Mixed-case entries in raw parquets are handled.
- **`|AND` artifact in actor_list** — SQL normalises commas→pipes but the word AND remains as a name prefix (e.g. `AND YOON KYE-SANG`). Both `cast_main.py` and `cast_encode.py` strip this with `re.compile(r'^AND\s+', re.IGNORECASE)`.
- **`cast_encode.py` must be run as a script, not imported** — module-level code is guarded under `if __name__ == '__main__'`. Import `encode_cast_features` safely from `refresh.py`.
- **`conflict_type` task is disabled** in both `prompts.yaml` and `prompts_v2.yaml`.
- **`flatten_extraction` collision handling** — if two tasks produce the same key with different values, the second is namespaced as `{task_name}__{key}`.
- **Films where any task errors are dropped** — `_error` set → film excluded from output parquet entirely.
- **LiteLLM async callbacks are suppressed** at import time to prevent coroutine leak warnings in sync contexts.
- **`ingest.py` VARIANT column detection** uses a 20%-threshold heuristic on a 2 000-row sample — may miss sparse array columns.
- **Hardcoded absolute paths** in `main.py` and `encode_synopsis.py` (`evt_back_up/base` sys.path insert, RSA key path, data dir) — update if the repo or data directory moves.
