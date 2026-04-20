# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

LLM-based pipeline that extracts structured metadata from film synopses (themes, genres, protagonists, tone, franchise status, etc.) using OpenAI API calls via LiteLLM. This is the **replacement** for the old local-Llama pipeline at `evt_back_up/synopses` — same extraction logic, ChatGPT API instead of downloaded model.

**Downstream dependency:** Output (`synopses_extracted.parquet`) feeds `encode_llm_features.py` in the box office model (`evt_back_up/box_office`). After any extraction run, re-run `encode_llm_features.py` → `build_data.py` → `model_train.py` before the new features affect predictions.

**Box office project context** (full notes in `/Users/jonathanchapman/Documents/data/claude_chats/CLAUDE.md`):
- Box office model: PyTorch MLP ensemble, predicts NZ cinema admissions weeks 1–4
- LLM features are merged via `inner` join in `build_train.py` line 833 — films with no synopsis entry are silently dropped from training. Prefer `left` join + 0-fill to avoid coverage gaps.
- Hard overrides for specific film_ids live in `encode_llm_features.py` (IP strength, genres, audience, sequel flags) — change there and re-run encode before retraining.

## Running the Pipeline

```bash
# Full extraction run (edit date range / sample_size in __main__ block first)
python main.py

# Sync local parquet ↔ Snowflake
python ingest.py

# Benchmark multiple models on a sample
python compare_models.py

# Analyse title regex patterns, output title_patterns.xlsx
python analyse_title_patterns.py
```

There is no test suite, linting config, or `requirements.txt`. Dependencies are managed externally.

## Architecture

### Pipeline flow

```
get_films_sources()          # external `films` module → DataFrame
        ↓
title_cleaner.clean_title_for_llm()   # strip festival codes, format tags, language suffixes
        ↓
load_prompts.py → prompts.yaml        # 11 ExtractionTask objects (one per field group)
        ↓
LlmJsonExtractor (main.py)
  ├── API path:   litellm.completion / acompletion (async, max_concurrency=20)
  └── Local path: llama_cpp.create_chat_completion (sequential)
        ↓
extraction.extract_json()             # parse JSON from raw LLM output
extraction.flatten_extraction()       # merge per-task dicts into single film record
        ↓
Merge with existing parquet (dedupe on film_id)
        ↓
synopses/outputs/synopses_extracted.parquet + .xlsx
        ↓
ingest.py → Snowflake staging table + MERGE
```

### Key modules

- **extractor.py** — `LlmJsonExtractor` class; `run_multiple_synopses` (sync) and `arun_multiple_synopses` (async); cost tracking; genre merging from source data.
- **main.py** — Orchestration only: loads config, filters films, calls extractor, merges with existing parquet, saves output.
- **models.py** — LLM registry (`MODELS` dict) with pricing and inference params. `DEFAULT_MODEL = 'gpt-4.1-nano'`, fallback chain to `gpt-4o-mini`.
- **extraction.py** — `ExtractionTask` dataclass; `extract_json()` for robust JSON parsing from LLM output; `flatten_extraction()` for merging task results.
- **prompts.yaml** — All LLM system prompts. Each task has `enabled`, `temperature`, `max_tokens`, `system_prompt`. Edit here to tune extraction behaviour.
- **title_cleaner.py** — Regex preprocessing to remove noise before LLM sees the title (`clean_title_for_llm()`, `tag_title()`).
- **ingest.py** — Snowflake sync: `get_synopses_differences()`, `ingest_local_to_snowflake()`, `persist_snowflake_to_local()`. Uses staging table + MERGE for idempotent writes.
- **config.py** — `SYNOPSES_EXTRACTED_PATH` (output parquet location).

### External dependencies (not in this repo)

- **`films` module** — `films.main.get_films_sources()` returns the input DataFrame (film_id, film_title, synopsis, film_nat_open_date, etc.)
- **`tools` module** — `tools.connections.SnowflakeDB` connection manager
- **Snowflake table** — `ENT_FORECAST_PRD.CURATED.FILM_SYNOPSES_EXTRACTED`; role `ent_forecast_owner`

### Environment variables

- `OPENAI_KEY` — loaded via `.env` (python-dotenv)
- `SNOWFLAKE_*` — Snowflake credentials via `tools.connections`

### Output schema (flattened per-film record)

Array fields (`themes`, `genres`, `subgenres`, `people`, `tone`, `setting_types`, `time_periods`, `protagonist_archetypes`, `secondary_audiences`, `language_cues`, `intellectual_property`) are Python lists locally and Snowflake VARIANT JSON in the warehouse. Boolean fields: `is_franchise`, `is_sequel`. String fields: `protagonist_type`, `primary_audience`.

Error fields `_error`, `_error_message`, `_raw_output` are set per task when JSON parsing or postprocessing fails.

### Controlling the run

Parameters are set directly in the `if __name__ == '__main__'` block of each script rather than via CLI flags:

```python
# main.py — filter by date, set sample size
df_films = df_films.loc[df_films['film_nat_open_date'].between('2026-04-15', '2026-04-30')]
main(df_films, sample_size=0)  # 0 = all new films

# compare_models.py — choose models and sample size
run_comparison(n_films=25, models_to_run=['gpt-4.1-nano', 'gpt-4o-mini'])
```
