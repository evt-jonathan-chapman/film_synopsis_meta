"""
Compare synopsis extraction quality across all configured models.

Output: synopses/outputs/model_comparison.xlsx
        One row per (film x model). Columns: model | film_title | synopsis | <extracted fields>

Console: error rate, avg field counts, timing, cross-model Jaccard similarity.

Usage:
    from compare_models import run_comparison
    run_comparison(n_films=10)

Or directly:
    python compare_models.py
"""
import asyncio
import gc
import os
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from llama_cpp import Llama

from films.main import get_films_sources
from load_prompts import load_tasks_from_yaml
from extractor import LlmJsonExtractor
from models import MODELS

PROMPTS_PATH = Path('prompts', 'prompts.yaml')
OUTPUT_PATH = Path('synopses', 'outputs', 'model_comparison.xlsx')

LIST_FIELDS = [
    'themes', 'genres', 'subgenres', 'setting_types', 'time_periods',
    'protagonist_archetypes', 'tone', 'language_cues', 'secondary_audiences',
    'people', 'intellectual_property',
]


# ── Output helpers ────────────────────────────────────────────────────────────

def _fmt(value) -> str | bool:
    """Flatten lists to sorted comma-separated strings; pass scalars through.
    Booleans are returned as-is so they write as proper Excel booleans."""
    if isinstance(value, bool):
        return value
    if isinstance(value, list):
        return ', '.join(sorted(str(v) for v in value))
    if value is None:
        return ''
    return str(value)


def _normalise_bool_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Convert string-encoded bool columns ('True'/'False'/'TRUE'/'FALSE') to Python bool."""
    _bool_strings = {'true', 'false'}
    for col in df.columns:
        if df[col].dtype == object:
            non_null = df[col].dropna()
            if len(non_null) > 0 and set(non_null.astype(str).str.lower().unique()) <= _bool_strings:
                df[col] = df[col].map(lambda x: str(x).lower() == 'true' if pd.notna(x) else x)
    return df


def _results_to_rows(model_name: str, results: dict, df_films: pd.DataFrame, run_timestamp: str) -> list[dict]:
    """One Excel row per film for a given model."""
    lookup_cols = ['film_title', 'synopsis'] + (['alt_synopsis'] if 'alt_synopsis' in df_films.columns else [])
    film_lookup = df_films.set_index('film_id')[lookup_cols].to_dict('index')
    rows = []
    for film_id, rec in results.items():
        info = film_lookup.get(film_id, {})

        # Metrics: total extracted list items and whether extraction errored
        has_error = '_error' in rec
        total_list_items = sum(len(rec.get(f) or []) for f in LIST_FIELDS) if not has_error else 0

        alt = info.get('alt_synopsis')
        n_synopses = 2 if (isinstance(alt, str) and alt.strip()) else 1

        row = {
            'run_timestamp': run_timestamp,
            'model': model_name,
            'film_id': film_id,
            'film_title': info.get('film_title', rec.get('title', '')),
            'synopsis': (
                info.get('synopsis', rec.get('synopsis', ''))
                if not (isinstance(info.get('alt_synopsis'), str) and info['alt_synopsis'].strip())
                else f"[1] {info['synopsis']}\n\n[2] {info['alt_synopsis']}"
            ),
            'metric_total_items': total_list_items,
            'metric_error': has_error,
            'n_synopses': n_synopses,
        }
        for k, v in rec.items():
            if k in ('film_id', 'title', 'synopsis'):
                continue
            row[k] = _fmt(v)
        rows.append(row)
    return rows


def _build_summary_rows(
    models_to_run: list[str],
    all_results: dict[str, dict],
    timings: dict[str, float],
    token_usages: dict[str, dict | None],
    run_timestamp: str,
) -> list[dict]:
    """One summary row per model for the current run."""
    rows = []
    for model_name in models_to_run:
        results = all_results[model_name]
        n = len(results)
        errors = sum(1 for r in results.values() if '_error' in r)
        avg_items = sum(
            sum(len(r.get(f) or []) for f in LIST_FIELDS)
            for r in results.values()
        ) / max(n, 1)
        elapsed = timings[model_name]
        usage = token_usages.get(model_name) or {}
        row = {
            'run_timestamp': run_timestamp,
            'model': model_name,
            'n_films': n,
            'n_errors': errors,
            'avg_list_items': round(avg_items, 1),
            'elapsed_s': round(elapsed, 1),
            'elapsed_per_film_s': round(elapsed / max(n, 1), 1),
            'prompt_tokens': usage.get('prompt_tokens') or None,
            'completion_tokens': usage.get('completion_tokens') or None,
            'cost_usd': round(usage['cost_usd'], 6) if usage.get('cost_usd') else None,
        }
        rows.append(row)
    return rows


# ── Diagnostics ───────────────────────────────────────────────────────────────

def _print_model_summary(model_name: str, results: dict, elapsed: float) -> None:
    n = len(results)
    errors = sum(1 for r in results.values() if '_error' in r)
    avg_items = sum(
        sum(len(r.get(f) or []) for f in LIST_FIELDS)
        for r in results.values()
    ) / max(n, 1)
    print(
        f'  {model_name:20s} | Films: {n} | Errors: {errors} | '
        f'Avg list items: {avg_items:.1f} | '
        f'Time: {elapsed:.0f}s ({elapsed / max(n, 1):.1f}s/film)'
    )


def _jaccard(a: set, b: set) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def _print_variation(all_results: dict[str, dict], film_ids: list) -> None:
    model_names = list(all_results.keys())
    if len(model_names) < 2:
        return
    print('\nAvg Jaccard similarity between model pairs across list fields (1.0 = identical):')
    for i, m1 in enumerate(model_names):
        for m2 in model_names[i + 1:]:
            scores = []
            for fid in film_ids:
                r1 = all_results[m1].get(fid, {})
                r2 = all_results[m2].get(fid, {})
                for field in LIST_FIELDS:
                    s1 = {v.lower() for v in (r1.get(field) or [])}
                    s2 = {v.lower() for v in (r2.get(field) or [])}
                    scores.append(_jaccard(s1, s2))
            avg = sum(scores) / len(scores) if scores else 0.0
            print(f'  {m1} vs {m2}: {avg:.3f}')


# ── Entry point ───────────────────────────────────────────────────────────────

def run_comparison(
    n_films: int = 10,
    models_to_run: list[str] | None = None,
    film_ids: list | None = None,
    film_source_persisted: bool = True
) -> None:
    load_dotenv()
    api_key = os.getenv('OPENAI_KEY')
    tasks = load_tasks_from_yaml(PROMPTS_PATH)

    if models_to_run is None:
        models_to_run = list(MODELS.keys())

    df_films = get_films_sources(persisted=film_source_persisted)
    df_valid = df_films.loc[df_films['synopsis'].str.len() >= 50].dropna(subset=['synopsis'])

    # Build sample: pinned IDs first, then random top-up to reach n_films
    pinned_ids = list(film_ids or [])
    if pinned_ids:
        df_pinned = df_valid.loc[df_valid['film_id'].isin(pinned_ids)]
        missing = [fid for fid in pinned_ids if fid not in df_valid['film_id'].values]
        if missing:
            print(f'Warning: film_ids not found in valid films (no synopsis?): {missing}')
        remaining = max(0, n_films - len(df_pinned))
        df_pool = df_valid.loc[~df_valid['film_id'].isin(pinned_ids)]
        df_random = df_pool.sample(n=min(remaining, len(df_pool)))
        df_sample = pd.concat([df_pinned, df_random], ignore_index=True)
    else:
        df_sample = df_valid.sample(n=min(n_films, len(df_valid)))

    sample_cols = ['film_id', 'film_title', 'synopsis']
    if 'alt_synopsis' in df_valid.columns:
        sample_cols.append('alt_synopsis')
    df_sample = df_sample[sample_cols]
    sampled_film_ids = df_sample['film_id'].tolist()
    print(f'Sampled {len(df_sample)} films for comparison')

    run_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    all_results: dict[str, dict] = {}
    timings: dict[str, float] = {}
    token_usages: dict[str, dict | None] = {}
    all_rows: list[dict] = []

    for model_name in models_to_run:
        model_cfg = MODELS[model_name]
        provider = model_cfg['provider']
        print(f'\n{"─" * 55}')
        print(f'Model: {model_name}  ({provider})')
        print(f'{"─" * 55}')

        t0 = time.time()

        if provider == 'llama_cpp':
            llm = Llama.from_pretrained(
                repo_id=model_cfg['repo_id'],
                filename=model_cfg['filename'],
                n_ctx=2048,
                n_threads=8,
                verbose=False,
                seed=42,
                n_gpu_layers=model_cfg.get('n_gpu_layers', 0),
                main_gpu=model_cfg.get('main_gpu', 0),
                n_batch=256,
            )
            extractor = LlmJsonExtractor(tasks=tasks, llm=llm)
            results = extractor.run_multiple_synopses(
                df_sample, id_col='film_id', title_col='film_title', synopsis_col='synopsis', alt_synopsis_col='alt_synopsis'
            )
            llm.close()
        else:
            extractor = LlmJsonExtractor(
                tasks=tasks,
                model=model_cfg['model'],
                api_key=api_key,
                cost_per_1m_input=model_cfg.get('cost_per_1m_input'),
                cost_per_1m_output=model_cfg.get('cost_per_1m_output'),
            )
            results = asyncio.run(extractor.arun_multiple_synopses(
                df_sample, id_col='film_id', title_col='film_title', synopsis_col='synopsis', alt_synopsis_col='alt_synopsis'
            ))

        elapsed = time.time() - t0
        timings[model_name] = elapsed
        all_results[model_name] = results
        token_usages[model_name] = extractor.token_usage
        all_rows.extend(_results_to_rows(model_name, results, df_sample, run_timestamp))

        del extractor
        gc.collect()

    # Console diagnostics
    print(f'\n{"═" * 55}')
    print('SUMMARY')
    print(f'{"═" * 55}')
    for model_name in models_to_run:
        _print_model_summary(model_name, all_results[model_name], timings[model_name])

    _print_variation(all_results, sampled_film_ids)

    # Build summary rows for this run
    df_new_summary = pd.DataFrame(_build_summary_rows(models_to_run, all_results, timings, token_usages, run_timestamp))

    # Excel — append to existing file, deduplicating results on (model, film_id); summary is append-only
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_new_results = pd.DataFrame(all_rows)

    if OUTPUT_PATH.exists():
        try:
            df_existing_results = _normalise_bool_cols(pd.read_excel(OUTPUT_PATH, sheet_name='results', engine='openpyxl'))
        except Exception:
            # Pre-migration file has no named sheets — treat as empty to trigger fresh write
            df_existing_results = pd.DataFrame()
        try:
            df_existing_summary = pd.read_excel(OUTPUT_PATH, sheet_name='summary', engine='openpyxl')
        except Exception:
            df_existing_summary = pd.DataFrame()

        if df_existing_results.empty:
            df_out_results = df_new_results.sort_values(['run_timestamp', 'film_title', 'model'], ascending=[False, True, True]).reset_index(drop=True)
            n_appended = len(df_out_results)
        else:
            # Align columns then deduplicate on (model, film_id)
            all_cols = list(df_existing_results.columns) + [
                c for c in df_new_results.columns if c not in df_existing_results.columns
            ]
            df_combined = pd.concat(
                [df_existing_results.reindex(columns=all_cols), df_new_results.reindex(columns=all_cols)],
                ignore_index=True,
            )
            df_out_results = (
                df_combined
                .drop_duplicates(subset=['model', 'film_id'], keep='last')
                .sort_values(['run_timestamp', 'film_title', 'model'], ascending=[False, True, True])
                .reset_index(drop=True)
            )
            n_appended = len(df_out_results) - len(df_existing_results)

        # Summary: pure append (each run_timestamp is unique)
        df_out_summary = pd.concat([df_existing_summary, df_new_summary], ignore_index=True)
    else:
        df_out_results = df_new_results.sort_values(['run_timestamp', 'film_title', 'model'], ascending=[False, True, True]).reset_index(drop=True)
        df_out_summary = df_new_summary
        n_appended = len(df_out_results)

    with pd.ExcelWriter(OUTPUT_PATH, engine='openpyxl') as writer:
        df_out_results.to_excel(writer, sheet_name='results', index=False)
        df_out_summary.to_excel(writer, sheet_name='summary', index=False)

    print(f'\nSaved → {OUTPUT_PATH}  ({n_appended} new rows, {len(df_out_results)} total)')


if __name__ == '__main__':
    models = ['llama-3.1', 'gpt-4.1-nano', 'gpt-4o-mini', 'gpt-5.4-nano']
    run_comparison(n_films=25, models_to_run=models, film_source_persisted=True)
