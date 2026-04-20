import gc
from typing import Iterable, Any, Mapping
import time
from pathlib import Path

from llama_cpp import Callable, Llama
import pandas as pd
from synopses.models import MODELS
from synopses.extraction import ExtractionTask, extract_json, flatten_extraction
from synopses.load_prompts import load_tasks_from_yaml
from films.main import get_films_sources

# SYNOPSES_EXTRACTED_PATH = Path('synopses', 'source_data', 'synopses_extracted.parquet')
SYNOPSES_EXTRACTED_PATH = Path('synopses', 'outputs', 'synopses_extracted.parquet')


class LlmJsonExtractor:
    def __init__(
        self,
        llm: Llama,
        tasks: Mapping[str, ExtractionTask],
        postprocessors: Mapping[str, Callable[[dict[str, Any]], dict[str, Any]]] | None = None,
    ):
        """
        llm: configured Llama instance
        tasks: mapping of task_name -> ExtractionTask (loaded from YAML)
        postprocessors: mapping of postprocess_name -> callable(payload) -> payload
        """
        self.llm = llm
        self.tasks = dict(tasks)
        self.postprocessors = dict(postprocessors or {})

    @staticmethod
    def _build_user_content(title: str, synopsis: str) -> str:
        # Title as a separate, explicit field for better context
        return f'Title: "{title.strip()}"\n\nSynopsis:\n{synopsis.strip()}'

    @staticmethod
    def merge_genres(df_dest: pd.DataFrame, df_src: pd.DataFrame) -> pd.DataFrame:
        # 1) build a minimal source lookup keyed by film_id
        src = df_src[["film_id", "genre_1", "genre_2", "genre_3"]].copy()

        for c in ["genre_1", "genre_2", "genre_3"]:
            src[c] = (
                src[c]
                .astype("string")
                .str.strip()
                .str.lower()
                .where(lambda s: s.notna() & (s != ""))
            )

        src = src.drop_duplicates("film_id")
        src_map = src.set_index("film_id")

        # --- PATCH: prepend missing genres only ---

        def _patch_genres(row):
            dest_genres = row["genres"]          # already a list
            dest_subgenres = row["subgenres"]    # already a list

            universe = set(dest_genres) | set(dest_subgenres)

            try:
                s = src_map.loc[row["film_id"]]
            except KeyError:
                return dest_genres

            src_genres = [
                g for g in (s["genre_1"], s["genre_2"], s["genre_3"])
                if pd.notna(g)
            ]

            to_prepend = [g for g in src_genres if g not in universe]

            # prepend, preserve existing, no duplicates
            return to_prepend + [g for g in dest_genres if g not in to_prepend]

        df_dest["genres"] = df_dest.apply(_patch_genres, axis=1)

        return df_dest

    def _run_task_for_synopsis(
        self,
        task: ExtractionTask,
        title: str,
        synopsis: str,
    ) -> dict[str, Any]:
        messages = [
            {"role": "system", "content": task.system_prompt},
            {
                "role": "user",
                "content": self._build_user_content(title=title, synopsis=synopsis),
            },
        ]

        resp = self.llm.create_chat_completion(
            messages=messages,
            temperature=task.temperature,
            top_p=task.top_p,
            max_tokens=task.max_tokens,
            repeat_penalty=task.repeat_penalty,
        )

        raw = resp["choices"][0]["message"]["content"]

        try:
            data = extract_json(raw)
        except ValueError as e:
            # Don't crash the pipeline: return an error payload for this task
            return {
                "_error": "json_parse_error",
                "_error_message": str(e),
                "_raw_output": raw,
            }

        # Optional postprocessing
        if task.postprocess:
            fn = self.postprocessors.get(task.postprocess)
            if fn is not None:
                try:
                    data = fn(data)
                except Exception as e:
                    # Again, don't kill the pipeline if a cleaner explodes
                    return {
                        "_error": "postprocess_error",
                        "_error_message": str(e),
                        "_raw_output": raw,
                    }

        return data

    def run_single_synopsis(
        self,
        title: str,
        synopsis: str,
        task_names: Iterable[str] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """
        Run all (or a subset of) extraction tasks on a single film.

        Returns:
            dict[task_name] -> parsed JSON payload for that task.
        """
        if task_names is None:
            task_names = self.tasks.keys()

        results: dict[str, dict[str, Any]] = {}
        for name in task_names:
            task = self.tasks[name]
            results[name] = self._run_task_for_synopsis(task, title, synopsis)

        return results

    def run_multiple_synopses(
        self,
        df: pd.DataFrame,
        id_col: str = "film_id",
        title_col: str = "title",
        synopsis_col: str = "synopsis",
        task_names: Iterable[str] | None = None,
        flatten: bool = True,
    ) -> dict[str, dict[str, Any]]:
        """
        Run extraction tasks for multiple films in a DataFrame.

        Args:
            df: DataFrame with at least id_col, title_col, synopsis_col.
            id_col: column name for unique film identifier.
            title_col: column with film titles.
            synopsis_col: column with synopses.
            task_names: optional subset of tasks to run.
            flatten: if True, merge all task outputs directly into the film dict.
                    if False, keep the old nested structure under "results".

        Returns:
            If flatten=True:
                {
                film_id_1: {
                    "title": ...,
                    "synopsis": ...,
                    <merged task keys: e.g. is_franchise, characters, themes, ...>
                },
                film_id_2: { ... },
                ...
                }

            If flatten=False:
                {
                film_id_1: {
                    "title": ...,
                    "synopsis": ...,
                    "results": {
                        "franchise_sequel": {...},
                        "characters": {...},
                        ...
                    }
                },
                ...
                }
        """
        if task_names is None:
            task_names = self.tasks.keys()

        task_names = list(task_names)

        output: dict[str, dict[str, Any]] = {}

        synopses_len = len(df)
        task_len = len(task_names)

        # Handle trivial case early
        if synopses_len == 0 or task_len == 0:
            return output

        total_task_calls = synopses_len * task_len
        start_time = time.time()

        def _fmt_secs(seconds: float) -> str:
            """Format seconds as H:MM:SS or M:SS for nicer printing."""
            seconds = int(seconds)
            h, rem = divmod(seconds, 3600)
            m, s = divmod(rem, 60)
            if h:
                return f"{h:d}:{m:02d}:{s:02d}"
            else:
                return f"{m:d}:{s:02d}"

        for sidx, row in enumerate(df.itertuples(index=False), start=1):
            film_id = getattr(row, id_col)
            title = getattr(row, title_col)
            synopsis = getattr(row, synopsis_col)

            if not isinstance(synopsis, str) or len(str(synopsis)) < 5:
                continue

            film_results: dict[str, dict[str, Any]] = {}
            for tidx, name in enumerate(task_names, start=1):
                # Progress counters
                completed_task_calls = (sidx - 1) * task_len + tidx
                elapsed = time.time() - start_time

                # Averages
                avg_per_task = elapsed / completed_task_calls
                # Convert tasks completed to "synopsis-equivalents" (tasks per synopsis)
                effective_synopses_done = completed_task_calls / task_len
                avg_per_synopsis = elapsed / effective_synopses_done

                # ETA
                remaining_task_calls = total_task_calls - completed_task_calls
                eta = remaining_task_calls * avg_per_task

                # Percentages
                synopsis_pct = sidx / synopses_len
                task_pct = tidx / task_len

                print(
                    (
                        f"Extracting - "
                        f"Synopses: {synopsis_pct:.1%} ({sidx}/{synopses_len}) | "
                        f"Tasks: {task_pct:.1%} ({tidx}/{task_len}) | "
                        f"Elapsed: {_fmt_secs(elapsed)} | "
                        f"Avg/task: {avg_per_task:.1f}s | "
                        f"Avg/synopsis: {avg_per_synopsis:.1f}s | "
                        f"ETA: {_fmt_secs(eta)}"
                        "      "
                    ),
                    end="\r",
                    flush=True,
                )

                task = self.tasks[name]
                film_results[name] = self._run_task_for_synopsis(task, title, synopsis)

            if flatten:
                merged = flatten_extraction(film_results)
                film_record: dict[str, Any] = {
                    "film_id": film_id,
                    "title": title,
                    "synopsis": synopsis,
                    **merged,
                }
            else:
                film_record = {
                    "film_id": film_id,
                    "title": title,
                    "synopsis": synopsis,
                    "results": film_results,
                }

            output[film_id] = film_record

        # Print a final newline so the last progress line doesn't overwrite the next log
        total_elapsed = time.time() - start_time
        print(f"\nDone. Total time: {_fmt_secs(total_elapsed)}")

        return output


def main(df_films: pd.DataFrame, sample_size: int = 100, sample_head: bool = True) -> None:
    # model = MODELS.get('phi-3.5')
    model = MODELS.get('llama-3.1')
    tasks = load_tasks_from_yaml(Path('synopses', "prompts.yaml"))

    df_new = df_films.loc[df_films['synopsis'].str.len() >= 5].copy(deep=True)

    if SYNOPSES_EXTRACTED_PATH.exists():
        df_exisitng = pd.read_parquet(SYNOPSES_EXTRACTED_PATH)
        existing_ids = df_exisitng['film_id'].tolist()
        df_new_diff = df_new.loc[~df_new['film_id'].isin(existing_ids)]
    else:
        df_exisitng = pd.DataFrame()
        df_new_diff = df_new.copy()

    if sample_size == 0:
        sample_kwarg = {'n': len(df_new_diff)}
    elif 1 > sample_size > 0:
        sample_kwarg = {'n': round(len(df_new_diff) * sample_size, 0)}
    else:
        sample_kwarg = {'n': sample_size}

    if sample_head:
        df_sampled = df_new_diff.sort_values(by='film_open_date', ascending=False).head(**sample_kwarg)
    else:
        df_sampled = df_new_diff.sample(**sample_kwarg)

    print(f'Input | Imported: {len(df_films)} | Synopsis > 5: {len(df_new)} | New: {len(df_new_diff)} | Sampled {len(df_sampled)}')

    if df_sampled.empty:
        print('No new titles with valid synopses found')
        return

    print(f'Extracting meta from {len(df_sampled)} titles')

    llm = Llama.from_pretrained(
        repo_id=model.get('repo_id'),
        filename=model.get('filename'),
        n_ctx=2048,
        n_threads=8,
        verbose=False,
        seed=42,
        n_gpu_layers=model.get('n_gpu_layers', 0),
        main_gpu=1,
        n_batch=256,
    )

    extractor = LlmJsonExtractor(
        llm=llm,
        tasks=tasks,
    )

    multi_results = extractor.run_multiple_synopses(df_sampled, id_col="film_id", title_col="film_title", synopsis_col="synopsis")
    df_extraction = pd.DataFrame.from_dict(multi_results, orient='index')

    if '_error' in df_extraction.columns:
        df_extraction = df_extraction.loc[df_extraction['_error']]

    if not df_exisitng.empty:
        df_result = pd.concat([df_exisitng, df_extraction]).drop_duplicates(subset='film_id', keep='first')
    else:
        df_result = df_extraction

    out_df = extractor.merge_genres(df_result, df_sampled)

    out_df.to_parquet(SYNOPSES_EXTRACTED_PATH, index=False)
    out_df.to_excel(SYNOPSES_EXTRACTED_PATH.with_suffix('.xlsx'), index=False)
    print(f'Output saved | Before: {len(df_exisitng)} | After: {len(out_df)} | New {len(df_extraction)}')

    llm.close()
    del extractor
    gc.collect()


if __name__ == '__main__':
    df_films = get_films_sources(persisted=False)
    df_films = df_films.loc[df_films['film_nat_open_date'].between('2026-06-01', '2026-08-01', inclusive='both')]
    # print(df_films)
    main(df_films, sample_size=0)
