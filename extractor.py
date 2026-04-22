import asyncio
import time
from typing import Iterable, Any, Mapping

import litellm
import pandas as pd

from extraction import ExtractionTask, extract_json, flatten_extraction
from title_cleaner import clean_title_for_llm


class LlmJsonExtractor:
    def __init__(
        self,
        tasks: Mapping[str, ExtractionTask],
        model: str | None = None,
        llm=None,  # llama_cpp.Llama instance for direct local inference
        fallbacks: list[dict] | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        postprocessors: Mapping[str, Any] | None = None,
        cost_per_1m_input: float | None = None,
        cost_per_1m_output: float | None = None,
    ):
        """
        Provide either:
          - model (str): LiteLLM model string for API inference (e.g. 'gpt-4.1-nano')
          - llm (Llama): a llama_cpp.Llama instance for direct local inference

        tasks: mapping of task_name -> ExtractionTask (loaded from YAML)
        fallbacks: LiteLLM fallback list, e.g. [{'model': 'gpt-4o-mini'}]
        api_key: API key for LiteLLM; if None, reads from env
        api_base: override base URL for LiteLLM
        cost_per_1m_input/output: fallback pricing (USD/1M tokens) when LiteLLM doesn't know the model
        """
        if llm is None and model is None:
            raise ValueError('Provide either model (LiteLLM) or llm (Llama instance)')
        self.model = model
        self.llm = llm
        self.tasks = dict(tasks)
        self.fallbacks = fallbacks
        self.api_key = api_key
        self.api_base = api_base
        self.postprocessors = dict(postprocessors or {})
        self._cost_per_1m_input = cost_per_1m_input
        self._cost_per_1m_output = cost_per_1m_output
        # Accumulated token usage for API calls (None for local llm)
        self.token_usage: dict[str, int | float] | None = None if llm is not None else {
            'prompt_tokens': 0, 'completion_tokens': 0, 'cost_usd': 0.0,
        }

    @staticmethod
    def _build_user_content(title: str, synopsis: str, alt_synopsis: str | None = None) -> str:
        cleaned = clean_title_for_llm(title)
        if alt_synopsis and alt_synopsis.strip():
            return (
                f'Title: "{cleaned}"\n\n'
                f'Synopsis (source 1):\n{synopsis.strip()}\n\n'
                f'Synopsis (source 2):\n{alt_synopsis.strip()}'
            )
        return f'Title: "{cleaned}"\n\nSynopsis:\n{synopsis.strip()}'

    @staticmethod
    def merge_genres(df_dest: pd.DataFrame, df_src: pd.DataFrame) -> pd.DataFrame:
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

        def _patch_genres(row):
            dest_genres = row["genres"]
            dest_subgenres = row["subgenres"]
            universe = set(dest_genres) | set(dest_subgenres)

            try:
                s = src_map.loc[row["film_id"]]
            except KeyError:
                return dest_genres

            src_genres = [g for g in (s["genre_1"], s["genre_2"], s["genre_3"]) if pd.notna(g)]
            to_prepend = [g for g in src_genres if g not in universe]
            return to_prepend + [g for g in dest_genres if g not in to_prepend]

        df_dest["genres"] = df_dest.apply(_patch_genres, axis=1)
        return df_dest

    def _run_task_for_synopsis(
        self,
        task: ExtractionTask,
        title: str,
        synopsis: str,
        alt_synopsis: str | None = None,
    ) -> dict[str, Any]:
        messages = [
            {"role": "system", "content": task.system_prompt},
            {"role": "user", "content": self._build_user_content(title=title, synopsis=synopsis, alt_synopsis=alt_synopsis)},
        ]

        if self.llm is not None:
            resp = self.llm.create_chat_completion(
                messages=messages,
                temperature=task.temperature,
                top_p=task.top_p,
                max_tokens=task.max_tokens,
                repeat_penalty=task.repeat_penalty,
            )
            raw = resp['choices'][0]['message']['content']
        else:
            kwargs: dict[str, Any] = dict(
                model=self.model,
                messages=messages,
                temperature=task.temperature,
                top_p=task.top_p,
                max_tokens=task.max_tokens,
            )
            if self.api_key:
                kwargs['api_key'] = self.api_key
            if self.api_base:
                kwargs['api_base'] = self.api_base
            if self.fallbacks:
                kwargs['fallbacks'] = self.fallbacks
            resp = litellm.completion(**kwargs)
            raw = resp.choices[0].message.content
            if self.token_usage is not None and resp.usage:
                prompt_tokens = resp.usage.prompt_tokens or 0
                completion_tokens = resp.usage.completion_tokens or 0
                self.token_usage['prompt_tokens'] += prompt_tokens
                self.token_usage['completion_tokens'] += completion_tokens
                cost = None
                try:
                    cost = litellm.completion_cost(completion_response=resp)
                except Exception:
                    pass
                if not cost and self._cost_per_1m_input is not None and self._cost_per_1m_output is not None:
                    cost = (prompt_tokens * self._cost_per_1m_input + completion_tokens * self._cost_per_1m_output) / 1_000_000
                if cost:
                    self.token_usage['cost_usd'] += cost

        try:
            data = extract_json(raw)
        except ValueError as e:
            return {"_error": "json_parse_error", "_error_message": str(e), "_raw_output": raw}

        if task.postprocess:
            fn = self.postprocessors.get(task.postprocess)
            if fn is not None:
                try:
                    data = fn(data)
                except Exception as e:
                    return {"_error": "postprocess_error", "_error_message": str(e), "_raw_output": raw}

        return data

    def run_single_synopsis(
        self,
        title: str,
        synopsis: str,
        alt_synopsis: str | None = None,
        task_names: Iterable[str] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Run all (or a subset of) extraction tasks on a single film."""
        if task_names is None:
            task_names = self.tasks.keys()

        results: dict[str, dict[str, Any]] = {}
        for name in task_names:
            task = self.tasks[name]
            results[name] = self._run_task_for_synopsis(task, title, synopsis, alt_synopsis=alt_synopsis)

        return results

    def run_multiple_synopses(
        self,
        df: pd.DataFrame,
        id_col: str = "film_id",
        title_col: str = "title",
        synopsis_col: str = "synopsis",
        alt_synopsis_col: str | None = None,
        task_names: Iterable[str] | None = None,
        flatten: bool = True,
    ) -> dict[str, dict[str, Any]]:
        """
        Run extraction tasks for multiple films in a DataFrame.

        Returns:
            If flatten=True:  {film_id: {merged task keys...}, ...}
            If flatten=False: {film_id: {"title": ..., "synopsis": ..., "results": {...}}, ...}
        """
        if task_names is None:
            task_names = self.tasks.keys()

        task_names = list(task_names)
        output: dict[str, dict[str, Any]] = {}

        synopses_len = len(df)
        task_len = len(task_names)

        if synopses_len == 0 or task_len == 0:
            return output

        total_task_calls = synopses_len * task_len
        start_time = time.time()

        def _fmt_secs(seconds: float) -> str:
            seconds = int(seconds)
            h, rem = divmod(seconds, 3600)
            m, s = divmod(rem, 60)
            return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"

        for sidx, row in enumerate(df.itertuples(index=False), start=1):
            film_id = getattr(row, id_col)
            title = getattr(row, title_col)
            synopsis = getattr(row, synopsis_col)
            alt_synopsis = getattr(row, alt_synopsis_col, None) if alt_synopsis_col else None
            if isinstance(alt_synopsis, float):  # NaN from pandas
                alt_synopsis = None

            if not isinstance(synopsis, str) or len(str(synopsis)) < 5:
                continue

            film_results: dict[str, dict[str, Any]] = {}
            for tidx, name in enumerate(task_names, start=1):
                completed_task_calls = (sidx - 1) * task_len + tidx
                elapsed = time.time() - start_time
                avg_per_task = elapsed / completed_task_calls
                effective_synopses_done = completed_task_calls / task_len
                avg_per_synopsis = elapsed / effective_synopses_done
                remaining_task_calls = total_task_calls - completed_task_calls
                eta = remaining_task_calls * avg_per_task

                print(
                    (
                        f"Extracting - "
                        f"Synopses: {sidx / synopses_len:.1%} ({sidx}/{synopses_len}) | "
                        f"Tasks: {tidx / task_len:.1%} ({tidx}/{task_len}) | "
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
                film_results[name] = self._run_task_for_synopsis(task, title, synopsis, alt_synopsis=alt_synopsis)

            if flatten:
                merged = flatten_extraction(film_results)
                film_record: dict[str, Any] = {"film_id": film_id, "title": title, "synopsis": synopsis, **merged}
            else:
                film_record = {"film_id": film_id, "title": title, "synopsis": synopsis, "results": film_results}

            output[film_id] = film_record

        total_elapsed = time.time() - start_time
        print(f"\nDone. Total time: {_fmt_secs(total_elapsed)}")

        return output

    async def _arun_task_for_synopsis(
        self,
        task: ExtractionTask,
        title: str,
        synopsis: str,
        semaphore: asyncio.Semaphore,
        alt_synopsis: str | None = None,
    ) -> dict[str, Any]:
        """Async version of _run_task_for_synopsis for API providers."""
        async with semaphore:
            messages = [
                {"role": "system", "content": task.system_prompt},
                {"role": "user", "content": self._build_user_content(title=title, synopsis=synopsis, alt_synopsis=alt_synopsis)},
            ]
            kwargs: dict[str, Any] = dict(
                model=self.model,
                messages=messages,
                temperature=task.temperature,
                top_p=task.top_p,
                max_tokens=task.max_tokens,
            )
            if self.api_key:
                kwargs['api_key'] = self.api_key
            if self.api_base:
                kwargs['api_base'] = self.api_base
            if self.fallbacks:
                kwargs['fallbacks'] = self.fallbacks

            resp = await litellm.acompletion(**kwargs)
            raw = resp.choices[0].message.content

            if self.token_usage is not None and resp.usage:
                prompt_tokens = resp.usage.prompt_tokens or 0
                completion_tokens = resp.usage.completion_tokens or 0
                self.token_usage['prompt_tokens'] += prompt_tokens
                self.token_usage['completion_tokens'] += completion_tokens
                cost = None
                try:
                    cost = litellm.completion_cost(completion_response=resp)
                except Exception:
                    pass
                if not cost and self._cost_per_1m_input is not None and self._cost_per_1m_output is not None:
                    cost = (prompt_tokens * self._cost_per_1m_input + completion_tokens * self._cost_per_1m_output) / 1_000_000
                if cost:
                    self.token_usage['cost_usd'] += cost

        try:
            data = extract_json(raw)
        except ValueError as e:
            return {"_error": "json_parse_error", "_error_message": str(e), "_raw_output": raw}

        if task.postprocess:
            fn = self.postprocessors.get(task.postprocess)
            if fn is not None:
                try:
                    data = fn(data)
                except Exception as e:
                    return {"_error": "postprocess_error", "_error_message": str(e), "_raw_output": raw}

        return data

    async def arun_multiple_synopses(
        self,
        df: pd.DataFrame,
        id_col: str = "film_id",
        title_col: str = "title",
        synopsis_col: str = "synopsis",
        alt_synopsis_col: str | None = None,
        task_names: Iterable[str] | None = None,
        flatten: bool = True,
        max_concurrency: int = 20,
        min_synopsis_len: int = 5,
    ) -> dict[str, dict[str, Any]]:
        """
        Async version of run_multiple_synopses for API providers.
        Runs all (film × task) calls concurrently, bounded by max_concurrency.
        """
        if task_names is None:
            task_names = list(self.tasks.keys())
        task_names = list(task_names)

        semaphore = asyncio.Semaphore(max_concurrency)
        start_time = time.time()
        completed = 0
        valid_rows = [
            row for row in df.itertuples(index=False)
            if isinstance(getattr(row, title_col), str) and len(str(getattr(row, title_col))) >= 1
            and (min_synopsis_len == 0 or (isinstance(getattr(row, synopsis_col), str) and len(str(getattr(row, synopsis_col))) >= min_synopsis_len))
        ]
        total = len(valid_rows)

        def _fmt_secs(seconds: float) -> str:
            seconds = int(seconds)
            h, rem = divmod(seconds, 3600)
            m, s = divmod(rem, 60)
            return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"

        async def process_film(row) -> tuple | None:
            nonlocal completed
            film_id = getattr(row, id_col)
            title = getattr(row, title_col)
            synopsis = getattr(row, synopsis_col)
            alt_synopsis = getattr(row, alt_synopsis_col, None) if alt_synopsis_col else None
            if isinstance(alt_synopsis, float):  # NaN from pandas
                alt_synopsis = None

            coros = [
                self._arun_task_for_synopsis(self.tasks[name], title, synopsis, semaphore, alt_synopsis=alt_synopsis)
                for name in task_names
            ]
            task_results = await asyncio.gather(*coros)
            film_results = dict(zip(task_names, task_results))

            completed += 1
            elapsed = time.time() - start_time
            rate = completed / elapsed if elapsed > 0 else 0
            eta = (total - completed) / rate if rate > 0 else 0
            print(
                f"Extracting - {completed}/{total} films | Elapsed: {_fmt_secs(elapsed)} | ETA: {_fmt_secs(eta)}      ",
                end="\r", flush=True,
            )

            if flatten:
                merged = flatten_extraction(film_results)
                return film_id, {"film_id": film_id, "title": title, "synopsis": synopsis, **merged}
            return film_id, {"film_id": film_id, "title": title, "synopsis": synopsis, "results": film_results}

        results_list = await asyncio.gather(*[process_film(row) for row in valid_rows])

        print(f"\nDone. Total time: {_fmt_secs(time.time() - start_time)}")

        return dict(results_list)
