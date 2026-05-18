"""
film_meta_extractor.py
----------------------
Per-film metadata extraction via OpenAI Responses API with the web_search tool.

Returns the schema defined in prompts/film_meta_prompts.yaml::film_meta —
title, release_date, director, writers, genres, description, budget_usd,
budget_source, studios[order/name/role], cast[order/actor/character],
trailers[type/channel/date/url].

This module uses the OpenAI Python SDK directly (not LiteLLM) because the
web_search tool is only exposed through the Responses endpoint, not Chat
Completions.

Mirrors the interface of LlmJsonExtractor.arun_multiple_synopses() loosely so
it slots into main.py's pattern alongside the other enrichers.

Usage:
    from film_meta_extractor import FilmMetaExtractor
    from load_prompts import load_tasks_from_yaml

    tasks  = load_tasks_from_yaml('prompts/film_meta_prompts.yaml')
    runner = FilmMetaExtractor(task=tasks['film_meta'], api_key=os.getenv('OPENAI_KEY'))
    results = await runner.arun(df_films, max_concurrency=4)
"""

import asyncio
import json
import os
import re
from typing import Any

import pandas as pd
from openai import AsyncOpenAI

from extraction import ExtractionTask


# Web-search-capable model. Override via FILM_META_MODEL env var.
#
# Choice notes (as of May 2026 — verify against your account's tool access):
#   gpt-5.4-mini cheap + fast, confirmed-compatible with hosted web_search.
#                Good default for bulk runs. (Replaces gpt-4o-mini, being deprecated.)
#   gpt-5.5      highest quality reasoning; slowest. Last resort for tricky films.
#   gpt-5.4-nano same model the synopsis/cast/director extractors use, BUT does
#                NOT support hosted web_search — do not use for film_meta.
DEFAULT_MODEL = os.environ.get("FILM_META_MODEL", "gpt-5.4-mini")


def _strip_json(text: str) -> str | None:
    """Pull the first JSON object out of a response (in case prose precedes)."""
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    return m.group() if m else None


def _parse_response(raw: str) -> dict | None:
    """Salvage-tolerant JSON parse — strips trailing commas as a second pass."""
    js = _strip_json(raw)
    if not js:
        return None
    try:
        return json.loads(js)
    except json.JSONDecodeError:
        cleaned = re.sub(r",(\s*[}\]])", r"\1", js)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return None


class FilmMetaExtractor:
    """Async per-film extractor using OpenAI Responses API + web_search."""

    def __init__(
        self,
        task: ExtractionTask,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        cost_per_1m_input: float | None = None,
        cost_per_1m_output: float | None = None,
        use_web_search: bool = True,
    ):
        self.task   = task
        self.model  = model
        self.client = AsyncOpenAI(api_key=api_key)
        self._cost_in  = cost_per_1m_input
        self._cost_out = cost_per_1m_output
        self.use_web_search = use_web_search
        self.token_usage = {
            "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0,
        }

    def _build_user_msg(self, title: str, rel_at: Any, dstbtr: Any) -> str:
        rel_str    = pd.to_datetime(rel_at).strftime("%Y-%m-%d") if pd.notna(rel_at) else "unknown"
        dstbtr_str = str(dstbtr).strip() if pd.notna(dstbtr) else "unknown"
        return (
            f'Title: "{title}"\n'
            f"Estimated release date: {rel_str}\n"
            f"Distributor (Australia): {dstbtr_str}\n"
            f"\nExtract the metadata JSON now."
        )

    async def _extract_one(
        self,
        film_id: int,
        title: str,
        rel_at: Any,
        dstbtr: Any,
        semaphore: asyncio.Semaphore,
    ) -> tuple[int, dict[str, Any]]:
        user_msg = self._build_user_msg(title, rel_at, dstbtr)
        async with semaphore:
            try:
                kwargs = {
                    "model": self.model,
                    "input": [
                        {"role": "system", "content": self.task.system_prompt},
                        {"role": "user",   "content": user_msg},
                    ],
                }
                if self.use_web_search:
                    kwargs["tools"] = [{"type": "web_search"}]
                resp = await self.client.responses.create(**kwargs)
                raw = (resp.output_text or "").strip()

                # Token accounting (best-effort)
                usage = getattr(resp, "usage", None)
                if usage is not None:
                    in_tok  = getattr(usage, "input_tokens",  0) or 0
                    out_tok = getattr(usage, "output_tokens", 0) or 0
                    self.token_usage["prompt_tokens"]     += in_tok
                    self.token_usage["completion_tokens"] += out_tok
                    if self._cost_in is not None and self._cost_out is not None:
                        self.token_usage["cost_usd"] += (
                            in_tok  * self._cost_in  / 1_000_000 +
                            out_tok * self._cost_out / 1_000_000
                        )

                data = _parse_response(raw)
                if not data:
                    return film_id, {"_error": "json_parse_failed", "_raw_output": raw[:1000]}
                return film_id, data
            except Exception as e:
                return film_id, {"_error": f"{type(e).__name__}: {e}"[:300]}

    async def arun(
        self,
        df: pd.DataFrame,
        id_col: str = "film_id",
        title_col: str = "film_title",
        rel_at_col: str = "rel_at",
        dstbtr_col: str = "dstbtr",
        max_concurrency: int = 4,
    ) -> dict[int, dict]:
        """Run extraction on every row in df. Returns {film_id: result_dict}.

        Result is either the parsed schema (success) or a dict containing
        `_error` (and optionally `_raw_output`) on failure — matches the
        convention used by LlmJsonExtractor so the main.py checkpoint plumbing
        keeps working.
        """
        sem = asyncio.Semaphore(max_concurrency)
        coros = [
            self._extract_one(
                int(row[id_col]),
                str(row[title_col]),
                row.get(rel_at_col),
                row.get(dstbtr_col),
                sem,
            )
            for _, row in df.iterrows()
        ]
        results = await asyncio.gather(*coros, return_exceptions=False)
        return {fid: data for fid, data in results}
