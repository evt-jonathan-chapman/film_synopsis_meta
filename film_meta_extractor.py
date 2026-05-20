"""
film_meta_extractor.py
----------------------
Extractors built on the OpenAI Responses API + optional web_search tool.

Three concrete extractors:
  FilmMetaExtractor      — per-film metadata (film_meta_prompts.yaml::film_meta)
  ActorMetaExtractor     — per-actor profile  (cast_prompts.yaml::actor_profile)
  DirectorMetaExtractor  — per-director profile (director_prompts.yaml::director_profile)

All three share ResponsesExtractor for the API call, JSON parse, and token
accounting. They differ only in how the user message is built from the input
row and in the dataframe column conventions used by `arun`.

We use the OpenAI Python SDK directly (not LiteLLM) because the web_search
tool is exposed only on the Responses endpoint, not Chat Completions.
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
#   gpt-5.4-mini cheap + fast, confirmed-compatible with hosted web_search.
#                Default for all three Responses extractors.
#   gpt-5.5      highest quality reasoning; slowest. Last resort.
#   gpt-5.4-nano does NOT support hosted web_search — never use here.
DEFAULT_MODEL = os.environ.get("FILM_META_MODEL", "gpt-5.4-mini")


def _strip_json(text: str) -> str | None:
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    return m.group() if m else None


def _parse_response(raw: str) -> dict | None:
    """Salvage-tolerant JSON parse — strips trailing commas on a second pass."""
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


def _first_sentence(text: str, max_chars: int = 250) -> str:
    """First sentence of a synopsis, used as a disambiguation hint.

    Looks for a sentence-ending punctuation followed by whitespace and an
    uppercase letter, but only after position 20 to skip leading abbreviations
    like "Mr." or "Dr.". Falls back to first max_chars.
    """
    text = str(text).strip()
    m = re.search(r"[.!?]\s+(?=[A-Z])", text)
    if m and m.start() >= 20:
        return text[: m.start() + 1]
    return text[:max_chars]


class ResponsesExtractor:
    """Base class — wraps a Responses API call with optional web_search and
    JSON-parsing. Subclasses implement `_build_user_msg` and `arun`."""

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
        # max_retries=8 — TPM-rolling-window saturation can take 30-60s to drain;
        # the SDK honours Retry-After and uses exponential backoff between attempts.
        self.client = AsyncOpenAI(api_key=api_key, max_retries=8)
        self._cost_in  = cost_per_1m_input
        self._cost_out = cost_per_1m_output
        self.use_web_search = use_web_search
        self.token_usage = {
            "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0,
        }

    async def _call_api(self, user_msg: str) -> dict[str, Any]:
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
            return {"_error": "json_parse_failed", "_raw_output": raw[:1000]}
        return data


class FilmMetaExtractor(ResponsesExtractor):
    """Per-film metadata. Inputs: title + release_date + director + first
    sentence of EVT synopsis. Director + synopsis are disambiguation anchors —
    two films can share a title and year, but rarely share a director and
    almost never share an opening sentence."""

    def _build_user_msg(
        self,
        title: str,
        rel_at: Any,
        director: Any = None,
        synopsis: Any = None,
    ) -> str:
        rel_str = pd.to_datetime(rel_at).strftime("%Y-%m-%d") if pd.notna(rel_at) else "unknown"
        dir_str = str(director).strip() if pd.notna(director) and str(director).strip() else "unknown"
        syn_str = _first_sentence(synopsis) if pd.notna(synopsis) and str(synopsis).strip() else "unknown"
        return (
            f'Title: "{title}"\n'
            f"Estimated release date: {rel_str}\n"
            f"Director (per EVT records): {dir_str}\n"
            f'Synopsis opening (per EVT records): "{syn_str}"\n'
            f"\nExtract the metadata JSON now."
        )

    async def _extract_one(self, film_id, title, rel_at, director, synopsis, semaphore):
        user_msg = self._build_user_msg(title, rel_at, director, synopsis)
        async with semaphore:
            try:
                return film_id, await self._call_api(user_msg)
            except Exception as e:
                return film_id, {"_error": f"{type(e).__name__}: {e}"[:300]}

    async def arun(
        self,
        df: pd.DataFrame,
        id_col: str = "film_id",
        title_col: str = "film_title",
        rel_at_col: str = "rel_at",
        director_col: str = "director",
        synopsis_col: str = "synopsis",
        max_concurrency: int = 4,
    ) -> dict[int, dict]:
        sem = asyncio.Semaphore(max_concurrency)
        coros = [
            self._extract_one(
                int(row[id_col]),
                str(row[title_col]),
                row.get(rel_at_col),
                row.get(director_col),
                row.get(synopsis_col),
                sem,
            )
            for _, row in df.iterrows()
        ]
        results = await asyncio.gather(*coros, return_exceptions=False)
        return {fid: data for fid, data in results}


class ActorMetaExtractor(ResponsesExtractor):
    """Per-actor profile. Input: actor name (string key)."""

    def _build_user_msg(self, actor_name: str) -> str:
        return f'Actor name: "{actor_name}"\n\nReturn the JSON now.'

    async def _extract_one(self, actor_name, semaphore):
        user_msg = self._build_user_msg(actor_name)
        async with semaphore:
            try:
                return actor_name, await self._call_api(user_msg)
            except Exception as e:
                return actor_name, {"_error": f"{type(e).__name__}: {e}"[:300]}

    async def arun(
        self,
        df: pd.DataFrame,
        name_col: str = "actor_name",
        max_concurrency: int = 4,
    ) -> dict[str, dict]:
        sem = asyncio.Semaphore(max_concurrency)
        names = [str(row[name_col]) for _, row in df.iterrows()]
        coros  = [self._extract_one(n, sem) for n in names]
        results = await asyncio.gather(*coros, return_exceptions=False)
        return {name: data for name, data in results}


class DirectorMetaExtractor(ResponsesExtractor):
    """Per-director profile. Input: director name (string key)."""

    def _build_user_msg(self, director_name: str) -> str:
        return f'Director name: "{director_name}"\n\nReturn the JSON now.'

    async def _extract_one(self, director_name, semaphore):
        user_msg = self._build_user_msg(director_name)
        async with semaphore:
            try:
                return director_name, await self._call_api(user_msg)
            except Exception as e:
                return director_name, {"_error": f"{type(e).__name__}: {e}"[:300]}

    async def arun(
        self,
        df: pd.DataFrame,
        name_col: str = "director_name",
        max_concurrency: int = 4,
    ) -> dict[str, dict]:
        sem = asyncio.Semaphore(max_concurrency)
        names = [str(row[name_col]) for _, row in df.iterrows()]
        coros  = [self._extract_one(n, sem) for n in names]
        results = await asyncio.gather(*coros, return_exceptions=False)
        return {name: data for name, data in results}
