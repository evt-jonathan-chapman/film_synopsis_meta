"""
test_film_meta.py
-----------------
Smoke-test FilmMetaExtractor on a single film. Hits the OpenAI Responses API
with the web_search tool; ~10-30s per call.

Usage:
    python test_film_meta.py                          # uses TEST_FILM defaults
    python test_film_meta.py "Marty Supreme" 2026-12-25
    python test_film_meta.py "Crime 101"    2026-11-21 "Sony Pictures"

Edit the TEST_FILM block below to change the default film. Output is printed
and also saved to data/film_meta/test_run.json so you can diff against
data/film_meta_gpt/sample_spider_man_brand_new_day.json.
"""

import asyncio
import json
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

import nest_asyncio
nest_asyncio.apply()

from dotenv import load_dotenv
load_dotenv()

import pandas as pd

from film_meta_extractor import FilmMetaExtractor
from load_prompts import load_tasks_from_yaml
from config import DATA_DIR

# ── Defaults (override via CLI args) ──────────────────────────────────────────
TEST_FILM = {
    "film_id":  59530,
    "title":    "THE HOUSEMAID",
    "rel_at":   "2025-02-04",
    "dstbtr":   "Lionsgate",
}
PROMPTS_PATH = str(_REPO / 'prompts' / 'film_meta_prompts.yaml')
OUT_PATH     = DATA_DIR / "film_meta" / "test_run.json"

# ── Parse CLI overrides ───────────────────────────────────────────────────────
if len(sys.argv) >= 2:
    TEST_FILM["title"] = sys.argv[1]
    TEST_FILM["film_id"] = abs(hash(sys.argv[1])) % 100000  # fake id for ad-hoc runs
if len(sys.argv) >= 3:
    TEST_FILM["rel_at"] = sys.argv[2]
if len(sys.argv) >= 4:
    TEST_FILM["dstbtr"] = sys.argv[3]

print(f"Testing extraction on:")
print(f"  film_id : {TEST_FILM['film_id']}")
print(f"  title   : {TEST_FILM['title']}")
print(f"  rel_at  : {TEST_FILM['rel_at']}")
print(f"  dstbtr  : {TEST_FILM['dstbtr']}")

# ── Build extractor ───────────────────────────────────────────────────────────
tasks = load_tasks_from_yaml(PROMPTS_PATH)
if "film_meta" not in tasks:
    raise SystemExit(f"film_meta task missing from {PROMPTS_PATH}")

api_key = os.getenv("OPENAI_KEY")
if not api_key:
    raise SystemExit("OPENAI_KEY not set — check .env")

extractor = FilmMetaExtractor(
    task=tasks["film_meta"],
    api_key=api_key,
    # Cost tracking optional — leave None unless you want exact figures
    cost_per_1m_input=None,
    cost_per_1m_output=None,
)
print(f"\nModel: {extractor.model}  (override with FILM_META_MODEL env var)")

# ── Run on a 1-row DataFrame ──────────────────────────────────────────────────
df = pd.DataFrame([{
    "film_id":    TEST_FILM["film_id"],
    "film_title": TEST_FILM["title"],
    "rel_at":     pd.Timestamp(TEST_FILM["rel_at"]),
    "dstbtr":     TEST_FILM["dstbtr"],
}])

print("\nCalling Responses API (web_search enabled) — this can take 10-30s...")
results = asyncio.run(
    extractor.arun(
        df=df,
        id_col="film_id",
        title_col="film_title",
        rel_at_col="rel_at",
        dstbtr_col="dstbtr",
        max_concurrency=1,
    )
)

# ── Inspect + persist ─────────────────────────────────────────────────────────
fid    = TEST_FILM["film_id"]
result = results.get(fid, {})

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_PATH, "w") as f:
    json.dump(result, f, indent=2, default=str)

print(f"\nResult written to → {OUT_PATH}")
print("─" * 70)
if "_error" in result:
    print(f"EXTRACTION FAILED: {result['_error']}")
    if "_raw_output" in result:
        print(f"\nRaw model output (first 1000 chars):\n{result['_raw_output']}")
else:
    print(json.dumps(result, indent=2, default=str))

# ── Token / cost summary ──────────────────────────────────────────────────────
u = extractor.token_usage
print("─" * 70)
print(f"Tokens — input: {u['prompt_tokens']:,}  output: {u['completion_tokens']:,}")
if u.get("cost_usd"):
    print(f"Cost (if pricing was set): ${u['cost_usd']:.4f}")
