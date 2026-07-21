# Comscore Matching

Maps EVT `film_id` → Comscore `title_global_id` so that `IBOE_TITLES` + `IBOE_FLASH_GROSS` box office data can be joined onto EVT films.

There's a second, independent source matched the same way — Gower box-office
estimates, see `GOWER.md`. Both run their own fuzzy pass straight against
the EVT catalogue; `id_bridge.py` joins the two results together afterwards
(`film_id` / `cs_id` / `gower_id`). The matching engine itself
(`ComscoreMatcher` below) is a thin subclass of
`title_matcher.py::FuzzyTitleMatcher`, the shared base `GowerMatcher` also
subclasses — everything in this file describing normalisation, scoring,
variant propagation, and confidence tiers applies to Gower unchanged.

---

## Key files

| File | Purpose |
|---|---|
| `sql/comscore_extract.sql` | One-off Snowflake pull of Comscore titles + release dates |
| `title_matcher.py::FuzzyTitleMatcher` | Shared matching engine (normalisation, scoring, variant propagation, cache/review/override I/O) — used by both Comscore and Gower |
| `comscore_matcher.py::ComscoreMatcher` | Comscore-specific column names/paths on top of `FuzzyTitleMatcher` |
| `rematch_comscore.py` | Driver script — pulls SQL, loads EVT films, runs matcher |
| `diagnostics/inspect_comscore_unmatched.py` | Audit unmatched/borderline films by admits + adaptation_type |
| `id_bridge.py` | Outer-joins `comscore_cache.parquet` + `gower_cache.parquet` on `film_id` |

Outputs under `~/Documents/data/comscore/`:

| File | Description |
|---|---|
| `comscore_cache.parquet` | All matched films keyed on EVT `film_id` — this is the checkpoint |
| `comscore_review_needed.parquet` | Borderline + unmatched rows with top-5 candidates, for human triage |
| `comscore_manual_overrides.parquet` | Human-authored — fill `manual_override_cs_id` and save here |

---

## Comscore SQL extract (`sql/comscore_extract.sql`)

Pulls from `IBOE_TITLES` joined to `IBOE_FLASH_GROSS_STATE_TITLE` (week 1 rows only, so `release_date = MIN(exhibition_date)` at week 1).

**Filters applied:**
- `CNTRY_ID = 'AU'` — AU only
- Release date windows: **2018-01-01 → 2020-02-01** (pre-COVID) **OR 2021-12-01 → ~now+28 days** (post-COVID). Films outside these windows won't be in the extract — they'll appear unmatched even if the title matches perfectly.

**Title columns returned** (all five are scored during matching):

| Column | Notes |
|---|---|
| `film_name` / `upper_name` | Primary title; upper_name is the same uppercased |
| `title_aka` | Alternate/local title |
| `us_title_name` | US release title |
| `short_name` | Abbreviated title |

`is_alt_content` flags concert films, sports, etc. — these are dropped before matching.

---

## Matching logic (`ComscoreMatcher`)

### 1. Title normalisation

Applied to both EVT and Comscore titles before scoring:
- Strip variant/format prefixes: `GC`, `3D`, `IMAX`, `VMAX`, `XD`, `EVENT`, `GOLD`, `GOLDCLASS`, `VIP`
- Strip festival prefixes: `TFF -`, `MIFF -`, etc. (2–4 uppercase letters + dash)
- Strip variant suffixes: `3D`, `IMAX`, `VMAX`, `XD` at end of title
- Strip year parentheticals: `(2023)`
- NFKD unicode normalisation + strip combining characters (handles accents)
- Lowercase + collapse whitespace

### 2. Article transposition

Before scoring, EVT titles are expanded to article-transposed variants so that `THE MEG` also tries `MEG, THE` and vice versa. Articles handled: `THE`, `A`, `AN`.

Examples:
- `THE MEG` → tries both `THE MEG` and `MEG, THE`
- `MEG, THE` → tries both `MEG, THE` and `THE MEG`
- `A QUIET PLACE` → tries both `A QUIET PLACE` and `QUIET PLACE, A`

### 3. Fuzzy scoring

For each Comscore row, all five title columns are scored against all EVT title variants. The best score across all column × variant combinations wins.

Score = `max(ratio, token_sort_ratio)` from rapidfuzz (falls back to `difflib` if not installed), divided by 100.

**Length-ratio guard:** if `min(len_a, len_b) / max(len_a, len_b) < MIN_LENGTH_RATIO` (0.5), score is forced to 0.0. This prevents short-title impostors (e.g. `AVATAR` matching `TÁR`).

### 4. Year window filter

Only Comscore rows within ±1 year of the EVT `rel_at` date are considered. If EVT has no release date, all Comscore rows are scored.

### 5. Confidence tiers

| Tier | Condition |
|---|---|
| **high** | score ≥ 0.92 AND days_diff ≤ 365 |
| **borderline** | score ≥ 0.80 (match threshold) but not high |
| **unmatched** | score < 0.80 |
| **manual** | User-supplied override — always wins, never re-scored |
| **variant** | Inherited from a matched film with the same base title (step 4 below) |

`match_thresh = 0.80` — anything at or above this is cached and skipped on future runs.
`HIGH_CONFIDENCE_SCORE = 0.92` — the higher bar for the `high` label.

### 6. Tie-breaking

When multiple Comscore rows tie on score, the one with the smallest `days_diff` (closest release date) wins.

### 7. Variant propagation

After the fuzzy match loop, `ComscoreMatcher._propagate_variants()` inherits a matched cs_id to all EVT format/event variants of the same base film.

A film is a **variant** if `_strip_variant(film)` ≠ the original title (uppercased). Patterns stripped include:

- Format prefixes/suffixes: `3D …`, `… - 3D`, `… - IMAX`, `… - IMAX 3D`, `… (3D)`, `… (IMAX)`, `… - SCREEN X`
- Event suffixes: `… - SPECIAL SCREENING`, `… - SING-ALONG`, `… - BONUS CONTENT`, `… - EVENT CINEMA`, `… - SPECIAL Q AND A`
- Festival prefixes: `TFF -`, `FFF -`, `MIFF -`, `CFF -`, `MF -`

For each unmatched variant, the matcher finds a **keeper** — any matched film in scope with the same base title within ±1 year. Keeper preference: manual > high > borderline. The variant inherits the keeper's `cs_id` with `confidence='variant'`, `match_score=1.0` (so it's skipped on future incremental runs).

These patterns are copied from `cinema_admits_models/encode_helper.py::_VARIANT_STRIP` so the Dagster pipeline can run without depending on the sibling repo.

---

## Films excluded before matching

Both sides drop content that won't have a cross-side match:

**EVT side** (`rematch_comscore.py`):
- Festival/event/sports distributors (e.g. `ZZ Japanese Film Festival`, `AU FATHOM EVENTS`) — full list in `SKIP_DISTRIBUTORS`
- `adaptation_type == "concert_film"` from `film_meta_enriched.parquet`

**Comscore side:**
- `is_alt_content = TRUE` — concerts, sports, etc.

---

## Checkpoint / resume behaviour

`comscore_cache.parquet` is the checkpoint — there is no separate progress JSON.

- Films with `match_score >= 0.80` are **skipped** on re-run (`build_mapping`)
- Films with `match_score < 0.80` are **retried** — useful when the extract is refreshed or matching rules change
- Manual overrides are **always applied first** and **never touched** by `re_score()`

To force a full re-match: delete both `comscore_cache.parquet` and `comscore_review_needed.parquet`.

---

## Running

```bash
# Full run (pull Comscore from Snowflake, match all EVT films)
python rematch_comscore.py                  # LIMIT_FILMS=None at top of file

# Sample run for testing
# Set LIMIT_FILMS = 200 at top of rematch_comscore.py, then:
python rematch_comscore.py

# Re-score existing cache with updated algorithm (no Snowflake re-pull needed)
# Run in Python:
from rematch_comscore import pull_comscore, load_evt_films
from comscore_matcher import ComscoreMatcher
cs = pull_comscore()
films = load_evt_films()
matcher = ComscoreMatcher()
matcher.re_score(cs, films)

# Via Dagster (runs build_mapping — skips already-matched films)
# ./start_dagster_matching.sh, then Jobs → comscore_job → Launch Run
# (separate Dagster process/code location from the LLM paths — see DAGSTER.md)
```

**`build_mapping` vs `re_score`:**

| Method | What it re-scores | When to use |
|---|---|---|
| `build_mapping()` | Only score < 0.80 (unmatched + borderline) | Normal incremental run, new films |
| `re_score()` | Everything except manual overrides | After algorithm changes (e.g. new title logic) |

---

## Manual overrides

For films that the matcher can't resolve automatically:

1. Open `comscore_review_needed.parquet` — each row has `candidate_1_cs_id` through `candidate_5_cs_id` with scores
2. Fill `manual_override_cs_id` with the correct Comscore ID
3. Save the file as `comscore_manual_overrides.parquet` in the same directory
4. Re-run — overrides are applied first, forced to `confidence=manual`, `score=1.0`

Manual overrides survive `re_score()` and are never overwritten by the matcher.

---

## Auditing unmatched films

```bash
python diagnostics/inspect_comscore_unmatched.py
```

Shows unmatched films bucketed by best-candidate score, with total admits and adaptation_type joined in. High-admits unmatched films are the priority to fix via manual override. Low-admits unmatched films (< 1,000 total) can generally be ignored.
