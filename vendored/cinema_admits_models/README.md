# vendored/cinema_admits_models

**Do not edit these files directly.** They are vendored copies of code from a
sibling repository — edits will be overwritten the next time we re-vendor.

## Source

Upstream repo: `/Users/jonathan_chapman/Documents/git/cinema_admits_models`
Vendored on: 2026-05-21
Re-vendored on: 2026-08-11 — upstream had added `strip_format_variant` /
`consolidate_all_admits` / `_VARIANT_STRIP` to `encode_helper.py` (used by
`build_train.py::BuildTrain.create_merged_admits_column` upstream to merge
format-variant *admits* rows) which our copy predated; nothing else changed
in either file (see `git diff --stat` at the time: only that ~100-line
addition, plus the local import patch below).
Re-vendored on: 2026-08-13 — fixed upstream `_VARIANT_STRIP` in
`encode_helper.py`: the old `^GC\s+` pattern only stripped plain "GC Title"
bookings, not the equally-common "GC - Title" / "GC -Title" forms, leaving a
dangling "- " that never matched the base title (found while investigating
why 62 real format-variant pairs — Deadpool and Wolverine, The Marvels,
Indiana Jones and the Dial of Destiny, Avatar: The Way of Water, Guardians of
the Galaxy Vol 3, Star Wars: Episode VII, and ~15 others — weren't grouping).
Also added a `BTQ` prefix pattern (previously unrecognized entirely). Fixed
62 → 7 mismatches; the remaining 7 are genuine internal-title punctuation
inconsistencies (e.g. Spider-Man: Brand New Day's colon-vs-hyphen subtitle
separator across its 4 booking variants) unrelated to prefix stripping —
deliberately not auto-fixed, since blanket punctuation normalization risks
merging genuinely distinct titles. Nothing else changed in either file.

## Files

| File | Upstream path | Used by |
|---|---|---|
| `re_release_filter.py` | `cinema_admits_models/re_release_filter.py` | `film_variant_merge.py::filter_variants` (fuzzy-matched reschedule detection), `main.py::enrich_film_meta` — filters/merges re-release and duplicate-booking titles before LLM extraction |
| `encode_helper.py` | `cinema_admits_models/encode_helper.py` | `strip_format_variant` used directly by `film_variant_merge.py::filter_variants` (format-variant detection); also imported transitively by `re_release_filter.py` (`EncHelper`) |

## Local modifications

- `re_release_filter.py` line 4: changed `from encode_helper import EncHelper`
  to `from .encode_helper import EncHelper` so the package self-resolves
  without relying on `sys.path` injection.

## Re-vendoring procedure

If the upstream changes and you want to pick up the update:

```bash
cp /Users/jonathan_chapman/Documents/git/cinema_admits_models/re_release_filter.py vendored/cinema_admits_models/
cp /Users/jonathan_chapman/Documents/git/cinema_admits_models/encode_helper.py vendored/cinema_admits_models/
# Re-apply the local modification documented above.
```

## Why vendor instead of importing?

Originally `main.py` and `refresh.py` injected `cinema_admits_models` onto
`sys.path` with an absolute hard-coded path. That broke any non-local checkout
(Dagster running headless, CI, a teammate's machine). Vendoring makes this repo
self-contained — same pattern as `base_snowflake.py` at the repo root.
