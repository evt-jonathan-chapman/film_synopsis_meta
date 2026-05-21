# vendored/cinema_admits_models

**Do not edit these files directly.** They are vendored copies of code from a
sibling repository — edits will be overwritten the next time we re-vendor.

## Source

Upstream repo: `/Users/jonathanchapman/Documents/git/cinema_admits_models`
Vendored on: 2026-05-21

## Files

| File | Upstream path | Used by |
|---|---|---|
| `re_release_filter.py` | `cinema_admits_models/re_release_filter.py` | `main.py::enrich_film_meta`, `refresh.py::_enrich_film_meta` — filters out re-release titles before LLM extraction |
| `encode_helper.py` | `cinema_admits_models/encode_helper.py` | Imported transitively by `re_release_filter.py` (`EncHelper`) |

## Local modifications

- `re_release_filter.py` line 4: changed `from encode_helper import EncHelper`
  to `from .encode_helper import EncHelper` so the package self-resolves
  without relying on `sys.path` injection.

## Re-vendoring procedure

If the upstream changes and you want to pick up the update:

```bash
cp /Users/jonathanchapman/Documents/git/cinema_admits_models/re_release_filter.py vendored/cinema_admits_models/
cp /Users/jonathanchapman/Documents/git/cinema_admits_models/encode_helper.py vendored/cinema_admits_models/
# Re-apply the local modification documented above.
```

## Why vendor instead of importing?

Originally `main.py` and `refresh.py` injected `cinema_admits_models` onto
`sys.path` with an absolute hard-coded path. That broke any non-local checkout
(Dagster running headless, CI, a teammate's machine). Vendoring makes this repo
self-contained — same pattern as `base_snowflake.py` at the repo root.
