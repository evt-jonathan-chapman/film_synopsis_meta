"""
inspect_comscore_unmatched.py
-----------------------------
Bucketise the unmatched/borderline rows in comscore_review_needed.parquet
by candidate_1_score so we know whether the right Comscore row is "close"
(needs a tuning tweak) or genuinely absent (no fix possible).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from comscore_matcher import ComscoreMatcher
from config import DATA_DIR, RAW_PARQUET_GLOBS_ALL, FILM_META_ENRICHED_PATH


from rematch_comscore import pull_comscore

cs_all = pull_comscore()

cs_cols = ["title_global_id", "film_name", "upper_name", "title_aka", "us_title_name", "short_name", "synopsis",
           "is_alt_content", "orig_cntry", "cntry_id", "distr_global_id", "release_date"]

cs = cs_all[cs_cols].drop_duplicates()

OUT_PATH = "/Users/jonathanchapman/Documents/data/comscore/output"

EXAMPLES_PER_BUCKET = 10

BUCKETS = [
    ("close   (>=0.6)",      lambda s: s >= 0.6),
    ("fuzzy   (0.3-0.6)",    lambda s: (s >= 0.3) & (s < 0.6)),
    ("far     (<0.3)",       lambda s: (s > 0) & (s < 0.3)),
    ("no candidates",        lambda s: s.isna() | (s == 0)),
]

# cs.groupby("match_confidence").agg({"film_id": "nunique"})

# 1. Sum admits per film across all parquets
# parts = []
# for pattern in RAW_PARQUET_GLOBS_ALL:
#     for p in sorted(glob.glob(pattern)):
#         parts.append(pd.read_parquet(p, columns=['film_id', 'week_admits']))

film_lookup_all = pd.read_parquet("/Users/jonathanchapman/Documents/data/look_ups/film_lookup.parquet")

fim_lookup = film_lookup_all[film_lookup_all["rel_at"] >= cs["release_date"].min()]

adaptation = pd.read_parquet(FILM_META_ENRICHED_PATH, columns=['film_id', 'adaptation_type'])

admits = fim_lookup[["film_id", "rel_at", "week1_admits"]]

# 2. Load comscore cache
cache = (
    pd.read_parquet(DATA_DIR / 'comscore' / 'comscore_cache.parquet')
    .merge(admits, how="left", on="film_id")
    .merge(adaptation, on='film_id', how='left')
)

unmatched = (
    cache[cache['match_confidence'] == 'unmatched']
    .sort_values('week1_admits', ascending=False)
    [['film_id', 'film', 'week1_admits', 'match_score', 'adaptation_type']]
)

print(unmatched.head(30))
print(f"\nTotal unmatched: {len(unmatched)}")
print(f"Unmatched with <1000 admits: {(unmatched['week1_admits'] < 1000).sum()}")
print(f"Unmatched with <100 admits:  {(unmatched['week1_admits'] < 100).sum()}")
print(f"\nUnmatched by adaptation_type:")
print(unmatched.groupby('adaptation_type', dropna=False)['week1_admits'].agg(['count', 'sum']).sort_values('sum', ascending=False).to_string())

result = cache.groupby(["match_confidence", cache["rel_at"].dt.year]).agg(
    {"film_id": "nunique"}
)

print(result.to_markdown())

print(cache.groupby(["match_confidence"]).agg({"film_id": "nunique"}))


print(result.to_markdown())

match_cat = ["borderline", "high", "unmatched"]

for m in match_cat:
    
    _df = cache[cache["match_confidence"]==m]
    
    _filename = f"{OUT_PATH}/{m}.csv"
    _df.sort_values("week1_admits", ascending=False).to_csv(_filename)
    

def main():
    review = pd.read_parquet(ComscoreMatcher.REVIEW_PATH)
    adaptation = pd.read_parquet(FILM_META_ENRICHED_PATH, columns=['film_id', 'adaptation_type'])
    review = review.merge(adaptation, on='film_id', how='left')

    print(f"Review file: {len(review):,} rows "
          f"({(review['current_match_confidence'] == 'unmatched').sum()} unmatched, "
          f"{(review['current_match_confidence'] == 'borderline').sum()} borderline)\n")

    unmatched = review[review["current_match_confidence"] == "unmatched"].copy()
    print(f"=== Unmatched ({len(unmatched):,}) by candidate_1_score ===\n")

    scores = unmatched["candidate_1_score"]
    for label, mask_fn in BUCKETS:
        sub = unmatched[mask_fn(scores)]
        print(f"  {label:<22} n={len(sub):>4}  ({len(sub) / len(unmatched):.0%})")
    print()

    for label, mask_fn in BUCKETS:
        sub = unmatched[mask_fn(scores)].head(EXAMPLES_PER_BUCKET)
        if sub.empty:
            continue
        print(f"--- {label} — {EXAMPLES_PER_BUCKET} examples ---")
        cols = ["film", "rel_at", "dstbtr", "adaptation_type",
                "candidate_1_title", "candidate_1_year", "candidate_1_score", "candidate_1_days"]
        print(sub[cols].to_string(index=False))
        print()


if __name__ == "__main__":
    main()
