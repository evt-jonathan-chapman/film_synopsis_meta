"""
s3_checkpoint.py
-----------------
Direct-to-S3 checkpoint I/O for refresh.py's four LLM extraction paths and
film_data_merge.py's merged output — no local-disk persistence, so this
works under Dagster/Kubernetes pods with no shared or persistent disk
between separate asset runs (replaces the old pattern of writing locally
then syncing to S3 as a separate step — see s3_sync.py, now unused by
these paths).

Why not one big mutable JSON checkpoint rewritten wholesale on every flush
(the old local-disk pattern — still used by main.py's ad-hoc path): S3 has
no partial-write/append primitive, so every flush would re-upload the
ENTIRE progress object (film_meta's local equivalent is already 5.9MB for
~4,400 films). That's expensive and slow, and a real correctness risk once
concurrent writers are involved (film_meta runs at concurrency=2, synopsis
at 8): last-write-wins on a full-object PUT means one coroutine's flush can
silently clobber another's still-unflushed progress, since there's no
merge on write.

Pattern instead — write-ahead log + periodic compaction:
    s3://{bucket}/{prefix}/{name}/cache/snapshot.json      compacted state
    s3://{bucket}/{prefix}/{name}/cache/deltas/{key}.json  one object per flush batch
    s3://{bucket}/{prefix}/{name}/cache/errors.json        small, rewritten wholesale
    s3://{bucket}/{prefix}/{name}/{filename}.parquet       final output, rewritten wholesale

`load_checkpoint()` merges the snapshot with every delta (in delta-key
order — keys are timestamp-prefixed, so lexicographic order is also
chronological order) into one dict; a later delta wins over an earlier one
on the same id (e.g. a film re-extracted after an earlier error).
`append_checkpoint()` only ever writes ONE new small delta object per
flush — O(batch size), not O(corpus size) — and never collides with
another writer's delta, since each gets its own key. `compact_checkpoint()`
folds everything into a fresh snapshot and deletes the deltas, keeping the
delta count (and therefore `load_checkpoint()`'s list+download cost)
bounded. Call it once at the end of a run, not on every flush.

Auth: boto3 session using the AWS profile in config.yaml's `s3.profile` if
set (currently the Stax SSO profile, for local testing — run `stax2aws
login` first), falling back to boto3's default credential chain (picks up
an IAM role / IRSA automatically) if `s3.profile` is unset — same fallback
s3_sync.py already documents for a production deployment. Clear
`s3.profile` in config.yaml when this runs in prod.
"""

import io
import json
import time
import uuid

import boto3
from botocore.exceptions import ClientError
import pandas as pd

from config import S3_BUCKET, S3_PREFIX, S3_PROFILE


def _client():
    session = boto3.Session(profile_name=S3_PROFILE) if S3_PROFILE else boto3.Session()
    return session.client("s3")


def _snapshot_key(name: str) -> str:
    return f"{S3_PREFIX}/{name}/cache/snapshot.json"


def _delta_prefix(name: str) -> str:
    return f"{S3_PREFIX}/{name}/cache/deltas/"


def _errors_key(name: str) -> str:
    return f"{S3_PREFIX}/{name}/cache/errors.json"


def _is_not_found(e: ClientError) -> bool:
    return e.response.get("Error", {}).get("Code") in ("NoSuchKey", "404")


def _get_json(s3, key: str, default):
    try:
        obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
        return json.loads(obj["Body"].read())
    except ClientError as e:
        if _is_not_found(e):
            return default
        raise


def _put_json(s3, key: str, data: dict, **kwargs) -> None:
    s3.put_object(Bucket=S3_BUCKET, Key=key, Body=json.dumps(data, default=str, **kwargs).encode())


def _list_keys(s3, prefix: str) -> list[str]:
    keys = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
        keys.extend(o["Key"] for o in page.get("Contents", []))
    return keys


def load_checkpoint(name: str) -> dict:
    """Merge the compacted snapshot with every delta written since, in
    chronological order — a later delta wins over an earlier one for the
    same id."""
    s3 = _client()
    merged = _get_json(s3, _snapshot_key(name), {})
    for key in sorted(_list_keys(s3, _delta_prefix(name))):
        merged.update(_get_json(s3, key, {}))
    return merged


def append_checkpoint(name: str, new_items: dict) -> None:
    """Write one small immutable delta object for a batch of newly-completed
    items. No-op if new_items is empty — never writes an empty delta."""
    if not new_items:
        return
    s3 = _client()
    key = f"{_delta_prefix(name)}{time.time():020.6f}_{uuid.uuid4().hex[:8]}.json"
    _put_json(s3, key, new_items)


def compact_checkpoint(name: str) -> int:
    """Fold the snapshot + all current deltas into one fresh snapshot, then
    delete the deltas. Call at the end of a run (not on every flush) to keep
    load_checkpoint()'s list+download cost bounded. Returns the number of
    deltas folded in."""
    s3 = _client()
    merged = load_checkpoint(name)
    _put_json(s3, _snapshot_key(name), merged)
    delta_keys = _list_keys(s3, _delta_prefix(name))
    for i in range(0, len(delta_keys), 1000):  # delete_objects caps at 1000 keys/call
        s3.delete_objects(Bucket=S3_BUCKET, Delete={"Objects": [{"Key": k} for k in delta_keys[i:i + 1000]]})
    return len(delta_keys)


def load_errors(name: str) -> dict:
    return _get_json(_client(), _errors_key(name), {})


def save_errors(name: str, errors: dict) -> None:
    """Errors are rewritten wholesale — this dict only holds failures, so it
    stays small; no delta pattern needed."""
    _put_json(_client(), _errors_key(name), errors, indent=2)


def read_parquet(name: str, filename: str, columns: "list[str] | None" = None,
                  prefix: "str | None" = None) -> "pd.DataFrame | None":
    s3 = _client()
    key = f"{prefix if prefix is not None else S3_PREFIX}/{name}/{filename}"
    try:
        obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
        return pd.read_parquet(io.BytesIO(obj["Body"].read()), columns=columns)
    except ClientError as e:
        if _is_not_found(e):
            return None
        raise


def write_parquet(name: str, filename: str, df: pd.DataFrame, prefix: "str | None" = None) -> None:
    s3 = _client()
    key = f"{prefix if prefix is not None else S3_PREFIX}/{name}/{filename}"
    buf = io.BytesIO()
    df.to_parquet(buf, engine="pyarrow", index=False)
    s3.put_object(Bucket=S3_BUCKET, Key=key, Body=buf.getvalue())


def s3_uri(name: str, filename: str, prefix: "str | None" = None) -> str:
    return f"s3://{S3_BUCKET}/{prefix if prefix is not None else S3_PREFIX}/{name}/{filename}"
