"""
s3_sync.py
----------
NOTE — scope narrowed: refresh.py's four extraction paths and
film_data_merge.py now read/write S3 directly via s3_checkpoint.py (no local
disk, no separate sync step, no Stax SSO dependency — see s3_checkpoint.py's
module docstring). This module is no longer part of that path and is NOT
wired into dagster_defs.py. It still exists for main.py's local-disk ad-hoc
workflow: if you run main.py locally and want to push its output to S3
manually, this uploads it.

Uploads the four LLM meta-output checkpoints (synopsis, film_meta, cast_meta,
director_meta — enriched/extracted parquet + progress json) plus the derived
film_data_merged parquet to S3.

CAUTION: the parquet files land at the SAME S3 keys refresh.py's direct
writes use (e.g. film_meta/film_meta_enriched.parquet) — running this after
main.py against a different/older local work-set can overwrite the Dagster
path's output with main.py's local version. The *_progress.json files this
uploads are main.py's local checkpoint format; they are NOT read by
s3_checkpoint.py's cache (that lives under film_meta/cache/, etc.) — they're
uploaded here purely as a backup/inspection artifact, not a live checkpoint.

Auth: boto3 session using the AWS profile in config.yaml's `s3.profile`
(currently `stax-stax-au1-event`, a Stax SSO profile — credentials expire
hourly, so run `stax2aws login` first if uploads fail with a credentials
error). No automatic refresh; that's a manual step for now.
"""

from pathlib import Path

import boto3
from boto3.exceptions import S3UploadFailedError
from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError

from config import DATA_DIR, S3_BUCKET, S3_PREFIX, S3_PROFILE

# (local dir under DATA_DIR, s3 folder name, files to upload from that dir)
SYNC_SPECS = [
    ("meta_data/synopsis_v2",       "synopsis",         ["synopses_extracted.parquet", "synopsis_progress.json"]),
    ("meta_data/film_meta",         "film_meta",         ["film_meta_enriched.parquet", "film_meta_progress.json"]),
    ("meta_data/cast_meta",         "cast_meta",         ["cast_enriched.parquet", "cast_progress.json"]),
    ("meta_data/director_meta",     "director_meta",     ["director_enriched.parquet", "director_progress.json"]),
    # No progress json — film_data_merge.py is stateless, re-derived fresh every run.
    ("meta_data/film_data_merged",  "film_data_merged",  ["film_data_merged.parquet"]),
]


def sync_meta_outputs_to_s3() -> dict:
    """Upload each file in SYNC_SPECS to s3://{S3_BUCKET}/{S3_PREFIX}/{s3_folder}/.

    Returns a summary dict (uploaded / skipped / failed file lists) — printed
    and also handed back so the Dagster asset can surface it in the run log.
    """
    if not S3_BUCKET:
        raise RuntimeError("config.yaml is missing an `s3.bucket` value")

    session = boto3.Session(profile_name=S3_PROFILE)
    s3 = session.client("s3")

    uploaded, skipped, failed = [], [], []

    for local_dir, s3_folder, filenames in SYNC_SPECS:
        for filename in filenames:
            local_path = DATA_DIR / local_dir / filename
            if not local_path.exists():
                print(f"  skip (not found): {local_path}")
                skipped.append(str(local_path))
                continue

            key = f"{S3_PREFIX}/{s3_folder}/{filename}"
            try:
                s3.upload_file(str(local_path), S3_BUCKET, key)
                size_mb = local_path.stat().st_size / 1_000_000
                print(f"  uploaded: {local_path} -> s3://{S3_BUCKET}/{key}  ({size_mb:.1f} MB)")
                uploaded.append(key)
            except NoCredentialsError:
                raise RuntimeError(
                    f"No valid AWS credentials for profile '{S3_PROFILE}'. "
                    "Run `stax2aws login` and retry."
                )
            except (ClientError, BotoCoreError, S3UploadFailedError) as e:
                if "ExpiredToken" in str(e) or "token has expired" in str(e):
                    raise RuntimeError(
                        f"AWS session for profile '{S3_PROFILE}' has expired. "
                        "Run `stax2aws login` and retry."
                    )
                print(f"  FAILED: {local_path} -> s3://{S3_BUCKET}/{key}  ({e})")
                failed.append(key)

    summary = {"uploaded": uploaded, "skipped": skipped, "failed": failed}
    print(f"\ns3_sync → {len(uploaded)} uploaded, {len(skipped)} skipped, {len(failed)} failed")
    if failed:
        raise RuntimeError(f"s3_sync: {len(failed)} file(s) failed to upload: {failed}")
    return summary


if __name__ == "__main__":
    sync_meta_outputs_to_s3()
