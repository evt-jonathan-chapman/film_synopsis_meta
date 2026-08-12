#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate venv
if [[ ! -f ".venv/bin/activate" ]]; then
    echo "ERROR: .venv not found. Run: uv venv --python 3.11 && uv pip install -r requirements.txt"
    exit 1
fi
source .venv/bin/activate

# Stax SSO session for the s3.profile in config.yaml expires hourly with no
# auto-refresh (see CLAUDE.md's "S3 sync" section) — stax_login.sh opens the
# browser for the SSO flow and blocks until you complete it there, so this
# only proceeds to the actual sync once a valid session exists.
echo "Refreshing Stax AWS session (opens browser for SSO login)..."
./stax_login.sh

echo ""
echo "Stax session refreshed — running S3 sync..."
python s3_sync.py
